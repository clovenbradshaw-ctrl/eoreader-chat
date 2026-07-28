#!/usr/bin/env node
/**
 * EO Reader Proxy — resilient, tool-calling proxy for Ollama
 *
 * Fixes over the original:
 * - Request body size limits (prevents OOM)
 * - Graceful shutdown (SIGTERM/SIGINT drains connections)
 * - Upstream retry with backoff (resilient to Ollama restarts)
 * - Bounded memory store with TTL eviction
 * - Async I/O in all paths (no fs.writeFileSync in request handlers)
 * - Request timeouts (no hanging connections)
 * - Streaming support (SSE for chat completions)
 * - Health check endpoint
 * - MCP client integration
 * - Tool-calling loop (function calling compatible)
 * - All I/O errors are caught per-operation (no single-point crash)
 *
 * Usage:
 *   node proxy.js [options]
 *
 * Options:
 *   --port=<n>        Proxy port (default: 11435)
 *   --target=<url>    Ollama endpoint (default: http://localhost:11434)
 *   --limit=<n>       Token limit for context assembly (default: 3000)
 *   --max-body=<n>    Max request body in bytes (default: 5242880)
 *   --store-ttl=<n>   Store entry TTL in ms (default: 3600000 = 1hr)
 *   --store-max=<n>   Max store entries (default: 10000)
 */

import http from "http";
import fs from "fs";
import fsp from "fs/promises";
import path from "path";

import { createModelRouter } from "./model-router.js";

// ── CLI args with validation ──

function parseArg(name, def, parse = (v) => v) {
  const idx = process.argv.findIndex(a => a === `--${name}` || a.startsWith(`--${name}=`));
  if (idx < 0) return def;
  const arg = process.argv[idx];
  const val = arg.includes("=") ? arg.split("=").slice(1).join("=") : process.argv[idx + 1];
  if (val === undefined || val.startsWith("--")) return def;
  try { return parse(val); } catch { return def; }
}

const REPO_PATH = parseArg("repo", process.cwd());
const PORT = parseArg("port", 11435, Number);
const TARGET = parseArg("target", "http://localhost:11434");
const TOKEN_LIMIT = parseArg("limit", 3000, Number);
const MAX_BODY = parseArg("max-body", 5_242_880, Number);
const STORE_TTL = parseArg("store-ttl", 3_600_000, Number);
const STORE_MAX = parseArg("store-max", 10_000, Number);
const MEMORY_DIR = path.join(REPO_PATH, "memory");

// ── Model routing ──
const TINY_MODEL = parseArg("tiny-model", "phi4-mini:latest");
const MEDIUM_MODEL = parseArg("medium-model", "qwen2.5-coder:7b");

function selectModel(messages) {
  const text = (messages || []).map(m => m.content || "").join(" ");
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  const charCount = text.length;

  // Complex signals → medium model
  const codePatterns = [/```/, /function\s/, /class\s/, /import\s/, /export\s/, /=>/, /\(\)\s*=>/];
  const hasCode = codePatterns.some(p => p.test(text));
  const isLong = wordCount > 50;
  const isTechnical = /\b(refactor|implement|debug|architecture|algorithm|database|api|endpoint|test|deploy)\b/i.test(text);
  const hasManyTurns = messages.length > 4;

  if (hasCode || isTechnical || (isLong && hasManyTurns)) {
    return MEDIUM_MODEL;
  }

  // Simple signals → tiny model
  const isShort = wordCount < 15;
  const isGreeting = /^(hi|hey|hello|yo|sup|howdy|thanks|ok|okay|cool|nice)\b/i.test(text.trim());

  if (isShort && isGreeting) {
    return TINY_MODEL;
  }

  // Default: use message ratio to decide
  return charCount < 200 ? TINY_MODEL : MEDIUM_MODEL;
}

// Learned routing: reuses eoreader5's predictive-competency substrate to
// pick between TINY_MODEL/MEDIUM_MODEL from measured tool-loop outcomes
// instead of the heuristic above. selectModel() remains the cold-start
// fallback (see model-router.js) and the deterministic override path when a
// caller explicitly requests a model.
let modelRouter;
try {
  modelRouter = createModelRouter({
    candidates: [TINY_MODEL, MEDIUM_MODEL],
    ledgerPath: path.join(MEMORY_DIR, "model-router-ledger.jsonl"),
    heuristicFallback: selectModel,
  });
} catch (err) {
  console.error(`[proxy] model-router unavailable, falling back to heuristic only: ${err.message}`);
  modelRouter = null;
}

// ── Retry helper ──

async function withRetry(fn, { label = "operation", maxRetries = 2, baseMs = 500 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt < maxRetries) {
        const delay = baseMs * Math.pow(2, attempt) + Math.random() * 200;
        console.error(`[proxy] ${label} failed (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay.toFixed(0)}ms: ${err.message}`);
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }
  throw lastErr;
}

// ── Safe fetch with timeout ──

async function safeFetch(url, options = {}, timeoutMs = 30_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...options, signal: controller.signal });
    return resp;
  } finally {
    clearTimeout(timer);
  }
}

// ── EO Cube ──

const OPERATORS = ["NUL", "SEG", "DEF", "SIG", "CON", "EVA", "INS", "SYN", "REC"];
const TERRAINS = ["Void", "Entity", "Kind", "Field", "Link", "Network", "Atmosphere", "Lens", "Paradigm"];
const STANCES = ["Clearing", "Dissecting", "Unraveling", "Tending", "Binding", "Tracing", "Cultivating", "Making", "Composing"];

const DIAGONAL = {
  NUL: [{ terrain: "Void", stance: "Clearing" }, { terrain: "Entity", stance: "Dissecting" }, { terrain: "Kind", stance: "Unraveling" }],
  SEG: [{ terrain: "Field", stance: "Clearing" }, { terrain: "Link", stance: "Dissecting" }, { terrain: "Network", stance: "Unraveling" }],
  DEF: [{ terrain: "Atmosphere", stance: "Clearing" }, { terrain: "Lens", stance: "Dissecting" }, { terrain: "Paradigm", stance: "Unraveling" }],
  SIG: [{ terrain: "Void", stance: "Tending" }, { terrain: "Entity", stance: "Binding" }, { terrain: "Kind", stance: "Tracing" }],
  CON: [{ terrain: "Field", stance: "Tending" }, { terrain: "Link", stance: "Binding" }, { terrain: "Network", stance: "Tracing" }],
  EVA: [{ terrain: "Atmosphere", stance: "Tending" }, { terrain: "Lens", stance: "Binding" }, { terrain: "Paradigm", stance: "Tracing" }],
  INS: [{ terrain: "Void", stance: "Cultivating" }, { terrain: "Entity", stance: "Making" }, { terrain: "Kind", stance: "Composing" }],
  SYN: [{ terrain: "Field", stance: "Cultivating" }, { terrain: "Link", stance: "Making" }, { terrain: "Network", stance: "Composing" }],
  REC: [{ terrain: "Atmosphere", stance: "Cultivating" }, { terrain: "Lens", stance: "Making" }, { terrain: "Paradigm", stance: "Composing" }],
};

function classifyCode(text) {
  if (/^export\s|^module\.exports/.test(text)) return "DEF";
  if (/import\s|require\(/.test(text)) return "SIG";
  if (/function\s|const\s.*=\s*\(/.test(text)) return "INS";
  if (/class\s/.test(text)) return "DEF";
  if (/if\s*\(|switch\s*\(/.test(text)) return "EVA";
  if (/\.\w+\(/.test(text)) return "CON";
  if (/\/\/|\/\*/.test(text)) return "SEG";
  return "NUL";
}

function classifyMessage(text) {
  if (/what|how|why|explain|describe/i.test(text)) return "EVA";
  if (/add|create|make|implement|write/i.test(text)) return "INS";
  if (/fix|bug|error|issue/i.test(text)) return "SEG";
  if (/connect|link|relate|depend/i.test(text)) return "CON";
  if (/define|specify|declare/i.test(text)) return "DEF";
  if (/remove|delete|clear/i.test(text)) return "NUL";
  if (/learn|understand|pattern/i.test(text)) return "REC";
  if (/synthesize|combine|merge/i.test(text)) return "SYN";
  return "SIG";
}

function getCell(operator) {
  const cells = DIAGONAL[operator] || DIAGONAL.NUL;
  return cells[1] || cells[0];
}

// ── Content Index (structural codebase index) ──

import { ContentIndex } from "./content-index.js";

let contentIndex = null;

async function buildContentIndex() {
  const repoRoots = [
    "/Users/mlacy/Documents/Default Project/eoreader5",
    "/Users/mlacy/Documents/Default Project/eoPriors",
    "/Users/mlacy/Documents/Default Project/eoreader4.2",
    "/Users/mlacy/Documents/Default Project/eoreader-chat",
    "/Users/mlacy/Documents/Default Project/eoreader-mcp",
    "/Users/mlacy/Documents/Default Project/eoreader-proxy",
    "/Users/mlacy/Documents/Default Project/eoreaderapp",
  ].filter(p => { try { return fs.statSync(p).isDirectory(); } catch { return false; } });

  const idx = new ContentIndex();
  console.error(`[proxy] Building content index from ${repoRoots.length} repos...`);
  await idx.scan(repoRoots);
  contentIndex = idx;
  console.error(`[proxy] Content index built: ${idx.totalFiles} files, ${idx.entities.size} entities, ${idx.definitions.size} definitions in ${idx.scanTime}ms`);
}

// ── Bounded memory store with TTL eviction ──

class BoundedStore {
  #entries = [];
  #max;
  #ttl;

  constructor(max = 10000, ttl = 3_600_000) {
    this.#max = max;
    this.#ttl = ttl;
  }

  #evict() {
    const cutoff = Date.now() - this.#ttl;
    this.#entries = this.#entries.filter(e => e.ts > cutoff);
    if (this.#entries.length > this.#max) {
      this.#entries.sort((a, b) => b.ts - a.ts);
      this.#entries = this.#entries.slice(0, this.#max);
    }
  }

  #hash(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) h = ((h << 5) - h) + str.charCodeAt(i) | 0;
    return (h >>> 0).toString(16).padStart(8, "0");
  }

  ingest(text, type, meta = {}) {
    this.#evict();
    const operator = type === "code" ? classifyCode(text) : classifyMessage(text);
    this.#entries.push({
      id: this.#hash(text + Date.now() + Math.random().toString(36).slice(2, 6)),
      text,
      type,
      meta,
      cell: { operator, terrain: getCell(operator).terrain, stance: getCell(operator).stance },
      ts: Date.now(),
    });
  }

  search(query, topK = 5) {
    this.#evict();
    const queryOperator = classifyMessage(query);
    const qc = getCell(queryOperator);
    const words = (query || "").toLowerCase().split(/\s+/).filter(w => w.length > 1);

    return this.#entries.map(e => {
      const text = e.text.toLowerCase();
      let score = 0;
      for (const w of words) {
        if (text.includes(w)) score += 2;
        for (const t of text.split(/\s+/)) {
          if (t === w) score += 1;
          else if (t.includes(w) || w.includes(t)) score += 0.5;
        }
      }
      if (e.cell) {
        if (e.cell.terrain === qc.terrain) score += 3;
        if (e.cell.stance === qc.stance) score += 2;
      }
      return { ...e, score };
    })
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
  }

  get size() { return this.#entries.length; }
}

const store = new BoundedStore(STORE_MAX, STORE_TTL);

// ── Load code (async, error-isolated per file) ──

async function loadCode(repo) {
  const ignore = new Set(["node_modules", ".git", "dist", "build", "__pycache__", ".opencode"]);
  const skip = new Set([".json", ".lock", ".map", ".png", ".jpg", ".gif", ".ico", ".svg", ".woff", ".woff2", ".mp3", ".mp4", ".wasm"]);

  async function walk(dir) {
    let entries;
    try { entries = await fsp.readdir(dir, { withFileTypes: true }); }
    catch { return; }

    const tasks = entries.map(async e => {
      if (ignore.has(e.name)) return;
      const p = path.join(dir, e.name);
      try {
        if (e.isDirectory()) { await walk(p); return; }
        if (!e.isFile()) return;
        const ext = path.extname(e.name);
        if (skip.has(ext) || e.name.includes(".test.") || e.name.includes(".spec.")) return;

        const content = await fsp.readFile(p, "utf8");
        if (content.length < 20) return;
        const rel = path.relative(repo, p);
        const lines = content.split("\n");
        let chunk = [], size = 0;
        for (const line of lines) {
          chunk.push(line);
          size += line.length;
          if (size > 1200 || /^(module\.exports|export\s)/.test(line)) {
            const text = chunk.join("\n").trim();
            if (text.length > 30) store.ingest(text, "code", { file: rel });
            chunk = []; size = 0;
          }
        }
        if (chunk.length > 0) {
          const text = chunk.join("\n").trim();
          if (text.length > 30) store.ingest(text, "code", { file: rel });
        }
      } catch {}
    });
    await Promise.all(tasks);
  }

  console.error(`[proxy] Loading ${repo}...`);
  const start = Date.now();
  await walk(repo);
  console.error(`[proxy] ${store.size} chunks loaded in ${Date.now() - start}ms`);
}

// ── URL fetch + snapshot ──

const SNAPSHOT_MAX_CHARS = 1400;

function detectUrls(text) {
  return [...new Set((text || "").match(/https?:\/\/[^\s<>"')\]]+/g) || [])];
}

function contentSnapshot(text, url) {
  if (!text || text.length < 20) return `[Empty content from ${url}]`;
  const lines = text.split(/\n+/).map(l => l.trim()).filter(l => l.length > 0);
  const title = lines[0] || '';
  const body = lines.slice(1).join(' ');
  let excerpt = body.slice(0, SNAPSHOT_MAX_CHARS);
  const lastPeriod = excerpt.lastIndexOf('.');
  if (lastPeriod > SNAPSHOT_MAX_CHARS * 0.6) excerpt = excerpt.slice(0, lastPeriod + 1);
  const wordCount = body.split(/\s+/).length;
  return [`[Source: ${url}]`, title ? `[Title: ${title}]` : '', `[${wordCount} words — excerpt below]`, '', excerpt].filter(Boolean).join('\n');
}

async function fetchAndSaveUrl(url) {
  try {
    const resp = await safeFetch(url, {
      headers: { "User-Agent": "EOReader-Proxy/2.0" },
    }, 15000);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
    const contentType = resp.headers.get("content-type") || "";
    let text = await resp.text();

    if (contentType.includes("text/html") || contentType.includes("application/xhtml") || text.trim().startsWith("<")) {
      text = text
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
        .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
        .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, "")
        .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
        .replace(/<!--[\s\S]*?-->/g, "")
        .replace(/<[^>]+>/g, " ")
        .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
        .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
        .replace(/\s+/g, " ").trim();
    }

    const hostname = new URL(url).hostname.replace(/^www\./, '');
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `url_${hostname}_${ts}.txt`;
    const filepath = path.join(MEMORY_DIR, filename);

    try {
      await fsp.mkdir(MEMORY_DIR, { recursive: true });
      await fsp.writeFile(filepath, text, "utf8");
    } catch {}
    console.error(`[proxy] Saved ${filename} (${text.length} chars)`);
    return { text, filename };
  } catch (e) {
    console.error(`[proxy] Fetch failed ${url}: ${e.message}`);
    return { text: null, filename: null, error: e.message };
  }
}

// ── Context assembly ──

const tok = (t) => Math.ceil((t || "").length / 3.5);

async function assemble(messages) {
  const latest = [...messages].reverse().find(m => m.role === "user");
  if (!latest) return messages;
  const query = latest.content || "";
  const qc = getCell(classifyMessage(query));

  let ctx = [], t = 0;
  const sys = messages.find(m => m.role === "system");
  if (sys) { t += tok(sys.content); ctx.push(sys); }
  else {
    const d = "You are EO, a focused research and engineering assistant. Use the available code and context.";
    t += tok(d); ctx.push({ role: "system", content: d });
  }

  // Content index enrichment: if query looks like codebase exploration,
  // add structural context from the content index
  if (contentIndex && (query.includes("modules") || query.includes("packages") || query.includes("entity") || query.includes("organ") || query.includes("engine") || query.includes("index") || query.includes("search") || query.includes("presence") || query.includes("store") || query.includes("fold") || query.includes("discourse") || query.includes("spine") || query.includes("reaction"))) {
    const codeResults = contentIndex.find(query, { limit: 5 });
    if (codeResults.length > 0) {
      const ctxStr = "\n[Codebase context from content index]\n" + codeResults.slice(0, 5).map(r => {
        const parts = [`[${r.type}] ${r.name || r.path}`];
        if (r.repo) parts.push(`(${r.repo})`);
        if (r.line) parts.push(`:${r.line}`);
        if (r.header) parts.push(` — ${r.header.slice(0, 100)}`);
        return parts.join(" ");
      }).join("\n");
      if (t + tok(ctxStr) < TOKEN_LIMIT) { t += tok(ctxStr); ctx.push({ role: "system", content: ctxStr }); }
    }
  }

  // URL fetch (isolated per URL — one failure doesn't block others)
  const urls = detectUrls(query);
  if (urls.length > 0) {
    const results = await Promise.allSettled(urls.map(url => fetchAndSaveUrl(url)));
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      if (r.status === "fulfilled" && r.value.text) {
        const snapshot = contentSnapshot(r.value.text, urls[i]);
        if (t + tok(snapshot) < TOKEN_LIMIT) {
          t += tok(snapshot);
          ctx.push({ role: "system", content: snapshot });
        }
      }
    }
  }

  const results = store.search(query, 5);
  if (results.length) {
    const c = "\n[Context: " + qc.terrain + "/" + qc.stance + "]\n" +
      results.map(r => r.type === "code"
        ? `--- ${r.meta.file || "?"} ---\n${r.text.slice(0, 500)}`
        : `[${r.type}]: ${r.text.slice(0, 300)}`
      ).join("\n\n");
    if (t + tok(c) < TOKEN_LIMIT) { t += tok(c); ctx.push({ role: "system", content: c }); }
  }

  ctx.push({ role: "user", content: query });
  console.error(`[proxy] ${t} tokens, terrain=${qc.terrain}/${qc.stance}`);
  return ctx;
}

// ── Tool definitions (OpenAI-compatible function calling) ──

const TOOL_DEFINITIONS = [
  {
    type: "function",
    function: {
      name: "bash",
      description: "Execute a shell command. Returns stdout and stderr.",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "The bash command to execute" },
          workdir: { type: "string", description: "Working directory (default cwd)" },
          timeout: { type: "number", description: "Timeout in ms (default 30000)" },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read a file from disk and return its contents.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Absolute path to the file" },
          offset: { type: "number", description: "Line offset (1-indexed)" },
          limit: { type: "number", description: "Max lines to return" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description: "Write content to a file on disk.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Absolute path to write to" },
          content: { type: "string", description: "Content to write" },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "edit_file",
      description: "Edit a file by replacing exact string matches.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Absolute path to the file" },
          old_string: { type: "string", description: "Text to replace" },
          new_string: { type: "string", description: "Replacement text" },
        },
        required: ["path", "old_string", "new_string"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "glob",
      description: "Find files by glob pattern.",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "Glob pattern (e.g. **/*.js)" },
          path: { type: "string", description: "Root directory (default cwd)" },
        },
        required: ["pattern"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "grep",
      description: "Search file contents with regex.",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "Regex pattern" },
          include: { type: "string", description: "File glob filter (e.g. *.js)" },
          path: { type: "string", description: "Root directory (default cwd)" },
        },
        required: ["pattern"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web for information.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          count: { type: "number", description: "Number of results (default 5)" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_fetch",
      description: "Fetch a URL and return its content.",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string", description: "URL to fetch" },
        },
        required: ["url"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "ls",
      description: "List contents of a directory.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Directory path (default .)" },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "ingest",
      description: "Read a file or directory into the engine's memory store.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "File or directory path to ingest" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_memory",
      description: "Search the engine's memory store for relevant context.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "number", description: "Max results (default 5)" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "memory_stats",
      description: "Get memory store statistics.",
      parameters: { type: "object", properties: {} },
    },
  },

  // ── Content Index tools (high-level codebase traversal) ──

  {
    type: "function",
    function: {
      name: "codebase_structure",
      description: "Show the tree structure of the codebase. Optionally filter by path prefix to zoom into a package or directory.",
      parameters: {
        type: "object",
        properties: {
          prefix: { type: "string", description: "Path prefix to filter (e.g. 'packages/engine/search' or 'packages/engine/emergence/store'). Omit for root." },
          depth: { type: "number", description: "Max depth to show (default: all)" },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "codebase_find",
      description: "Find definitions, exports, modules, and content matches by name across the entire codebase. Returns ranked results with paths and line numbers.",
      parameters: {
        type: "object",
        properties: {
          term: { type: "string", description: "Name or term to search for (function name, class name, file name fragment)" },
          limit: { type: "number", description: "Max results (default 20)" },
        },
        required: ["term"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "codebase_lookup",
      description: "Get detailed information about a specific module file: its imports, exports, definitions, entities implemented, and cross-references.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Relative path to the module (e.g. 'eoreader5/packages/engine/search/index.js' or 'packages/engine/emergence/store/index.js')" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "codebase_search",
      description: "Full-text search across the entire codebase. Searches both source text and structural metadata (exports, definitions, module headers). Returns ranked module-level results.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query (terms or phrase)" },
          limit: { type: "number", description: "Max results (default 20)" },
          repo: { type: "string", description: "Filter by repo name (e.g. 'eoreader5', 'eoPriors')" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "codebase_related",
      description: "Show what a module imports, what imports it, and what entities it implements. Good for understanding dependencies and impact analysis.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Relative path to the module" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "codebase_entities",
      description: "List all eoreader5 conceptual entities (cube, presence, fold, store, discourse, spine, reaction, etc.) mapped to their implementation files.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "codebase_api",
      description: "Show the API surface (exports + definitions) for a package or directory.",
      parameters: {
        type: "object",
        properties: {
          prefix: { type: "string", description: "Package or directory prefix (e.g. 'packages/engine/search')" },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "codebase_summary",
      description: "Show overall stats: repos, files, entities, definitions.",
      parameters: { type: "object", properties: {} },
    },
  },
];

// ── Tool handlers ──

function _renderTree(node, maxDepth, indent) {
  if (maxDepth <= 0) return indent + "...";
  if (node.type === "file") {
    const tags = [];
    if (node.entities?.length) tags.push(`[${node.entities.join(", ")}]`);
    if (node.exports > 0) tags.push(`${node.exports} exports`);
    if (node.defs > 0) tags.push(`${node.defs} defs`);
    const tagStr = tags.length ? ` ${tags.join(" ")}` : "";
    return `${indent}${node.name}${tagStr}`;
  }
  const parts = [`${indent}${node.name}/`];
  if (node.children) {
    const sorted = [...node.children].sort((a, b) => {
      if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    for (const child of sorted) {
      parts.push(_renderTree(child, maxDepth - 1, indent + "  "));
    }
  }
  return parts.join("\n");
}

const toolHandlers = {
  async bash(args) {
    const { execSync } = await import("child_process");
    try {
      const out = execSync(args.command, {
        cwd: args.workdir || process.cwd(),
        timeout: args.timeout || 30000,
        encoding: "utf8",
        maxBuffer: 10_485_760,
        stdio: ["pipe", "pipe", "pipe"],
      });
      return out || "(no output)";
    } catch (err) {
      const std = err.stdout || "";
      const errOut = err.stderr || err.message;
      return std ? `${std}\n\nSTDERR:\n${errOut}` : `Error: ${errOut}`;
    }
  },

  async read_file(args) {
    try {
      const content = await fsp.readFile(args.path, "utf8");
      const lines = content.split("\n");
      const start = args.offset ? Math.max(0, args.offset - 1) : 0;
      const end = args.limit ? start + args.limit : lines.length;
      return lines.slice(start, end).join("\n");
    } catch (err) {
      return `[Error reading ${args.path}: ${err.message}]`;
    }
  },

  async write_file(args) {
    try {
      await fsp.mkdir(path.dirname(args.path), { recursive: true });
      await fsp.writeFile(args.path, args.content, "utf8");
      return `Written ${args.content.length} bytes to ${args.path}`;
    } catch (err) {
      return `[Error writing ${args.path}: ${err.message}]`;
    }
  },

  async edit_file(args) {
    try {
      const content = await fsp.readFile(args.path, "utf8");
      if (!content.includes(args.old_string)) {
        return `[Error: old_string not found in ${args.path}]`;
      }
      const updated = content.replace(args.old_string, args.new_string);
      await fsp.writeFile(args.path, updated, "utf8");
      return `Edited ${args.path}`;
    } catch (err) {
      return `[Error editing ${args.path}: ${err.message}]`;
    }
  },

  async glob(args) {
    const { globSync } = await import("glob");
    try {
      return globSync(args.pattern, { cwd: args.path || process.cwd() }).join("\n") || "(no matches)";
    } catch (err) {
      return `[Error: ${err.message}]`;
    }
  },

  async grep(args) {
    const { execSync } = await import("child_process");
    const cmd = `rg --no-heading -n ${args.include ? `-g '${args.include}'` : ''} '${args.pattern}' ${args.path || '.'}`;
    try {
      return execSync(cmd, { encoding: "utf8", timeout: 15000, maxBuffer: 5_242_880 }) || "(no matches)";
    } catch (err) {
      if (err.status === 1) return "(no matches)";
      return `[Error: ${err.message}]`;
    }
  },

  async web_search(args) {
    const { execSync } = await import("child_process");
    try {
      const result = execSync(
        `curl -s "https://html.duckduckgo.com/html/?q=${encodeURIComponent(args.query)}" 2>/dev/null | sed -n 's/.*<a[^>]*class="result__a"[^>]*>//p' | sed 's/<\\/a>//' | head -${args.count || 5}`,
        { encoding: "utf8", timeout: 15000, maxBuffer: 5_242_880 }
      );
      return result || "(no results)";
    } catch {
      try {
        const resp = await safeFetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(args.query)}&format=json`, {}, 10000);
        const data = await resp.json();
        return data.AbstractText || data.RelatedTopics?.slice(0, args.count || 5).map(t => t.Text || t).join("\n") || "(no results)";
      } catch (err) {
        return `[Search failed: ${err.message}]`;
      }
    }
  },

  async web_fetch(args) {
    try {
      const resp = await safeFetch(args.url, { headers: { "User-Agent": "EOReader/2.0" } }, 15000);
      const text = await resp.text();
      const isHtml = (resp.headers.get("content-type") || "").includes("text/html") || text.trim().startsWith("<");
      if (isHtml) {
        return text
          .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
          .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
          .replace(/<[^>]+>/g, " ")
          .replace(/\s+/g, " ").trim().slice(0, 10000);
      }
      return text.slice(0, 10000);
    } catch (err) {
      return `[Error fetching ${args.url}: ${err.message}]`;
    }
  },

  async ls(args) {
    try {
      const entries = await fsp.readdir(args.path || ".", { withFileTypes: true });
      return entries.map(e => {
        let size = "";
        if (e.isFile()) try { size = ` ${fs.statSync(path.join(args.path || ".", e.name)).size}`; } catch {}
        return `${e.isDirectory() ? "d" : "-"} ${size.padStart(10)}  ${e.name}`;
      }).join("\n");
    } catch (err) {
      return `[Error: ${err.message}]`;
    }
  },

  async ingest(args) {
    try {
      const stats = fs.statSync(args.path);
      if (stats.isDirectory()) { await loadCode(args.path); return `Ingested directory ${args.path}`; }
      const content = await fsp.readFile(args.path, "utf8");
      store.ingest(content, "file", { path: args.path });
      return `Ingested ${args.path} (${content.length} chars)`;
    } catch (err) {
      return `[Error ingesting ${args.path}: ${err.message}]`;
    }
  },

  async search_memory(args) {
    const results = store.search(args.query, args.limit || 5);
    if (!results.length) return "(no matches in memory)";
    return results.map((r, i) =>
      `--- ${i + 1}. (score: ${r.score.toFixed(2)}) ${r.meta.file || r.meta.path || "?"} ---\n${r.text.slice(0, 500)}`
    ).join("\n\n");
  },

  async memory_stats() {
    return JSON.stringify({ entries: store.size, max: STORE_MAX, ttl_ms: STORE_TTL });
  },

  // ── Content Index handlers ──

  async codebase_structure(args) {
    if (!contentIndex) return "[Content index not built]";
    const tree = contentIndex.structure(args.prefix);
    return _renderTree(tree, args.depth || 99, "");
  },

  async codebase_find(args) {
    if (!contentIndex) return "[Content index not built]";
    const results = contentIndex.find(args.term, { limit: args.limit || 20 });
    if (!results.length) return `No matches for "${args.term}"`;
    return results.map((r, i) => {
      const line = [`${i + 1}. [${r.type}] ${r.name || r.path}`];
      if (r.repo) line.push(`     Repo: ${r.repo}/${r.path}`);
      if (r.line) line.push(`     Line: ${r.line}`);
      if (r.excerpt) line.push(`     ${r.excerpt}`);
      if (r.header) line.push(`     ${r.header.slice(0, 120)}`);
      if (r.entities && r.entities.length) line.push(`     Entities: ${r.entities.join(", ")}`);
      if (r.description) line.push(`     ${r.description.slice(0, 200)}`);
      if (r.files) line.push(`     Files: ${r.files.map(f => f.path).join(", ")}`);
      if (r.context) line.push(`     ...${r.context.slice(0, 200)}...`);
      return line.join("\n");
    }).join("\n\n");
  },

  async codebase_lookup(args) {
    if (!contentIndex) return "[Content index not built]";
    const mod = contentIndex.lookup(args.path);
    if (!mod) return `Module not found: ${args.path}`;
    const lines = [
      `Module: ${mod.repoRel}`,
      `Repo: ${mod.repoName}  Pkg: ${mod.pkgName}`,
      `Size: ${mod.size} bytes, ${mod.lines} lines`,
      `Header: ${mod.header || "(none)"}`,
    ];
    if (mod.entities?.length) lines.push(`Entities: ${mod.entities.join(", ")}`);
    if (mod.definitions?.length) lines.push(`Definitions: ${mod.definitions.map(d => `${d.name} (${d.type}:${d.line})`).join(", ")}`);
    if (mod.exports?.length) lines.push(`Exports: ${mod.exports.map(e => `${e.name} (${e.type}:${e.line})`).join(", ")}`);
    if (mod.imports?.length) lines.push(`Imports: ${mod.imports.map(i => i.spec).join(", ")}`);
    if (mod.importedBy?.length) lines.push(`Imported by: ${mod.importedBy.map(i => i.by).join(", ")}`);
    return lines.join("\n");
  },

  async codebase_search(args) {
    if (!contentIndex) return "[Content index not built]";
    const results = contentIndex.search(args.query, { limit: args.limit || 20, repo: args.repo });
    if (!results.length) return `No matches for "${args.query}"`;
    return results.map((r, i) => {
      const parts = [`${i + 1}. ${r.path} (score: ${r.score})`];
      if (r.header) parts.push(`     ${r.header.slice(0, 120)}`);
      if (r.entities?.length) parts.push(`     Entities: ${r.entities.join(", ")}`);
      parts.push(`     ${r.definitions} defs, ${r.exports} exports, ${r.lines} lines`);
      return parts.join("\n");
    }).join("\n\n");
  },

  async codebase_related(args) {
    if (!contentIndex) return "[Content index not built]";
    const rel = contentIndex.related(args.path);
    if (rel.error) return rel.error;
    const lines = [
      `=== ${rel.module.repoRel} ===`,
      `Header: ${rel.module.header || "(none)"}`,
      `Entities: ${(rel.module.entities || []).join(", ") || "none"}`,
    ];
    if (rel.imports?.length) {
      lines.push(`\nImports:`);
      for (const m of rel.imports.slice(0, 15)) lines.push(`  ${m.repoRel} — ${(m.header || "").slice(0, 80)}`);
    }
    if (rel.importedBy?.length) {
      lines.push(`\nImported by:`);
      for (const m of rel.importedBy.slice(0, 15)) lines.push(`  ${m.repoRel} — ${(m.header || "").slice(0, 80)}`);
    }
    if (rel.entities?.length) {
      lines.push(`\nEntity implementations:`);
      for (const e of rel.entities) {
        lines.push(`  ${e.name}: ${e.description || "(no description)"}`);
        for (const f of e.files) lines.push(`    → ${f.path}`);
      }
    }
    return lines.join("\n");
  },

  async codebase_entities() {
    if (!contentIndex) return "[Content index not built]";
    const ents = contentIndex.entityIndex();
    const names = Object.keys(ents).sort();
    return names.map(name => {
      const e = ents[name];
      const desc = e.description ? `— ${e.description.slice(0, 150)}` : "";
      const files = e.files.map(f => `  ${f.repo}:${f.path}`).join("\n");
      return `${name} ${desc}\n${files}`;
    }).join("\n\n");
  },

  async codebase_api(args) {
    if (!contentIndex) return "[Content index not built]";
    const apis = contentIndex.apiSurface(args.prefix);
    if (!apis.length) return `No API surface found for prefix: ${args.prefix}`;
    return apis.map(a => {
      const parts = [`${a.path}`];
      if (a.entities?.length) parts.push(`  Entities: ${a.entities.join(", ")}`);
      if (a.exports?.length) parts.push(`  Exports: ${a.exports.map(e => `${e.name}(${e.type})`).join(", ")}`);
      if (a.definitions?.length) parts.push(`  Defs: ${a.definitions.map(d => `${d.name}(${d.type})`).join(", ")}`);
      return parts.join("\n");
    }).join("\n\n");
  },

  async codebase_summary() {
    if (!contentIndex) return "[Content index not built]";
    const s = contentIndex.summary();
    const lines = [
      `Content Index Summary`,
      `Scan time: ${s.scanTime}ms`,
      `Total files: ${s.totalFiles}`,
      `Total entities: ${s.totalEntities}`,
      `Total definitions: ${s.totalDefinitions}`,
      ``,
      `Repos:`,
    ];
    for (const r of s.repos) {
      lines.push(`  ${r.name}: ${r.files} files, ${r.lines} lines, ${r.packages} packages`);
      lines.push(`    ${(r.description || "").slice(0, 120)}`);
    }
    return lines.join("\n");
  },
};

// ── MCP Client ──

const mcpClients = new Map(); // serverName -> { tools, callTool }

async function connectMcpServer(name, command, args = []) {
  if (mcpClients.has(name)) return mcpClients.get(name).tools;
  try {
    const { spawn } = await import("child_process");
    const proc = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    let buf = "";
    let pending = new Map();
    let msgId = 0;

    proc.stdout.on("data", chunk => {
      buf += chunk.toString();
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.id && pending.has(msg.id)) {
            const { resolve, reject } = pending.get(msg.id);
            pending.delete(msg.id);
            if (msg.error) reject(new Error(msg.error.message));
            else resolve(msg.result);
          }
        } catch {}
      }
    });

    proc.stderr.on("data", chunk => {
      const text = chunk.toString().trim();
      if (text) console.error(`[mcp:${name}] ${text}`);
    });

    proc.on("exit", (code) => {
      console.error(`[mcp:${name}] exited with code ${code}`);
      mcpClients.delete(name);
      for (const [, { reject }] of pending) reject(new Error("MCP server exited"));
    });

    async function send(method, params = {}) {
      const id = ++msgId;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        const msg = JSON.stringify({ jsonrpc: "2.0", id, method, params });
        proc.stdin.write(msg + "\n");
        setTimeout(() => {
          if (pending.has(id)) {
            pending.delete(id);
            reject(new Error(`MCP request ${method} timed out`));
          }
        }, 30000);
      });
    }

    // Initialize
    const init = await send("initialize", {
      protocolVersion: "0.1.0",
      capabilities: {},
      clientInfo: { name: "eo-proxy", version: "2.0" },
    });

    // Get tools
    const toolResult = await send("tools/list");
    const tools = (toolResult?.tools || []).map(t => ({
      type: "function",
      function: {
        name: `mcp_${name}_${t.name}`,
        description: `[MCP:${name}] ${t.description || t.name}`,
        parameters: t.inputSchema || { type: "object", properties: {} },
      },
    }));

    const client = {
      tools,
      async callTool(toolName, args) {
        const mcpName = toolName.replace(/^mcp_/, "");
        const result = await send("tools/call", { name: mcpName, arguments: args });
        const text = result?.content?.[0]?.text || result?.content?.[0] || JSON.stringify(result || {});
        return text;
      },
      async close() {
        proc.stdin.end();
        proc.kill();
      },
    };

    mcpClients.set(name, client);
    console.error(`[mcp] Connected: ${name} (${tools.length} tools)`);
    return tools;
  } catch (err) {
    console.error(`[mcp] Failed to connect ${name}: ${err.message}`);
    return [];
  }
}

async function getAllTools() {
  const tools = [...TOOL_DEFINITIONS];
  for (const [, client] of mcpClients) {
    tools.push(...client.tools);
  }
  return tools;
}

// ── Tool calling loop ──

async function runToolLoop(messages, tools, onEvent = null, maxRounds = 8, forceModel = null) {
  const effectiveTools = tools && tools.length > 0 ? tools : await getAllTools();

  // An explicit forceModel is a deliberate human/client override — it wins
  // outright and never enters the learned-routing ledger (there'd be no
  // honest "the router chose this" claim to score).
  let model, routerCtx;
  if (forceModel) {
    model = forceModel;
    routerCtx = null;
  } else if (modelRouter) {
    ({ model, ctx: routerCtx } = modelRouter.pick(messages));
  } else {
    model = selectModel(messages);
    routerCtx = null;
  }

  const revealOutcome = async (outcome) => {
    if (!routerCtx) return;
    try { await modelRouter.reveal(routerCtx, outcome); }
    catch (err) { console.error(`[proxy] model-router reveal failed: ${err.message}`); }
  };

  try {
  for (let round = 0; round < maxRounds; round++) {
    const body = {
      model,
      messages,
      tools: effectiveTools,
      stream: false,
      options: { temperature: 0.7, num_predict: 4096 },
    };

    if (onEvent) onEvent({ type: "llm_call", round, tools: effectiveTools.length, model });

    const resp = await withRetry(() => safeFetch(`${TARGET}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }, 120000), { label: "Ollama chat", maxRetries: 2 });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => resp.statusText);
      throw new Error(`Ollama ${resp.status}: ${errText}`);
    }

    const data = await resp.json();
    const msg = data.message || {};

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      messages.push({ role: "assistant", content: msg.content || "" });
      if (onEvent) onEvent({ type: "response", content: msg.content || "", model });
      await revealOutcome("success");
      return msg.content || "";
    }

    const toolCalls = msg.tool_calls.map(tc => ({
      id: tc.id || `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      type: "function",
      function: { name: tc.function?.name || tc.name, arguments: typeof tc.function?.arguments === 'string' ? tc.function.arguments : JSON.stringify(tc.function?.arguments || {}) },
    }));

    if (onEvent) onEvent({ type: "tool_calls", calls: toolCalls.map(tc => ({ name: tc.function.name, args: tc.function.arguments })) });

    messages.push({ role: "assistant", content: msg.content || "", tool_calls: toolCalls });

    for (const tc of msg.tool_calls) {
      const name = tc.function?.name || tc.name;
      let args = {};
      try { args = typeof tc.function?.arguments === 'string' ? JSON.parse(tc.function.arguments) : (tc.function?.arguments || {}); } catch {}

      console.error(`[proxy] Tool call: ${name}(${JSON.stringify(args).slice(0, 200)})`);

      let result;
      const isMcp = name.startsWith("mcp_");

      if (isMcp) {
        const serverName = name.split("_")[1];
        const toolName = name.split("_").slice(2).join("_");
        const client = mcpClients.get(serverName);
        if (client) {
          try { result = await client.callTool(name, args); }
          catch (err) { result = `[MCP error: ${err.message}]`; }
        } else {
          result = `[MCP server "${serverName}" not connected]`;
        }
      } else {
        const handler = toolHandlers[name];
        if (handler) {
          try { result = await handler(args); }
          catch (err) { result = `[Error calling ${name}: ${err.message}]`; }
        } else {
          result = `[Unknown tool: ${name}]`;
        }
      }

      const resultStr = typeof result === 'string' ? result : JSON.stringify(result);
      store.ingest(`Tool ${name}: ${resultStr.slice(0, 300)}`, "tool", { name });

      if (onEvent) onEvent({ type: "tool_result", name, result: resultStr.slice(0, 500) });

      messages.push({
        role: "tool",
        content: resultStr.slice(0, 10000),
        tool_call_id: tc.id || `call_${Date.now()}`,
      });
    }
  }

  const last = messages[messages.length - 1];
  if (last?.role === "tool") {
    messages.push({ role: "assistant", content: "[Max tool rounds reached. Please continue based on the results above.]" });
  }
  const finalContent = messages[messages.length - 1]?.content || "";
  if (onEvent) onEvent({ type: "response", content: finalContent, model });
  await revealOutcome("failure");
  return finalContent;
  } catch (err) {
    await revealOutcome("failure");
    throw err;
  }
}

// ── Streaming tool-calling endpoint (SSE) ──

async function handleToolStream(res, messages, tools, forceModel = null) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const sendSSE = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  // Push tool definitions first
  const effectiveTools = tools && tools.length > 0 ? tools : await getAllTools();
  sendSSE("tools_available", { count: effectiveTools.length });

  try {
    const content = await runToolLoop(messages, effectiveTools, (evt) => {
      sendSSE(evt.type, evt);
    }, 8, forceModel);
    sendSSE("done", { content });
  } catch (err) {
    sendSSE("error", { message: err.message });
  }
  res.end();
}

// ── Server ──

let connections = new Set();
let shuttingDown = false;

const server = http.createServer((req, res) => {
  // Track connection for graceful shutdown
  connections.add(res);
  res.on("close", () => connections.delete(res));

  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") { res.writeHead(200); res.end(); return; }

  // Health check
  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", store_size: store.size, uptime: process.uptime().toFixed(1) }));
    return;
  }

  // Stats
  if (req.method === "GET" && req.url === "/stats") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ store_size: store.size, max_store: STORE_MAX, ttl_seconds: STORE_TTL / 1000, uptime: process.uptime().toFixed(1) }));
    return;
  }

  // List available models (tiny/medium routing tiers)
  if (req.method === "GET" && req.url === "/v1/models") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ tiny: TINY_MODEL, medium: MEDIUM_MODEL }));
    return;
  }

  // List MCP servers
  if (req.method === "GET" && req.url === "/mcp/servers") {
    const servers = [];
    for (const [name, client] of mcpClients) {
      servers.push({ name, tools: client.tools.length });
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ servers }));
    return;
  }

  // Connect an MCP server
  if (req.method === "POST" && req.url === "/mcp/connect") {
    let body = "";
    req.on("data", c => { body += c; });
    req.on("end", async () => {
      try {
        const { name, command, args } = JSON.parse(body);
        const tools = await connectMcpServer(name, command, args || []);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ connected: true, name, tools }));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // Streaming tool-calling endpoint
  if (req.method === "POST" && req.url === "/api/chat/tools") {
    let body = "";
    let bodySize = 0;
    req.on("data", chunk => {
      bodySize += chunk.length;
      if (bodySize > MAX_BODY) { req.destroy(new Error("Request body too large")); return; }
      body += chunk.toString("utf8");
    });
    req.on("end", async () => {
      try {
        const data = JSON.parse(body);
        const messages = data.messages || [];
        const tools = data.tools;

        // Ingest user input
        for (const m of messages) {
          if (m.content?.length > 5 && m.role === "user") store.ingest(m.content, m.role, {});
        }

        await handleToolStream(res, messages, tools, data.model || null);
      } catch (err) {
        if (!res.headersSent) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err.message }));
        }
      }
    });
    return;
  }

  // Chat completions (with tool calling)
  if (req.method === "POST" && (req.url === "/v1/chat/completions" || req.url === "/api/chat")) {
    let body = "";
    let bodySize = 0;

    req.on("data", chunk => {
      bodySize += chunk.length;
      if (bodySize > MAX_BODY) {
        req.destroy(new Error("Request body too large"));
        return;
      }
      body += chunk.toString("utf8");
    });

    req.on("end", async () => {
      let data;
      try { data = JSON.parse(body); } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: `Invalid JSON: ${e.message}` }));
        return;
      }

      // Tools: use provided tools or default to our internal tools
      const tools = data.tools && data.tools.length > 0 ? data.tools : undefined;
      const useToolLoop = tools || data.use_tools;

      try {
        if (useToolLoop) {
          // Non-streaming tool loop
          const result = await runToolLoop(
            data.messages || [],
            tools || TOOL_DEFINITIONS
          );

          // Ingest for memory
          for (const m of data.messages || []) {
            if (m.content?.length > 5) store.ingest(m.content, m.role, {});
          }
          if (result.length > 5) store.ingest(result, "assistant", {});

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            id: `chat-${Date.now()}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: data.model || "llama3.2",
            choices: [{
              index: 0,
              message: { role: "assistant", content: result },
              finish_reason: "stop",
            }],
          }));
        } else if (data.stream) {
          // Streaming passthrough with model routing
          const targetUrl = req.url === "/api/chat" ? `${TARGET}/api/chat` : `${TARGET}/v1/chat/completions`;
          // Remove use_tools before forwarding, route model
          const { use_tools, ...forwardData } = data;
          forwardData.model = selectModel(forwardData.messages || []);
          const upstreamResp = await withRetry(() => safeFetch(targetUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(forwardData),
          }, 120000), { label: "Ollama stream", maxRetries: 1 });

          res.writeHead(upstreamResp.status, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
          });

          const reader = upstreamResp.body.getReader();
          const decoder = new TextDecoder();
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              res.write(decoder.decode(value, { stream: true }));
            }
          } catch (err) {
            console.error(`[proxy] Stream error: ${err.message}`);
          }
          res.end();
        } else {
          // Non-streaming passthrough with model routing
          const { use_tools, ...forwardData } = data;
          if (!forwardData.messages) forwardData.messages = [];

          // Route model intelligently
          forwardData.model = selectModel(forwardData.messages);

          // Assemble context (ingest, search memory)
          forwardData.messages = await assemble(forwardData.messages);

          const upstreamResp = await withRetry(() => safeFetch(`${TARGET}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...forwardData, stream: false }),
          }, 120000), { label: "Ollama chat", maxRetries: 2 });

          const upstreamText = await upstreamResp.text();
          try {
            const parsed = JSON.parse(upstreamText);
            const content = parsed.message?.content || parsed.choices?.[0]?.message?.content || "";
            if (content.length > 5) store.ingest(content, "assistant", {});
          } catch {}

          res.writeHead(upstreamResp.status, { "Content-Type": "application/json" });
          res.end(upstreamText);
        }
      } catch (err) {
        console.error(`[proxy] Request error: ${err.message}`);
        if (!res.headersSent) {
          res.writeHead(502, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err.message }));
        }
      }
    });

    req.on("error", (err) => {
      console.error(`[proxy] Request stream error: ${err.message}`);
    });
    return;
  }

  // Fallback: proxy to Ollama
  const targetUrl = `${TARGET}${req.url}`;
  safeFetch(targetUrl, {
    method: req.method,
    headers: { ...req.headers, host: new URL(TARGET).host },
  }, 30000)
    .then(async (upstreamResp) => {
      const text = await upstreamResp.text();
      res.writeHead(upstreamResp.status, { "Content-Type": upstreamResp.headers.get("content-type") || "application/json" });
      res.end(text);
    })
    .catch((err) => {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: `Upstream error: ${err.message}`, detail: `Could not reach ${TARGET}` }));
    });
});

// ── Graceful shutdown ──

async function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.error(`\n[proxy] ${signal}: draining connections (${connections.size} active)...`);

  server.close(() => {
    console.error("[proxy] Server closed");
    process.exit(0);
  });

  // Force-close remaining connections after timeout
  setTimeout(() => {
    console.error(`[proxy] Force closing ${connections.size} connections`);
    for (const res of connections) {
      try { res.destroy(); } catch {}
    }
    process.exit(1);
  }, 5000).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("uncaughtException", (err) => {
  console.error(`[proxy] Uncaught exception: ${err.message}`);
  console.error(err.stack);
});
process.on("unhandledRejection", (err) => {
  console.error(`[proxy] Unhandled rejection: ${err.message}`);
});

// ── Startup ──

server.keepAliveTimeout = 30000;
server.headersTimeout = 31000;

async function start() {
  // Ensure memory dir
  try { await fsp.mkdir(MEMORY_DIR, { recursive: true }); } catch {}

  // Load code (async, error-isolated)
  try {
    await loadCode(REPO_PATH);
  } catch (err) {
    console.error(`[proxy] Warning: code loading incomplete: ${err.message}`);
  }

  // Build content index (async, error-isolated)
  try {
    await buildContentIndex();
  } catch (err) {
    console.error(`[proxy] Warning: content index build failed: ${err.message}`);
  }

  // Verify upstream is reachable
  try {
    await safeFetch(`${TARGET}/api/tags`, {}, 5000);
    console.error(`[proxy] Ollama reachable at ${TARGET}`);
  } catch (err) {
    console.error(`[proxy] Warning: Ollama not reachable at ${TARGET}: ${err.message}`);
    console.error(`[proxy] The proxy will start but upstream calls will fail until Ollama is available.`);
  }

  server.listen(PORT, () => {
    console.error(`[proxy] Ready on port ${PORT} (target: ${TARGET}, store: ${store.size}/${STORE_MAX})`);
    console.error(`[proxy] Tool calling: ${Object.keys(toolHandlers).length} tools loaded`);
    // Print ready message on stdout for consumers
    process.stdout.write(`EO_PROXY_READY:${PORT}\n`);
  });
}

start().catch(err => {
  console.error(`[proxy] Fatal startup error: ${err.message}`);
  process.exit(1);
});
