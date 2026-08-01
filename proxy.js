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
import { cellOf } from "@eoreader/engine/operators";
import { ensureSession, engineIngestFile, engineIngestText, engineGroundQuery, engineSearch, engineReadSpan, engineReadSegment, engineReadSourceBytes, engineReadContext, engineStats, engineListSources, engineFoldSource, engineDeleteSource, engineListRecycleBin, engineRestoreSource, enginePurgeSource, enginePurgeRecycleBin, engineRecycleBinStats, engineTerrainReport, outlineOfText, engineOutlineOfSource } from "./engine-ground.js";
import { loadCorefPrior, activatePriors } from "./priors-bridge.js";
// Static, not dynamic: the request handler is synchronous, and the module only
// catalogs on import — ingest stays lazy behind ensurePriorsIngested().
import * as priorsSource from "./priors-source.js";
import { HolonicTask } from "./holonic-task.js";
import { validateCitations, citedNumbers, buildGroundedSystemMessage, voidedAnswer } from "./grounding-gate.js";

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

// Boot-time corpus ingest runs in the background so the server can listen
// immediately. Until it finishes, the engine holds only part of the corpus — a
// query in that window returned the ordinary `no_evidence_matched` gap, which
// is indistinguishable from "your sources genuinely do not say this". They are
// different facts and must not read alike, so warmup is tracked and reported.
const corpusWarmup = { started: false, ready: false };

// ── Model routing ──
const TINY_MODEL = parseArg("tiny-model", "phi4-mini:latest");
const MEDIUM_MODEL = parseArg("medium-model", "qwen2.5-coder:7b");
// A turn that finishes but blows past this reads as a routing FAILURE, not a
// success — "fast every time" is the actual product requirement, and a
// success signal blind to latency cannot route toward it. 8s is a chat
// reply's outer bound before it reads as broken, not a generation-quality
// target.
const LATENCY_BUDGET_MS = parseArg("latency-budget-ms", 8000, Number);

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

// ── Prompt budgeting from measured latency ──
//
// The grounding prompt used to be sized by a de facto budget (2400 tokens)
// no matter which model served the turn. A slow model under that budget blows
// past LATENCY_BUDGET_MS; a fast model is under-fed and answers with less of
// the evidence it could have had. Both are fixed the same way: measure each
// model's actual prefill speed on real turns (Ollama reports the timings for
// every /api/chat response) and size the grounding prompt so its prefill
// time — the part of the turn that prompt length actually moves — stays
// inside the latency budget. The budget is a budget on the PROMPT, never on
// the retrieval: everything retrieved is still served whole to the reader.
const DEFAULT_GROUND_BUDGET = 2400;

// model -> { calls, prefillTokensPerSec, avgOutputMs } — measured, per model
// id, from Ollama's own per-call timings. Averages, not fits: enough signal
// to trend a budget over a session without pretending the variance is noise.
const latencyProfiles = new Map();

function recordModelLatency(model, data) {
  if (!model || !data || typeof data !== "object") return;
  const prefillMs = Number(data.prompt_eval_duration) / 1e6;
  const evalMs = Number(data.eval_duration) / 1e6;
  const prefillTokens = Number(data.prompt_eval_count);
  if (!Number.isFinite(prefillMs) || prefillMs <= 0 || !Number.isFinite(prefillTokens) || prefillTokens <= 0) return;
  const prof = latencyProfiles.get(model) || { calls: 0, prefillTokensPerSec: 0, avgOutputMs: 0 };
  prof.calls += 1;
  const rate = (prefillTokens / prefillMs) * 1000;
  prof.prefillTokensPerSec = prof.calls === 1 ? rate : (prof.prefillTokensPerSec * (prof.calls - 1) + rate) / prof.calls;
  if (Number.isFinite(evalMs) && evalMs > 0) {
    prof.avgOutputMs = prof.calls === 1 ? evalMs : (prof.avgOutputMs * (prof.calls - 1) + evalMs) / prof.calls;
  }
  latencyProfiles.set(model, prof);
}

// Grounding prompt token budget for `model`, sized so the prompt's prefill
// fits LATENCY_BUDGET_MS once the model's own measured generation time is
// reserved. Cold (fewer than 3 measured turns) keeps the default rather than
// guessing a rate from nothing.
function groundingPromptBudget(model) {
  const prof = latencyProfiles.get(model);
  if (!prof || prof.calls < 3) return DEFAULT_GROUND_BUDGET;
  const reserveForGeneration = prof.avgOutputMs > 0 ? prof.avgOutputMs : LATENCY_BUDGET_MS * 0.6;
  const prefillMs = Math.max(250, LATENCY_BUDGET_MS - reserveForGeneration);
  const tokens = Math.floor((prof.prefillTokensPerSec * prefillMs) / 1000);
  return Math.max(400, Math.min(tokens, 20000));
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
//
// The cell (terrain, stance) for an operator is derived from the engine's own
// algebra (cellOf, @eoreader/engine/operators) — never a hand-copied table.
// classifyCode/classifyMessage below only guess WHICH operator a piece of
// content or a chat message performs; the cell that operator earns is fixed
// by the algebra, not re-guessed here.

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
  return cellOf(operator, "Figure");
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

// ── Discourse store — persisted conversation state ──
//
// Every turn (user message + assistant response) is appended to a JSONL file
// keyed by session id. On startup the file is replayed: each observation is
// re-admitted into the engine session so the discourse survives restart.
//
// When the token budget fills up, older turns are folded into a keyword-based
// summary. Attachments (ingested files) are tracked as first-class objects
// rather than injected as inline text — the LLM sees attachment cards
// (name + size + excerpt) and can FETCH full content on demand.
//
// This mirrors eoreader-mcp/lib/chat-history.js but uses async I/O and
// integrates with the proxy's BoundedStore and engine-ground bridge.

const DISCOURSE_DIR = path.join(MEMORY_DIR, "discourse");
const DISCOURSE_CONTEXT_WINDOW = 32768;
const DISCOURSE_FOLD_THRESHOLD = 0.55;
const DISCOURSE_RECENT_KEEP = 8;
const ATTACHMENT_SNAPSHOT_CHARS = 1400;

class DiscourseStore {
  #sessions = new Map();

  #sessionPath(sessionId) {
    return path.join(DISCOURSE_DIR, `${sessionId}.jsonl`);
  }

  async #ensureDir() {
    await fsp.mkdir(DISCOURSE_DIR, { recursive: true });
  }

  /** Append a single entry (message or attachment) to the JSONL log. */
  async #append(sessionId, entry) {
    await this.#ensureDir();
    const line = JSON.stringify(entry) + "\n";
    await fsp.appendFile(this.#sessionPath(sessionId), line).catch(() => {});
  }

  /** Load a session from disk, replaying turns into BoundedStore. */
  async load(sessionId) {
    if (this.#sessions.has(sessionId)) return this.#sessions.get(sessionId);

    await this.#ensureDir();
    const p = this.#sessionPath(sessionId);
    let lines = [];
    try {
      const text = await fsp.readFile(p, "utf8");
      lines = text.trim().split("\n").filter(Boolean).map(l => JSON.parse(l));
    } catch { /* no log yet */ }

    const session = {
      messages: [],
      attachments: new Map(),
      priorSummary: null,
      foldCount: 0,
      totalTokens: 0,
    };

    for (const entry of lines) {
      if (entry.type === "attachment") {
        session.attachments.set(entry.name, entry);
      } else if (entry.role) {
        session.messages.push(entry);
      }
    }

    // Re-ingest prior turns into the BoundedStore so search_memory works
    // across restarts (the engine-ground bridge handles its own replay via
    // the engine session, which survives in-process)
    for (const msg of session.messages) {
      if (msg.content?.length > 5) {
        store.ingest(msg.content, msg.role, { session: sessionId });
      }
    }

    session.totalTokens = this.#sessionTokens(session);
    this.#sessions.set(sessionId, session);
    return session;
  }

  #sessionTokens(session) {
    let total = 0;
    for (const msg of session.messages) total += tok(msg.content);
    total += tok(session.priorSummary || "");
    return total;
  }

  /** Add a message turn. Returns { folded, foldCount, messageCount, tokens }. */
  async addMessage(sessionId, role, content) {
    const session = await this.load(sessionId);
    const msg = { role, content, timestamp: Date.now() };
    session.messages.push(msg);
    session.totalTokens = this.#sessionTokens(session);

    await this.#append(sessionId, msg);

    let folded = false;
    if (
      session.totalTokens > DISCOURSE_CONTEXT_WINDOW * DISCOURSE_FOLD_THRESHOLD &&
      session.messages.length > DISCOURSE_RECENT_KEEP + 4
    ) {
      this.#foldSession(session);
      folded = true;
    }

    return {
      folded,
      foldCount: session.foldCount,
      messageCount: session.messages.length,
      tokens: session.totalTokens,
    };
  }

  /** Fold older messages into a mechanical keyword summary. */
  #foldSession(session) {
    const splitIdx = session.messages.length - DISCOURSE_RECENT_KEEP;
    const toSummarize = session.messages.slice(0, splitIdx);
    const recent = session.messages.slice(splitIdx);

    const priorCtx = session.priorSummary
      ? `[Prior summary]\n${session.priorSummary}\n\n`
      : "";

    const dialogue = toSummarize
      .filter(m => m.role === "user" || m.role === "assistant")
      .map(m => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.slice(0, 300)}`)
      .join("\n");

    const allText = toSummarize.map(m => m.content).join(" ");
    const topics = this.#extractTopics(allText);

    session.priorSummary = [
      priorCtx,
      `Topics discussed: ${topics.join(", ")}`,
      `Exchange count: ${toSummarize.length}`,
      ``,
      `Key exchanges (compressed):`,
      dialogue.slice(0, 2000),
    ].filter(Boolean).join("\n");

    session.messages = recent;
    session.foldCount++;
    session.totalTokens = this.#sessionTokens(session);
  }

  #extractTopics(text) {
    const stops = new Set([
      "the","a","an","and","or","but","in","on","at","to","for","of","with",
      "by","from","as","is","was","are","were","be","been","has","had","have",
      "do","does","did","will","would","could","should","may","might","can",
      "that","this","it","its","i","you","we","they","he","she","me","my",
      "your","our","their","his","her","him","not","no","so","if","then",
      "just","about","like","what","when","where","how","which","who",
    ]);
    const words = text.toLowerCase().split(/\s+/);
    const freq = {};
    for (const w of words) {
      const clean = w.replace(/[^a-z0-9]/g, "");
      if (clean.length > 3 && !stops.has(clean)) freq[clean] = (freq[clean] || 0) + 1;
    }
    return Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([w]) => w);
  }

  /** Build model-ready context from a session. */
  async buildContext(sessionId, systemPrompt, userMessage) {
    const session = await this.load(sessionId);
    const messages = [];

    if (systemPrompt) messages.push({ role: "system", content: systemPrompt });

    if (session.priorSummary) {
      messages.push({
        role: "system",
        content: `[Conversation context — ${session.foldCount} prior folds]\n${session.priorSummary}`,
      });
    }

    // Attachment references (not inline content — the LLM sees cards)
    if (session.attachments.size > 0) {
      const attCards = [...session.attachments.values()].map(a => {
        const excerpt = (a.text || "").slice(0, ATTACHMENT_SNAPSHOT_CHARS);
        return `[${a.name}] ${a.type || "file"} (${Math.round((a.size || 0) / 1024)}KB) — ${a.ingestedAt || ""}\n${excerpt}`;
      }).join("\n\n");
      const attMsg = `Available attachments (use FETCH:<name> to retrieve full content):\n\n${attCards}`;
      if (tok(attMsg) < DISCOURSE_CONTEXT_WINDOW * 0.4) {
        messages.push({ role: "system", content: attMsg });
      } else {
        // Too many attachments — collapse to a name-only index
        const idx = `Available attachments: ${[...session.attachments.keys()].join(", ")}`;
        messages.push({ role: "system", content: idx });
      }
    }

    for (const msg of session.messages) {
      if (msg.role !== "system") messages.push({ role: msg.role, content: msg.content });
    }

    if (userMessage && !messages.some(m => m.role === "user" && m.content === userMessage)) {
      messages.push({ role: "user", content: userMessage });
    }

    return messages;
  }

  /** Store an ingested file as an attachment (not inline text). */
  async addAttachment(sessionId, { name, content, type, size, ingestedAt }) {
    await this.#ensureDir();
    const session = await this.load(sessionId);
    const entry = { type: "attachment", name, content: content?.slice(0, 100000), type, size, text: content?.slice(0, ATTACHMENT_SNAPSHOT_CHARS), ingestedAt: ingestedAt || new Date().toISOString(), contentHash: this.#hash(content?.slice(0, 1000) || "") };
    session.attachments.set(name, entry);

    // Also save full content to a sidecar file so FETCH can retrieve it
    const sidecar = path.join(DISCOURSE_DIR, `${sessionId}_attach_${name.replace(/[^a-zA-Z0-9._-]/g, "_")}`);
    await fsp.writeFile(sidecar, content?.slice(0, 500000) || "", "utf8").catch(() => {});

    await this.#append(sessionId, entry);
    return entry;
  }

  /** Retrieve full attachment content (for FETCH: tool calls). */
  async getAttachmentContent(sessionId, name) {
    await this.load(sessionId);
    const sidecar = path.join(DISCOURSE_DIR, `${sessionId}_attach_${name.replace(/[^a-zA-Z0-9._-]/g, "_")}`);
    try {
      return await fsp.readFile(sidecar, "utf8");
    } catch {
      const session = this.#sessions.get(sessionId);
      return session?.attachments.get(name)?.content || null;
    }
  }

  #hash(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) h = ((h << 5) - h) + str.charCodeAt(i) | 0;
    return (h >>> 0).toString(16).padStart(8, "0");
  }

  async getStats(sessionId) {
    const session = await this.load(sessionId);
    return {
      messageCount: session.messages.length,
      attachmentCount: session.attachments.size,
      foldCount: session.foldCount,
      tokens: session.totalTokens,
      contextWindow: DISCOURSE_CONTEXT_WINDOW,
      usagePercent: Math.round((session.totalTokens / DISCOURSE_CONTEXT_WINDOW) * 100),
      attachmentNames: [...session.attachments.keys()],
    };
  }

  async clearSession(sessionId) {
    this.#sessions.delete(sessionId);
    try { await fsp.unlink(this.#sessionPath(sessionId)); } catch {}
    // Also clean sidecar files
    const dir = DISCOURSE_DIR;
    try {
      const files = await fsp.readdir(dir);
      for (const f of files) {
        if (f.startsWith(`${sessionId}_attach_`)) {
          await fsp.unlink(path.join(dir, f)).catch(() => {});
        }
      }
    } catch {}
  }
}

const discourse = new DiscourseStore();

// ── Load code (async, error-isolated per file) ──

async function loadCode(repo) {
  const ignore = new Set(["node_modules", ".git", "dist", "build", "__pycache__", ".opencode", "colbert-venv", ".claude", ".venv", "venv", ".mypy_cache", ".pytest_cache"]);
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

// ── Web history ──
//
// Every page a turn pulls in is appended here before it can be cited. The
// engine holds the text; this holds the provenance — which URL, which query
// pulled it, when, and where the raw bytes landed. Append-only and on disk, so
// "why was this in my answer?" stays answerable after the process that fetched
// it is gone. Ingesting from the web without this leaves the reader holding
// citations to sources they never chose and cannot audit.
const WEB_HISTORY_PATH = path.join(import.meta.dirname, "web-history.jsonl");
const webHistory = [];

function loadWebHistory() {
  try {
    for (const line of fs.readFileSync(WEB_HISTORY_PATH, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try { webHistory.push(JSON.parse(line)); } catch {}
    }
  } catch { /* no history yet — the first web fetch creates it */ }
}
loadWebHistory();

function recordWebHistory(record) {
  webHistory.push(record);
  try {
    fs.appendFileSync(WEB_HISTORY_PATH, JSON.stringify(record) + "\n");
  } catch (err) {
    console.error(`[proxy] web-history append failed: ${err.message}`);
  }
}

// DuckDuckGo wraps every result href in a redirect
// (//duckduckgo.com/l/?uddg=<encoded real url>&rut=…). Unwrap it so what we
// hand back — and later ingest and cite — is the actual page, not a tracker.
// Sponsored rows go through the same wrapper but resolve to y.js ad links;
// they return "" so an ad never becomes a cited source.
function decodeDdgHref(href) {
  if (!href) return "";
  let h = href.replace(/&amp;/g, "&");
  const m = h.match(/[?&]uddg=([^&]+)/);
  if (m) {
    try { h = decodeURIComponent(m[1]); } catch { return ""; }
  }
  if (/\/y\.js\?|[?&]ad_provider=|[?&]ad_domain=/.test(h)) return "";
  return h;
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

// ── Citation validation ──
//
// The model sometimes cites [N] where N exceeds the number of grounding
// passages — a fabricated reference. Replace those with a visible gap marker
// so the reader never sees a fake citation. Valid citations are left alone.

// One bounded retry that demands the model actually ground its answer. Fires
// only when a turn produced NO citation despite offered passages — a total
// failure state — so the extra call is strictly cheaper than serving a
// model-only answer. Lower temperature: the task is mechanical, not creative.
async function retryGrounded(messages, model, maxCitation, onEvent) {
  const nudge = {
    role: "system",
    content: `Your previous answer used no citation from the SOURCE MATERIAL, so it could not be served. ` +
      `Rewrite it now. Every factual claim must be grounded in a numbered source passage — cite with the ` +
      `bracket numbers provided ([1]…[${maxCitation}]). If the sources do not contain the answer, say so in ` +
      `one sentence and cite the closest passage you found. Do NOT answer from your own knowledge.`,
  };
  const body = {
    model,
    messages: [...messages, nudge],
    stream: false,
    options: { temperature: 0.2, num_predict: 4096 },
  };
  try {
    if (onEvent) onEvent({ type: "llm_call", round: "grounding-retry", tools: 0, model });
    const resp = await withRetry(() => safeFetch(`${TARGET}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }, 120000), { label: "Ollama grounding retry", maxRetries: 1 });
    if (!resp.ok) return null;
    const data = await resp.json();
    recordModelLatency(model, data);
    const text = (data.message?.content || "").trim();
    return text || null;
  } catch (err) {
    console.error(`[proxy] grounding retry failed: ${err.message}`);
    return null;
  }
}

// ── Context assembly ──

const tok = (t) => Math.ceil((t || "").length / 3.5);

async function assemble(messages, sessionId = "default") {
  const latest = [...messages].reverse().find(m => m.role === "user");
  if (!latest) return messages;
  const query = latest.content || "";
  const qc = getCell(classifyMessage(query));

  let ctx = [], t = 0;
  const sys = messages.find(m => m.role === "system");
  if (sys) { t += tok(sys.content); ctx.push(sys); }
  else {
    const d = [
      "You are EO, a focused research and engineering assistant with access to web search.",
      "",
      "## Web Search Strategy",
      "Use web_search and web_fetch when you need current information, facts, or data not in local context.",
      "- Formulate keyword-rich queries — be specific",
      "- Start with type='fast', then use type='deep' for comprehensive research",
      "- Read results, then web_fetch promising URLs for full content",
      "- If results are thin, reformulate the query",
      "- Cite sources when presenting facts",
      "",
      "Use the available code and context from memory when relevant.",
    ].join("\n");
    t += tok(d); ctx.push({ role: "system", content: d });
  }

  // Discourse context: fold in past conversation from persisted store.
  // This is how the "discourse channel remembers what we're chatting about."
  try {
    const discourseCtx = await discourse.buildContext(sessionId, null, null);
    // Skip the first message (system prompt) and last message (current query);
    // fold the in-between conversation history into context
    const historyMsgs = discourseCtx.filter(m => m.role !== "system" && m.content !== query);
    if (historyMsgs.length > 0) {
      let histStr = "";
      for (const m of historyMsgs.slice(-10)) {
        const roleTag = m.role === "user" ? "[User]" : "[Assistant]";
        const folded = m.content.length > 400
          ? m.content.slice(0, 400) + "..."
          : m.content;
        histStr += `${roleTag} ${folded}\n`;
      }
      const histCtx = `[Discourse context — recent conversation]\n${histStr}`;
      if (t + tok(histCtx) < TOKEN_LIMIT) {
        t += tok(histCtx);
        ctx.push({ role: "system", content: histCtx });
      }
    }
  } catch (err) {
    console.error(`[proxy] discourse context load error: ${err.message}`);
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
        const fullText = r.value.text;
        // Fold: if content is near the token budget, snapshot it instead of
        // injecting the full text. The full content is saved to disk and
        // retrievable via FETCH:.
        const remaining = TOKEN_LIMIT - t;
        if (tok(fullText) > remaining * 0.6) {
          // Content is too big for remaining context — fold it down
          const snapshot = contentSnapshot(fullText, urls[i]);
          if (t + tok(snapshot) < TOKEN_LIMIT) {
            t += tok(snapshot);
            ctx.push({ role: "system", content: snapshot });
          }
        } else {
          // Content fits — include as-is
          if (t + tok(fullText) < TOKEN_LIMIT) {
            t += tok(fullText);
            ctx.push({ role: "system", content: `[Source: ${urls[i]}]\n${fullText.slice(0, 3000)}` });
          }
        }
      }
    }
  }

  // Store search — fold results if we're near the limit
  const budgetForSearch = TOKEN_LIMIT - t;
  const results = store.search(query, 5);
  if (results.length) {
    let c = "\n[Context: " + qc.terrain + "/" + qc.stance + "]\n" +
      results.map(r => r.type === "code"
        ? `--- ${r.meta.file || "?"} ---\n${r.text.slice(0, 500)}`
        : `[${r.type}]: ${r.text.slice(0, 300)}`
      ).join("\n\n");

    // Fold: if search results would hog the budget, truncate each harder
    if (tok(c) > budgetForSearch * 0.7) {
      c = "\n[Context: " + qc.terrain + "/" + qc.stance + " — truncated for budget]\n" +
        results.map(r => r.type === "code"
          ? `--- ${r.meta.file || "?"} ---\n${r.text.slice(0, 200)}`
          : `[${r.type}]: ${r.text.slice(0, 120)}`
        ).join("\n\n");
    }

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
      description: "Search the web for information. Performs real-time web searches across multiple sources. Use when you need current information, facts, news, or data not available in local context. Always consider using web_fetch after web_search to get detailed content from specific results.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "The search query. Formulate this like you would for a search engine — specific, keyword-rich queries return better results." },
          numResults: { type: "number", description: "Number of search results to return (default: 8, max: 20)" },
          type: { type: "string", enum: ["auto", "fast", "deep"], description: "Search type - 'auto': balanced (default), 'fast': quick snippet results, 'deep': comprehensive search with longer excerpts" },
          livecrawl: { type: "string", enum: ["fallback", "preferred"], description: "Live crawl mode - 'fallback': use cached results if available (default), 'preferred': prioritize live-fetched content" },
          contextMaxCharacters: { type: "number", description: "Maximum characters for the formatted result string (default: 8000). Use lower values for tight token budgets, higher when you need full page excerpts." },
          site: { type: "string", description: "Optional: restrict search to a specific domain (e.g. 'arxiv.org', 'github.com'). Leave empty for all sources." },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_fetch",
      description: "Fetch a URL and return its content as readable text. Use this after web_search to get detailed content from specific URLs. Automatically strips HTML tags, scripts, and styles — returns clean text.",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string", description: "URL to fetch (full URL including https://)" },
          maxChars: { type: "number", description: "Maximum characters to return (default: 10000, max: 50000)" },
          format: { type: "string", enum: ["text", "markdown", "html"], description: "Format for the returned content: 'text' for plain text (default), 'markdown' for rendered markdown, 'html' for raw HTML" },
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
      name: "verbatim_search",
      description: "Search the engine for EXACT verbatim spans from ingested source texts. Returns byte-offset anchored passages with exact text — no model hallucination. Use this when you need to retrieve EXACT quotes from ingested documents like War and Peace.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query for finding relevant spans" },
          limit: { type: "number", description: "Max results (default 5, max 40)" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "verbatim_read",
      description: "Read the full verbatim text of a previously-searched span by its span_id. Returns exact byte-offset anchored text from the source document.",
      parameters: {
        type: "object",
        properties: {
          span_id: { type: "string", description: "The span_id returned by verbatim_search" },
          max_bytes: { type: "number", description: "Maximum bytes to return (default 4000)" },
        },
        required: ["span_id"],
      },
    },
  },
  // NOTE: there are deliberately NO priors_* tools here. Priors are witness-
  // tier knowledge that STEERS retrieval; they are never model context. The
  // model absorbs a prior's effect through the evidence it widens (see
  // holonic-task.js's researchSubtask), never through a rule stated to it.
  // Priors are surfaced to the USER instead — /api/priors* for browsing, and
  // per-surf activation provenance for "what shaped this answer".
  {
    type: "function",
    function: {
      name: "fetch_attachment",
      description: "Fetch the full content of an attached file from the discourse store. Use this when you need to read a file that was uploaded as an attachment.",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string", description: "The attachment filename to retrieve" },
          session: { type: "string", description: "Session ID (default: 'default')" },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "terrain_report",
      description: "Show the engine's own measured cell occupancy across the 9-operator × 3-grain cube: which (operator, grain) cells are earned by a real organ, and which remain open questions. Not a per-file classification — that reading was refuted; this is a fact about the engine itself. Fully mechanical — no model call.",
      parameters: {
        type: "object",
        properties: {},
      },
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
      description: "Get detailed information about a specific module file: its imports (resolved to the modules they point to), what imports it, exports, definitions, and entities implemented. Good for understanding a module and its dependencies in one call.",
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
      name: "codebase_entities",
      description: "List all eoreader5 conceptual entities (cube, presence, fold, store, discourse, spine, reaction, etc.) mapped to their implementation files.",
      parameters: { type: "object", properties: {} },
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
  {
    type: "function",
    function: {
      name: "holonic_task",
      description: "Decompose and execute a complex writing/research task using holonic task decomposition. Given any task description, the system plans sub-tasks, researches each via the engine, generates content with mechanical inline citations, and assembles the final output with a unified references section. Works best for essays, reports, analyses, or any multi-section document.",
      parameters: {
        type: "object",
        properties: {
          task: { type: "string", description: "The high-level task or topic to decompose. Be specific: 'write a 5-page essay about X covering Y, with citations'." },
          model: { type: "string", description: "Ollama model to use (default: gemma2:2b)" },
          output_path: { type: "string", description: "Optional file path to write the output to" },
        },
        required: ["task"],
      },
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

// Shared by the LLM-callable `holonic_task` tool and the dedicated
// `/api/holonic` endpoint (the Compose UI surface) — one adapter
// construction, two callers.
function buildHolonicEngineAdapter() {
  try {
    return {
      search(query, { limit = 5 } = {}) {
        const result = engineSearch(query, limit);
        return (result.passages || []).slice(0, limit).map(p => ({
          text: (p.text || p.preview || "").slice(0, 800),
          source: p.source || p.source_id || "?",
          score: p.score || 0,
          span_id: p.span_id,
          byte_start: p.byte_start,
          byte_end: p.byte_end,
        })).filter(r => r.text.length > 20);
      },
      // Real per-text coref prior activation (priors-bridge.js), replacing
      // the previously-missing method that left every production run with
      // zero activated priors. Steering only — see holonic-task.js's
      // executeSubtask; nothing here is ever shown to the model as text.
      getPriors(text, sourceId) {
        try {
          const prior = loadCorefPrior(sourceId || "");
          return activatePriors(text, prior);
        } catch (err) {
          return { activated: [], gap: `priors-bridge error: ${err.message}` };
        }
      },
    };
  } catch {
    return null;
  }
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
    // ── Intelligent search with multi-backend support ──
    // Backends tried in order (configurable via env):
    //   1. Brave Search API (BRAVE_API_KEY) — best quality, free tier: 2000/mo
    //   2. Serper.dev (SERPER_API_KEY) — Google results via API, free tier: 2500/mo
    //   3. DuckDuckGo (no key needed) — fallback HTML scraper
    //
    // Returns structured results optimized for LLM consumption.
    const numResults = Math.min(args.numResults || 8, 20);
    const searchType = args.type || "auto";
    const livecrawl = args.livecrawl || "fallback";
    const maxChars = args.contextMaxCharacters || 8000;
    const siteFilter = args.site || "";

    // Build the query with optional site filter
    const query = siteFilter ? `${args.query} site:${siteFilter}` : args.query;

    // ── Backend 1: Brave Search (best quality, free tier) ──
    const braveKey = process.env.BRAVE_API_KEY;
    if (braveKey) {
      try {
        const count = searchType === "deep" ? Math.min(numResults, 20) : Math.min(numResults, 10);
        const braveUrl = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${count}&safesearch=off${
          livecrawl === "preferred" ? "&freshness=week" : ""
        }${searchType === "deep" ? "&extra_snippets=true" : ""}`;
        const resp = await safeFetch(braveUrl, {
          headers: { "Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": braveKey },
        }, 10000);
        if (resp.ok) {
          const data = await resp.json();
          const web = data.web || {};
          const results = (web.results || []).slice(0, numResults);
          if (results.length > 0) {
            const lines = [`[Brave Search] "${args.query}" — ${web.total_results || results.length} results (type: ${searchType})`, ""];
            for (let i = 0; i < results.length; i++) {
              const r = results[i];
              const snippet = (r.description || r.snippet || "").slice(0, maxChars / numResults);
              lines.push(`[${i + 1}] ${r.title}`);
              lines.push(`    URL: ${r.url}`);
              if (r.page_age) lines.push(`    Age: ${r.page_age}`);
              if (r.profile) lines.push(`    Source: ${r.profile.name}`);
              if (snippet) lines.push(`    ${snippet}`);
              lines.push("");
            }
            const output = lines.join("\n");
            if (output.length > maxChars) return output.slice(0, maxChars) + "\n…[truncated]";
            return output;
          }
        }
      } catch (err) {
        console.error(`[proxy] Brave Search failed, falling back: ${err.message}`);
      }
    }

    // ── Backend 2: Serper.dev (Google results, free tier) ──
    const serperKey = process.env.SERPER_API_KEY;
    if (serperKey) {
      try {
        const count = searchType === "deep" ? Math.min(numResults, 20) : numResults;
        const resp = await safeFetch("https://google.serper.dev/search", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-KEY": serperKey },
          body: JSON.stringify({
            q: query,
            num: count,
            gl: "us",
            hl: "en",
          }),
        }, 10000);
        if (resp.ok) {
          const data = await resp.json();
          const results = (data.organic || []).slice(0, numResults);
          if (results.length > 0) {
            const lines = [`[Serper/Google] "${args.query}" — ${data.searchParameters?.totalResults || results.length} results`, ""];
            for (let i = 0; i < results.length; i++) {
              const r = results[i];
              const snippet = (r.snippet || "").slice(0, maxChars / numResults);
              lines.push(`[${i + 1}] ${r.title}`);
              lines.push(`    URL: ${r.link}`);
              if (r.date) lines.push(`    Date: ${r.date}`);
              if (r.source) lines.push(`    Source: ${r.source}`);
              if (snippet) lines.push(`    ${snippet}`);
              lines.push("");
            }
            if (data.knowledgeGraph) {
              const kg = data.knowledgeGraph;
              lines.push(`[Knowledge Graph] ${kg.title || ""}`);
              if (kg.description) lines.push(`    ${kg.description}`);
              if (kg.attributes) {
                for (const [k, v] of Object.entries(kg.attributes)) lines.push(`    ${k}: ${v}`);
              }
              lines.push("");
            }
            if (data.peopleAlsoAsk?.length) {
              lines.push("[People also ask]");
              for (const q of data.peopleAlsoAsk.slice(0, 3)) lines.push(`  ${q.question}`);
              lines.push("");
            }
            const output = lines.join("\n");
            if (output.length > maxChars) return output.slice(0, maxChars) + "\n…[truncated]";
            return output;
          }
        }
      } catch (err) {
        console.error(`[proxy] Serper failed, falling back: ${err.message}`);
      }
    }

    // ── Backend 3: DuckDuckGo (no key needed, HTML scraper) ──
    try {
      const ddgUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
      const resp = await safeFetch(ddgUrl, {
        headers: {
          "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          "Accept": "text/html,application/xhtml+xml",
        },
      }, 15000);

      if (resp.ok) {
        const html = await resp.text();
        const results = [];

        // Parse DuckDuckGo HTML results — extract result blocks.
        // Split on the class TOKEN, not a literal attribute: DDG emits
        // class="links_main links_deep result__body", so matching the exact
        // string '<div class="result__body">' silently found zero results.
        const resultBlocks = html.split(/<div[^>]*\bclass="[^"]*\bresult__body\b[^"]*"[^>]*>/);
        // Skip first split (content before any result)
        for (let i = 1; i < resultBlocks.length && results.length < numResults; i++) {
          const block = resultBlocks[i];
          try {
            // Extract title
            const titleMatch = block.match(/<a[^>]*\bclass="[^"]*\bresult__a\b[^"]*"[^>]*>(.*?)<\/a>/s);
            const title = titleMatch ? titleMatch[1].replace(/<[^>]+>/g, "").trim() : "";

            // Extract URL
            const urlMatch = block.match(/<a[^>]*\bclass="[^"]*\bresult__a\b[^"]*"[^>]*href="(.*?)"/)
              || block.match(/<a[^>]*href="(.*?)"[^>]*\bclass="[^"]*\bresult__a\b/);
            let url = urlMatch ? decodeDdgHref(urlMatch[1]) : "";
            if (url.startsWith("//")) url = "https:" + url;
            
            // Extract snippet
            const snippetMatch = block.match(/<a[^>]*\bclass="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)<\/a>/s);
            let snippet = snippetMatch ? snippetMatch[1].replace(/<[^>]+>/g, "").trim() : "";
            if (!snippet) {
              const altMatch = block.match(/\bclass="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)<\/(?:span|div|td)>/s);
              snippet = altMatch ? altMatch[1].replace(/<[^>]+>/g, "").trim() : "";
            }

            if (title && url) {
              results.push({ title, url, snippet });
            }
          } catch {}
        }

        if (results.length > 0) {
          const lines = [`[DuckDuckGo] "${args.query}" — ${results.length} results`, ""];
          for (let i = 0; i < results.length; i++) {
            const r = results[i];
            const snippet = r.snippet.slice(0, Math.floor(maxChars / results.length));
            lines.push(`[${i + 1}] ${r.title}`);
            lines.push(`    URL: ${r.url}`);
            if (snippet) lines.push(`    ${snippet}`);
            lines.push("");
          }
          const output = lines.join("\n");
          if (output.length > maxChars) return output.slice(0, maxChars) + "\n…[truncated]";
          return output;
        }
      }
    } catch (err) {
      console.error(`[proxy] DuckDuckGo search failed: ${err.message}`);
    }

    // ── All backends failed ──
    return `[Search failed: all backends exhausted for "${args.query}". Try a different query or check network connectivity.]`;
  },

  async web_fetch(args) {
    const maxChars = Math.min(args.maxChars || 10000, 50000);
    const format = args.format || "text";
    try {
      const resp = await safeFetch(args.url, {
        headers: { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" },
      }, 15000);
      const text = await resp.text();
      const contentType = resp.headers.get("content-type") || "";
      const isHtml = contentType.includes("text/html") || contentType.includes("application/xhtml") || text.trim().startsWith("<");

      if (!isHtml && format === "text") {
        return text.slice(0, maxChars);
      }

      // Strip HTML tags for text format
      const clean = text
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
        .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
        .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
        .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, "")
        .replace(/<!--[\s\S]*?-->/g, "")
        .replace(/<[^>]+>/g, " ")
        .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
        .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
        .replace(/&[a-z]+;/g, " ")
        .replace(/\s+/g, " ").trim();

      if (!clean || clean.length < 20) {
        return `[${args.url}] fetched but content appears empty or is behind a paywall/login.`;
      }

      const result = clean.slice(0, maxChars);
      return result.length < clean.length
        ? result + "\n\n…[content truncated — use web_fetch with maxChars higher to see more]"
        : result;
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

      // Read as bytes — works for text, binary, anything
      const bytes = await fsp.readFile(args.path);
      const content = new TextDecoder("utf-8", { fatal: false }).decode(bytes);

      // The engine's own measured cell occupancy — not a per-file classification
      // (see engineTerrainReport in engine-ground.js: that reading is refuted).
      let terrainInfo = null;
      try {
        terrainInfo = engineTerrainReport();
      } catch (err) {
        terrainInfo = { error: err.message };
      }

      store.ingest(content.slice(0, 50000), "file", { path: args.path, terrain: terrainInfo });
      const terrainSummary = terrainInfo?.occupied
        ? ` Engine coverage: ${terrainInfo.counts.occupied}/${terrainInfo.counts.total} cells occupied.`
        : "";
      return `Ingested ${args.path} (${bytes.length} bytes).${terrainSummary}`;
    } catch (err) {
      return `[Error ingesting ${args.path}: ${err.message}]`;
    }
  },

  async search_memory(args) {
    const results = store.search(args.query, args.limit || 5);
    if (!results.length) return "(no matches in memory)";
    return results.map((r, i) => {
      const terrain = r.meta?.terrain;
      const terrainHint = terrain?.counts ? ` [engine coverage: ${terrain.counts.occupied}/${terrain.counts.total}]` : "";
      return `--- ${i + 1}. (score: ${r.score.toFixed(2)}) ${r.meta.file || r.meta.path || "?"}${terrainHint} ---\n${r.text.slice(0, 500)}`;
    }).join("\n\n");
  },

  async verbatim_search(args) {
    try {
      const result = engineSearch(args.query, Math.min(args.limit || 5, 40));
      if (!result.passages.length) return "(no verbatim spans found)";
      const lines = result.passages.map((p, i) => {
        return `[${i + 1}] span:${p.span_id} score:${p.score.toFixed(2)} source:${p.source.slice(0, 60)} byte:${p.byte_start}-${p.byte_end}\n${p.text.slice(0, 600)}`;
      });
      return `Found ${result.total} verbatim spans:\n\n` + lines.join("\n\n") +
        (result.gaps?.length ? `\n\n[Gaps: ${result.gaps.join("; ")}]` : "");
    } catch (err) {
      return `[Error: ${err.message}]`;
    }
  },

  async verbatim_read(args) {
    try {
      const result = engineReadSpan(args.span_id, args.max_bytes || 4000);
      if (result.error) return `[Error: ${result.error}]`;
      return `span:${result.span_id} source:${result.source_id} byte:${result.byte_start}-${result.byte_end} verbatim:${result.verbatim} truncated:${result.truncated}\n\n${result.text}`;
    } catch (err) {
      return `[Error: ${err.message}]`;
    }
  },

  async fetch_attachment(args) {
    const sessionId = args.session || "default";
    const content = await discourse.getAttachmentContent(sessionId, args.name);
    if (!content) return `[Attachment "${args.name}" not found in discourse store]`;
    return content.slice(0, 15000);
  },

  async terrain_report() {
    try {
      const report = engineTerrainReport();
      const cellLabel = (c) => `${c.op}·${c.grain}=${c.terrain}/${c.stance}`;
      const lines = [
        `Engine cell occupancy (epoch ${report.epoch}): ${report.counts.occupied}/${report.counts.total} earned, ${report.counts.empty} open questions`,
        ``,
        `Occupied:`,
        ...report.occupied.map((c) => `  ${cellLabel(c)} (${c.organs.map((o) => o.id).join(", ")})`),
        ``,
        `Empty (open questions):`,
        ...report.empty.map((c) => `  ${cellLabel(c)}`),
      ];
      return lines.join("\n");
    } catch (err) {
      return `[Error building terrain report: ${err.message}]`;
    }
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

    // Resolve imports/importedBy to the modules they point to (not just raw specs)
    const rel = contentIndex.related(args.path);
    if (!rel.error && rel.imports?.length) {
      lines.push(`\nImports:`);
      for (const m of rel.imports.slice(0, 15)) lines.push(`  ${m.repoRel} — ${(m.header || "").slice(0, 80)}`);
    } else if (mod.imports?.length) {
      lines.push(`Imports: ${mod.imports.map(i => i.spec).join(", ")}`);
    }
    if (!rel.error && rel.importedBy?.length) {
      lines.push(`\nImported by:`);
      for (const m of rel.importedBy.slice(0, 15)) lines.push(`  ${m.repoRel} — ${(m.header || "").slice(0, 80)}`);
    } else if (mod.importedBy?.length) {
      lines.push(`Imported by: ${mod.importedBy.map(i => i.by).join(", ")}`);
    }
    if (!rel.error && rel.entities?.length) {
      lines.push(`\nEntity implementations:`);
      for (const e of rel.entities) {
        lines.push(`  ${e.name}: ${e.description || "(no description)"}`);
        for (const f of e.files) lines.push(`    → ${f.path}`);
      }
    }
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

  async holonic_task(args, onEvent) {
    const taskDescription = args.task || args.description || "";
    if (!taskDescription) return "Error: 'task' parameter is required.";

    const model = args.model || "gemma2:2b";
    const outputPath = args.output_path || null;
    const engineAdapter = buildHolonicEngineAdapter();

    const task = new HolonicTask({
      task: taskDescription,
      model,
      engine: engineAdapter,
      outputPath,
    });

    let lastEvent = "planning";
    try {
      const result = await task.run({
        onProgress: (phase, msg, data = {}) => {
          lastEvent = `${phase}: ${msg.slice(0, 80)}`;
          // Forward as its own named SSE event (holonic_plan, holonic_subtask_start,
          // holonic_subtask_priors, holonic_subtask_iteration, holonic_subtask_done,
          // holonic_replan_start, holonic_replan_done, holonic_assemble,
          // holonic_done) — the generic sendSSE(evt.type, evt) in
          // handleToolStream forwards any type verbatim.
          //
          // holonic_subtask_priors is the USER's window onto which priors are
          // steering this surf. It travels the SSE channel only. Note that the
          // `summary` returned below becomes a tool result and therefore enters
          // the model's context — which is exactly why no prior information is
          // put in it. Priors steer retrieval; they are never model context.
          if (onEvent) onEvent({ type: `holonic_${phase}`, msg, ...data });
        },
      });

      const totalMc = result.results.reduce((a, r) => a + (r.citations ? r.citations.length : 0), 0);
      const totalSurf = result.results.reduce((a, r) => a + (r.surf ? r.surf.length : 0), 0);
      const summary = {
        sections: result.results.length,
        chars: result.output.length,
        pages: Math.round(result.output.length / 3000),
        mechanicalCitations: totalMc,
        surfPassages: totalSurf,
        gaps: result.gaps.length,
        metrics: result.metrics,
        output_path: result.path,
        output_preview: result.output.slice(0, 2000),
      };

      store.ingest(`holonic_task: ${taskDescription.slice(0, 100)} (${summary.sections} sections, ${summary.chars} chars)`, "tool", { type: "holonic_task" });
      return JSON.stringify(summary, null, 2);
    } catch (err) {
      return `[holonic_task failed at ${lastEvent}]: ${err.message}`;
    }
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

// web_search returns formatted text ("[i] Title\n    URL: ...\n    snippet"),
// not structured objects — parse the URLs back out so the pages behind them
// can be ingested into the engine rather than staying as untraceable text.
function parseWebSearchResults(resultStr) {
  const out = [];
  for (const block of resultStr.split(/\n\n+/)) {
    const urlMatch = block.match(/URL:\s*(\S+)/);
    if (!urlMatch) continue;
    const lines = block.split("\n").map(l => l.trim()).filter(Boolean);
    const titleLine = lines.find(l => /^\[\d+\]/.test(l));
    const title = titleLine ? titleLine.replace(/^\[\d+\]\s*/, "") : urlMatch[1];
    out.push({ title, url: urlMatch[1] });
  }
  return out;
}

const WEB_RESULT_FAILURE_RE = /^\[(Error|Search failed)/;

// Scan assistant prose for JSON objects shaped like a tool call and convert
// them into real calls. Only names the model was actually offered are
// accepted — an unknown name is prose that happens to look like JSON, and
// mistaking it for a call would invent a tool the turn never had.
// Returns { calls, remainder } where remainder is the content minus the
// consumed JSON, so any genuine prose around it survives.
function salvageTextToolCalls(content, tools) {
  const known = new Set(tools.map(t => t.function?.name || t.name).filter(Boolean));
  const calls = [];
  let remainder = content;

  // Walk brace-balanced candidates rather than regex-matching nested JSON.
  for (let i = 0; i < content.length; i++) {
    if (content[i] !== "{") continue;
    let depth = 0, inStr = false, esc = false, end = -1;
    for (let j = i; j < content.length; j++) {
      const ch = content[j];
      if (esc) { esc = false; continue; }
      if (ch === "\\") { esc = true; continue; }
      if (ch === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (ch === "{") depth++;
      else if (ch === "}") { depth--; if (depth === 0) { end = j; break; } }
    }
    if (end === -1) break;

    const raw = content.slice(i, end + 1);
    let parsed;
    try { parsed = JSON.parse(raw); } catch { continue; }
    const name = parsed?.name;
    if (typeof name === "string" && known.has(name) && ("arguments" in parsed || "parameters" in parsed)) {
      const rawArgs = parsed.arguments ?? parsed.parameters ?? {};
      calls.push({
        id: `salvaged_${calls.length}_${Math.random().toString(36).slice(2, 8)}`,
        type: "function",
        function: {
          name,
          arguments: typeof rawArgs === "string" ? rawArgs : JSON.stringify(rawArgs),
        },
      });
      remainder = remainder.replace(raw, "");
      i = end;
    }
  }

  // Strip the code fences that wrapped the consumed JSON, if any.
  if (calls.length) {
    remainder = remainder.replace(/```(?:json|tool_code)?\s*```/g, "").replace(/\s+/g, " ").trim();
  }
  return { calls, remainder };
}

// onWebContent(urls) — invoked after a web tool returns so the caller can
// ingest those pages into the engine and re-ground. There is no separate
// "web citation" record: a fetched page becomes a source with span_ids and
// byte offsets exactly like an uploaded file, or it is not citable at all.
// Search-guidance system message. Shared between handleToolStream (which must
// inject it BEFORE the serving model is picked, so the pick sees the same
// message history it will serve) and runToolLoop (whose own guard keeps a
// double injection from ever happening).
const WEB_SEARCH_STRATEGY_SYSTEM = [
  "You are EO, a focused research and engineering assistant with access to web search.",
  "",
  "## Web Search Strategy",
  "You have web_search and web_fetch tools. Use them when you need current information, facts, or data not in your training or the local context.",
  "",
  "**When to search:**",
  "- The user asks about current events, recent developments, or time-sensitive information",
  "- You need specific data (prices, stats, specifications, APIs, documentation)",
  "- The question requires domain knowledge you're uncertain about",
  "- The local codebase or memory doesn't contain the answer",
  "",
  "**How to search effectively:**",
  "- Formulate keyword-rich queries — be specific, not vague",
  "- Start broad, then narrow: use type='fast' for quick orientation, type='deep' for comprehensive research",
  "- Read search results first, then use web_fetch to get full content from promising URLs",
  "- If results are thin, try different query formulations or use site: to target known domains",
  "- Use livecrawl='preferred' for breaking news or frequently updated content",
  "",
  "**How to use results:**",
  "- Synthesize information from multiple sources — don't rely on a single result",
  "- Cite sources when presenting facts",
  "- If search returns nothing useful, try reformulating the query before giving up",
].join("\n");

async function runToolLoop(messages, tools, onEvent = null, maxRounds = 8, forceModel = null, webSearch = true, onWebContent = null, prepicked = null) {
  const effectiveTools = tools && tools.length > 0 ? tools : await getAllTools();

  // Inject intelligent search guidance as a system message.
  // This tells the model HOW to use web_search effectively — when to search,
  // how to formulate queries, and how to iterate on results.
  // Only injected when web search is enabled. When the caller pre-picked the
  // model (handleToolStream), it already injected this before the pick; the
  // guard below makes that idempotent.
  if (webSearch && !messages.some(m => m.role === "system" && m.content?.includes("Web Search Strategy"))) {
    messages.unshift({ role: "system", content: WEB_SEARCH_STRATEGY_SYSTEM });
  }

  // An explicit forceModel is a deliberate human/client override — it wins
  // outright and never enters the learned-routing ledger (there'd be no
  // honest "the router chose this" claim to score). `prepicked` is the
  // handleToolStream path: the model and its router commitment were chosen
  // BEFORE grounding so the grounding prompt could be sized to that model's
  // measured latency; the commitment still reveals here, once.
  let model, routerCtx;
  if (prepicked) {
    model = prepicked.model;
    routerCtx = prepicked.ctx;
  } else if (forceModel) {
    model = forceModel;
    routerCtx = null;
  } else if (modelRouter) {
    ({ model, ctx: routerCtx } = modelRouter.pick(messages));
    console.error(`[proxy] router picked ${model} (msgs=${messages.length}, chars=${messages.reduce((n, m) => n + (m.content || "").length, 0)})`);
  } else {
    model = selectModel(messages);
    routerCtx = null;
  }

  const turnStartedAt = Date.now();

  // "Success" for the router meant only "the tool loop finished cleanly" —
  // measured directly: a 256-second reply and a 2-second reply from the same
  // model both recorded as unqualified successes, so nothing in the learned
  // routing signal could ever prefer the fast candidate over the slow one.
  // Latency IS the thing being optimized for here, so it has to be part of
  // the outcome the router scores, not a side effect nobody reveals to it.
  const revealOutcome = async (outcome) => {
    if (!routerCtx) return;
    const elapsedMs = Date.now() - turnStartedAt;
    const gated = outcome === "success" && elapsedMs > LATENCY_BUDGET_MS ? "failure" : outcome;
    try { await modelRouter.reveal(routerCtx, gated); }
    catch (err) { console.error(`[proxy] model-router reveal failed: ${err.message}`); }
    if (prepicked) prepicked.revealed = true;
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

      // Measure the model's real prefill/generation speed on this call. This
      // is the "actual lag" the next turn's grounding prompt budget is sized
      // from — never the fixed default once a model has been measured.
      recordModelLatency(model, data);

      const msg = data.message || {};

      // Smaller local models routinely emit a tool call as prose — a bare or
      // fenced {"name":…,"arguments":…} in `content` with `tool_calls` empty.
      // Left alone, that JSON streams to the reader AS the answer and the turn
      // silently loses its grounding. Recover it into a real call instead.
      if ((!msg.tool_calls || msg.tool_calls.length === 0) && msg.content) {
        const salvaged = salvageTextToolCalls(msg.content, effectiveTools);
        if (salvaged.calls.length) {
          msg.tool_calls = salvaged.calls;
          msg.content = salvaged.remainder;
          if (onEvent) onEvent({ type: "tool_call_salvaged", count: salvaged.calls.length, names: salvaged.calls.map(c => c.function.name) });
        }
      }

      if (!msg.tool_calls || msg.tool_calls.length === 0) {
        messages.push({ role: "assistant", content: msg.content || "" });
        if (onEvent) onEvent({ type: "response", content: msg.content || "", model, elapsedMs: Date.now() - turnStartedAt });
        await revealOutcome("success");
        return msg.content || "";
      }

      // Ollama expects `arguments` as an object on the way back in — handing
      // it a JSON *string* makes its parser fail with "Value looks like
      // object, but can't find closing '}'" and kills the whole turn.
      const toolCalls = msg.tool_calls.map(tc => {
        const rawArgs = tc.function?.arguments ?? tc.arguments ?? {};
        let argsObj = rawArgs;
        if (typeof rawArgs === "string") {
          try { argsObj = JSON.parse(rawArgs); } catch { argsObj = {}; }
        }
        return {
          id: tc.id || `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          type: "function",
          function: { name: tc.function?.name || tc.name, arguments: argsObj },
        };
      });

      if (onEvent) onEvent({ type: "tool_calls", calls: toolCalls.map(tc => ({ name: tc.function.name, args: JSON.stringify(tc.function.arguments) })) });

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
            // Handlers receive onEvent as a second (optional) argument so a
            // long-running tool (e.g. holonic_task) can push its own named
            // SSE events mid-flight instead of only returning a final
            // result. Handlers that ignore the second argument (the vast
            // majority) are unaffected.
            try { result = await handler(args, onEvent); }
            catch (err) { result = `[Error calling ${name}: ${err.message}]`; }
          } else {
            result = `[Unknown tool: ${name}]`;
          }
        }

        const resultStr = typeof result === 'string' ? result : JSON.stringify(result);
        store.ingest(`Tool ${name}: ${resultStr.slice(0, 300)}`, "tool", { name });

        if (onEvent) onEvent({ type: "tool_result", name, result: resultStr.slice(0, 500) });

        // A web hit is only worth anything once it is in the engine — hand the
        // URLs to the caller to ingest and re-ground, then tell the model to
        // cite the renumbered engine passages rather than the raw tool text.
        let citeHint = "";
        if (onWebContent && (name === "web_search" || name === "web_fetch") && !WEB_RESULT_FAILURE_RE.test(resultStr)) {
          const urls = name === "web_search"
            ? parseWebSearchResults(resultStr).map(r => r.url)
            : (args.url ? [args.url] : []);
          if (urls.length) citeHint = (await onWebContent(urls)) || "";
        }

        messages.push({
          role: "tool",
          content: resultStr.slice(0, 10000) + citeHint,
          tool_call_id: tc.id || `call_${Date.now()}`,
        });
      }
    }

    const last = messages[messages.length - 1];
    if (last?.role === "tool") {
      messages.push({ role: "assistant", content: "[Max tool rounds reached. Please continue based on the results above.]" });
    }
    const finalContent = messages[messages.length - 1]?.content || "";
    if (onEvent) onEvent({ type: "response", content: finalContent, model, elapsedMs: Date.now() - turnStartedAt });
    await revealOutcome("failure");
    return finalContent;
  } catch (err) {
    await revealOutcome("failure");
    throw err;
  }
}

// ── Streaming tool-calling endpoint (SSE) ──

async function handleToolStream(res, messages, tools, forceModel = null, opts = {}) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const sendSSE = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  let effectiveTools = tools && tools.length > 0 ? tools : await getAllTools();

  // Filter web search tools when the toggle is off
  if (opts.webSearch === false) {
    const webTools = new Set(["web_search", "web_fetch"]);
    effectiveTools = effectiveTools.filter(t => {
      const name = t.function?.name || t.name || "";
      return !webTools.has(name);
    });
  }

  sendSSE("tools_available", { count: effectiveTools.length });

  // Pick the serving model BEFORE grounding. The grounding prompt is the one
  // variable-sized block in the request, and it is sized to this model's
  // measured latency — so the pick has to happen first, and the commitment
  // rides along into runToolLoop (which reveals it once the turn finishes).
  // The search-strategy message is injected first so the router sees the same
  // history it will serve; runToolLoop's own guard keeps this idempotent.
  if (opts.webSearch !== false && !messages.some(m => m.role === "system" && m.content?.includes("Web Search Strategy"))) {
    messages.unshift({ role: "system", content: WEB_SEARCH_STRATEGY_SYSTEM });
  }
  let prepicked = null;
  if (forceModel) {
    prepicked = { model: forceModel, ctx: null, revealed: false };
  } else if (modelRouter) {
    const picked = modelRouter.pick(messages);
    prepicked = { model: picked.model, ctx: picked.ctx, revealed: false };
    console.error(`[proxy] router picked ${picked.model} pre-grounding (msgs=${messages.length}, chars=${messages.reduce((n, m) => n + (m.content || "").length, 0)})`);
  } else {
    prepicked = { model: selectModel(messages), ctx: null, revealed: false };
  }

  // Engine-grounded context: search + fold before the LLM sees anything.
  // Inject as a system message so the model answers from source material
  // with inline citations, not from training-data recollection. Mandatory —
  // there is no code path that skips this to answer from the model alone.
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const query = lastUser ? (lastUser.content || "") : null;

  // One grounding pass. Re-runnable: after web pages are ingested, the same
  // call re-searches the (now larger) engine and replaces the injected system
  // message, so a web-sourced passage is numbered and cited by the identical
  // mechanism as a file-sourced one.
  let groundedSystemMsg = null;
  let groundingEmptyNote = null;
  const groundNow = () => {
    if (query === null) return null;
    // The fold budget IS the prompt budget — everything retrieved is still
    // served whole to the reader (engineGroundQuery's `retrieved[].text`).
    // What the model gets is trimmed to the serving model's measured latency.
    const promptBudget = opts.groundBudget ?? groundingPromptBudget(prepicked.model);
    const prof = latencyProfiles.get(prepicked.model);
    const budgetSource = opts.groundBudget != null ? "explicit" : (prof && prof.calls >= 3 ? "measured" : "default");
    const groundResult = engineGroundQuery(query, {
      budget: promptBudget,
      maxUnits: opts.groundMaxUnits ?? 16,
      limit: opts.groundLimit ?? 30,
      source: opts.groundSource,
    });

            const built = buildGroundedSystemMessage(groundResult, query, corpusWarmup.started && !corpusWarmup.ready);

    // Install — or, on web re-grounding, replace in place (the model must not
    // see two competing tables) — the system message the model answers from.
    if (groundedSystemMsg) {
      groundedSystemMsg.content = built.message.content;
      groundedSystemMsg._citationCount = built.message._citationCount;
    } else {
      groundedSystemMsg = built.message;
      const userIdx = messages.findIndex((m) => m.role === "user" && m.content === query);
      if (userIdx >= 0) messages.splice(userIdx, 0, groundedSystemMsg);
      else messages.unshift(groundedSystemMsg);
    }

    if (!groundResult.context) {
      // Two different facts, reported differently: the corpus is still loading,
      // or the sources really are silent on this. Either way no citation is
      // possible, and under the grounding contract the answer must be a typed
      // gap — the model was told it must not fall back on its own knowledge.
      groundingEmptyNote = built.warming
        ? "Document index still loading — no source passage could be retrieved yet."
        : "No passage in your sources matched this question — the answer could not be grounded.";

      sendSSE("grounding", {
        sourceCount: 0,
        empty: true,
        warming: built.warming,
        systemContext: groundedSystemMsg.content,
        retrieved: [],
        queryTerms: String(query || "").trim().toLowerCase().split(/\s+/).filter(Boolean),
        gaps: groundResult.gaps || [],
        note: groundingEmptyNote,
      });
      return groundResult;
    }

    sendSSE("grounding", {
      sourceCount: groundResult.total,
      foldedCount: groundResult.folded,
      tokens: groundResult.tokens,
      budget: groundResult.budget,
      // Where the prompt budget came from: the caller's explicit value, the
      // model's measured latency, or the cold-start default.
      budgetSource,
      dropped: groundResult.dropped,
      // The query as the engine tokenized it, so the reader can see which of
      // their words actually drove retrieval.
      queryTerms: String(query || "").trim().toLowerCase().split(/\s+/).filter(Boolean),
      // Every retrieved span, kept or dropped, with its ranking evidence. This
      // is the whole retrieval step, emitted before the model is called at all.
      retrieved: groundResult.retrieved || [],
      // The verbatim system context injected above — clients surface this
      // as the prompt actually sent, rather than reconstructing it.
      systemContext: groundedSystemMsg.content,
      citations: groundResult.citations.map((c, i) => ({
        index: i + 1,
        span_id: c.span_id,
        source_id: c.source_id,
        byte_start: c.byte_start,
        byte_end: c.byte_end,
        score: Math.round(c.score * 100) / 100,
        text: c.text,
      })),
      gaps: groundResult.gaps || [],
    });
    return groundResult;
  };

  groundNow();

  // Web pages are ingested as ordinary sources, then re-grounded. A page the
  // engine cannot admit yields no citation at all rather than a bare URL the
  // reader cannot check.
  const ingestedUrls = new Set();
  const onWebContent = async (urls) => {
    const fresh = urls.filter(u => !ingestedUrls.has(u)).slice(0, 3);
    if (!fresh.length) return "";
    const admitted = [];
    for (const url of fresh) {
      ingestedUrls.add(url);
      try {
        const fetched = await fetchAndSaveUrl(url);
        if (!fetched.text || fetched.text.length < 200) continue;
        let label;
        try {
          const u = new URL(url);
          label = (u.hostname.replace(/^www\./, "") + u.pathname).replace(/\/+$/, "").replace(/\//g, "_");
        } catch { label = url.replace(/\//g, "_"); }
        engineIngestText(fetched.text.slice(0, 500000), `source:${label}`, label);
        admitted.push({ url, label });
        const record = {
          name: label, url, size: fetched.text.length,
          session: opts.session || "default",
          query: query || "",
          ingestedAt: new Date().toISOString(),
          savedPath: fetched.path || null,
        };
        recordWebHistory(record);
        sendSSE("source_added", record);
      } catch (err) {
        sendSSE("gap", { type: "web_ingest_failed", url, reason: err.message });
      }
    }
    if (!admitted.length) return "\n\n(None of these pages could be ingested — do not cite them.)";
  // A grounding failure must not kill the turn — but the router commitment
  // made above still needs its one reveal, or the step is silently unobserved.
  try {
    groundNow();
  } catch (err) {
    console.error(`[proxy] grounding failed: ${err.message}`);
    sendSSE("gap", { type: "grounding_failed", reason: err.message });
    if (prepicked?.ctx && !prepicked.revealed) {
      try { await modelRouter.reveal(prepicked.ctx, "failure"); }
      catch (e) { console.error(`[proxy] model-router reveal failed: ${e.message}`); }
      prepicked.revealed = true;
    }
  }
    return `\n\n(Ingested: ${admitted.map(a => a.label).join(", ")}. The SOURCE MATERIAL above has been refreshed — cite those numbered passages, not this tool output.)`;
  };

  try {
    const rawContent = await runToolLoop(messages, effectiveTools, (evt) => {
      sendSSE(evt.type, evt);
    }, 8, forceModel, opts.webSearch, onWebContent, prepicked);

    // Validate citations: replace fabricated [N] with a visible gap marker
    // so the reader never sees a fake citation number.
    const maxCitation = groundedSystemMsg
      ? (groundedSystemMsg._citationCount || 0)
      : 0;

    // Grounding gate: an answer that cites none of the offered passages is
    // MODEL-tier — it came from the model's own knowledge, not the reader's
    // sources. It is voided, never served. "Everything must be grounded."
    let content = maxCitation > 0 ? validateCitations(rawContent, maxCitation) : rawContent;
    let voided = false;
    let voidReason = null;

    if (maxCitation === 0) {
      // The reader's sources are silent (or still warming). No citation could
      // possibly ground this answer, so whatever the model produced is voided
      // and replaced with the typed gap itself.
      voided = true;
      voidReason = groundingEmptyNote
        || "No passage in your sources matched this question — the answer could not be grounded and was not served.";
      content = null;
    } else if (citedNumbers(content, maxCitation).length === 0) {
      // Passages were retrieved and offered, yet the model answered without a
      // single citation. One bounded retry that demands grounding; a second
      // uncited answer is voided rather than served.
      const retried = await retryGrounded(messages, prepicked.model, maxCitation, (evt) => sendSSE(evt.type, evt));
      if (retried && citedNumbers(retried, maxCitation).length > 0) {
        content = validateCitations(retried, maxCitation);
      } else {
        voided = true;
        voidReason = `The model cited none of the ${maxCitation} retrieved passage(s) — its answer came from its own knowledge and was voided rather than served.`;
        content = null;
      }
    }

    if (voided) {
      content = voidedAnswer(voidReason);
      sendSSE("gap", {
        type: "ungrounded_answer_voided",
        reason: voidReason,
        availableCitations: maxCitation,
      });
    }

    sendSSE("done", { content, voided });

    // Persist assistant response in discourse store. A voided answer is a
    // typed gap, not an assistant statement — nothing to remember.
    if (!voided && content?.length > 5 && opts.session) {
      try {
        await discourse.addMessage(opts.session, "assistant", content);
      } catch (err) {
        console.error(`[proxy] discourse persist error (tools): ${err.message}`);
      }
    }
  } catch (err) {
    // runToolLoop reveals on its own internal failures; anything that threw
    // before it ran (or was never reached) still owes the router its reveal.
    if (prepicked?.ctx && !prepicked.revealed) {
      try { await modelRouter.reveal(prepicked.ctx, "failure"); }
      catch (e) { console.error(`[proxy] model-router reveal failed: ${e.message}`); }
      prepicked.revealed = true;
    }
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

  // Learned model-router state (competency ledger snapshot, read-only)
  if (req.method === "GET" && req.url === "/v1/router") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(modelRouter ? modelRouter.describe() : { error: "model-router unavailable" }));
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

  // ══════════════════════════════════════════════════════════════
  // Discourse endpoints — persisted conversation + attachments
  // ══════════════════════════════════════════════════════════════

  // Load discourse context for a session
  if (req.method === "GET" && req.url.startsWith("/api/discourse")) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const sessionId = url.searchParams.get("session") || "default";
    const pathPart = url.pathname;

    (async () => {
      try {
        if (pathPart === "/api/discourse/stats") {
          const stats = await discourse.getStats(sessionId);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(stats));
          return;
        }

        if (pathPart === "/api/discourse/context") {
          const sysPrompt = url.searchParams.get("system") || "You are a helpful assistant with access to a memory store.";
          const messages = await discourse.buildContext(sessionId, sysPrompt);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ messages, sessionId }));
          return;
        }

        // Default: load full session
        const session = await discourse.load(sessionId);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          sessionId,
          messages: session.messages,
          attachments: [...session.attachments.entries()].map(([name, a]) => ({
            name, type: a.type, size: a.size, ingestedAt: a.ingestedAt,
          })),
          foldCount: session.foldCount,
          tokens: session.totalTokens,
        }));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    })();
    return;
  }

  // Add a message to discourse (used by browser chat for persistence)
  if (req.method === "POST" && req.url === "/api/discourse/message") {
    let body = "";
    req.on("data", c => { body += c; });
    req.on("end", async () => {
      try {
        const { sessionId, role, content } = JSON.parse(body);
        const result = await discourse.addMessage(sessionId || "default", role, content);
        store.ingest(content, role, { session: sessionId || "default" });
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // Clear a session
  if (req.method === "POST" && req.url === "/api/discourse/clear") {
    let body = "";
    req.on("data", c => { body += c; });
    req.on("end", async () => {
      try {
        const { sessionId } = JSON.parse(body);
        await discourse.clearSession(sessionId || "default");
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ cleared: true }));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // Attachment endpoints
  if (req.method === "GET" && req.url.startsWith("/api/attachments")) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const sessionId = url.searchParams.get("session") || "default";

    (async () => {
      if (url.pathname === "/api/attachments/content") {
        const name = url.searchParams.get("name");
        if (!name) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'name' parameter" }));
          return;
        }
        const content = await discourse.getAttachmentContent(sessionId, name);
        res.writeHead(200, { "Content-Type": "text/plain" });
        res.end(content || "[Attachment not found]");
        return;
      }

      const session = await discourse.load(sessionId);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        attachments: [...session.attachments.entries()].map(([name, a]) => ({
          name, type: a.type, size: a.size, ingestedAt: a.ingestedAt,
          excerpt: (a.text || "").slice(0, 200),
        })),
      }));
    })();
    return;
  }

  // Ingest a file or text content into the engine session for grounded search
  if (req.method === "POST" && req.url === "/api/ingest") {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", async () => {
      try {
        const { path: ingestPath, url: ingestUrl, content, name, session } = JSON.parse(body);
        const sessionId = session || "default";

        // URL ingestion — fetch, strip markup, then fall through the same
        // content path so a page becomes a first-class citable source, not a
        // one-turn context injection.
        if (ingestUrl && !content) {
          const fetched = await fetchAndSaveUrl(ingestUrl);
          if (!fetched.text) {
            res.writeHead(502, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: `Fetch failed for ${ingestUrl} — ${fetched.error || "no text"}` }));
            return;
          }
          // The engine derives a display name by stripping everything up to the
          // last slash, so a raw URL would come back "(unnamed)". Flatten
          // host+path into one slash-free label; the URL rides alongside it.
          let label;
          try {
            const u = new URL(ingestUrl);
            label = (u.hostname.replace(/^www\./, "") + u.pathname).replace(/\/+$/, "").replace(/\//g, "_");
          } catch { label = ingestUrl.replace(/\//g, "_"); }
          const srcName = name || label || ingestUrl;
          const sourceId = `source:${srcName}`;
          const { engineIngestText } = await import("./engine-ground.js");
          const result = engineIngestText(fetched.text.slice(0, 500000), sourceId, srcName);
          const att = await discourse.addAttachment(sessionId, {
            name: srcName,
            content: fetched.text,
            type: "url",
            size: fetched.text.length,
            ingestedAt: new Date().toISOString(),
          });
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            ...result, name: srcName, url: ingestUrl,
            attachment: { name: att.name, type: att.type, size: att.size },
          }));
          return;
        }

        // Content-based ingestion (from browser file picker)
        if (content) {
          const sourceId = `source:${name || "upload"}:${Date.now()}`;
          const { engineIngestText } = await import("./engine-ground.js");
          const result = engineIngestText(content.slice(0, 500000), sourceId, name || "upload");

          // Register as attachment in discourse
          const att = await discourse.addAttachment(sessionId, {
            name: name || `upload_${Date.now()}.txt`,
            content,
            type: name ? (name.endsWith(".txt") ? "text" : name.endsWith(".json") ? "json" : name.endsWith(".js") ? "javascript" : name.endsWith(".py") ? "python" : "file") : "text",
            size: content.length,
            ingestedAt: new Date().toISOString(),
          });

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ...result, name: name || "upload", attachment: { name: att.name, type: att.type, size: att.size } }));
          return;
        }

        // Path-based ingestion (from server filesystem)
        if (!ingestPath) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'url', 'path' or 'content' field" }));
          return;
        }
        try {
          const result = engineIngestFile(ingestPath);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(result));
        } catch (err) {
          // Idempotent: if already ingested, return success with existing source info
          if (err.message?.includes("duplicate")) {
            const existing = engineListSources().find(s => s.path === ingestPath);
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({
              path: ingestPath,
              alreadyIngested: true,
              chunks: existing?.chunks || 0,
              pool: existing?.pool || "corpus",
              note: "Source already ingested",
            }));
          } else {
            throw err;
          }
        }
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // Engine session stats
  if (req.method === "GET" && req.url === "/api/grounded/stats") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(engineStats()));
    return;
  }

  // List ingested sources. Entries carry `kind` ("corpus" | "prior-raw" |
  // "prior-card") and `pool`, so the UI can pill priors distinctly from texts
  // instead of presenting witness-tier artifacts as if they were source material.
  if (req.method === "GET" && req.url.startsWith("/api/sources")) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const poolFilter = url.searchParams.get("pool");
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(engineListSources(poolFilter ? { pool: poolFilter } : {})));
    return;
  }

  // Delete (soft-delete) a source — moves it to the recycle bin.
  // Sources can be restored or permanently purged through /api/recycle-bin endpoints.
  if (req.method === "DELETE" && req.url.startsWith("/api/sources")) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const sourceKey = url.searchParams.get("source");
    if (!sourceKey) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing 'source' query parameter" }));
      return;
    }
    const result = engineDeleteSource(sourceKey, { pool: url.searchParams.get("pool") || undefined });
    const status = result.error ? 404 : 200;
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(JSON.stringify(result));
    return;
  }

  // ── Recycle bin endpoints ──

  // List deleted sources
  if (req.method === "GET" && req.url === "/api/recycle-bin") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(engineListRecycleBin()));
    return;
  }

  // Recycle bin stats
  if (req.method === "GET" && req.url === "/api/recycle-bin/stats") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(engineRecycleBinStats()));
    return;
  }

  // Restore a source from the recycle bin
  if (req.method === "POST" && req.url === "/api/recycle-bin/restore") {
    let body = "";
    req.on("data", (chunk) => body += chunk);
    req.on("end", () => {
      let parsed;
      try { parsed = JSON.parse(body); } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Invalid JSON body" }));
        return;
      }
      const sourceKey = parsed.source;
      if (!sourceKey) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Missing 'source' field" }));
        return;
      }
      const result = engineRestoreSource(sourceKey, { pool: parsed.pool || undefined });
      const status = result.error ? 404 : 200;
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(result));
    });
    return;
  }

  // Permanently delete one source or empty the entire recycle bin
  if (req.method === "DELETE" && req.url === "/api/recycle-bin") {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const sourceKey = url.searchParams.get("source");
    if (sourceKey) {
      const result = enginePurgeSource(sourceKey);
      const status = result.error ? 404 : 200;
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(result));
    } else {
      const result = enginePurgeRecycleBin();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(result));
    }
    return;
  }

  // The append-only provenance ledger for everything a web search pulled in.
  // Independent of any chat turn, so the reader can audit what the engine was
  // fed long after the turn that fed it.
  if (req.method === "GET" && req.url.startsWith("/api/web-history")) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ count: webHistory.length, entries: webHistory }));
    return;
  }

  // The reading fold for one ingested source: its cast (referents typed
  // holon/emanon/protogon/field/apparatus), its divisions, and how much of the
  // file was folded at all. This is what eochat's buildEntityMatcher — a regex
  // over capitalized words, top-20 by frequency — is meant to be replaced by.
  //
  // Anything the engine cannot supply arrives as a typed gap in `gaps`, and a
  // referent no per-text prior has individuated arrives in `withheld` with
  // `aliasesResolved: false` rather than in `referents` under a guessed label.
  // A client that renders `referents` therefore cannot show a fabricated cast;
  // one that wants the withheld candidates must ask for them by name.
  if (req.method === "GET" && req.url.startsWith("/api/fold")) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const source = url.searchParams.get("source");
    if (!source) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing 'source' parameter" }));
      return;
    }
    try {
      // `z` tunes how readily the novelty curve calls a boundary. It is handed
      // straight to detectBoundaries rather than post-filtered here, so a finer
      // read is the engine's reading at another sensitivity — not this endpoint
      // second-guessing the one it got.
      const z = url.searchParams.has("z") ? Number(url.searchParams.get("z")) : undefined;
      const result = engineFoldSource(source, {
        pool: url.searchParams.get("pool") || undefined,
        limit: parseInt(url.searchParams.get("limit") || "40", 10),
        anchors: parseInt(url.searchParams.get("anchors") || "3", 10),
        zThreshold: Number.isFinite(z) ? z : undefined,
      });
      // An unresolvable source is a 404, not a 200 carrying an error body — a
      // client polling this must be able to tell "no such source" from "a fold
      // with nothing in it", and those mean very different things here.
      res.writeHead(result.error ? 404 : 200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // Read a byte range of an ingested source. The reader's body text for corpus
  // sources: /api/attachments/content only knows session uploads, and the books
  // ingested at startup are not uploads. Byte ranges are the same coordinates
  // /api/fold's divisions carry, so paging by division needs no translation.
  if (req.method === "GET" && req.url.startsWith("/api/source/text")) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const source = url.searchParams.get("source");
    if (!source) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing 'source' parameter" }));
      return;
    }
    try {
      const num = (name) => {
        const raw = url.searchParams.get(name);
        if (raw == null || raw === "") return null;
        const n = Number(raw);
        return Number.isFinite(n) ? n : null;
      };
      const result = engineReadSourceBytes(source, {
        pool: url.searchParams.get("pool") || undefined,
        start: num("start") ?? 0,
        end: num("end"),
        maxBytes: num("max") ?? undefined,
      });
      // Same 404-vs-200 split as /api/fold: "no such source" and "a source with
      // nothing at that offset" are different answers and the client acts on
      // them differently.
      res.writeHead(result.error ? 404 : 200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // ══════════════════════════════════════════════════════════════
  // Priors — eoPriors artifacts as a browsable, searchable source.
  // Separate pool: these are never returned by corpus grounding.
  // ══════════════════════════════════════════════════════════════
  if (req.method === "GET" && req.url.startsWith("/api/priors")) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    try {
      const priors = priorsSource;

      if (url.pathname === "/api/priors/read") {
        const id = url.searchParams.get("id");
        if (!id) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'id' parameter" }));
          return;
        }
        const result = priors.readPrior(id, {
          layer: url.searchParams.get("layer") === "raw" ? "raw" : "card",
          byteStart: parseInt(url.searchParams.get("start") || "0", 10),
          maxBytes: parseInt(url.searchParams.get("max") || "40000", 10),
        });
        res.writeHead(result.error ? 404 : 200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
        return;
      }

      if (url.pathname === "/api/priors/search") {
        const q = url.searchParams.get("q");
        if (!q) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Missing 'q' parameter" }));
          return;
        }
        const result = priors.searchPriors(q, parseInt(url.searchParams.get("limit") || "8", 10), {
          maxChars: parseInt(url.searchParams.get("max_chars") || "900", 10),
          prior: url.searchParams.get("prior") || undefined,
        });
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
        return;
      }

      // Catalog. Entries carry metadata only (size, family, scope, key names);
      // the parsed artifacts are never retained, so this stays small.
      const state = priors.ensurePriorsIngested();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        pool: state.pool,
        count: state.priors,
        gaps: state.gaps,
        priors: priors.priorsCatalog(),
      }));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // Structural outline of a document, for the reader's section navigation.
  //
  // POST rather than GET, and it takes the text itself, because the reader's
  // content arrives from three different places — session attachments, prior
  // artifacts, client-side blobs from an upload the engine never saw — and an
  // outline that only worked for one of them would quietly leave the other two
  // rendering as a single unnavigable blob. The caller already holds the exact
  // string it is about to slice, so sending it back is what makes the returned
  // offsets provably in the same coordinate system as the reader's own text.
  //
  // `{ name, session }` is the fallback when the caller has a name but no text.
  if (req.method === "POST" && req.url === "/api/verbatim/outline") {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", async () => {
      try {
        const { text, name, session, source: sourceRef, pool } = JSON.parse(body || "{}");
        // An ingested corpus source resolves through the engine and comes back
        // byte-addressed, which is what the reader pages with. Attachments and
        // raw text keep the code-unit-only path — they have no file behind them.
        if (sourceRef) {
          const result = engineOutlineOfSource(sourceRef, { pool: pool || undefined });
          res.writeHead(result.error ? 404 : 200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(result));
          return;
        }
        let content = typeof text === "string" ? text : null;
        if (content == null && name) {
          content = await discourse.getAttachmentContent(session || "default", name);
        }
        if (content == null) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Provide 'source' (ingested), 'text', or a 'name' that resolves to an attachment" }));
          return;
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(outlineOfText(content)));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // ══════════════════════════════════════════════════════════════
  // Verbatim endpoints — direct engine search, NO model call.
  // Returns exact byte-offset anchored spans from ingested text.
  // ══════════════════════════════════════════════════════════════

  // Search the engine for verbatim spans matching a query.
  if (req.method === "GET" && req.url.startsWith("/api/verbatim")) {
    const url = new URL(req.url, `http://${req.headers.host}`);

    // Read a specific span by ID
    if (url.pathname === "/api/verbatim/read") {
      // UX-DESIGN.md documents this as `span_id`; the implementation only ever
      // read `id`, so every caller written against the doc got a 400. Accept
      // both — the documented name is not the wrong name.
      const spanId = url.searchParams.get("id") || url.searchParams.get("span_id");
      const maxBytes = parseInt(url.searchParams.get("max") || "4000", 10);
      if (!spanId) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Missing 'id' (or 'span_id') parameter" }));
        return;
      }
      try {
        const result = engineReadSpan(spanId, maxBytes);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // Read a segment by query — omnimodal, discovers dynamic boundaries
    if (url.pathname === "/api/verbatim/segment") {
      const q = url.searchParams.get("q");
      const maxBytes = parseInt(url.searchParams.get("max") || "50000", 10);
      const source = url.searchParams.get("source") || null;
      if (!q) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Missing 'q' parameter" }));
        return;
      }
      try {
        const result = engineReadSegment(q, maxBytes, source);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // Read context around a span: expand before/after
    if (url.pathname === "/api/verbatim/context") {
      const id = url.searchParams.get("id");
      const before = parseInt(url.searchParams.get("before") || "0", 10);
      const after = parseInt(url.searchParams.get("after") || "0", 10);
      if (!id) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Missing 'id' parameter" }));
        return;
      }
      try {
        const result = engineReadContext(id, { beforeBytes: before, afterBytes: after });
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // Search verbatim spans
    let query = url.searchParams.get("q");
    const limit = parseInt(url.searchParams.get("limit") || "10", 10);
    const maxChars = parseInt(url.searchParams.get("max_chars") || "800", 10);
    let source = url.searchParams.get("source") || null;
    if (!query) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing 'q' parameter" }));
      return;
    }
    // `source=priors` selects the priors POOL rather than filtering the corpus
    // pool by filename — asking for priors is a different question, not a
    // narrower one. `source=priors:lens-fold` narrows within that pool.
    let searchPool = undefined;
    if (source === "priors" || source?.startsWith("priors:")) {
      priorsSource.ensurePriorsIngested();
      searchPool = priorsSource.PRIORS_POOL;
      source = source.startsWith("priors:") ? source.slice("priors:".length) : null;
    }
    try {
      // Try the query as-is first (the engine's dense retrieval may find
      // semantically related passages even with diacritic differences)
      let result = engineSearch(query, Math.min(limit, 40), { maxChars, source, pool: searchPool });

      // If the query returned gaps (no_evidence_matched) and the query has
      // diacritics or the query might differ from stored text's diacritics,
      // retry with a broadened query: strip diacritics so "Natasha" matches
      // "Natásha", then re-search. This is the only model-free fix for the
      // Natásha↔Natasha problem — the engine's dense embedder treats them
      // as different tokens.
      if (result.passages.length === 0) {
        const stripped = query.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
        if (stripped !== query) {
          result = engineSearch(stripped, Math.min(limit, 40), { maxChars, source, pool: searchPool });
          if (result.passages.length > 0) {
            result.diacritic_fallback = true;
            result.diacritic_query = stripped;
          }
        }
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        query,
        pool: result.pool,
        total: result.total,
        passages: result.passages,
        gaps: result.gaps,
        verbatim: true,
        note: "These are exact verbatim spans from the engine — byte-accurate, no model involved.",
      }));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
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

  // Streaming tool-calling endpoint (always engine-grounded — not a client toggle)
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
        let messages = data.messages || [];
        const tools = data.tools;
        const sessionId = data.session || "default";

        // Ingest user input into discourse store
        for (const m of messages) {
          if (m.content?.length > 5 && m.role === "user") {
            store.ingest(m.content, m.role, { session: sessionId });
            await discourse.addMessage(sessionId, m.role, m.content);
          }
        }

        // Fold discourse context into messages so the LLM remembers
        // the conversation across turns
        try {
          const discourseCtx = await discourse.buildContext(sessionId, null, null);
          const historyMsgs = discourseCtx.filter(m =>
            m.role !== "system" && !messages.some(existing =>
              existing.role === m.role && existing.content === m.content
            )
          );
          if (historyMsgs.length > 0) {
            // Insert history before the current batch of messages
            messages = [...discourseCtx.filter(m => m.role === "system"), ...historyMsgs.slice(-10), ...messages];
          }
        } catch (err) {
          console.error(`[proxy] discourse context injection error: ${err.message}`);
        }

        await handleToolStream(res, messages, tools, data.model || null, {
          webSearch: data.webSearch !== false,
          session: sessionId,
          groundBudget: data.groundBudget ?? 2400,
          groundMaxUnits: data.groundMaxUnits ?? 16,
          groundLimit: data.groundLimit ?? 30,
          groundSource: data.groundSource || null,
        });
      } catch (err) {
        if (!res.headersSent) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err.message }));
        }
      }
    });
    return;
  }

  // Dedicated holonic-task generation endpoint (the Compose UI surface) —
  // always runs holonic_task directly rather than routing through the LLM's
  // own tool-choice, since this is a deliberate, distinct action, not an
  // ambient chat capability. Streams the same holonic_* SSE events the
  // LLM-callable tool emits via onEvent.
  if (req.method === "POST" && req.url === "/api/holonic") {
    let body = "";
    let bodySize = 0;
    req.on("data", chunk => {
      bodySize += chunk.length;
      if (bodySize > MAX_BODY) { req.destroy(new Error("Request body too large")); return; }
      body += chunk.toString("utf8");
    });
    req.on("end", async () => {
      let data;
      try {
        data = JSON.parse(body);
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: `Invalid JSON: ${err.message}` }));
        return;
      }

      const taskDescription = data.task || "";
      if (!taskDescription) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "'task' is required" }));
        return;
      }

      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      });
      const sendSSE = (event, payload) => {
        res.write(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
      };

      const task = new HolonicTask({
        task: taskDescription,
        model: data.model || "gemma2:2b",
        engine: buildHolonicEngineAdapter(),
        outputPath: data.output_path || null,
      });

      try {
        const result = await task.run({
          onProgress: (phase, msg, progressData = {}) => sendSSE(`holonic_${phase}`, { msg, ...progressData }),
        });
        sendSSE("done", {
          sections: result.results.length,
          chars: result.output.length,
          mechanicalCitations: result.results.reduce((a, r) => a + r.citations.length, 0),
          gaps: result.gaps.length,
          replanHistory: result.replanHistory,
          output: result.output,
          results: result.results,
        });
      } catch (err) {
        sendSSE("error", { message: err.message });
      }
      res.end();
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
      const sessionId = data.session || "default";

      try {
        if (useToolLoop) {
          // Non-streaming tool loop — same grounding contract as the SSE
          // surface: ground the question first, run the loop, then void any
          // answer that never cited the reader's sources.
          const messages = data.messages || [];
          const lastUser = [...messages].reverse().find((m) => m.role === "user");
          const query = lastUser ? (lastUser.content || "") : null;
          let maxCitation = 0;
          if (query) {
            const groundResult = engineGroundQuery(query, {
              budget: DEFAULT_GROUND_BUDGET,
              maxUnits: 16,
              limit: 30,
            });
    const built = buildGroundedSystemMessage(groundResult, query, corpusWarmup.started && !corpusWarmup.ready);
            maxCitation = built.message._citationCount;
            const userIdx = messages.findIndex((m) => m.role === "user" && m.content === query);
            if (userIdx >= 0) messages.splice(userIdx, 0, built.message);
            else messages.unshift(built.message);
          }

          const result = await runToolLoop(messages, tools || TOOL_DEFINITIONS);

          let content = maxCitation > 0 ? validateCitations(result, maxCitation) : result;
          let voided = false;
          let voidReason = null;
          if (maxCitation === 0) {
            voided = true;
            voidReason = "No passage in your sources matched this question — the answer could not be grounded and was not served.";
            content = null;
          } else if (citedNumbers(content, maxCitation).length === 0) {
            voided = true;
            voidReason = `The model cited none of the ${maxCitation} retrieved passage(s) — its answer came from its own knowledge and was voided rather than served.`;
            content = null;
          }
          if (voided) content = voidedAnswer(voidReason);

          // Persist to discourse store
          for (const m of messages) {
            if (m.content?.length > 5 && m.role === "user") {
              await discourse.addMessage(sessionId, m.role, m.content);
              store.ingest(m.content, m.role, { session: sessionId });
            }
          }
          if (!voided && content?.length > 5) {
            await discourse.addMessage(sessionId, "assistant", content);
            store.ingest(content, "assistant", { session: sessionId });
          }

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            id: `chat-${Date.now()}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: data.model || "llama3.2",
            choices: [{
              index: 0,
              message: { role: "assistant", content },
              finish_reason: "stop",
            }],
          }));
        } else if (data.stream) {
          // Streaming passthrough with model routing
          const targetUrl = req.url === "/api/chat" ? `${TARGET}/api/chat` : `${TARGET}/v1/chat/completions`;
          // Remove use_tools before forwarding, route model
          const { use_tools, session: _sess, ...forwardData } = data;
          forwardData.model = selectModel(forwardData.messages || []);

          // Persist user messages to discourse before forwarding
          for (const m of forwardData.messages || []) {
            if (m.content?.length > 5 && m.role === "user") {
              await discourse.addMessage(sessionId, m.role, m.content);
            }
          }

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
          let fullResponse = "";
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              const chunk = decoder.decode(value, { stream: true });
              fullResponse += chunk;
              res.write(chunk);
            }
          } catch (err) {
            console.error(`[proxy] Stream error: ${err.message}`);
          }
          res.end();

          // Persist assistant response after stream completes
          try {
            const lines = fullResponse.split("\n");
            let content = "";
            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const parsed = JSON.parse(line.slice(6));
                  if (parsed.message?.content) content += parsed.message.content;
                  else if (parsed.choices?.[0]?.delta?.content) content += parsed.choices[0].delta.content;
                } catch {}
              }
            }
            if (content.length > 5) {
              await discourse.addMessage(sessionId, "assistant", content);
              store.ingest(content, "assistant", { session: sessionId });
            }
          } catch (err) {
            console.error(`[proxy] Discourse persist error (stream): ${err.message}`);
          }
        } else {
          // Non-streaming passthrough with model routing
          const { use_tools, session: _sess, ...forwardData } = data;
          if (!forwardData.messages) forwardData.messages = [];

          // Persist user messages
          for (const m of forwardData.messages) {
            if (m.content?.length > 5 && m.role === "user") {
              await discourse.addMessage(sessionId, m.role, m.content);
            }
          }

          // Route model intelligently
          forwardData.model = selectModel(forwardData.messages);

          // Assemble context (ingest, search memory, discourse history)
          forwardData.messages = await assemble(forwardData.messages, sessionId);

          const upstreamResp = await withRetry(() => safeFetch(`${TARGET}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...forwardData, stream: false }),
          }, 120000), { label: "Ollama chat", maxRetries: 2 });

          const upstreamText = await upstreamResp.text();
          let responseContent = "";
          try {
            const parsed = JSON.parse(upstreamText);
            responseContent = parsed.message?.content || parsed.choices?.[0]?.message?.content || "";
            if (responseContent.length > 5) {
              await discourse.addMessage(sessionId, "assistant", responseContent);
              store.ingest(responseContent, "assistant", { session: sessionId });
            }
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

  // Start server FIRST so it accepts connections immediately
  server.listen(PORT, () => {
    console.error(`[proxy] Ready on port ${PORT} (target: ${TARGET}, store: ${store.size}/${STORE_MAX})`);
    console.error(`[proxy] Tool calling: ${Object.keys(toolHandlers).length} tools loaded`);
    // Print ready message on stdout for consumers
    process.stdout.write(`EO_PROXY_READY:${PORT}\n`);
  });

  // Load code (async, error-isolated) — happens AFTER server starts
  try {
    await loadCode(REPO_PATH);
  } catch (err) {
    console.error(`[proxy] Warning: code loading incomplete: ${err.message}`);
  }

  // Build content index (server starts first, index builds in background
  // — deferred via setImmediate so synchronous file I/O inside the scan
  // doesn't block the event loop before server.listen)
  setImmediate(() => {
    buildContentIndex().catch(err => {
      console.error(`[proxy] Warning: content index build failed: ${err.message}`);
    });
  });

  // Auto-ingest large files AFTER server is listening (non-blocking)
  // These run in the background and don't prevent the server from accepting requests
  setImmediate(async () => {
    corpusWarmup.started = true;
    // Auto-ingest War and Peace (pg2600.txt) for verbatim span retrieval
    const WAR_AND_PEACE_PATHS = [
      path.resolve(REPO_PATH, "pg2600.txt"),
      path.resolve(REPO_PATH, "..", "pg2600.txt"),
      path.resolve(process.env.HOME || "/Users/mlacy", "Downloads", "pg2600.txt"),
      path.resolve(process.env.HOME || "/Users/mlacy", "Desktop", "pg2600.txt"),
    ];
    for (const wpPath of WAR_AND_PEACE_PATHS) {
      try {
        if (fs.existsSync(wpPath)) {
          const wpResult = engineIngestFile(wpPath);
          console.error(`[proxy] Ingested War and Peace: ${wpPath} (${wpResult.chunks} chunks)`);
          break;
        }
      } catch (err) {
        if (!err.message?.includes("duplicate")) {
          console.error(`[proxy] War and Peace ingest skipped at ${wpPath}: ${err.message}`);
        }
      }
    }

    // Also look for Frankenstein (pg84.txt) in the repo dir
    const FRANKENSTEIN_PATHS = [
      path.resolve(REPO_PATH, "pg84.txt"),
      path.resolve(REPO_PATH, "..", "pg84.txt"),
      path.resolve(process.env.HOME || "/Users/mlacy", "Downloads", "pg84.txt"),
    ];
    for (const frPath of FRANKENSTEIN_PATHS) {
      try {
        if (fs.existsSync(frPath)) {
          const frResult = engineIngestFile(frPath);
          console.error(`[proxy] Ingested Frankenstein: ${frPath} (${frResult.chunks} chunks)`);
          break;
        }
      } catch (err) {
        if (!err.message?.includes("duplicate")) {
          console.error(`[proxy] Frankenstein ingest skipped at ${frPath}: ${err.message}`);
        }
      }
    }

    // Also look for the King James Bible (pg10.txt)
    const BIBLE_PATHS = [
      path.resolve(REPO_PATH, "pg10.txt"),
      path.resolve(REPO_PATH, "..", "pg10.txt"),
      path.resolve(process.env.HOME || "/Users/mlacy", "Downloads", "pg10.txt"),
    ];
    for (const bibPath of BIBLE_PATHS) {
      try {
        if (fs.existsSync(bibPath)) {
          const bibResult = engineIngestFile(bibPath);
          console.error(`[proxy] Ingested King James Bible: ${bibPath} (${bibResult.chunks} chunks)`);
          break;
        }
      } catch (err) {
        if (!err.message?.includes("duplicate")) {
          console.error(`[proxy] Bible ingest skipped at ${bibPath}: ${err.message}`);
        }
      }
    }

    corpusWarmup.ready = true;
    console.error(`[proxy] Corpus warm — grounding is now complete`);
  });

  // Verify upstream is reachable — retry with backoff because the
  // concurrent engine ingest of large texts (War and Peace, Bible) may
  // temporarily make Ollama unresponsive during embedding.
  for (let attempt = 1; attempt <= 4; attempt++) {
    const ok = await safeFetch(`${TARGET}/api/tags`, {}, attempt < 4 ? 5000 : 10000).then(r => true).catch(() => false);
    if (ok) {
      console.error(`[proxy] Ollama reachable at ${TARGET}`);
      break;
    }
    if (attempt < 4) {
      console.error(`[proxy] Waiting for Ollama (attempt ${attempt}/4)...`);
      await new Promise(r => setTimeout(r, 3000 * attempt));
    } else {
      console.error(`[proxy] Warning: Ollama not reachable at ${TARGET} after 4 attempts`);
      console.error(`[proxy] The proxy will start but upstream calls will fail until Ollama is available.`);
    }
  }
}

start().catch(err => {
  console.error(`[proxy] Fatal startup error: ${err.message}`);
  process.exit(1);
});
