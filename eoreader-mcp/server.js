#!/usr/bin/env node
/**
 * EOReader MCP Server — merged
 *
 * All engine work via direct eoreader5 imports.
 * All model work via direct Ollama calls.
 * Compression scoring via sweep.js (zlib only).
 * No proxy dependency.
 *
 * Tools: fetch_url, fetch_webpage, fetch_text, read_file, read_text, ingest,
 * ingest_binary, scout, fold, search_memory, list_documents, get_memory_state,
 * think, plan, speak, craft, grow, evaluate, revise, cite, compile, steer,
 * patch, run_pipeline, snip, set_focus, set_lens, chat_turn, chat_stats,
 * chat_clear.
 *
 * Generic dev-agent affordances (bash, read/write/edit/glob/grep) were
 * removed — an EO memory server duplicating a coding agent's own file/shell
 * tools is scope creep; hosts that need those already provide them.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import * as engine from "./lib/engine-bridge.js";
import * as model from "./lib/model-bridge.js";
import * as log from "./lib/log.js";
import * as chatHistory from "./lib/chat-history.js";
import { sweep, worstWindows, uncoveredTarget } from "./lib/sweep.js";
import { patchLoop } from "./lib/patch-loop.js";
import { grow as growArtifact } from "./lib/grow.js";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const CRAFT_MODEL = process.env.CRAFT_MODEL || "qwen2.5-coder:7b";

// ── Helpers ──

function turnId() { return `turn:${Date.now()}`; }

function textResult(text) {
  return { content: [{ type: "text", text }] };
}

function jsonResult(obj) {
  return textResult(JSON.stringify(obj, null, 2));
}

// Progress notification via MCP notifications/progress
let transportRef = null;
function notifyProgress(progressToken, message, percentage) {
  if (transportRef) {
    transportRef.send({
      jsonrpc: "2.0",
      method: "notifications/progress",
      params: { progressToken, progress: { message, percentage } },
    }).catch(() => {});
  }
}

// Stream a token to the client via MCP notifications/message.
// Used by speak and chat_turn to deliver partial results during generation.
function notifyToken(session, token) {
  if (transportRef) {
    transportRef.send({
      jsonrpc: "2.0",
      method: "notifications/message",
      params: {
        level: "info",
        data: JSON.stringify({ type: "token", session, token }),
      },
    }).catch(() => {});
  }
}

// ── MCP Server ──

const server = new McpServer({ name: "eoreader", version: "3.0.0" });

// ══════════════════════════════════════════════════════════════
// fetch_url — fetch URL, extract readable text, ingest into engine
// ══════════════════════════════════════════════════════════════

function stripHtml(html) {
  // Remove scripts, styles, nav, header, footer, ads
  let text = html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<nav[\s\S]*?<\/nav>/gi, "")
    .replace(/<header[\s\S]*?<\/header>/gi, "")
    .replace(/<footer[\s\S]*?<\/footer>/gi, "")
    .replace(/<aside[\s\S]*?<\/aside>/gi, "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
  return text;
}

server.tool(
  "fetch_url",
  "Fetch a URL, extract readable text content, and ingest it into memory. Strips HTML tags, scripts, and navigation. Returns extracted text and chunk count.",
  {
    url: z.string().describe("URL to fetch"),
    session: z.string().describe("Session ID for grouping"),
    save_path: z.string().optional().describe("Where to save the extracted text (default: /tmp/fetched-{timestamp}.txt)"),
  },
  async ({ url, session, save_path }) => {
    try {
      const res = await fetch(url, {
        headers: { "User-Agent": "EOReader/1.0" },
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const contentType = res.headers.get("content-type") || "";
      const raw = await res.text();

      let text;
      let mediaType;
      if (contentType.includes("text/html") || raw.trim().startsWith("<")) {
        text = stripHtml(raw);
        mediaType = "text/html";
      } else {
        text = raw;
        mediaType = contentType || "text/plain";
      }

      const outPath = save_path || `/tmp/fetched-${Date.now()}.txt`;
      fs.writeFileSync(outPath, text, "utf8");

      const sourceId = `url:${url}`;
      const result = engine.ingestContent(text, sourceId, session);
      log.write({ type: "fetch", layer: 1, session, url, path: outPath, media_type: mediaType, size: text.length, chunks: result.chunks });
      return jsonResult({ path: outPath, media_type: mediaType, size: text.length, chunks: result.chunks, url, preview: text.slice(0, 500) });
    } catch (err) {
      return textResult(`Error fetching ${url}: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// read_file — read a local file into memory
// ══════════════════════════════════════════════════════════════

server.tool(
  "read_file",
  "Read a file from disk into memory. Chunked and ingested into Layer 1.",
  {
    file_path: z.string().describe("Absolute path to the file"),
    session: z.string().describe("Session ID"),
    tags: z.array(z.string()).optional().describe("Optional tags"),
  },
  async ({ file_path, session, tags }) => {
    try {
      const result = await engine.ingestFile(file_path, session);
      log.write({ type: "read_file", layer: 1, session, path: file_path, chunks: result.chunks, tags: tags || [] });
      return jsonResult({ path: file_path, chunks: result.chunks });
    } catch (err) {
      return textResult(`Error reading ${file_path}: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// read_text — store arbitrary text into memory
// ══════════════════════════════════════════════════════════════

server.tool(
  "read_text",
  "Store arbitrary text into memory with a title. Useful for code snippets, notes, conversation.",
  {
    title: z.string().describe("A title for this text"),
    text: z.string().describe("The text content"),
    session: z.string().describe("Session ID"),
    tags: z.array(z.string()).optional().describe("Optional tags"),
  },
  async ({ title, text, session, tags }) => {
    try {
      const sourceId = `note:${title.replace(/\s+/g, "_")}`;
      const result = engine.ingestContent(text, sourceId, session);
      log.write({ type: "read_text", layer: 1, session, title, source: sourceId, chunks: result.chunks, tags: tags || [] });
      return jsonResult({ title, source: sourceId, chunks: result.chunks });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// ingest — ingest files/dirs into engine (NUL)
// ══════════════════════════════════════════════════════════════

server.tool(
  "ingest",
  "NUL — Ingest files or directories into the engine. Writes source entries to memory (Layer 1). Supports text and binary files (audio, video, images) — binary content is routed through signal extraction.",
  {
    path: z.string().describe("File or directory path"),
    session: z.string().describe("Session ID"),
    extensions: z.array(z.string()).optional().describe("File extensions to include (dirs only)"),
  },
  async ({ path: targetPath, session, extensions }) => {
    try {
      const stats = fs.statSync(targetPath);
      let result;
      if (stats.isDirectory()) {
        result = await engine.ingestDir(targetPath, session, extensions);
      } else {
        result = await engine.ingestFile(targetPath, session);
      }
      log.write({ type: "ingest", layer: 1, session, path: targetPath, chunks: result.chunks });
      return jsonResult({ path: targetPath, chunks: result.chunks });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// ingest_binary — ingest binary files with CV model support
// ══════════════════════════════════════════════════════════════

server.tool(
  "ingest_binary",
  "Ingest any binary file (images, audio, video, documents). Extracts metadata and runs CV model for images. Returns searchable text representation.",
  {
    file_path: z.string().describe("Absolute path to the binary file"),
    session: z.string().describe("Session ID"),
    caption: z.string().optional().describe("User-provided description (optional, used for images)"),
    model: z.string().optional().describe("Vision model to use (default: llava:13b)"),
    skip_cv: z.boolean().optional().describe("Skip CV model, metadata only"),
  },
  async ({ file_path, session, caption, model, skip_cv }) => {
    try {
      const result = await engine.ingestBinary(file_path, session, {
        caption,
        model,
        skipCV: skip_cv,
      });
      log.write({ 
        type: "ingest_binary", 
        layer: 1, 
        session, 
        path: file_path, 
        ext: result.ext,
        chunks: result.chunks,
        has_caption: !!caption,
        cv_used: !skip_cv,
      });
      return jsonResult({ 
        path: file_path, 
        ext: result.ext,
        chunks: result.chunks, 
        signal_length: result.signal_length,
        preview: result.preview,
      });
    } catch (err) {
      return textResult(`Error ingesting binary: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// scout — search ingested material (SIG)
// ══════════════════════════════════════════════════════════════

server.tool(
  "scout",
  "SIG — Search ingested material for relevant passages. Algorithmic search, no model call.",
  {
    query: z.string().describe("Search query"),
    session: z.string().describe("Session ID"),
    limit: z.number().optional().describe("Max passages (default 10, max 40)"),
  },
  async ({ query, session, limit }) => {
    try {
      const result = engine.searchQuery(query, limit || 10);
      log.write({ type: "scout", layer: 1, session, query, total: result.total });
      return jsonResult({
        total: result.total,
        passages: result.passages.slice(0, limit || 10).map(p => ({
          text: p.text.slice(0, 500),
          source: p.source,
          score: p.score,
        })),
        gaps: result.gaps,
      });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// fold — compress passages into summary (INS)
// ══════════════════════════════════════════════════════════════

server.tool(
  "fold",
  "INS — Compress passages into a token-budget-constrained summary. Pure keyword-density scoring.",
  {
    passages: z.array(z.object({
      text: z.string(),
      source: z.string().optional(),
    })).describe("Passages to compress"),
    query: z.string().describe("Original query for relevance scoring"),
    session: z.string().describe("Session ID"),
    budget: z.number().optional().describe("Max tokens (default 600)"),
    max_units: z.number().optional().describe("Max snippets (default 8)"),
  },
  async ({ passages, query, session, budget, max_units }) => {
    try {
      const result = engine.foldUnits(passages, query, budget || 600, max_units || 8);
      const entry = log.write({ type: "fold", layer: 2, session, query, selected: result.selected, tokens: result.tokens });
      return jsonResult({ ...result, log_id: entry.id });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// search_memory — search all layers
// ══════════════════════════════════════════════════════════════

server.tool(
  "search_memory",
  "Search across memory. Returns scored results with source info.",
  {
    query: z.string().describe("Search query"),
    session: z.string().describe("Session ID"),
    limit: z.number().optional().describe("Max results (default 10)"),
  },
  async ({ query, session, limit }) => {
    try {
      const result = engine.searchQuery(query, limit || 10);
      if (!result.passages.length) return textResult("No matching memory found.");
      const formatted = result.passages.slice(0, limit || 10).map((r, i) =>
        `--- ${i + 1}. (score: ${r.score.toFixed(2)}) ${r.source} ---\n${r.text.slice(0, 500)}`
      ).join("\n\n");
      return textResult(formatted);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// list_documents — list all ingested docs
// ══════════════════════════════════════════════════════════════

server.tool(
  "list_documents",
  "List all documents that have been read into memory.",
  {},
  async () => {
    try {
      const entries = log.read({ type: "source" });
      if (!entries.length) return textResult("No documents in memory.");
      const bySource = {};
      for (const e of entries) {
        const src = e.source || e.path || "unknown";
        if (!bySource[src]) bySource[src] = 0;
        bySource[src]++;
      }
      const formatted = Object.entries(bySource).map(([src, chunks]) =>
        `${src} (${chunks} chunks)`
      ).join("\n");
      return textResult(formatted);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// get_memory_state — dump memory stats
// ══════════════════════════════════════════════════════════════

server.tool(
  "get_memory_state",
  "Get memory stats: total log entries, sources, sessions.",
  {},
  async () => {
    try {
      const all = log.read({});
      const types = {};
      for (const e of all) { types[e.type] = (types[e.type] || 0) + 1; }
      const sessions = new Set(all.filter(e => e.session).map(e => e.session));
      return jsonResult({ total_entries: all.length, by_type: types, sessions: [...sessions] });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// set_focus — set EO focus coordinates
// ══════════════════════════════════════════════════════════════

server.tool(
  "set_focus",
  "Set the EO Cube focus. Logs the focus shift for subsequent operations.",
  {
    operator: z.string().optional().describe("Operator: NUL, SIG, INS, SEG, CON, SYN, DEF, EVA, REC"),
    terrain: z.string().optional().describe("Terrain: Void, Entity, Kind, Field, Link, Network, Atmosphere, Lens, Paradigm"),
    stance: z.string().optional().describe("Stance: Clearing, Dissecting, Unraveling, Tending, Binding, Tracing, Cultivating, Making, Composing"),
  },
  async ({ operator, terrain, stance }) => {
    log.write({ type: "focus", layer: "meta", operator, terrain, stance });
    return textResult(`Focus set: ${operator || "-"}/${terrain || "-"}/${stance || "-"}`);
  }
);

// ══════════════════════════════════════════════════════════════
// set_lens — set viewing lens
// ══════════════════════════════════════════════════════════════

server.tool(
  "set_lens",
  "Switch the viewing lens for content filtering.",
  {
    lens: z.string().describe("Lens name: neutral, code, narrative, technical, etc."),
  },
  async ({ lens }) => {
    log.write({ type: "lens", layer: "meta", lens });
    return textResult(`Lens set to: ${lens}`);
  }
);

// ══════════════════════════════════════════════════════════════
// think — verify evidence sufficiency (SEG)
// ══════════════════════════════════════════════════════════════

server.tool(
  "think",
  "SEG — Verify if a folded summary contains specific evidence for a question. Suggests reformulation if insufficient.",
  {
    query: z.string().describe("The question"),
    summary: z.string().describe("Folded summary"),
    session: z.string().describe("Session ID"),
    round: z.number().optional().describe("Thinking round"),
  },
  async ({ query, summary, session, round }) => {
    try {
      const ctx = chatHistory.getContext(session);
      const result = await model.think(query, summary, round || 1, ctx);
      const entry = log.write({ type: "think", layer: 3, session, query, round: round || 1, sufficient: result.sufficient, evidence: result.evidence, gap: result.gap });
      return jsonResult({ ...result, log_id: entry.id });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// plan — predict next step (CON)
// ══════════════════════════════════════════════════════════════

server.tool(
  "plan",
  "CON — Predict the next error-reducing step given current state and goal.",
  {
    query: z.string().describe("The overall goal"),
    session: z.string().describe("Session ID"),
  },
  async ({ query, session }) => {
    try {
      const priorPlan = log.latest({ type: "plan", session, superseded: false });
      const ctx = chatHistory.getContext(session);
      const result = await model.plan(session, log, query, priorPlan?.text || null, ctx);
      const entry = log.write({ type: "plan", layer: 2, session, query, text: result, supersedes: priorPlan?.id || null });
      return textResult(result);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// craft — generate file from spec (DEF)
// ══════════════════════════════════════════════════════════════

server.tool(
  "craft",
  "DEF — Generate a file (HTML, CSS, JS, etc.) from a specification. Writes to disk.",
  {
    spec: z.string().describe("What to build"),
    content: z.string().optional().describe("Grounding content / source material"),
    output_path: z.string().describe("Where to write the file"),
    file_type: z.string().optional().describe("Type: html, css, js, json, svg, txt"),
    session: z.string().describe("Session ID"),
  },
  async ({ spec, content, output_path, file_type, session }) => {
    try {
      const result = await model.callModel(CRAFT_MODEL, [
        {
          role: "system",
          content: `You are a craft organ — you generate ${file_type || "files"} from specifications. Output ONLY the raw ${file_type || "file"} content, no explanations or markdown fences.`,
        },
        {
          role: "user",
          content: `SPEC: ${spec}\n\nCONTEXT:\n${(content || "(none)").slice(0, 3000)}\n\nGenerate the ${file_type || "file"}. Pure output only.`,
        },
      ], 4096);
      const cleaned = result.replace(/^```[\w]*\n?/, "").replace(/\n?```$/, "").trim();
      fs.writeFileSync(output_path, cleaned, "utf8");
      log.write({ type: "craft", layer: "artifact", session, output_path, file_type: file_type || "txt", size: cleaned.length });
      return jsonResult({ path: output_path, size: cleaned.length, preview: cleaned.slice(0, 2000) });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// grow — holonic recursive fold generation (REC/Paradigm/Cultivating)
// ══════════════════════════════════════════════════════════════

server.tool(
  "grow",
  "REC/Paradigm/Cultivating — holonic recursive fold generator. Takes a short seed and grows it into a complete artifact through recursive fold-compressed generation. Never cuts off: decomposes the seed into a task tree, generates each piece with fold-compressed context, validates each piece, retries on failure. The fundamental unit of longform generation.",
  {
    seed: z.string().describe("Short seed specification — what to build. Examples: 'reddit for dolphin lovers', 'personal homepage with dark theme'"),
    file_type: z.string().optional().describe("Type of artifact to generate (default: html)"),
    session: z.string().describe("Session ID"),
    output_path: z.string().optional().describe("Where to write the final artifact"),
    max_tasks: z.number().optional().describe("Max sub-tasks to generate (default 20, safety limit)"),
  },
  async ({ seed, file_type, session, output_path, max_tasks }) => {
    try {
      const result = await growArtifact(seed, {
        fileType: file_type || "html",
        outputPath: output_path || null,
        session: session || "default",
        maxTasks: max_tasks || 20,
      });
      return jsonResult({
        ok: result.ok,
        pieces: result.pieces?.length || 0,
        total_chars: result.stats?.totalChars || 0,
        duration: result.stats?.duration || "0s",
        errors: result.errors || [],
        final_validation: result.finalValidation,
        output_path: result.outputPath,
        preview: result.assembled?.slice(0, 2000) || "",
      });
    } catch (err) {
      return textResult(`Grow error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// speak — generate answer from verified fold (DEF)
// ══════════════════════════════════════════════════════════════

server.tool(
  "speak",
  "DEF — Generate a final answer from verified source material.",
  {
    query: z.string().describe("The question"),
    summary: z.string().describe("The verified fold summary"),
    session: z.string().describe("Session ID"),
    format: z.string().optional().describe("Answer format: direct, detailed, quote"),
  },
  async ({ query, summary, session, format }) => {
    try {
      const ctx = chatHistory.getContext(session);
      const result = await model.speak(query, summary, format || "direct", ctx, {
        onToken: (token) => notifyToken(session, token),
      });
      log.write({ type: "answer", layer: 3, session, query, text: result.slice(0, 1000) });
      return textResult(result);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// evaluate — test answer against provenance (EVA)
// ══════════════════════════════════════════════════════════════

server.tool(
  "evaluate",
  "EVA — Test an answer against its provenance chain. Detect fabrication, polarity flips, thesis injection.",
  {
    session: z.string().describe("Session ID"),
    answer_id: z.string().optional().describe("Specific answer to evaluate (defaults to latest)"),
  },
  async ({ session, answer_id }) => {
    try {
      const answer = answer_id ? log.read({ id: answer_id })[0] : log.latest({ type: "answer", session });
      if (!answer) return textResult("No answer to evaluate.");
      const evidence = answer.text || "(no evidence)";
      const ctx = chatHistory.getContext(session);
      const result = await model.evaluate(answer.text, evidence, ctx);
      log.write({ type: "evaluate", layer: 3, session, parent: answer.id, passes: result.passes, findings: result.findings });
      return jsonResult(result);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// revise — restructure when evaluation breaks (REC)
// ══════════════════════════════════════════════════════════════

server.tool(
  "revise",
  "REC — Propose revisions to plan/thesis/approach when evaluation finds a problem.",
  {
    session: z.string().describe("Session ID"),
    evaluation_id: z.string().optional().describe("The failed evaluation entry"),
  },
  async ({ session, evaluation_id }) => {
    try {
      const evalEntry = evaluation_id
        ? log.read({ id: evaluation_id })[0]
        : log.latest({ type: "evaluate", session, passes: false });
      if (!evalEntry) return textResult("No failed evaluation to revise from.");
      const planEntry = log.latest({ type: "plan", session, superseded: false });
      const ctx = chatHistory.getContext(session);
      const result = await model.revise(session, log, evalEntry, planEntry?.text || null, ctx);
      log.write({ type: "revise", layer: 3, session, parent: evalEntry.id, text: result });
      return textResult(result);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// cite — mechanically bind assertions to sources (CON)
// ══════════════════════════════════════════════════════════════

const STOPWORDS = new Set([
  "the","a","an","and","or","but","in","on","at","to","for",
  "of","with","by","from","as","is","was","are","were","be",
  "been","has","had","have","do","does","did","will","would",
  "could","should","may","might","can","shall","what","which",
  "who","whom","when","where","why","how","this","that","these",
  "those","it","its","they","them","their","we","our","you",
  "your","he","she","his","her","him","me","my","not","no",
  "nor","so","if","then","than","just","about","into","over",
  "after","before","between","under",
]);

function extractKeyTerms(text) {
  return [...new Set(text.toLowerCase().split(/\s+/).filter(w => w.length > 3 && !STOPWORDS.has(w)))];
}

function sentences(text) {
  return (text || "").split(/(?<=[.?!])\s+/).map(s => s.trim()).filter(s => s.length > 20);
}

server.tool(
  "cite",
  "CON — Mechanically bind assertions to source spans. Pure term overlap, no model call.",
  {
    session: z.string().describe("Session ID"),
    answer_id: z.string().optional().describe("Answer to cite (defaults to latest)"),
  },
  async ({ session, answer_id }) => {
    try {
      const answer = answer_id ? log.read({ id: answer_id })[0] : log.latest({ type: "answer", session });
      if (!answer) return textResult("No answer to cite.");
      const sources = log.read({ type: "source", session });
      const sents = sentences(answer.text);
      const citations = [];
      for (let i = 0; i < sents.length; i++) {
        const terms = extractKeyTerms(sents[i]);
        if (terms.length < 2) continue;
        let best = null, bestScore = 0;
        for (const src of sources) {
          const srcLower = (src.text || "").toLowerCase();
          let matchCount = 0;
          for (const t of terms) { if (srcLower.includes(t)) matchCount++; }
          if (matchCount > bestScore) { bestScore = matchCount; best = src; }
        }
        if (best && bestScore >= 2) {
          citations.push({ sentence_idx: i, sentence: sents[i].slice(0, 100), source: best.path || best.source, score: bestScore });
        }
      }
      const entry = log.write({ type: "cite", layer: 2, session, parent: answer.id, citations, coverage: citations.length + "/" + sents.length });
      return jsonResult({ citations: citations.length + "/" + sents.length + " sentences cited", details: citations.slice(0, 5) });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// compile — assemble document from log (SYN)
// ══════════════════════════════════════════════════════════════

server.tool(
  "compile",
  "SYN — Assemble a complete document from session log entries. No model call.",
  {
    session: z.string().describe("Session ID"),
    format: z.string().optional().describe("Output format: markdown (default) or plain"),
  },
  async ({ session, format }) => {
    try {
      const answers = log.read({ type: "answer", session });
      if (!answers.length) return textResult("No answers to compile.");
      const isMarkdown = format !== "plain";
      const lines = [];
      for (const e of answers) {
        if (isMarkdown) lines.push(`## Answer\n`);
        lines.push(e.text + "\n");
      }
      return textResult(lines.join("\n"));
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// steer — compression-distance scoring
// ══════════════════════════════════════════════════════════════

server.tool(
  "steer",
  "Compare generated output against a golden target using compression distance. Pure mechanical — no model.",
  {
    generated_path: z.string().describe("Path to the generated file"),
    target_path: z.string().describe("Path to the golden target"),
    session: z.string().describe("Session ID"),
    k: z.number().optional().describe("Number of worst windows (default 3)"),
  },
  async ({ generated_path, target_path, session, k }) => {
    try {
      const gen = fs.readFileSync(generated_path, "utf8");
      const target = fs.readFileSync(target_path, "utf8");
      const channels = sweep(gen, target);
      const bad = worstWindows(gen, target, 0.1, k || 3);
      const uncovered = uncoveredTarget(gen, target, k || 3);
      log.write({ type: "steer", layer: "meta", session, generated_path, target_path, channels });
      return jsonResult({
        channels: channels.map(c => ({ channel: c.channel, invention: c.invention.toFixed(3), omission: c.omission.toFixed(3) })),
        worst_windows: bad.map(w => ({ at: w.pos, residual: w.inv.toFixed(3) })),
        uncovered: uncovered.map(u => ({ at: u.pos, omission: u.omission.toFixed(3), preview: u.preview.slice(0, 120) })),
        verdict: channels.some(c => c.omission > 0.5) ? "HIGH_OMISSION" :
                 channels.some(c => c.invention > 0.5) ? "HIGH_INVENTION" : "CLOSE",
      });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// patch — compression-guided error correction loop
// ══════════════════════════════════════════════════════════════

server.tool(
  "patch",
  "Iteratively steer toward golden using compression scoring. Model patches; steering is free.",
  {
    generated_path: z.string().describe("Path to the file to improve"),
    target_path: z.string().describe("Path to the golden target"),
    session: z.string().describe("Session ID"),
    max_rounds: z.number().optional().describe("Max patch rounds (default 6)"),
  },
  async ({ generated_path, target_path, session, max_rounds }) => {
    try {
      const craftHandler = async (args) => {
        const result = await model.callModel(CRAFT_MODEL, [
          { role: "system", content: `You are a craft organ. Generate ${args.file_type || "files"} from specifications. Output ONLY raw content, no markdown fences.` },
          { role: "user", content: `SPEC: ${args.spec}\n\nGenerate the ${args.file_type || "file"}. Pure output only.` },
        ], 4096);
        const cleaned = result.replace(/^```[\w]*\n?/, "").replace(/\n?```$/, "").trim();
        fs.writeFileSync(args.output_path, cleaned, "utf8");
        log.write({ type: "craft", layer: "artifact", session, output_path: args.output_path, size: cleaned.length });
        return { content: [{ type: "text", text: JSON.stringify({ path: args.output_path, size: cleaned.length, preview: cleaned.slice(0, 2000) }) }] };
      };
      const handlers = { craft: craftHandler };
      const result = await patchLoop(handlers, generated_path, target_path, session, {
        maxRounds: max_rounds || 6, targetOmission: 0.2, k: 3,
      });
      return jsonResult({
        final_path: result.path,
        converged: result.converged,
        rounds: result.history.length,
        history: result.history.map(h => ({ round: h.round, omission: h.avgOmission.toFixed(3), invention: h.avgInvention.toFixed(3) })),
      });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// run_pipeline — multi-turn scout→fold→think→speak orchestration
// ══════════════════════════════════════════════════════════════

server.tool(
  "run_pipeline",
  "Full multi-turn research pipeline. Executes: ingest (if URL) → scout → fold → think → (re-scout if insufficient) → speak → cite. Runs up to max_rounds thinking loops. Returns final answer with provenance.",
  {
    query: z.string().describe("The research question"),
    session: z.string().describe("Session ID"),
    source_url: z.string().optional().describe("URL to fetch and ingest before searching"),
    source_path: z.string().optional().describe("Local file path to ingest before searching"),
    max_rounds: z.number().optional().describe("Max scout→fold→think rounds (default 3)"),
    format: z.string().optional().describe("Answer format: direct, detailed, quote"),
  },
  async ({ query, session, source_url, source_path, max_rounds, format }) => {
    const rounds = max_rounds || 3;
    const progressToken = `pipeline:${session}:${Date.now()}`;
    const log_entries = [];

    try {
      // Phase 0: Ingest source material if provided
      if (source_url) {
        notifyProgress(progressToken, `Fetching ${source_url}...`, 5);
        const fetchResult = await fetch(source_url, {
          headers: { "User-Agent": "EOReader/1.0" },
          signal: AbortSignal.timeout(30000),
        });
        if (fetchResult.ok) {
          const html = await fetchResult.text();
          const text = stripHtml(html);
          const sourceId = `url:${source_url}`;
          engine.ingestContent(text, sourceId, session);
          log.write({ type: "fetch", layer: 1, session, url: source_url, chunks: text.length });
          log_entries.push(`ingest_url: ${text.length} chars from ${source_url}`);
        }
      }

      if (source_path) {
        notifyProgress(progressToken, `Ingesting ${source_path}...`, 10);
        const result = await engine.ingestFile(source_path, session);
        log_entries.push(`ingest_file: ${result.chunks} chunks from ${source_path}`);
      }

      // Phase 1-3: Scout → Fold → Think loop
      let currentQuery = query;
      let finalSummary = null;
      let sufficient = false;

      for (let round = 1; round <= rounds; round++) {
        const pct = 10 + (round / rounds) * 60;

        // Scout
        notifyProgress(progressToken, `Round ${round}: scouting for "${currentQuery.slice(0, 60)}..."`, pct);
        const scoutResult = engine.searchQuery(currentQuery, 15);
        log.write({ type: "scout", layer: 1, session, query: currentQuery, round, total: scoutResult.total });

        if (!scoutResult.passages.length) {
          log_entries.push(`round ${round}: scout found 0 passages`);
          if (round < rounds) {
            currentQuery = query; // reset to original
            continue;
          }
          break;
        }

        log_entries.push(`round ${round}: scout found ${scoutResult.passages.length} passages (best: ${scoutResult.passages[0].score.toFixed(2)})`);

        // Fold
        notifyProgress(progressToken, `Round ${round}: compressing ${scoutResult.passages.length} passages...`, pct + 5);
        const foldResult = engine.foldUnits(
          scoutResult.passages.map(p => ({ text: p.text, source: p.source })),
          currentQuery, 800, 10
        );
        log.write({ type: "fold", layer: 2, session, query: currentQuery, round, selected: foldResult.selected, tokens: foldResult.tokens });
        finalSummary = foldResult.summary;
        log_entries.push(`round ${round}: fold selected ${foldResult.selected} snippets (${foldResult.tokens} tokens)`);

        // Think
        notifyProgress(progressToken, `Round ${round}: verifying evidence sufficiency...`, pct + 10);
        const pipelineThinkCtx = chatHistory.getContext(session);
        const thinkResult = await model.think(query, finalSummary, round, pipelineThinkCtx);
        sufficient = thinkResult.sufficient;
        log.write({ type: "think", layer: 3, session, query, round, sufficient, evidence: thinkResult.evidence, gap: thinkResult.gap });
        log_entries.push(`round ${round}: think sufficient=${sufficient}`);

        if (sufficient) {
          log_entries.push(`round ${round}: sufficient evidence found`);
          break;
        }

        // Use reformulated query for next round
        if (thinkResult.reformulation && thinkResult.reformulation !== currentQuery) {
          currentQuery = thinkResult.reformulation;
          log_entries.push(`round ${round}: reformulating → "${currentQuery.slice(0, 80)}"`);
        }
      }

      // Phase 4: Speak
      notifyProgress(progressToken, "Generating answer...", 80);
      let answer;
      const pipelineCtx = chatHistory.getContext(session);
      if (sufficient && finalSummary) {
        answer = await model.speak(query, finalSummary, format || "direct", pipelineCtx, {
          onToken: (token) => notifyToken(session, token),
        });
      } else {
        answer = `[After ${rounds} rounds, insufficient evidence found. Partial summary:\n${(finalSummary || "none").slice(0, 500)}]`;
      }
      log.write({ type: "answer", layer: 3, session, query, text: answer.slice(0, 2000), sufficient });
      log_entries.push(`speak: ${answer.length} chars`);

      // Phase 5: Cite
      notifyProgress(progressToken, "Binding citations...", 90);
      const sources = log.read({ type: "source", session });
      const stopWords = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","as","is","was","are","were","be","been","has","had","have","do","does","did","will","would","could","should","may","might","can","shall","what","which","who","whom","when","where","why","how","this","that","these","those","it","its","they","them","their","we","our","you","your","he","she","his","her","him","me","my","not","no","nor","so","if","then","than","just","about","into","over","after","before","between","under"]);
      const sents = answer.split(/(?<=[.?!])\s+/).filter(s => s.length > 20);
      const citations = [];
      for (const sent of sents) {
        const terms = [...new Set(sent.toLowerCase().split(/\s+/).filter(w => w.length > 3 && !stopWords.has(w)))];
        if (terms.length < 2) continue;
        let best = null, bestScore = 0;
        for (const src of sources) {
          const srcLower = (src.text || "").toLowerCase();
          let score = 0;
          for (const t of terms) { if (srcLower.includes(t)) score++; }
          if (score > bestScore) { bestScore = score; best = src; }
        }
        if (best && bestScore >= 2) {
          citations.push({ sentence: sent.slice(0, 100), source: best.path || best.source, score: bestScore });
        }
      }
      log.write({ type: "cite", layer: 2, session, citations, coverage: `${citations.length}/${sents.length}` });

      // Phase 6: Evaluate
      notifyProgress(progressToken, "Evaluating provenance...", 95);
      let evaluation = null;
      if (finalSummary) {
        const evalCtx = chatHistory.getContext(session);
        evaluation = await model.evaluate(answer, finalSummary, evalCtx);
        log.write({ type: "evaluate", layer: 3, session, passes: evaluation.passes, findings: evaluation.findings });
        log_entries.push(`evaluate: passes=${evaluation.passes}`);
      }

      notifyProgress(progressToken, "Done", 100);

      // Record the pipeline's Q&A in chat history for future context
      chatHistory.addMessage(session, "user", query);
      chatHistory.addMessage(session, "assistant", answer.slice(0, 500));

      return jsonResult({
        answer,
        sufficient,
        rounds_used: rounds,
        citations: citations.length,
        evaluation: evaluation ? { passes: evaluation.passes, findings: evaluation.findings.slice(0, 200) } : null,
        log: log_entries,
      });

    } catch (err) {
      return textResult(`Pipeline error: ${err.message}\n\nProgress:\n${log_entries.join("\n")}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// chat_turn — conversational turn with rolling context fold
// ══════════════════════════════════════════════════════════════

server.tool(
  "chat_turn",
  "Record a user message and get an assistant response with rolling conversation context. Auto-folds old turns into summaries when context budget fills. Automatically searches ingested material for grounding and binds citations to the response.",
  {
    session: z.string().describe("Session ID"),
    message: z.string().describe("The user's message"),
  },
  async ({ session, message }) => {
    try {
      // ── Auto-scout: search ingested material for grounding ──
      let sourceContext = null;
      let sourcePassages = [];
      try {
        const scoutResult = engine.searchQuery(message, 8);
        if (scoutResult.passages && scoutResult.passages.length > 0) {
          sourcePassages = scoutResult.passages;
          // Fold passages into a compact summary for the model
          const foldResult = engine.foldUnits(
            scoutResult.passages.map(p => ({ text: p.text, source: p.source })),
            message, 800, 8
          );
          if (foldResult.summary) {
            sourceContext = foldResult.summary;
          }
        }
      } catch {}

      // ── Generate response (grounded if source material was found) ──
      const result = await model.chatTurn(session, message, chatHistory, {
        onToken: (token) => notifyToken(session, token),
        sourceContext,
      });

      // ── Auto-cite: mechanically bind assertions to sources ──
      const citations = [];
      const grounded = !!sourceContext;
      if (sourcePassages.length > 0 && result.response) {
        const sents = result.response.split(/(?<=[.?!])\s+/).filter(s => s.length > 20);
        for (const sent of sents) {
          const terms = extractKeyTerms(sent);
          if (terms.length < 2) continue;
          let best = null, bestScore = 0;
          for (const src of sourcePassages) {
            const srcLower = (src.text || "").toLowerCase();
            let matchCount = 0;
            for (const t of terms) { if (srcLower.includes(t)) matchCount++; }
            if (matchCount > bestScore) { bestScore = matchCount; best = src; }
          }
          if (best && bestScore >= 2) {
            citations.push({
              sentence: sent.slice(0, 120),
              source: best.source || "",
              score: bestScore,
            });
          }
        }
      }

      log.write({
        type: "chat_turn", layer: 3, session,
        message: message.slice(0, 200),
        response: result.response.slice(0, 200),
        folds: result.stats.foldCount,
        grounded,
        citations: citations.length,
      });

      return jsonResult({
        response: result.response,
        grounded,
        citations,
        stats: {
          messages: result.stats.messageCount,
          folds: result.stats.foldCount,
          context_usage: result.stats.usagePercent + "%",
          tokens: result.stats.tokens,
        },
      });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// chat_stats — show session conversation context usage
// ══════════════════════════════════════════════════════════════

server.tool(
  "chat_stats",
  "Show conversation context stats for a session: message count, fold count, context usage percentage, tokens used.",
  {
    session: z.string().describe("Session ID"),
  },
  async ({ session }) => {
    try {
      const stats = chatHistory.getSessionStats(session);
      return jsonResult(stats);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// chat_clear — reset session conversation history
// ══════════════════════════════════════════════════════════════

server.tool(
  "chat_clear",
  "Clear a session's conversation history and fold state. Fresh start.",
  {
    session: z.string().describe("Session ID"),
  },
  async ({ session }) => {
    try {
      chatHistory.clearSession(session);
      return textResult(`Session ${session} cleared.`);
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// fetch_webpage — fetch URL, return raw text (no engine ingest)
// ══════════════════════════════════════════════════════════════

server.tool(
  "fetch_webpage",
  "Fetch a URL and return readable text content. Strips HTML tags and scripts. Does NOT ingest into engine — use fetch_url for that.",
  {
    url: z.string().describe("URL to fetch"),
    format: z.string().optional().describe("Return format: text (default) or markdown"),
  },
  async ({ url, format }) => {
    try {
      const res = await fetch(url, {
        headers: { "User-Agent": "EOReader/1.0" },
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const contentType = res.headers.get("content-type") || "";
      const raw = await res.text();
      let text;
      if (contentType.includes("text/html") || raw.trim().startsWith("<")) {
        text = stripHtml(raw);
      } else {
        text = raw;
      }
      return textResult(text.slice(0, 50000));
    } catch (err) {
      return textResult(`Error fetching ${url}: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// snip — extract contiguous range from source file
// ══════════════════════════════════════════════════════════════

const HEADING_PATTERN = /^(?:(?:Chapter|CHAPTER|Capítulo|CAPÍTULO|Kapitulua|KAPITULUA|Letter|LETTER)\s+\d+|Book|BOOK|Part|PART)\b.*$/m;

function parseSourcePath(source) {
  const m = source.match(/^source:(.+?)(?::chunk-\d+)?$/);
  return m ? m[1] : source;
}

function findNextHeading(lines, startIdx) {
  for (let i = startIdx + 1; i < lines.length; i++) {
    if (HEADING_PATTERN.test(lines[i])) return i;
  }
  return lines.length;
}

function trimToLength(text, lengthSpec) {
  if (!lengthSpec) return text;
  const paragraphs = text.split(/\n\s*\n/);

  const wordCount = (s) => s.split(/\s+/).filter(Boolean).length;

  // "N words"
  const wordMatch = lengthSpec.match(/^(\d+)\s*words?$/i);
  if (wordMatch) {
    const target = parseInt(wordMatch[1]);
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length <= target) return text;
    return words.slice(0, target).join(" ") + " […]";
  }

  // "N paragraphs"
  const paraMatch = lengthSpec.match(/^(\d+)\s*paragraphs?$/i);
  if (paraMatch) {
    const n = parseInt(paraMatch[1]);
    return paragraphs.slice(0, n).join("\n\n");
  }

  // "N lines"
  const lineMatch = lengthSpec.match(/^(\d+)\s*lines?$/i);
  if (lineMatch) {
    const n = parseInt(lineMatch[1]);
    const lines = text.split("\n");
    return lines.slice(0, n).join("\n");
  }

  switch (lengthSpec.toLowerCase()) {
    case "brief":
    case "short":
      // ~1 paragraph or ~100 words
      const first = paragraphs[0];
      if (first && wordCount(first) <= 200) return first;
      return paragraphs.slice(0, Math.ceil(paragraphs.length * 0.15)).join("\n\n");

    case "normal":
    case "medium":
      // ~3 paragraphs or ~500 words
      const nParas = Math.min(3, paragraphs.length);
      return paragraphs.slice(0, nParas).join("\n\n");

    case "full":
    case "long":
    case "complete":
      return text;

    default:
      return text;
  }
}

// Generate search keys from a vague description using the model
async function generateSearchKeys(about) {
  try {
    const result = await model.callModel("llama3.2", [
      {
        role: "system",
        content: "You generate search keywords for finding text passages. Given a vague description, output 3-5 distinctive phrases (each on its own line) that would appear literally in the text. Output ONLY the phrases, one per line. Be precise — use words that are likely to appear verbatim.",
      },
      {
        role: "user",
        content: `Vague description: "${about}"\n\nGenerate 3-5 search phrases that would appear literally in the text:`,
      },
    ], 256);
    return result.trim().split("\n").filter(Boolean).map(s => s.trim().replace(/^[-*]\s*/, ""));
  } catch {
    // Fallback: use the description itself as the key
    return [about];
  }
}

// Cluster matching line numbers to find the best passage
function findBestCluster(matches, lines, minCluster = 3) {
  if (!matches.length) return null;
  matches.sort((a, b) => a - b);
  let best = { start: matches[0], end: matches[0], count: 1 };
  let cur = { start: matches[0], end: matches[0], count: 1 };
  for (let i = 1; i < matches.length; i++) {
    if (matches[i] - cur.end <= 50) {
      cur.end = matches[i];
      cur.count++;
    } else {
      if (cur.count > best.count) best = { ...cur };
      cur = { start: matches[i], end: matches[i], count: 1 };
    }
  }
  if (cur.count > best.count) best = { ...cur };
  // Expand to heading boundaries if cluster is dense enough
  if (best.count >= minCluster) {
    const beforeHeading = findPrevHeading(lines, best.start);
    const afterHeading = findNextHeading(lines, best.end);
    best.start = beforeHeading !== -1 ? beforeHeading : Math.max(0, best.start - 5);
    best.end = afterHeading;
  }
  return best;
}

function findPrevHeading(lines, startIdx) {
  for (let i = startIdx - 1; i >= 0; i--) {
    if (lines[i].trim() === "") continue;
    if (HEADING_PATTERN.test(lines[i])) return i;
    if (/^[A-Z\s]{4,}$/.test(lines[i])) return i;
  }
  return -1;
}

server.tool(
  "snip",
  "Extract a contiguous range of text from a source file. Supports: (1) heading-based via 'match' param, (2) semantic via 'about' param (describe vaguely, model finds it), (3) length control ('brief', 'normal', 'full', 'N words', 'N paragraphs'). Language-agnostic — works with any language content. Cross-language: describe in one language, find in another. Pure I/O + pattern matching for heading mode; semantic mode uses model for search-key generation only.",
  {
    source: z.string().describe("File path or scout source reference (e.g. 'source:/path/file.txt:chunk-117')"),
    match: z.string().optional().describe("Heading anchor (e.g. 'Chapter 2', 'CHAPTER I', 'Book One', 'Capítulo 3'). Finds by exact line match, then flexible match, then substring."),
    end: z.string().optional().describe("Explicit end boundary string. If omitted, snips to next heading or EOF."),
    about: z.string().optional().describe("Vague semantic description of what to extract. Uses model to generate search keys, then grep to find the passage. Cross-language: describe in English, find in Spanish text."),
    length: z.string().optional().describe("Output length: 'brief' (~1 para), 'normal' (~3 paras), 'full' (entire section), 'N words', 'N paragraphs' ('3 paragraphs'), 'N lines' ('20 lines'). Default: full section."),
    context_lines: z.number().optional().describe("Extra lines before the match to include (default 0)"),
    max_lines: z.number().optional().describe("Maximum lines to return (safety limit, default 1000)"),
  },
  async ({ source, match, end, about, length, context_lines, max_lines }) => {
    try {
      const filePath = parseSourcePath(source);
      if (!fs.existsSync(filePath)) {
        return textResult(`File not found: ${filePath}`);
      }
      const raw = fs.readFileSync(filePath, "utf8");
      const lines = raw.split(/\r?\n/);
      let startIdx = -1;
      let endIdx = -1;

      // ── Mode 1: Heading-based (match param) ──
      if (match) {
        const esc = match.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const exactPat = new RegExp(`^(?:${esc})\\s*$`, 'm');
        const flexPat = new RegExp(`^(?:\\s*${esc}\\s*)$`, 'm');
        let fallback = -1;
        for (let i = 0; i < lines.length; i++) {
          if (exactPat.test(lines[i])) { startIdx = i; break; }
          if (fallback === -1 && flexPat.test(lines[i])) fallback = i;
        }
        if (startIdx === -1) startIdx = fallback;
        if (startIdx === -1) {
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes(match)) { startIdx = i; break; }
          }
        }
        if (startIdx === -1) {
          return textResult(`Match "${match}" not found in ${filePath}`);
        }
        startIdx = Math.max(0, startIdx - (context_lines || 0));

        if (end) {
          for (let i = startIdx + 1; i < lines.length; i++) {
            if (lines[i].includes(end)) { endIdx = i; break; }
          }
          if (endIdx === -1) endIdx = lines.length;
        } else {
          endIdx = findNextHeading(lines, startIdx);
        }

      // ── Mode 2: Semantic about mode ──
      } else if (about) {
        const keys = await generateSearchKeys(about);
        // Grep for each key in the file
        const allMatches = [];
        for (const key of keys) {
          const lower = key.toLowerCase().trim();
          if (!lower || lower.length < 3) continue;
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].toLowerCase().includes(lower)) {
              allMatches.push(i);
            }
          }
        }
        if (allMatches.length === 0) {
          return textResult(`Could not find passage matching "${about}" in ${filePath}\nGenerated keys: ${keys.join(", ")}`);
        }
        const cluster = findBestCluster([...new Set(allMatches)], lines);
        if (cluster) {
          startIdx = Math.max(0, cluster.start - (context_lines || 0));
          endIdx = cluster.end;
        } else {
          startIdx = Math.max(0, allMatches[0] - (context_lines || 0));
          endIdx = lines.length;
        }

      // ── Mode 3: Raw (no match, no about) ──
      } else {
        startIdx = 0;
        endIdx = lines.length;
      }

      // Safety: respect max_lines
      const safeMax = max_lines || 1000;
      if (endIdx - startIdx > safeMax) {
        endIdx = startIdx + safeMax;
      }

      const slice = lines.slice(startIdx, endIdx).join("\n").trim();

      // Apply length trimming
      const trimmed = trimToLength(slice, length);

      const info = {
        source: filePath,
        mode: match ? "heading" : about ? "semantic" : "raw",
        start_line: startIdx + 1,
        end_line: Math.min(endIdx, startIdx + slice.split("\n").length),
        chars: trimmed.length,
      };

      return textResult(
        trimmed + `\n\n── snip: ${info.source} lines ${info.start_line}-${info.end_line} (${info.chars} chars, mode: ${info.mode}) ──`
      );
    } catch (err) {
      return textResult(`Error: ${err.message}`);
    }
  }
);

// ══════════════════════════════════════════════════════════════
// fetch_text — download plain text from a URL and save locally
// ══════════════════════════════════════════════════════════════

server.tool(
  "fetch_text",
  "Fetch a URL and save its readable text content to a local file. Strips HTML. Returns the file path and text preview. Use this to import news articles, academic papers, blog posts, or any web content for later snip extraction.",
  {
    url: z.string().describe("URL to fetch"),
    save_path: z.string().optional().describe("Where to save the text. Defaults to /tmp/fetched-{timestamp}.txt"),
    format: z.string().optional().describe("Output format: 'text' (default, plain text), 'markdown'"),
    max_chars: z.number().optional().describe("Max characters to save (default 100000)"),
  },
  async ({ url, save_path, format, max_chars }) => {
    try {
      const ts = Date.now();
      const outPath = save_path || `/tmp/fetched-${ts}.txt`;
      const res = await fetch(url, {
        headers: { "User-Agent": "Mozilla/5.0 (compatible; EOReader/1.0)" },
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const contentType = res.headers.get("content-type") || "";
      const raw = await res.text();
      let text;
      if (format === "markdown") {
        // Simple HTML→text with minimal structure
        text = raw
          .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
          .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
          .replace(/<[^>]+>/g, "")
          .replace(/&nbsp;/g, " ")
          .replace(/&amp;/g, "&")
          .replace(/&lt;/g, "<")
          .replace(/&gt;/g, ">")
          .replace(/&quot;/g, '"')
          .replace(/&#39;/g, "'")
          .replace(/\n{3,}/g, "\n\n")
          .trim();
      } else {
        // Simple HTML→text stripping (no external dependency)
        text = raw
          .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
          .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
          .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, "")
          .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
          .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
          .replace(/<[^>]+>/g, "")
          .replace(/&nbsp;/g, " ")
          .replace(/&amp;/g, "&")
          .replace(/&lt;/g, "<")
          .replace(/&gt;/g, ">")
          .replace(/&quot;/g, '"')
          .replace(/&#39;/g, "'")
          .replace(/\n[ \t]+/g, "\n")
          .replace(/\n{3,}/g, "\n\n")
          .trim();
      }
      const maxC = max_chars || 100000;
      if (text.length > maxC) text = text.slice(0, maxC) + "\n\n[... truncated at " + maxC + " chars]";
      const dir = path.dirname(outPath);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(outPath, text, "utf8");
      const preview = text.slice(0, 500);
      const lang = detectLanguage(text);
      return textResult(
        `Saved ${text.length} chars to ${outPath}\nLanguage: ${lang}\n\nPreview:\n${preview}`
      );
    } catch (err) {
      return textResult(`Error fetching ${url}: ${err.message}`);
    }
  }
);

// detect rough language of a text sample
function detectLanguage(text) {
  const sample = text.slice(0, 2000).toLowerCase();
  const hasBasque = /\b(zure|bat|eta|ez|du|dute|izan|bere|hau|ere|batez|baten|gabe|gure|nahi|orduan|baina|beraz|beste|edo|zein|ziren|zela|zuen|zuten|dela|diren|ditu|dute|izango|egongo)\b/i.test(sample);
  if (hasBasque) return "Basque";
  const esCount = (sample.match(/\b(ella|ellos|había|sido|estar|estaba|sobre|entre|sin|desde|hasta|porque|cuando|siempre|también|pero|muy|más|menos|tan|tanto|como|así|allí|aquí|ahora|nunca|jamás|algo|nada|todo|poco|mucho|otro|mismo|gran|buen|hermana|madre|padre|hijo|hija|mujer|hombre|vida|mundo|año|día|vez|parte|forma|historia|obra|arte|artista|país|familia)\b/gi) || []).length;
  if (esCount > 5) return "Spanish";
  const enCount = (sample.match(/\b(the|and|was|were|had|have|has|been|with|from|that|this|these|those|which|what|when|where|how|would|could|should|about|between|through|during|before|after|their|them|they|than|because|while|since|until|although|though|whether|or|nor|but|not)\b/gi) || []).length;
  if (enCount > 10) return "English";
  return "unknown";
}

// ══════════════════════════════════════════════════════════════
// Start server
// ══════════════════════════════════════════════════════════════

const transport = new StdioServerTransport();
transportRef = transport;
await server.connect(transport);
console.error(`[eoreader-mcp] v3 — merged server (no proxy)`);
console.error(`[eoreader-mcp] log: ${log.LOG_PATH}`);
console.error(`[eoreader-mcp] chat history: ${chatHistory.CHAT_DIR}`);
console.error(`[eoreader-mcp] ready`);
