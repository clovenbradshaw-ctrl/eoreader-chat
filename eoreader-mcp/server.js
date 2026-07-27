#!/usr/bin/env node
/**
 * EOReader MCP Server — merged
 *
 * All engine work via direct eoreader5 imports.
 * All model work via direct Ollama calls.
 * Compression scoring via sweep.js (zlib only).
 * No proxy dependency.
 *
 * Tools: fetch_url, read_file, read_text, ingest, scout, fold,
 * search_memory, list_documents, get_memory_state, think, plan,
 * speak, craft, evaluate, revise, cite, compile, steer, patch,
 * set_focus, set_lens.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import * as engine from "./lib/engine-bridge.js";
import * as model from "./lib/model-bridge.js";
import * as log from "./lib/log.js";
import { sweep, worstWindows, uncoveredTarget } from "./lib/sweep.js";
import { patchLoop } from "./lib/patch-loop.js";

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
      const result = engine.ingestFile(file_path, session);
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
        result = engine.ingestDir(targetPath, session, extensions);
      } else {
        result = engine.ingestFile(targetPath, session);
      }
      log.write({ type: "ingest", layer: 1, session, path: targetPath, chunks: result.chunks });
      return jsonResult({ path: targetPath, chunks: result.chunks });
    } catch (err) {
      return textResult(`Error: ${err.message}`);
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
      const result = await model.think(query, summary, round || 1);
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
      const result = await model.plan(session, log, query, priorPlan?.text || null);
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
      const result = await model.speak(query, summary, format || "direct");
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
      const result = await model.evaluate(answer.text, evidence);
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
      const result = await model.revise(session, log, evalEntry, planEntry?.text || null);
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
        const result = engine.ingestFile(source_path, session);
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
        const thinkResult = await model.think(query, finalSummary, round);
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
      if (sufficient && finalSummary) {
        answer = await model.speak(query, finalSummary, format || "direct");
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
        evaluation = await model.evaluate(answer, finalSummary);
        log.write({ type: "evaluate", layer: 3, session, passes: evaluation.passes, findings: evaluation.findings });
        log_entries.push(`evaluate: passes=${evaluation.passes}`);
      }

      notifyProgress(progressToken, "Done", 100);

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
// Start server
// ══════════════════════════════════════════════════════════════

const transport = new StdioServerTransport();
transportRef = transport;
await server.connect(transport);
console.error(`[eoreader-mcp] v3 — merged server (no proxy)`);
console.error(`[eoreader-mcp] log: ${log.LOG_PATH}`);
console.error(`[eoreader-mcp] ready`);
