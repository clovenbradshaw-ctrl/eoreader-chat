#!/usr/bin/env node
/**
 * Proxy with EO Cube coordinate system
 * 
 * Everything gets mapped to the cube:
 * - ACT: what operation (NUL/SEG/DEF/SIG/CON/EVA/INS/SYN/REC)
 * - SITE: where it lands (9 terrains)
 * - STANCE: how it resolves (9 stances)
 * 
 * Usage:
 *   node proxy.js --repo=/path/to/repo [--port 11435] [--target http://localhost:11434]
 */

import http from "http";
import fs from "fs";
import path from "path";

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const idx = args.findIndex(a => a === `--${name}` || a.startsWith(`--${name}=`));
  if (idx >= 0) {
    const arg = args[idx];
    if (arg.includes("=")) return arg.split("=").slice(1).join("=");
    return args[idx + 1];
  }
  return def;
};

const REPO_PATH = getArg("repo", process.cwd());
const PORT = parseInt(getArg("port", "11435"));
const TARGET = getArg("target", "http://localhost:11434");
const TOKEN_LIMIT = parseInt(getArg("limit", "3000"));

// ── EO Cube ──

const OPERATORS = ["NUL", "SEG", "DEF", "SIG", "CON", "EVA", "INS", "SYN", "REC"];
const TERRAINS = ["Void", "Entity", "Kind", "Field", "Link", "Network", "Atmosphere", "Lens", "Paradigm"];
const STANCES = ["Clearing", "Dissecting", "Unraveling", "Tending", "Binding", "Tracing", "Cultivating", "Making", "Composing"];

// Diagonal cells: operator → [{terrain, stance}]
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

// Map code patterns to operators
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

// Map message patterns to operators
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

// Get cell from operator
function getCell(operator, grain = "Figure") {
  const cells = DIAGONAL[operator] || DIAGONAL.NUL;
  return cells.find(c => c.grain === grain) || cells[1];
}

// ── Store ──

let store = [];
let position = { x: 0, y: 0 };
let momentum = 0;

function hash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = ((h << 5) - h) + str.charCodeAt(i) | 0;
  return h.toString(16).padStart(8, "0");
}

function ingest(text, type, meta = {}) {
  const operator = type === "code" ? classifyCode(text) : classifyMessage(text);
  const cell = getCell(operator);
  
  store.push({
    id: hash(text + Date.now()),
    text,
    type,
    meta,
    cell,
    ts: Date.now()
  });
  
  // Physics from ingestion
  const drift = (text || "").split(/\s+/).length * 0.01;
  position.x += (Math.random() - 0.5) * drift;
  position.y += (Math.random() - 0.5) * drift;
  momentum = Math.min(1.0, momentum + drift * 0.1);
}

function search(query, topK = 5) {
  const queryOperator = classifyMessage(query);
  const queryCell = getCell(queryOperator);
  const words = (query || "").toLowerCase().split(/\s+/).filter(w => w.length > 1);
  
  return store.map(e => {
    const text = e.text.toLowerCase();
    let score = 0;
    
    // Text relevance
    for (const w of words) {
      if (text.includes(w)) score += 2;
      for (const t of text.split(/\s+/)) {
        if (t === w) score += 1;
        else if (t.includes(w) || w.includes(t)) score += 0.5;
      }
    }
    
    // Cell proximity (diagonal = high score)
    if (e.cell && queryCell) {
      if (e.cell.terrain === queryCell.terrain) score += 3;
      if (e.cell.stance === queryCell.stance) score += 2;
    }
    
    return { ...e, score };
  })
  .filter(s => s.score > 0)
  .sort((a, b) => b.score - a.score)
  .slice(0, topK);
}

// ── Load code ──

function loadCode(repo) {
  const ignore = new Set(["node_modules", ".git", "dist", "build", "__pycache__"]);
  const skip = new Set([".json", ".lock", ".map", ".png", ".jpg", ".gif", ".ico"]);

  function walk(dir) {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if (ignore.has(e.name)) continue;
      const p = path.join(dir, e.name);
      if (e.isDirectory()) { walk(p); continue; }
      if (e.isFile() && !skip.has(path.extname(e.name)) && !e.name.includes(".test.") && !e.name.includes(".spec.")) {
        try {
          const content = fs.readFileSync(p, "utf8");
          if (content.length < 20) continue;
          const rel = p.replace(repo + "/", "");
          const lines = content.split("\n");
          let chunk = [], size = 0;
          for (const line of lines) {
            chunk.push(line);
            size += line.length;
            if (size > 1200 || line.match(/^(module\.exports|export\s)/)) {
              const text = chunk.join("\n");
              if (text.trim().length > 30) ingest(text, "code", { file: rel });
              chunk = [];
              size = 0;
            }
          }
          if (chunk.length > 0) {
            const text = chunk.join("\n");
            if (text.trim().length > 30) ingest(text, "code", { file: rel });
          }
        } catch {}
      }
    }
  }
  walk(repo);
}

// ── Tokens ──

const tok = (t) => Math.ceil((t || "").length / 3.5);

// ── Context ──

function assemble(messages) {
  const latest = [...messages].reverse().find(m => m.role === "user");
  if (!latest) return messages;
  const query = latest.content || "";
  const queryOperator = classifyMessage(query);
  const queryCell = getCell(queryOperator);
  
  momentum *= 0.9;
  
  let ctx = [], t = 0;
  const sys = messages.find(m => m.role === "system");
  if (sys) { t += tok(sys.content); ctx.push({ role: "system", content: sys.content }); }
  else {
    const d = "You are an expert software engineer. Use the provided code.";
    t += tok(d); ctx.push({ role: "system", content: d });
  }

  const results = search(query, 5);
  if (results.length) {
    const c = "\n[Context: " + queryCell.terrain + "/" + queryCell.stance + "]\n" + 
      results.map(r => r.type === "code" ? 
        `--- ${r.meta.file} [${r.cell?.operator || "?"}] ---\n${r.text.slice(0, 500)}` : 
        `[${r.type}]: ${r.text.slice(0, 300)}`
      ).join("\n\n");
    if (t + tok(c) < TOKEN_LIMIT) { t += tok(c); ctx.push({ role: "system", content: c }); }
  }

  ctx.push({ role: "user", content: query });
  console.log(`[proxy] ${t} tokens, query=${queryOperator}(${queryCell.terrain}/${queryCell.stance})`);
  return ctx;
}

// ── Server ──

console.log(`[proxy] Loading ${REPO_PATH}`);
loadCode(REPO_PATH);
console.log(`[proxy] ${store.length} chunks`);

const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") { res.writeHead(200); res.end(); return; }

  if (req.method === "POST" && req.url === "/v1/chat/completions") {
    let body = "";
    req.on("data", c => body += c);
    req.on("end", async () => {
      try {
        const data = JSON.parse(body);
        for (const m of data.messages || []) if (m.content?.length > 5) ingest(m.content, m.role);
        data.messages = assemble(data.messages);
        const r = await fetch(`${TARGET}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
        const text = await r.text();
        try {
          const parsed = JSON.parse(text);
          const content = parsed.choices?.[0]?.message?.content || "";
          if (content.length > 5) ingest(content, "assistant");
        } catch {}
        res.writeHead(r.status, { "Content-Type": "application/json" });
        res.end(text);
      } catch (e) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
  } else {
    try {
      const r = await fetch(`${TARGET}${req.url}`, { method: req.method, headers: { ...req.headers, host: new URL(TARGET).host } });
      res.writeHead(r.status, { "Content-Type": "application/json" });
      res.end(await r.text());
    } catch (e) {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: e.message }));
    }
  }
});

server.listen(PORT, () => console.log(`[proxy] Ready :${PORT}`));
