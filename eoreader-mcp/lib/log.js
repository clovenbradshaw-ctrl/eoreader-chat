import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const LOG_PATH = process.env.EO_LOG_PATH || "/tmp/eo-mcp-log.jsonl";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CHAT_DIR = path.join(__dirname, "..", "..", "chats");
function _ensureChatDir() {
  if (!fs.existsSync(CHAT_DIR)) fs.mkdirSync(CHAT_DIR, { recursive: true });
}
function _chatPath(session) {
  return path.join(CHAT_DIR, `${session}.jsonl`);
}
function _mirrorToChat(session, entry) {
  if (!session) return;
  try {
    _ensureChatDir();
    fs.appendFileSync(_chatPath(session), JSON.stringify(entry) + "\n");
  } catch {}
}

export function write(entry) {
  const e = { ...entry, t: Date.now(), id: entry.id || `entry:${Date.now()}:${Math.random().toString(36).slice(2, 8)}` };
  try {
    fs.appendFileSync(LOG_PATH, JSON.stringify(e) + "\n");
  } catch {}

  // Mirror to per-session chat file
  _mirrorToChat(entry.session, {
    role: "engine",
    type: e.type,
    layer: e.layer,
    data: _redact(e),
    timestamp: e.t,
  });

  return e;
}

function _redact(entry) {
  const skip = new Set(["type", "layer", "session", "t", "id", "tags"]);
  const out = {};
  for (const [k, v] of Object.entries(entry)) {
    if (skip.has(k)) continue;
    if (typeof v === "string" && v.length > 500) out[k] = v.slice(0, 500) + `... (${v.length} chars)`;
    else if (Array.isArray(v) && v.length > 20) out[k] = `[${v.length} items]`;
    else out[k] = v;
  }
  return out;
}

export function read(filter = {}) {
  try {
    const raw = fs.readFileSync(LOG_PATH, "utf8");
    const entries = raw.trim().split("\n").filter(Boolean).map(l => JSON.parse(l));
    return entries.filter(e => {
      if (filter.type && e.type !== filter.type) return false;
      if (filter.layer !== undefined && e.layer !== filter.layer) return false;
      if (filter.turn && e.turn !== filter.turn) return false;
      if (filter.session && e.session !== filter.session) return false;
      if (filter.tag && (!e.tags || !e.tags.includes(filter.tag))) return false;
      if (filter.superseded === false && e.supersedes) return false;
      return true;
    }).sort((a, b) => (a.t || 0) - (b.t || 0));
  } catch { return []; }
}

export function latest(filter = {}) {
  const entries = read(filter);
  return entries[entries.length - 1] || null;
}

export function session(sessionId) {
  return read({ session: sessionId });
}

export function clear() {
  try { fs.writeFileSync(LOG_PATH, ""); } catch {}
}

export function entriesByType(type, filter = {}) {
  return read({ ...filter, type });
}

export { LOG_PATH, CHAT_DIR };
