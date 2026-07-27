import fs from "fs";

const LOG_PATH = process.env.EO_LOG_PATH || "/tmp/eo-mcp-log.jsonl";

export function write(entry) {
  const e = { ...entry, t: Date.now(), id: entry.id || `entry:${Date.now()}:${Math.random().toString(36).slice(2, 8)}` };
  try {
    fs.appendFileSync(LOG_PATH, JSON.stringify(e) + "\n");
  } catch {}
  return e;
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

export { LOG_PATH };
