// priors-source.js — makes eoPriors browsable, searchable, and citable in
// eochat as a first-class source, WITHOUT letting priors leak into corpus
// grounding.
//
// Two layers per prior, both real files on disk so every citation carries a
// verifiable byte range (engine-ground.js's read paths fs.readFileSync the
// resolved path — a synthetic in-memory document would make /api/verbatim/
// context and /segment unresolvable, and citations unfalsifiable):
//
//   RAW  — the prior's own .json, ingested unmodified. Cite this and the
//          byte range opens the actual artifact.
//   CARD — a rendered markdown projection under .derived/prior-cards/,
//          ingested alongside. JSON tokenizes badly (keys, braces, and
//          numerics dominate the signal), so the card is what retrieval
//          actually matches on; it links back to the raw path.
//
// Both live in the "priors" POOL, never the corpus pool. That boundary is the
// point: a prior is witness-tier knowledge ABOUT a corpus, not evidence FROM
// one. If lens-ledger.json could be returned as a grounding passage for "what
// happens to the creature," the engine/model tier distinction the codebase is
// built on would be gone. Priors are retrieved only when explicitly asked for.
//
// The card RENDERS structure; it never interprets it. Every line is a
// mechanical projection of a JSON node — no summary sentences, no inferred
// significance. Anything a card omits (oversized raw files, truncated tables)
// is stated in the card as an explicit gap with the path to read instead,
// never silently dropped.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { engineIngestFile, engineSearch, DEFAULT_POOL } from "./engine-ground.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const PRIORS_POOL = "priors";
const PRIORS_ROOT = path.resolve(__dirname, "..", "eoPriors");
const PRIORS_DIR = path.join(PRIORS_ROOT, "priors");
const CARDS_DIR = path.join(PRIORS_ROOT, ".derived", "prior-cards");

// Above this, the raw JSON is catalogued and readable by byte range but not
// admitted to the engine. corpus-prior-cube.json is 7.8MB — ~3,900 chunks of
// mostly-numeric cells that would swamp the pool's retrieval for no gain. The
// card still carries its structure, and the omission is reported as a gap.
const RAW_INGEST_MAX_BYTES = 2 * 1024 * 1024;

// Table rows rendered per array before truncating. Cards are for finding your
// way to the right artifact, not for replacing it.
const MAX_TABLE_ROWS = 40;
const MAX_LIST_ITEMS = 30;

// ── Catalog ──

function walkJson(dir, base = "") {
  const out = [];
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (e.name.startsWith(".")) continue;
    const abs = path.join(dir, e.name);
    const rel = base ? `${base}/${e.name}` : e.name;
    if (e.isDirectory()) out.push(...walkJson(abs, rel));
    else if (e.name.endsWith(".json")) out.push({ id: rel.replace(/\.json$/, ""), rel, abs });
  }
  return out;
}

// A prior's "family" is its declared schema when it has one, else its version
// key. Declared, not guessed from the filename — a name is not an identity.
function familyOf(json) {
  if (!json || typeof json !== "object") return "unknown";
  if (typeof json.schema === "string") return json.schema;
  const versionKey = Object.keys(json).find((k) => /_version$/.test(k));
  if (versionKey && typeof json[versionKey] === "string") return json[versionKey];
  return "unstructured";
}

// What a prior is ABOUT, read off what it declares — never inferred from its
// filename or its schema. A prior scoped to one text is a different kind of
// evidence from one folded over a corpus: the first applies only while reading
// that text, the second conditions any reading. That distinction is the useful
// one at the surface, and both artifacts state it themselves.
//
// `kind` is one of:
//   text   — declares a single source text (`source: "… pg84 — Frankenstein …"`)
//   prior  — declares another prior as its source (holons-L2 ← holons-L1.json)
//   corpus — declares the corpus it was generated from, or the texts it covers
//   none   — declares no subject. A typed gap, not a guess: it groups under
//            "scope not declared" rather than being quietly filed as corpus-wide.
function scopeOf(json) {
  if (!json || typeof json !== "object") return { kind: "none", label: "scope not declared" };

  const src = json.source;
  if (typeof src === "string" && src.trim()) {
    // A `source` naming another artifact is a derivation, not a subject.
    if (/\.json$/i.test(src.trim())) {
      return { kind: "prior", label: "derived from another prior", detail: src.trim() };
    }
    return { kind: "text", label: src.trim() };
  }

  const gen = json.generated_from;
  if (gen && typeof gen === "object") {
    const corpus = gen.corpus_dir_basename || gen.corpus_dir;
    const books = typeof gen.books === "number" ? gen.books : null;
    if (corpus) {
      return {
        kind: "corpus",
        label: "corpus-wide",
        detail: books != null ? `${corpus} · ${books} books` : String(corpus),
      };
    }
  }

  if (Array.isArray(json.texts) && json.texts.length) {
    return { kind: "corpus", label: "corpus-wide", detail: `${json.texts.length} texts` };
  }

  // An artifact that states WHY it has no scope has said something real, and
  // the reason travels with the gap. Undeclared-and-explained beats
  // undeclared-and-silent: the reader can tell "nobody got to it yet" from
  // "there is genuinely no corpus behind this one".
  const why = typeof json.provenance_gap === "string" && json.provenance_gap.trim()
    ? json.provenance_gap.trim()
    : null;
  return { kind: "none", label: "scope not declared", gap: why };
}

function hashOf(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

let catalogCache = null;

// Parse a prior's JSON on demand. Deliberately NOT cached: the catalog is a
// long-lived module singleton, and holding every parsed artifact in it
// retained ~33MB — most of it corpus-prior-cube.json (7.7MB of text becoming
// a far larger object graph) — for the entire life of the proxy, to serve a
// listing that only needs each file's size, family, scope and key names.
// Cards are rebuilt only when a file's hash changes, so this parses rarely.
export function readPriorJson(item) {
  try {
    return { json: JSON.parse(fs.readFileSync(item.path, "utf8")), gap: null };
  } catch (err) {
    return { json: null, gap: `unparseable JSON: ${err.message}` };
  }
}

export function priorsCatalog({ refresh = false } = {}) {
  if (catalogCache && !refresh) return catalogCache;
  const items = [];
  for (const { id, rel, abs } of walkJson(PRIORS_DIR)) {
    const stat = fs.statSync(abs);
    let json = null;
    let parseGap = null;
    try {
      json = JSON.parse(fs.readFileSync(abs, "utf8"));
    } catch (err) {
      parseGap = `unparseable JSON: ${err.message}`;
    }
    items.push({
      id,
      rel,
      path: abs,
      bytes: stat.size,
      family: parseGap ? "unreadable" : familyOf(json),
      scope: parseGap ? { kind: "none", label: "scope not declared" } : scopeOf(json),
      keys: json && typeof json === "object" ? Object.keys(json) : [],
      cardPath: path.join(CARDS_DIR, `${id}.md`),
      rawIngestable: stat.size <= RAW_INGEST_MAX_BYTES,
      gap: parseGap
        || (stat.size > RAW_INGEST_MAX_BYTES
          ? `raw not admitted to the engine (${(stat.size / 1048576).toFixed(1)}MB > ${RAW_INGEST_MAX_BYTES / 1048576}MB cap); card is indexed, raw is readable by byte range`
          : null),
    });
    // `json` goes out of scope here rather than onto the cached entry — see
    // readPriorJson above. Card rendering re-reads it when it actually needs it.
  }
  catalogCache = items;
  return items;
}

export function findPrior(id) {
  const wanted = String(id || "").replace(/\.json$/, "");
  const cat = priorsCatalog();
  return cat.find((p) => p.id === wanted)
    || cat.find((p) => p.id.endsWith(`/${wanted}`))
    || cat.find((p) => p.path === id)
    || null;
}

// ── Card rendering ──
//
// A mechanical JSON → markdown projection. Three shapes get special handling
// because they carry the signal: scalars (inline), arrays of uniform objects
// (tables), and everything else (nested sections). Nothing is summarized.

const isScalar = (v) => v === null || ["string", "number", "boolean"].includes(typeof v);

// Slicing a JSON blob mid-token leaves the reader unable to tell a truncation
// from the artifact genuinely ending there. Always mark the cut.
function jsonExcerpt(value, maxChars) {
  const full = JSON.stringify(value, null, 2);
  if (full.length <= maxChars) return full;
  return `${full.slice(0, maxChars)}\n… truncated at ${maxChars} of ${full.length} chars`;
}

function fmtScalar(v) {
  if (v === null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(6).replace(/0+$/, "");
  const s = String(v);
  // Long hex digests are provenance, not prose: keep them findable but short.
  if (/^[0-9a-f]{32,}$/i.test(s)) return `${s.slice(0, 12)}… (${s.length} hex)`;
  return s.length > 300 ? `${s.slice(0, 300)}…` : s;
}

function cellFor(v) {
  if (isScalar(v)) return fmtScalar(v).replace(/\|/g, "\\|").replace(/\n/g, " ");
  if (Array.isArray(v)) return `[${v.length} items]`;
  return `{${Object.keys(v).slice(0, 4).join(", ")}}`;
}

// Columns are the union of keys across sampled rows, so a field present on
// only some rows still gets a column rather than vanishing.
function columnsOf(rows) {
  const cols = [];
  const seen = new Set();
  for (const r of rows.slice(0, 50)) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k)) { seen.add(k); cols.push(k); }
    }
  }
  return cols.slice(0, 10);
}

function renderArray(key, arr, depth, ctx) {
  const out = [];
  const heading = "#".repeat(Math.min(depth + 2, 6));
  out.push(`${heading} ${key} — ${arr.length} ${arr.length === 1 ? "entry" : "entries"}`, "");

  if (arr.length === 0) { out.push("_(empty)_", ""); return out; }

  if (arr.every(isScalar)) {
    const shown = arr.slice(0, MAX_LIST_ITEMS).map(fmtScalar);
    out.push(shown.map((s) => `- ${s}`).join("\n"));
    if (arr.length > MAX_LIST_ITEMS) {
      const dropped = arr.length - MAX_LIST_ITEMS;
      out.push(`- _… ${dropped} more not rendered — read \`${ctx.rel}\` for all ${arr.length}_`);
      ctx.truncations.push(`${key}: ${dropped} of ${arr.length} items not rendered`);
    }
    out.push("");
    return out;
  }

  if (arr.every((v) => v && typeof v === "object" && !Array.isArray(v))) {
    const cols = columnsOf(arr);
    const rows = arr.slice(0, MAX_TABLE_ROWS);
    out.push(`| ${cols.join(" | ")} |`, `| ${cols.map(() => "---").join(" | ")} |`);
    for (const r of rows) out.push(`| ${cols.map((c) => cellFor(r[c])).join(" | ")} |`);
    out.push("");
    if (arr.length > MAX_TABLE_ROWS) {
      const dropped = arr.length - MAX_TABLE_ROWS;
      out.push(`_… ${dropped} more rows not rendered — read \`${ctx.rel}\` for all ${arr.length}._`, "");
      ctx.truncations.push(`${key}: ${dropped} of ${arr.length} rows not rendered`);
    }
    // One expanded exemplar, so nested structure the table flattened to
    // "{a, b}" is still visible somewhere in the card.
    const nested = cols.filter((c) => arr[0][c] && typeof arr[0][c] === "object");
    if (nested.length) {
      out.push(`First entry expanded:`, "", "```json", jsonExcerpt(arr[0], 1400), "```", "");
    }
    return out;
  }

  out.push("```json", jsonExcerpt(arr.slice(0, 5), 1600), "```", "");
  if (arr.length > 5) {
    out.push(`_… ${arr.length - 5} more of ${arr.length} mixed-shape entries not rendered — read \`${ctx.rel}\`._`, "");
    ctx.truncations.push(`${key}: ${arr.length - 5} of ${arr.length} mixed-shape entries not rendered`);
  }
  return out;
}

function renderNode(key, value, depth, ctx) {
  if (isScalar(value)) return [`- **${key}**: ${fmtScalar(value)}`];
  if (Array.isArray(value)) return renderArray(key, value, depth, ctx);

  const entries = Object.entries(value);
  const scalars = entries.filter(([, v]) => isScalar(v));
  const complex = entries.filter(([, v]) => !isScalar(v));
  const out = [];
  const heading = "#".repeat(Math.min(depth + 2, 6));
  out.push(`${heading} ${key}`, "");
  if (scalars.length) {
    out.push(...scalars.map(([k, v]) => `- **${k}**: ${fmtScalar(v)}`), "");
  }
  // Depth guard: past three levels the projection stops being readable and
  // the raw file is the better artifact to send the reader to.
  if (depth >= 3) {
    if (complex.length) {
      out.push(`_${complex.length} nested field(s) not expanded at this depth — read \`${ctx.rel}\`._`, "");
      ctx.truncations.push(`${key}: ${complex.length} nested fields beyond depth 3`);
    }
    return out;
  }
  for (const [k, v] of complex) out.push(...renderNode(k, v, depth + 1, ctx));
  return out;
}

// `json` is passed in rather than read off the catalog entry, so the parsed
// artifact lives only as long as the render that needs it.
export function renderCard(item, json = readPriorJson(item).json) {
  const ctx = { rel: `eoPriors/priors/${item.rel}`, truncations: [] };
  const body = [];

  if (!json) {
    return [
      `# ${item.id}`, "",
      `- **source**: \`${ctx.rel}\``,
      `- **bytes**: ${item.bytes}`,
      "",
      `## Gap`, "", item.gap || "prior could not be read", "",
    ].join("\n");
  }

  const entries = Object.entries(json);
  const scalars = entries.filter(([, v]) => isScalar(v));
  const complex = entries.filter(([, v]) => !isScalar(v));

  body.push(`# ${item.id}`, "");
  body.push(
    `- **family**: ${item.family}`,
    `- **source**: \`${ctx.rel}\``,
    `- **bytes**: ${item.bytes}`,
    `- **top-level keys**: ${entries.map(([k]) => k).join(", ")}`,
    "",
  );
  if (scalars.length) {
    body.push(`## Declared`, "", ...scalars.map(([k, v]) => `- **${k}**: ${fmtScalar(v)}`), "");
  }
  for (const [k, v] of complex) body.push(...renderNode(k, v, 1, ctx));

  const gaps = [...ctx.truncations];
  if (item.gap) gaps.unshift(item.gap);
  body.push(
    `## Gaps`, "",
    gaps.length
      ? gaps.map((g) => `- ${g}`).join("\n")
      : `- none — this card renders every field of \`${ctx.rel}\``,
    "",
    `_This card is a mechanical projection of \`${ctx.rel}\`. It renders structure; it does not interpret it. The raw artifact is the authority._`,
    "",
  );
  return body.join("\n");
}

// Write cards for every catalogued prior. A card is rewritten only when the
// raw artifact's hash changes, so cards are stable across restarts and their
// byte offsets don't churn under previously-issued citations.
export function buildCards({ force = false } = {}) {
  fs.mkdirSync(CARDS_DIR, { recursive: true });
  const written = [];
  for (const item of priorsCatalog()) {
    // Hash the bytes without holding them: the stamp decides whether this card
    // needs rebuilding at all, and most runs rebuild nothing.
    const stamp = `<!-- source-sha256: ${hashOf(fs.readFileSync(item.path))} -->`;
    fs.mkdirSync(path.dirname(item.cardPath), { recursive: true });
    if (!force && fs.existsSync(item.cardPath)) {
      const existing = fs.readFileSync(item.cardPath, "utf8");
      if (existing.startsWith(stamp)) continue;
    }
    // Parsed here, dropped when the loop iteration ends.
    fs.writeFileSync(item.cardPath, `${stamp}\n${renderCard(item, readPriorJson(item).json)}`, "utf8");
    written.push(item.id);
  }
  return written;
}

// ── Ingest ──

let ingested = null;

export function ensurePriorsIngested({ force = false } = {}) {
  if (ingested && !force) return ingested;
  buildCards({ force });

  const sources = [];
  const gaps = [];
  for (const item of priorsCatalog()) {
    try {
      const card = engineIngestFile(item.cardPath, {
        pool: PRIORS_POOL,
        kind: "prior-card",
        displayName: `${item.id} (card)`,
      });
      sources.push({ id: item.id, layer: "card", chunks: card.chunks, path: item.cardPath });
    } catch (err) {
      gaps.push(`card ingest failed for ${item.id}: ${err.message}`);
    }

    if (!item.rawIngestable) {
      gaps.push(`${item.id}: ${item.gap}`);
      continue;
    }
    try {
      const raw = engineIngestFile(item.path, {
        pool: PRIORS_POOL,
        kind: "prior-raw",
        displayName: `${item.id}.json`,
      });
      sources.push({ id: item.id, layer: "raw", chunks: raw.chunks, path: item.path });
    } catch (err) {
      gaps.push(`raw ingest failed for ${item.id}: ${err.message}`);
    }
  }

  ingested = { pool: PRIORS_POOL, priors: priorsCatalog().length, sources, gaps };
  return ingested;
}

// ── Read & search ──

// Read a prior's bytes directly — the escape hatch for artifacts too large to
// admit, and the way to verify any citation this module produced.
export function readPrior(id, { layer = "raw", byteStart = 0, maxBytes = 40000 } = {}) {
  const item = findPrior(id);
  if (!item) return { error: `unknown prior "${id}"` };
  const target = layer === "card" ? item.cardPath : item.path;
  if (!fs.existsSync(target)) return { error: `${layer} not built for "${item.id}" (${target})` };
  const stat = fs.statSync(target);
  const start = Math.max(0, Math.min(byteStart, stat.size));
  const length = Math.min(maxBytes, stat.size - start);
  const fd = fs.openSync(target, "r");
  const buf = Buffer.alloc(length);
  try {
    fs.readSync(fd, buf, 0, length, start);
  } finally {
    fs.closeSync(fd);
  }
  return {
    id: item.id,
    layer,
    path: target,
    family: item.family,
    byte_start: start,
    byte_end: start + length,
    total_bytes: stat.size,
    truncated: start + length < stat.size,
    gap: item.gap,
    text: buf.toString("utf8"),
  };
}

// Search the priors pool. Never touches the corpus pool — asking about the
// priors and asking about the texts are different questions with different
// evidence, and the caller has to say which one it means.
export function searchPriors(query, limit = 8, { maxChars = 900, prior } = {}) {
  ensurePriorsIngested();
  if (DEFAULT_POOL === PRIORS_POOL) throw new Error("priors pool must not be the default pool");
  // Filter by the prior's ID, not its path: engineSearch matches on the
  // filter's basename, and the id (no extension) is the one string that
  // matches BOTH layers — `lens-fold.json` and `lens-fold.md`. Passing the raw
  // path would silently exclude the card, which is the layer retrieval is
  // most likely to have hit.
  const source = prior ? (findPrior(prior)?.id ?? prior) : null;
  return engineSearch(query, limit, { maxChars, source, pool: PRIORS_POOL });
}
