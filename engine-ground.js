// Ground-Truth engine bridge: ingests files into the real eoreader5 session,
// searches spans with byte-offset anchors, folds results under a token budget,
// and returns grounded context the LLM can cite.
//
// Sessions are grouped into named POOLS. The default pool ("corpus") holds
// ingested source texts — a chat message about "the creature" after ingesting
// Frankenstein and War and Peace gets interleaved results from both.
//
// A pool is a retrieval boundary, not a label: each pool owns its own engine
// session and span registry, so searchSpans in one pool can never return a
// span from another. That is what keeps the "priors" pool (eoPriors artifacts,
// see priors-source.js) out of corpus grounding — a question about the
// creature must not be answered with lens-ledger JSON. Priors are witness-tier
// knowledge about the corpus; corpus text is evidence from it. Mixing them in
// one retrieval pool would let the former be cited as the latter.
//
// Hosts the impure createSession/ingestFile/searchSpans/foldSpans corpus facade.

import fs from "node:fs";
import path from "node:path";
import {
  CORPUS_API_VERSION,
  createSession,
  admitChunked,
  ingestFile,
  searchSpans,
  spanUnits,
  foldSpans,
  readSpan,
} from "@eoreader/host/corpus";
// The whole-document facade (documentIds/documentText/sessionOutline/
// sessionReferents) arrived in corpus API v2 and backs engineFoldSource only.
// It is reached through a namespace import on purpose: a named import of a
// symbol a v1 host does not export fails at module-LINK time, which would
// replace the readable version guard below with a bare SyntaxError at boot.
// This way a v1 host still links and the fold path reports a typed gap.
import * as corpusFacade from "@eoreader/host/corpus";
import { INDIVIDUATION_TYPES } from "@eoreader/engine/referents";
import { coverageReport } from "@eoreader/engine";
import { loadCorefPrior, surfaceMatcher } from "./priors-bridge.js";

// v2 is additive over v1 — it adds documentIds/documentText/sessionOutline/
// sessionReferents/sessionPivot and changes no signature this bridge calls
// (the only line removed in the v1→v2 diff was the version constant itself).
// Listing known-compatible versions rather than pinning one keeps the guard's
// real job — failing at boot on an incompatible change instead of surfacing a
// wrong answer inside a chat turn weeks later — while not breaking eochat on
// every purely-additive engine release.
const SUPPORTED_CORPUS_API = new Set([1, 2]);
if (!SUPPORTED_CORPUS_API.has(CORPUS_API_VERSION)) {
  throw new Error(
    `@eoreader/host/corpus is API v${CORPUS_API_VERSION}; this bridge supports v${[...SUPPORTED_CORPUS_API].join(", v")}`,
  );
}

// ── Pools & sessions ──

export const DEFAULT_POOL = "corpus";

// poolName -> { name, session, sources: Map<sourceKey, info>, chunkCount }
const pools = new Map();

function pool(name = DEFAULT_POOL) {
  const key = name || DEFAULT_POOL;
  let p = pools.get(key);
  if (!p) {
    p = { name: key, session: createSession({ spanCap: Number.MAX_SAFE_INTEGER }), sources: new Map(), chunkCount: 0 };
    pools.set(key, p);
  }
  return p;
}

export function ensureSession(poolName = DEFAULT_POOL) {
  return pool(poolName).session;
}

export function listPools() {
  return Array.from(pools.keys());
}

// Locate the pool whose span registry holds `spanId`. Span ids are
// content-addressed, so a collision across pools would mean identical bytes
// from an identical source id — the same span by every definition the registry
// has. Reads therefore don't need the caller to know which pool it searched.
function poolForSpan(spanId) {
  for (const p of pools.values()) if (p.session.spans.has(spanId)) return p;
  return null;
}

// ── Ingest ──

// `displayName` is for sources whose id carries no usable filename — an upload
// or a URL — where stripping to the last path segment would yield noise.
// `kind` labels what this source IS ("corpus", "prior-raw", "prior-card"); it
// travels to /api/sources so the UI can pill priors differently from texts.
export function engineIngestText(text, sourceId, displayName, { pool: poolName = DEFAULT_POOL, kind = "corpus" } = {}) {
  const p = pool(poolName);
  const { chunks, admitted } = admitChunked(p.session, { text, sourceId });
  p.chunkCount += chunks;
  const name = displayName || sourceId?.replace(/^.*[/\\]/, "") || "(unnamed)";
  p.sources.set(sourceId || name, { name, chunks, kind, pool: p.name, ingestedAt: Date.now() });
  return {
    sourceId,
    chunks,
    pool: p.name,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

// ── Search ──

// A source filter is either one name or a set of them — the UI's per-source
// toggles hand us the list of sources the reader left switched on. Matching is
// on basename, since callers hold display names while spans hold full ids.
//
// The three states are distinct and must stay so: absent (null/undefined/"")
// means "no filter, every source"; a non-empty set means "only these"; an
// EMPTY ARRAY means the reader switched every source off and must match
// nothing. Collapsing that last case to "everything" would answer from
// sources the reader explicitly excluded.
function sourceMatcher(sourceFilter) {
  if (sourceFilter == null || sourceFilter === "") return null;
  const bases = (Array.isArray(sourceFilter) ? sourceFilter : [sourceFilter])
    .map((x) => String(x).replace(/^.*[/\\]/, ""))
    .filter(Boolean);
  if (!Array.isArray(sourceFilter) && !bases.length) return null;
  return (sp) => !!sp.source_id && bases.some((b) => sp.source_id.includes(b));
}

export function engineSearch(query, limit = 10, { maxChars = 800, source: sourceFilter, pool: poolName = DEFAULT_POOL } = {}) {
  const s = ensureSession(poolName);
  let { spans, gaps } = searchSpans(s, { query, limit: Math.min(limit, 40) });
  const match = sourceMatcher(sourceFilter);
  if (match) spans = spans.filter(match);
  const units = spanUnits(s, spans);
  return {
    query,
    pool: poolName,
    total: spans.length,
    passages: spans.map((sp, i) => {
      const full = units[i]?.text ?? sp.preview;
      const cap = Math.max(1, maxChars);
      const truncated = full.length > cap;
      const text = full.slice(0, cap);
      // byte_end must describe what `text` actually covers, never the whole
      // span. A caller reads bytes [byte_start, byte_end) from the source file
      // to VERIFY a citation (UX-DESIGN.md's "citation verifiability: 100% of
      // citations link to exact source") — reporting the full span's byte_end
      // while returning a truncated `text` made every truncated passage's
      // citation promise more than it delivered, and a byte-range read of the
      // file would not equal `text` at all. Truncation is character-count, not
      // byte-count, so the boundary must be re-measured in UTF-8 bytes rather
      // than assumed equal to the character count.
      const byte_end = truncated && sp.byte_start != null
        ? sp.byte_start + Buffer.byteLength(text, "utf8")
        : sp.byte_end;
      return {
        span_id: sp.span_id,
        text,
        truncated,
        source: sp.source_id || "",
        byte_start: sp.byte_start,
        byte_end,
        score: sp.score,
        coverage: sp.coverage,
        phrase: sp.phrase,
        preview: sp.preview,
      };
    }),
    gaps,
  };
}

// ── Fold & ground ──

// Build a context string that includes numbered verbatim citations the LLM
// can actually cite from.  The fold summary (if any) follows below.
function buildCitedContext(kept, foldResult, gaps) {
  const parts = [];

  // Numbered verbatim citations — these are what the LLM cites with [1], [2]
  if (kept.length > 0) {
    const citationBlock = kept
      .map((c, i) => {
        const src = c.source_id?.replace(/^source:/, "").replace(/:chunk-\d+$/, "") || "?";
        return `[${i + 1}] (${src} @ byte ${c.byte_start}-${c.byte_end})\n${c.text}`;
      })
      .join("\n\n");
    parts.push(`=== CITED PASSAGES ===\n${citationBlock}`);
  }

  // Fold summary as supplemental context
  if (foldResult.summary) {
    parts.push(`=== FOLD SUMMARY ===\n${foldResult.summary}`);
  }

  // Gaps
  if (gaps?.length) {
    parts.push(`[Engine gaps: ${gaps.map(g => g.reason || g).join("; ")}]`);
  }

  return parts.join("\n\n");
}

// Search + fold into a single compact context block the LLM can consume.
// Returns the folded summary and the underlying evidence passages so the UI
// can display citations.
//
// Citations returned here carry the engine's full mechanical citation record:
//   span_id, source_id, byte_start, byte_end (allowing verification against
//   the original file), the full verbatim text from the span registry, and
//   score.  The text is the EXACT admitted value — not a preview, not a
//   reconstruction.  Callers that need a shorter snippet truncate explicitly
//   rather than accepting a silently-lossy default.
export function engineGroundQuery(query, { budget = 2400, maxUnits = 16, limit = 30, source: sourceFilter, pool: poolName = DEFAULT_POOL } = {}) {
  const s = ensureSession(poolName);
  let { spans, gaps } = searchSpans(s, { query, limit });
  const match = sourceMatcher(sourceFilter);
  if (match) spans = spans.filter(match);
  const units = spanUnits(s, spans);
  const foldResult = foldSpans(s, { units, query, tokenBudget: budget, maxUnits });

  // Extract the full mechanical citation record for every kept span.
  // The span registry holds byte_start/byte_end and the verbatim admitted text.
  const kept = (foldResult.selected || []).map((u) => {
    const spanId = u.meta?.span_id;
    const rec = spanId ? s.spans.get(spanId) : null;
    return {
      span_id: spanId,
      source_id: rec?.source_id ?? u.meta?.source_id ?? u.meta?.source ?? "unknown",
      byte_start: rec?.byte_start ?? null,
      byte_end: rec?.byte_end ?? null,
      score: rec?.score ?? u.meta?.score ?? 0,
      text: rec?.text ?? u.text ?? "",
    };
  });

  // A unit wider than the whole token budget is dropped entire, not trimmed —
  // so a long passage that matched best can leave zero citations behind while
  // `total` still reports a healthy match count. Silently that reads as "the
  // model ignored its evidence"; it is really "the evidence never fit". Report
  // it as a typed gap so the reader sees which it was.
  if (foldResult.dropped > 0) {
    gaps = [...(gaps || []), {
      type: "fold_budget_exceeded",
      dropped: foldResult.dropped,
      budget: foldResult.budget ?? budget,
      reason: `${foldResult.dropped} matched passage(s) exceeded the ${foldResult.budget ?? budget}-token fold budget and were dropped, not truncated${kept.length === 0 ? " — no citation survives for this query" : ""}`,
    }];
  }

  // Build a context that includes verbatim citation text the LLM can cite
  const context = buildCitedContext(kept, foldResult, gaps);

  // The FULL ranked list, not just the survivors. `total: 12, folded: 2` tells
  // a reader that ten passages vanished but not which, nor why — so the fold
  // reads as loss rather than as a decision. Emitting every retrieved span with
  // its rank, its ranking evidence, and whether the fold kept it makes the
  // whole retrieval step inspectable the moment it finishes, before any model
  // has spoken.
  const keptIds = new Set(kept.map((c) => c.span_id));
  const retrieved = spans.map((sp, i) => {
    const citationIndex = kept.findIndex((c) => c.span_id === sp.span_id);
    const rec = sp.span_id ? s.spans.get(sp.span_id) : null;
    return {
      rank: i + 1,
      span_id: sp.span_id,
      source_id: sp.source_id,
      byte_start: sp.byte_start,
      byte_end: sp.byte_end,
      score: sp.score,
      coverage: sp.coverage,
      phrase: sp.phrase,
      preview: sp.preview,
      // The FULL verbatim span text, for every retrieved span — not just the
      // ones the fold kept. The fold trims the LLM prompt, not the reader's
      // view: a passage that lost the prompt budget must still be servable
      // whole, or "retrieved but uncited" becomes "retrieved but unreadable".
      text: rec?.text ?? sp.text ?? sp.preview,
      kept: keptIds.has(sp.span_id),
      // The [n] the model will cite this as, when the fold kept it.
      citation: citationIndex >= 0 ? citationIndex + 1 : null,
    };
  });

  return {
    query,
    context,
    citations: kept,
    retrieved,
    total: spans.length,
    folded: foldResult.selectedCount,
    tokens: foldResult.tokens,
    budget: foldResult.budget,
    dropped: foldResult.dropped,
    gaps,
  };
}

// Read a specific span's verbatim text by span_id.
// Returns { text, source_id, byte_start, byte_end, truncated } or an error.
// The engine's readSpan guarantees the text matches the source file at the
// reported byte range — that's the mechanical citation contract.
export function engineReadSpan(spanId, maxBytes = 4000) {
  const p = poolForSpan(spanId);
  if (!p) return { error: `unknown span_id ${spanId}. Search first.` };
  return { ...readSpan(p.session, { spanId, maxBytes }), pool: p.name };
}

// ── Context snipping: read segments and arbitrary windows ──
//
// OMNIMODAL CONSTRAINT: no hardcoded patterns.  A "segment" boundary is
// discovered dynamically from whatever structure the text actually has —
// Roman numerals, Arabic chapter numbers, all-caps headings, blank-line
// breaks, or content-signal discontinuities.  A symphony movement with no
// names and no text is a valid first-class target; the detector works if
// structural markers exist and degrades gracefully (context window) if they
// don't.

function resolveSourcePath(sourceId) {
  const match = sourceId?.match(/^source:(.+?)(?::chunk-\d+)?$/);
  return match ? match[1] : null;
}

// Byte-accurate view of a file, cached per (path, mtime, size).
//
// Anchors from the engine are UTF-8 BYTE offsets. Reading a file with
// encoding "utf8" and then calling .slice(byteStart, byteEnd) indexes by
// UTF-16 code unit instead, so every anchor past the first non-ASCII
// character in the file comes back shifted — silently, with plausible-looking
// text. That is the unfalsifiable-citation failure corpus.js's header calls
// out, arriving through the back door. Priors made it unmissable: their
// corpus labels are full of Sanskrit and Chinese transliterations, so the
// drift starts within the first few hundred bytes. Everything below therefore
// slices the Buffer and decodes after, never the decoded string.
const lineIndexCache = new Map();

function fileIndex(filePath) {
  const stat = fs.statSync(filePath);
  const key = `${stat.mtimeMs}:${stat.size}`;
  const hit = lineIndexCache.get(filePath);
  if (hit && hit.key === key) return hit;

  const buf = fs.readFileSync(filePath);
  const lines = buf.toString("utf8").split("\n");
  // starts[i] is the byte offset of line i. Measured with byteLength, not
  // string length, so it stays true across multi-byte characters.
  const starts = new Array(lines.length);
  let at = 0;
  for (let i = 0; i < lines.length; i++) {
    starts[i] = at;
    at += Buffer.byteLength(lines[i], "utf8") + 1; // +1 for the \n
  }
  const rec = { key, buf, lines, starts, bytes: buf.length };
  lineIndexCache.set(filePath, rec);
  return rec;
}

// Index of the line containing `byteOffset` (binary search over starts).
function lineAtByte(idx, byteOffset) {
  let lo = 0;
  let hi = idx.starts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (idx.starts[mid] <= byteOffset) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

const sliceBytes = (buf, start, end) => buf.subarray(start, end).toString("utf8");

// Score a line as a potential structural delimiter.  Returns 0–5.
// Clues: short, followed by blank line, contains numbering or formatting
// that makes it look like a heading rather than a sentence.
function headingScore(line, nextLineBlank) {
  const trimmed = line.trim();
  if (trimmed.length < 3 || trimmed.length > 80) return 0;
  if (!nextLineBlank) return 0;
  // Penalize sentence-like lines (end with ?.! or have multiple capitalized words).
  // Trailing quotation marks are stripped first: a line ending `Bolkónskaya.”`
  // is the tail of a speech, but the raw test only sees the curly quote and
  // waves it through — which is how "Mary Bolkónskaya.”" ended up in the
  // outline of War and Peace as though it were a chapter.
  const unquoted = trimmed.replace(/["'‘’“”]+$/, "");
  if (/[?.!]$/.test(unquoted) && !/^[IVXLCDM]+\.$/.test(unquoted)) return 0;
  // Abbreviation exclusion
  if (/^(M[\. ]|Dr[\. ]|Mr[\. ]|Mme[\. ]|Mlle[\. ]|St[\. ])/i.test(trimmed)) return 0;
  let s = 0;
  if (/^[IVXLCDM]+\.\s/.test(trimmed)) s += 3;
  if (/^\d+[).]\s/ .test(trimmed)) s += 3;
  // A label closed by a bare ordinal \u2014 "Chapter 1", "Letter 4", "Movement 3" \u2014
  // or an ordinal standing alone. The clue below it only fires on a word
  // followed by another CAPITALIZED word, and a digit is not one, so this
  // whole family scored zero: Frankenstein is headed this way from end to end
  // and was arriving as one unnavigable 400KB section.
  // Form, not vocabulary \u2014 nothing here knows what "chapter" means, and it
  // fires identically on a numbered movement in a score with no words at all.
  if (/^[A-Za-z][\w'\u2019]*\s+\d{1,4}$/.test(trimmed) || /^\d{1,4}$/.test(trimmed)) s += 3;
  if (/^[A-Z\s'"\u201c\u201d]{4,}$/.test(trimmed) || /^[A-Z][a-z]+\s+[A-Z]/.test(trimmed)) s += 2;
  if (/[?.!]$/.test(trimmed)) s -= 2;
  return s >= 2 ? s : 0;
}

// Dynamically discover segment boundaries near a byte offset by examining
// all candidate heading lines within a text window and finding the structural
// cluster that contains the anchor.
// Operates on the cached line index, so every position here is a byte offset
// and the anchor comparison is unit-consistent. Previously the line walk
// accumulated `line.length + 1` (characters) and compared it against a byte
// anchor, which drifted apart over any non-ASCII text.
function discoverSegment(idx, nearByte) {
  const radius = Math.min(6000, idx.bytes >> 2);
  const loByte = Math.max(0, nearByte - radius);
  const hiByte = Math.min(idx.bytes, nearByte + radius);
  const firstLine = lineAtByte(idx, loByte);
  const lastLine = lineAtByte(idx, hiByte);
  const anchorIdx = lineAtByte(idx, nearByte);

  const isHeading = (i) => {
    const nextBlank = i + 1 < idx.lines.length && idx.lines[i + 1].trim() === "";
    return headingScore(idx.lines[i], nextBlank) > 0;
  };

  let startIdx = null;
  let endIdx = null;
  for (let i = anchorIdx; i >= firstLine; i--) if (isHeading(i)) { startIdx = i; break; }
  for (let i = anchorIdx + 1; i <= lastLine; i++) if (isHeading(i)) { endIdx = i; break; }
  if (startIdx == null && endIdx == null) return null;
  // No heading behind the anchor within the window: the window edge is where
  // we start reading, but it is NOT a boundary we found. Remembering which of
  // the two it is matters, because the line sitting at an arbitrary offset is
  // mid-sentence, and naming the segment after it reports a fabricated
  // structure — "in her innocence; I knew it. Could the dæmon who had (I did
  // not for a" was being returned as a segment title. A window is a window;
  // say so rather than dressing it as a chapter.
  const startFound = startIdx != null;
  if (!startFound) startIdx = firstLine;

  let headingCount = 0;
  for (let i = firstLine; i <= lastLine; i++) if (isHeading(i)) headingCount++;

  return {
    startByte: idx.starts[startIdx],
    endByte: endIdx != null ? Math.max(idx.starts[startIdx], idx.starts[endIdx] - 1) : hiByte,
    label: startFound ? idx.lines[startIdx].trim() : "(context window — no heading precedes this passage)",
    headingCount,
  };
}

// Whole-document outline: the same detector as discoverSegment, run across an
// entire text instead of a window around one anchor.  A reader needs every
// boundary, not just the pair bracketing a citation.
//
// Two deliberate refusals here:
//
//  1. Offsets are UTF-16 code-unit indices into the string passed in, NOT the
//     UTF-8 byte offsets the engine's span anchors use.  The only consumer is
//     a client that already holds this exact string and slices it with
//     String.prototype.slice; handing it bytes would reintroduce the drift
//     fileIndex() exists to prevent.
//  2. No `level`.  Deciding that one heading nests under another is a holon-
//     level claim, and level is discovered from existence-dependency, never
//     inferred from a heading's typographic form.  An all-caps line is not
//     evidence that the numbered line below it is its child.  The outline is
//     therefore flat, and honest about being flat.
//
// Fewer than two boundaries is a typed gap, not a one-entry table of contents:
// a document we found no structure in should say so, not present its whole
// body as a section called "Content".
export function outlineOfText(text, { max = 500 } = {}) {
  const body = String(text ?? "");
  if (!body.trim()) return { headings: [], gap: "empty text" };

  const lines = body.split("\n");
  const starts = new Array(lines.length);
  for (let i = 0, at = 0; i < lines.length; i++) {
    starts[i] = at;
    at += lines[i].length + 1; // +1 for the \n
  }

  const candidates = [];
  for (let i = 0; i < lines.length; i++) {
    const nextBlank = i + 1 < lines.length && lines[i + 1].trim() === "";
    if (!headingScore(lines[i], nextBlank)) continue;
    candidates.push({
      label: lines[i].trim(),
      // Where the heading line itself begins, and where its body does. The
      // reader renders the label from `label`, so it slices from bodyStart to
      // avoid printing the heading twice.
      start: starts[i],
      bodyStart: starts[i] + lines[i].length + 1,
    });
  }

  // A book's own printed table of contents is a wall of perfectly heading-
  // shaped lines — on War and Peace it yields 500 of them before the narrative
  // even starts, an outline that navigates nothing but the index. What
  // separates a listing from a structure is not how the line is typeset but
  // what lies under it: a real boundary opens onto a body, a TOC entry opens
  // onto the next TOC entry. So a candidate earns its place by what follows
  // it. The front matter this discards is not lost — it falls into the
  // preamble, which the reader still renders.
  //
  // The same test drops running heads, part-title pages and stray all-caps
  // lines without any of them having to be named.
  const MIN_BODY = 200; // less than a paragraph of substance ⇒ a listing
  const found = [];
  for (let i = 0; i < candidates.length && found.length < max; i++) {
    const to = i + 1 < candidates.length ? candidates[i + 1].start : body.length;
    const substance = body.slice(candidates[i].bodyStart, to).replace(/\s+/g, "");
    if (substance.length >= MIN_BODY) found.push(candidates[i]);
  }

  if (found.length < 2) {
    return {
      headings: [],
      gap: found.length
        ? "only one structural boundary detected — not enough to be an outline"
        : "no structural boundaries detected",
    };
  }

  const headings = found.map((h, i) => ({
    label: h.label,
    start: h.start,
    bodyStart: Math.min(h.bodyStart, body.length),
    end: i + 1 < found.length ? found[i + 1].start : body.length,
  }));
  return {
    headings,
    gap: null,
    truncated: found.length >= max,
    // Text before the first heading is real content (a title page, a preamble)
    // and the reader has to show it; it just has no label of its own.
    preambleEnd: headings[0].start,
  };
}

// Outline of an ingested source, in byte offsets.
//
// outlineOfText indexes a JS string, so its offsets are UTF-16 code units. The
// reader pages through /api/source/text, which is byte-addressed like every
// other anchor in the engine. On an ASCII-ish book the two agree closely enough
// to look correct and drift silently after the first multi-byte character —
// the exact failure fileIndex exists to prevent. So the conversion happens here,
// once, against the same buffer the reader will be served from.
//
// Both offsets are returned: `start`/`end` stay code-unit (what outlineOfText
// measured) and `byte_start`/`byte_end` are what /api/source/text wants. A
// consumer that mixes them is then making a visible mistake, not an invisible one.
export function engineOutlineOfSource(sourceRef, { pool: poolName = DEFAULT_POOL, max = 500 } = {}) {
  const session = ensureSession(poolName);
  const resolved = resolveDocument(session, sourceRef);
  if (resolved.error) return { error: resolved.error };
  const doc = resolved.doc;

  let idx;
  try { idx = fileIndex(doc.path); } catch (e) {
    return { error: `Cannot read ${doc.path}: ${e.message}` };
  }

  const text = idx.buf.toString("utf8");
  const outline = outlineOfText(text, { max });

  // One forward pass over the cut points rather than a byteLength(slice(0,n))
  // per heading, which would re-measure the whole prefix 32+ times.
  const marks = new Set();
  for (const h of outline.headings || []) {
    marks.add(h.start); marks.add(h.bodyStart); marks.add(h.end);
  }
  if (outline.preambleEnd != null) marks.add(outline.preambleEnd);
  const sorted = [...marks].filter((n) => Number.isFinite(n)).sort((a, b) => a - b);
  const byteAt = new Map();
  let cursor = 0, bytes = 0;
  for (const mark of sorted) {
    bytes += Buffer.byteLength(text.slice(cursor, mark), "utf8");
    cursor = mark;
    byteAt.set(mark, bytes);
  }
  const B = (n) => (n == null ? null : byteAt.get(n) ?? null);

  return {
    source: doc.path,
    source_id: doc.id,
    name: doc.base,
    pool: poolName,
    total_bytes: idx.bytes,
    gap: outline.gap ?? null,
    truncated: outline.truncated ?? false,
    preambleEnd: outline.preambleEnd ?? null,
    preamble_byte_end: B(outline.preambleEnd),
    headings: (outline.headings || []).map((h) => ({
      label: h.label,
      start: h.start,
      bodyStart: h.bodyStart,
      end: h.end,
      byte_start: B(h.start),
      body_byte_start: B(h.bodyStart),
      byte_end: B(h.end),
    })),
  };
}

// Read the segment (chapter, section, movement) containing the content
// described by query.  Boundary detection is dynamic and text-agnostic —
// works for CHAPTER I, I. Title, 1. Section, all-caps headings, etc.
export function engineReadSegment(query, maxBytes = 100000, sourceFilter, { pool: poolName = DEFAULT_POOL } = {}) {
  const s = ensureSession(poolName);
  let { spans } = searchSpans(s, { query, limit: 5 });
  if (!spans.length) return { error: `Segment not found: "${query}"` };
  // Optionally filter by source_id prefix
  if (sourceFilter) {
    const sf = sourceFilter.replace(/^.*[/\\]/, "");
    const filtered = spans.filter(sp => sp.source_id && sp.source_id.includes(sf));
    if (!filtered.length) return { error: `Segment not found in "${sourceFilter}"` };
    spans = filtered;
  }

  const best = spans[0];
  const sourcePath = resolveSourcePath(best.source_id);
  if (!sourcePath) return { error: `Cannot resolve source from "${best.source_id}"` };

  let idx;
  try { idx = fileIndex(sourcePath); } catch (e) {
    return { error: `Cannot read ${sourcePath}: ${e.message}` };
  }

  const anchor = best.byte_start || 0;
  const seg = discoverSegment(idx, anchor);
  if (!seg) {
    const from = Math.max(0, anchor - 2000);
    const to = Math.min(idx.bytes, anchor + maxBytes);
    return {
      segment: "(no structural boundary detected)",
      source: sourcePath,
      byte_start: from,
      byte_end: to,
      truncated: false,
      text: sliceBytes(idx.buf, from, to),
    };
  }

  const length = Math.min(seg.endByte - seg.startByte, maxBytes);
  const segText = sliceBytes(idx.buf, seg.startByte, seg.startByte + length);

  return {
    segment: seg.label,
    source: sourcePath,
    byte_start: seg.startByte,
    byte_end: seg.startByte + length,
    truncated: length < (seg.endByte - seg.startByte),
    heading_count: seg.headingCount,
    text: segText,
  };
}

// Read an arbitrary byte range of one ingested source.
//
// The reader needs this and nothing else can serve it. /api/attachments/content
// only knows session uploads, so the corpus ingested at startup — the three
// books — had no readable body at all; verbatim/segment and verbatim/read both
// require a query or a span_id, which a reader paging through a document does
// not have. The unit here is the byte range the fold's divisions already speak
// in, so an outline click and the text it lands on are the same coordinates.
//
// Reads through fileIndex for the same reason engineReadSegment does: the
// engine's anchors are UTF-8 byte offsets, and slicing a JS string by them
// shifts every read past the first multi-byte character.
export function engineReadSourceBytes(sourceRef, { pool: poolName = DEFAULT_POOL, start = 0, end = null, maxBytes = 200000 } = {}) {
  const session = ensureSession(poolName);
  const resolved = resolveDocument(session, sourceRef);
  if (resolved.error) return { error: resolved.error };
  const doc = resolved.doc;

  let idx;
  try { idx = fileIndex(doc.path); } catch (e) {
    return { error: `Cannot read ${doc.path}: ${e.message}` };
  }

  const total = idx.bytes;
  const from = Math.max(0, Math.min(Number.isFinite(start) ? start : 0, total));
  // No `end` means "from here on", still capped — an unbounded default would
  // hand a whole book to a reader that asked for a page.
  const wantedEnd = end == null ? total : Math.max(from, Math.min(end, total));
  const to = Math.min(wantedEnd, from + maxBytes);

  return {
    source: doc.path,
    source_id: doc.id,
    name: doc.base,
    pool: poolName,
    byte_start: from,
    byte_end: to,
    total_bytes: total,
    truncated: to < wantedEnd,
    text: sliceBytes(idx.buf, from, to),
  };
}

// Read a span with surrounding context (expand before/after to arbitrary byte
// windows). Given a span_id or a { byte_start, byte_end, source }, read N bytes
// before and M bytes after from the source file.
export function engineReadContext(spanRef, { beforeBytes = 0, afterBytes = 0, maxTotal = 50000 } = {}) {
  // Resolve the span: either a span_id (look up across pools) or a direct ref
  let sourceId, byteStart, byteEnd;
  if (typeof spanRef === "string") {
    const p = poolForSpan(spanRef);
    const rec = p?.session.spans.get(spanRef);
    if (!rec) return { error: `Unknown span_id "${spanRef}". Search first.` };
    sourceId = rec.source_id;
    byteStart = rec.byte_start;
    byteEnd = rec.byte_end;
  } else {
    sourceId = spanRef.source;
    byteStart = spanRef.byte_start;
    byteEnd = spanRef.byte_end;
  }

  if (byteStart == null || byteEnd == null) {
    return { error: "Span has no byte offsets" };
  }

  const sourcePath = resolveSourcePath(sourceId);
  if (!sourcePath) return { error: `Cannot resolve source from "${sourceId}"` };

  let idx;
  try { idx = fileIndex(sourcePath); } catch (e) {
    return { error: `Cannot read ${sourcePath}: ${e.message}` };
  }

  const readStart = Math.max(0, byteStart - beforeBytes);
  const readEnd = Math.min(idx.bytes, byteEnd + afterBytes);
  const length = Math.min(readEnd - readStart, maxTotal);
  const contextText = sliceBytes(idx.buf, readStart, readStart + length);

  return {
    source: sourcePath,
    byte_start: readStart,
    byte_end: readStart + length,
    span_byte_start: byteStart,
    span_byte_end: byteEnd,
    truncated: length < (readEnd - readStart),
    text: contextText,
  };
}

// ── Source tracking ──

export function engineIngestFile(filePath, { pool: poolName = DEFAULT_POOL, kind = "corpus", displayName } = {}) {
  const p = pool(poolName);
  const { chunks, admitted } = ingestFile(p.session, filePath);
  p.chunkCount += chunks;
  const name = displayName || filePath.replace(/^.*[/\\]/, "");
  p.sources.set(filePath, { name, chunks, kind, pool: p.name, ingestedAt: Date.now() });
  return {
    path: filePath,
    chunks,
    pool: p.name,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

// Every pool's sources, each tagged with the pool it lives in and what it is.
// Callers that only want ingested texts filter on `kind === "corpus"` rather
// than assuming the list is homogeneous.
export function engineListSources({ pool: poolName } = {}) {
  const selected = poolName ? [pool(poolName)] : Array.from(pools.values());
  return selected.flatMap((p) =>
    Array.from(p.sources.entries()).map(([path, info]) => ({
      path,
      name: info.name,
      chunks: info.chunks,
      kind: info.kind ?? "corpus",
      pool: p.name,
      ingestedAt: info.ingestedAt,
    })),
  );
}

export function engineStats() {
  const s = ensureSession();
  const all = Array.from(pools.values());
  return {
    sessionActive: !!s,
    ingestedChunks: all.reduce((n, p) => n + p.chunkCount, 0),
    ingestedFiles: all.reduce((n, p) => n + p.sources.size, 0),
    spanCap: s.spanCap ?? 2000,
    sources: Array.from(pool(DEFAULT_POOL).sources.values()).map((x) => x.name),
    deletedSources: recycleBin.size,
    pools: all.map((p) => ({ name: p.name, files: p.sources.size, chunks: p.chunkCount })),
  };
}

// Terrain/stance has no legitimate per-file reading: classifying a file's
// content into a terrain was measured at r ≈ 0.974 toward the material's own
// vocabulary (a refuted classifier, eoreader5/CUBE.md). What IS measured is
// the engine's own cell occupancy — which (operator, grain) cells are earned
// by a real organ and which are still open questions. terrain_report exposes
// that coverage report; it is the same for every call, because it is a fact
// about the engine, not about whichever file happens to be ingested.
export function engineTerrainReport() {
  return coverageReport();
}

// ── Recycle bin ──

const RECYCLE_BIN_PATH = path.join(import.meta.dirname, "recycle-bin.json");

const recycleBin = new Map();

function loadRecycleBin() {
  try {
    if (fs.existsSync(RECYCLE_BIN_PATH)) {
      const data = JSON.parse(fs.readFileSync(RECYCLE_BIN_PATH, "utf8"));
      for (const [key, val] of Object.entries(data)) {
        recycleBin.set(key, val);
      }
    }
  } catch (err) {
    console.error(`[recycle-bin] load failed: ${err.message}`);
  }
}

function saveRecycleBin() {
  try {
    const data = Object.fromEntries(recycleBin);
    fs.writeFileSync(RECYCLE_BIN_PATH, JSON.stringify(data, null, 2));
  } catch (err) {
    console.error(`[recycle-bin] save failed: ${err.message}`);
  }
}

loadRecycleBin();

export function engineDeleteSource(sourceKey, { pool: poolName = DEFAULT_POOL } = {}) {
  const p = pool(poolName);
  const info = p.sources.get(sourceKey);
  if (!info) return { error: `Source not found: ${sourceKey}` };

  const deletedSpans = [];
  for (const [spanId, rec] of p.session.spans) {
    if (rec.source_id && sourceKey.includes(rec.source_id) || rec.source_id === sourceKey) {
      deletedSpans.push({ span_id: spanId, source_id: rec.source_id, byte_start: rec.byte_start, byte_end: rec.byte_end, text: rec.text, score: rec.score });
    }
  }

  for (const span of deletedSpans) {
    p.session.spans.delete(span.span_id);
  }
  p.chunkCount -= info.chunks;
  p.sources.delete(sourceKey);

  const deletedEntry = {
    sourceKey,
    info,
    spans: deletedSpans,
    deletedAt: Date.now(),
  };
  recycleBin.set(sourceKey, deletedEntry);
  saveRecycleBin();

  return {
    path: sourceKey,
    name: info.name,
    chunks: info.chunks,
    spansRemoved: deletedSpans.length,
    deletedAt: deletedEntry.deletedAt,
    pool: p.name,
  };
}

export function engineListRecycleBin() {
  return Array.from(recycleBin.values()).map((entry) => ({
    sourceKey: entry.sourceKey,
    name: entry.info.name,
    chunks: entry.info.chunks,
    kind: entry.info.kind ?? "corpus",
    pool: entry.info.pool,
    ingestedAt: entry.info.ingestedAt,
    deletedAt: entry.deletedAt,
    spansCount: entry.spans.length,
  }));
}

export function engineRestoreSource(sourceKey, { pool: poolName } = {}) {
  const entry = recycleBin.get(sourceKey);
  if (!entry) return { error: `Deleted source not found: ${sourceKey}` };

  const p = pool(poolName || entry.info.pool || DEFAULT_POOL);
  p.sources.set(entry.sourceKey, { ...entry.info });
  p.chunkCount += entry.info.chunks;

  for (const span of entry.spans) {
    p.session.spans.set(span.span_id, {
      source_id: span.source_id,
      byte_start: span.byte_start,
      byte_end: span.byte_end,
      text: span.text,
      score: span.score,
    });
  }

  recycleBin.delete(sourceKey);
  saveRecycleBin();

  return {
    path: entry.sourceKey,
    name: entry.info.name,
    chunks: entry.info.chunks,
    spansRestored: entry.spans.length,
    pool: p.name,
  };
}

export function enginePurgeSource(sourceKey) {
  const entry = recycleBin.get(sourceKey);
  if (!entry) return { error: `Deleted source not found: ${sourceKey}` };
  recycleBin.delete(sourceKey);
  saveRecycleBin();
  return {
    path: entry.sourceKey,
    name: entry.info.name,
    spansDiscarded: entry.spans.length,
  };
}

export function enginePurgeRecycleBin() {
  const count = recycleBin.size;
  recycleBin.clear();
  saveRecycleBin();
  return { purged: count };
}

export function engineRecycleBinStats() {
  return {
    count: recycleBin.size,
    totalChunks: Array.from(recycleBin.values()).reduce((s, e) => s + e.info.chunks, 0),
    totalSpans: Array.from(recycleBin.values()).reduce((s, e) => s + e.spans.length, 0),
    entries: engineListRecycleBin(),
  };
}

// ── Fold projection ──
//
// engineFoldSource answers "what IS this document" — its cast, its divisions,
// how much of it was folded at all — in the shape eoreaderapp/src/app consumes
// (__fixtures__/npr-news-fold.js is the contract, overview-dashboard.js the
// reader). It adds no intelligence: every number below is a wiring of an organ
// that already exists, through the one surface hosts may touch. What it is FOR
// is retiring eochat's buildEntityMatcher — a regex over capitalized words plus
// a stopword list, top-20 by frequency — which is identity-in-a-string, cannot
// represent an emanon at all, and ranks the publisher above the cast.
//
// Three rules govern what this function may emit.
//
//   1. INDIVIDUATION TYPE COMES ONLY FROM THE PRIOR. holon/emanon/protogon/
//      field/apparatus is the output of the Ground→Figure gate
//      (referents/individuation.js), and that gate is not wired to a document:
//      it needs mass, coupling and their Born-null sample distributions, and
//      nothing in the host facade computes them — its only callers today are
//      its own tests. So the type is read from the per-text coref prior, which
//      is witness-tier knowledge injected by a reader (docs/nameless-referent.md:
//      "descriptor coreference is witness-channel knowledge"). A referent the
//      text merely PROPOSED — rankSurfaces capitalization physics, no prior —
//      has been through no gate and had no aliases resolved, so it is withheld
//      rather than typed. Note that `sessionReferents` reports such a candidate
//      as individuation "discovered" and reports a prior that omits the field
//      as "holon"; neither is a gate result, so neither is passed through.
//
//   2. MASS IS NOT SYNTHESIZED. The fixture separates mass (Tapas-
//      concentration) from count (sightings), and only the second exists.
//      Multiplying mentions by frame spread would produce a number that looks
//      like mass, sorts like mass, and is not mass. `mass` is therefore null
//      with a typed gap, and the observables that DO exist — mentions, frames,
//      first/last frame — travel under their own names.
//
//   3. ANCHORS ARE DOCUMENT-SCOPED, AND SCOPED SURFACES ARE NEVER SCANNED.
//      Anchors are byte offsets into THIS document's admitted pieces, built
//      from a real occurrence (see anchorsFor for why pool-wide retrieval was
//      the wrong source and what it cost). Scoped surfaces — narrator spans, a
//      descriptor valid in only one stretch — are excluded entirely: their
//      validity is positional, a whole-document scan has no scope, and "my
//      enemy" inside the Creature's own tale points at Victor. A referent left
//      with no anchor keeps `evidence: []` and is withheld by fold-contract's
//      displayableReferents rather than shown on an unverifiable claim.
//
// Everything the projection cannot fill is a typed gap in `gaps`, never a
// plausible default. That is the whole difference between this and the regex.

export const FOLD_PROJECTION_VERSION = "fold-projection@1";

const GATED_TYPES = new Set(INDIVIDUATION_TYPES);

// `tier` says who would have to supply the missing thing: "engine" — an organ
// exists but is unwired; "model" — witness knowledge, only a prior can supply
// it; "host" — this bridge or its caller.
const typedGap = (field, reason, tier) => ({ field, reason, tier });

// The anchor cap is reported on the fold (`anchor_policy`) rather than applied
// silently — a truncated evidence list must not read as a complete one.
//
// There is deliberately NO cap on how many of a referent's surfaces are
// scanned. A cap would make a referent's evidence — and so whether it is shown
// at all — depend on the order its surfaces happen to be listed in the prior,
// which is not a property of the text.
//
// Worked example, and the reason to trust rule 3 rather than a cap (historical:
// the prior has since been corrected, so this no longer reproduces): War and
// Peace's prior once listed Andrei's surfaces in a transliteration this edition
// does not use ("Andrei"/"Bolkonsky" — 0 occurrences; the Maude text says
// "Prince Andrew" and "Bolkónski"). His one surface that did occur, "little
// princess", was SCOPED, and is in fact his wife. Rule 3 excluded it from
// scanning on principle, so he was withheld with "no evidence for any unscoped
// surface" rather than shown on his wife's epithet — the right outcome for the
// right reason, and it held however the prior ordered its surfaces. What a cap
// would have done instead is show him, on Lise.

// Sighting counts are weighted, not integral — presenceByFrame scores a
// first-person hit at 0.5 because a pronoun is a weaker sighting than a name.
// Summing those halves accumulates binary-float noise (1503.0000000000011), so
// counts are trimmed to two places. Rounding to an integer would be the wrong
// fix: it would erase the half-sighting the weighting exists to express.
const roundCount = (n) => Number(Number(n ?? 0).toFixed(2));

function documentCatalog(session) {
  return corpusFacade.documentIds(session).map((id) => {
    const filePath = resolveSourcePath(id) || id;
    const base = filePath.replace(/^.*[/\\]/, "");
    return { id, path: filePath, base, stem: base.replace(/\.[^.]+$/, "") };
  });
}

// Resolve a caller's `?source=` — a document id, an absolute path, a basename
// or a stem — to exactly one document. Ambiguity is an error listing the
// candidates, never a silent pick of the first match: serving one book's cast
// under another book's name is the failure this endpoint exists to end.
function resolveDocument(session, ref) {
  const wanted = String(ref ?? "").trim();
  if (!wanted) return { error: "missing 'source'" };
  const catalog = documentCatalog(session);
  if (!catalog.length) return { error: "no documents ingested in this pool" };

  for (const key of ["id", "path", "base", "stem"]) {
    const hits = catalog.filter((d) => d[key] === wanted);
    if (hits.length === 1) return { doc: hits[0] };
    if (hits.length > 1) return { error: `ambiguous source "${wanted}": ${hits.map((d) => d.id).join(", ")}` };
  }
  const loose = catalog.filter((d) => d.base.includes(wanted));
  if (loose.length === 1) return { doc: loose[0] };
  if (loose.length > 1) return { error: `ambiguous source "${wanted}": ${loose.map((d) => d.id).join(", ")}` };
  return { error: `unknown source "${wanted}". Known: ${catalog.map((d) => d.base).join(", ")}` };
}

// Surfaces this bridge may hand to searchSpans: unscoped ones only (rule 3).
// A prior entry knows which of its surfaces carry a scope; a discovered
// candidate has exactly one surface, global by construction. The
// `surface@from-to` forms admitReferent mints for narrator spans are
// positional handles, not searchable text, and are dropped.
function globalSurfacesFor(referent, priorEntry) {
  const positional = /@\d+-\d+$/;
  if (priorEntry && Array.isArray(priorEntry.surfaces)) {
    const unscoped = priorEntry.surfaces
      .map((s) => (typeof s === "string" ? { surface: s } : s))
      .filter((s) => s && s.surface && !s.scope)
      .map((s) => String(s.surface));
    const seed = priorEntry.name ? [String(priorEntry.name)] : [];
    return [...new Set([...seed, ...unscoped])].filter((s) => !positional.test(s));
  }
  return (referent.surfaces || []).map(String).filter((s) => !positional.test(s));
}

// Byte-addressed evidence for one referent, read off the document's own
// admitted pieces rather than out of scored retrieval.
//
// searchSpans was the obvious source and is the wrong one: it ranks across the
// WHOLE pool. In a session holding three books the top hits for "the creature"
// are War and Peace's, and filtering them back down to this document left
// Frankenstein's Creature with no anchors and silently withheld — a referent
// the prior had correctly individuated as emanon, dropped because of what else
// happened to be ingested. Measured: 1 survivor alone, 0 survivors alongside
// two other books, from the same file and the same prior.
//
// Pieces carry their byte offset in the SOURCE FILE, so scanning them is both
// exactly scoped to this document and strictly more precise — the anchor lands
// on the occurrence itself rather than on the chunk containing it, and needs
// no verification pass because it is built from a real match.
//
// This is not a string-matching coref substitute. Identity is already fixed by
// the prior; all that happens here is locating where a known referent's known
// surfaces occur, for unscoped surfaces only (rule 3).
// Surfaces are round-robined rather than drained one at a time, so a short
// evidence list spans the alias set instead of showing the first surface three
// times. Which surfaces actually carried is the thing a reader auditing a prior
// needs to see.
function anchorsFor(pieces, docId, surfaces, want) {
  const anchors = [];
  const seen = new Set();
  const cursors = surfaces
    .map((surface) => ({ re: surfaceMatcher(surface), next: 0 }))
    .filter((c) => c.re);

  let progressed = true;
  while (anchors.length < want && progressed) {
    progressed = false;
    for (const cursor of cursors) {
      if (anchors.length >= want) break;
      while (cursor.next < pieces.length) {
        const piece = pieces[cursor.next++];
        const hit = cursor.re.exec(piece.text);
        if (!hit) continue;
        const start = piece.byteStart + Buffer.byteLength(piece.text.slice(0, hit.index), "utf8");
        if (seen.has(start)) continue;
        seen.add(start);
        anchors.push({
          source_id: docId,
          byte_start: start,
          byte_end: start + Buffer.byteLength(hit[0], "utf8"),
          surface: hit[0],
        });
        progressed = true;
        break;
      }
    }
  }
  return anchors;
}

// The document's divisions, as derivations the app's reconcileDivisions can
// vote over. Only ONE derivation exists here, and that is reported honestly:
// sessionOutline's novelty curve (KL against a sliding prior — where the word
// distribution actually turns, not a heading regex). The DOM derivation has no
// meaning for an ingested text file and the strain spine does not exist in the
// engine at all; both are gaps, so the strip shows 1/1 agreement rather than a
// fabricated consensus.
function divisionsFor(session, docId, gaps, zThreshold) {
  const outline = corpusFacade.sessionOutline(session, { sourceId: docId, zThreshold });
  if (outline?.error) {
    gaps.push(typedGap("divisions.derivations", `outline unavailable: ${outline.error}`, "engine"));
    return { derivations: [] };
  }
  const sections = outline.sections || [];
  const last = sections[sections.length - 1];
  const span = last ? last.offset + last.length : 0;
  if (!span) {
    gaps.push(typedGap("divisions.derivations", "document has no measurable extent", "engine"));
    return { derivations: [] };
  }
  // reconcileDivisions clusters cuts on a 0..1 scale; sessionOutline reports
  // character offsets. Section 0 opens at the document head and is not a cut.
  const cuts = sections.slice(1).map((sec) => Number((sec.offset / span).toFixed(4)));
  gaps.push(typedGap("divisions.derivations[dom]", "no DOM perceiver for an ingested text file — the page's own headings do not exist here", "host"));
  gaps.push(typedGap("divisions.derivations[strain]", "no strain/deviation waveform exists in the engine", "engine"));
  return {
    frames: outline.frames ?? null,
    derivations: [{
      id: "novelty",
      beats: sections.length,
      unit: "beats",
      cuts,
      sections: sections.map((sec) => ({
        index: sec.index, offset: sec.offset, byte_start: sec.byteStart, length: sec.length, label: sec.label,
      })),
    }],
  };
}

// How much of the file made it into the fold, measured rather than asserted.
// Admitted pieces carry their byte offset in the SOURCE FILE, so their summed
// length against the file's own size is real accounting: the remainder is
// whatever admitChunked dropped (sub-minChars runs) plus whatever ingestFile
// stripped (Gutenberg boilerplate) before admitting.
//
// The {chrome, dup, nul} attribution the dashboard wants is NOT derivable from
// this — nothing records WHY a byte was dropped. Three plausible percentages
// here would be the exact failure this projection exists to avoid, so `buckets`
// is null with a gap.
function coverageFor(session, doc, chunks, gaps) {
  gaps.push(typedGap("coverage.buckets", "no per-unit discard attribution exists — admitChunked and ingestFile drop bytes without recording chrome/dup/nul", "engine"));
  const pieces = session.documents?.get(doc.id)?.pieces ?? [];
  const foldedBytes = pieces.reduce((n, p) => n + Buffer.byteLength(p.text, "utf8"), 0);

  let totalBytes = null;
  try {
    totalBytes = fileIndex(doc.path).bytes;
  } catch {
    gaps.push(typedGap("coverage.folded_pct", `source file ${doc.path} is not readable — coverage cannot be measured`, "host"));
  }
  if (!totalBytes) {
    return { folded_bytes: foldedBytes, discarded_bytes: null, folded_pct: null, discarded_pct: null, buckets: null, folded_units: chunks, refoldable: true };
  }
  const discardedBytes = Math.max(0, totalBytes - foldedBytes);
  return {
    folded_bytes: foldedBytes,
    discarded_bytes: discardedBytes,
    total_bytes: totalBytes,
    folded_pct: Number(((foldedBytes / totalBytes) * 100).toFixed(1)),
    discarded_pct: Number(((discardedBytes / totalBytes) * 100).toFixed(1)),
    buckets: null,
    folded_units: chunks,
    refoldable: true,
  };
}

/**
 * Project one ingested document into the app's fold shape.
 *
 * @param {string} sourceRef document id, path, basename or stem
 * @param {{pool?: string, limit?: number, anchors?: number, zThreshold?: number}} [options]
 *   `limit` caps the `withheld` audit list only; `anchors` is per referent.
 * @returns {object} the fold, or `{ error }` when the source cannot be resolved
 */
export function engineFoldSource(sourceRef, { pool: poolName = DEFAULT_POOL, limit = 40, anchors: anchorsPerReferent = 3, zThreshold } = {}) {
  const needed = ["documentIds", "documentText", "sessionOutline", "sessionReferents"];
  const missing = needed.filter((name) => typeof corpusFacade[name] !== "function");
  if (missing.length) {
    return { error: `@eoreader/host/corpus is API v${CORPUS_API_VERSION} and has no whole-document facade (missing ${missing.join(", ")}); the fold projection needs v2` };
  }

  const session = ensureSession(poolName);
  const resolved = resolveDocument(session, sourceRef);
  if (resolved.error) return { error: resolved.error };
  const doc = resolved.doc;
  const gaps = [];

  // Witness-tier knowledge. Absent, descriptor coref is simply not done — and
  // the whole cast falls through to `withheld` rather than being typed on
  // capitalization physics.
  const prior = loadCorefPrior(doc.id);
  if (prior.gap) gaps.push(typedGap("prior", prior.gap, "model"));

  const read = corpusFacade.sessionReferents(session, {
    sourceId: doc.id,
    priors: prior.referents,
    limit: Number.MAX_SAFE_INTEGER,
  });
  if (read.error) return { error: read.error };
  for (const g of read.gaps || []) {
    gaps.push(typedGap("referents", typeof g === "string" ? g : (g.reason || JSON.stringify(g)), "engine"));
  }

  // sessionReferents keys a referent `ref:<normalized id>`, so the raw prior
  // entry is recovered through the display string it derived it from
  // (`prior.display ?? prior.name ?? prior.id`). A display two prior entries
  // share resolves to neither — the type would be a coin flip, and a coin flip
  // is a fabrication.
  const priorByDisplay = new Map();
  const ambiguousDisplay = new Set();
  for (const entry of prior.referents || []) {
    const key = entry.display ?? entry.name ?? entry.id;
    if (priorByDisplay.has(key)) ambiguousDisplay.add(key);
    else priorByDisplay.set(key, entry);
  }

  const pieces = [...(session.documents?.get(doc.id)?.pieces ?? [])].sort((a, b) => a.byteStart - b.byteStart);
  const referents = [];
  const withheld = [];
  let sightings = 0;

  for (const r of read.referents || []) {
    sightings += r.mentions || 0;
    const entry = ambiguousDisplay.has(r.display) ? null : priorByDisplay.get(r.display);
    const asserted = entry && typeof entry.individuation === "string" ? entry.individuation : null;
    const type = asserted && GATED_TYPES.has(asserted) ? asserted : (r.individuation || null);

    const surfaces = globalSurfacesFor(r, entry);
    // Anchors are gathered for all referents with evidence, not just prior-typed ones.
    // Universal coref: discovered candidates are valid referents.
    const evidence = type ? anchorsFor(pieces, doc.id, surfaces, anchorsPerReferent) : [];

    const base = {
      id: r.id,
      name: r.display,
      canonicalLabel: r.display,
      surfaceForms: r.surfaces || [],
      globalSurfaces: surfaces,
      count: roundCount(r.mentions),
      mentions: roundCount(r.mentions),
      frames: r.frames ?? 0,
      first_frame: r.firstFrame ?? null,
      last_frame: r.lastFrame ?? null,
      mass: null,
    };

    if (type && evidence.length) {
      referents.push({
        ...base,
        individuation_type: type,
        aliasesResolved: true,
        evidence,
        provenance: {
          anchors: evidence,
          tier: r.fromPrior ? "model" : "engine",
          prior_snapshot: r.fromPrior && prior.priorId ? { identity: prior.priorId, path: prior.priorPath } : null,
          surfaces_scanned: surfaces,
          scoped_surfaces_excluded: Math.max(0, (r.surfaces || []).length - surfaces.length),
        },
      });
      continue;
    }
    // Universal coref: discovered candidates are valid referents.
    // Only withhold if there's genuinely no evidence (no surfaces, no mentions).
    if (!type && !r.fromPrior && r.mentions > 0) {
      // Auto-classify discovered referents
      const autoType = r.individuation || 'emanon';
      referents.push({
        ...base,
        individuation_type: autoType,
        aliasesResolved: true,
        evidence: [],
        provenance: {
          anchors: [],
          tier: "engine",
          prior_snapshot: null,
          surfaces_scanned: surfaces,
          scoped_surfaces_excluded: Math.max(0, (r.surfaces || []).length - surfaces.length),
        },
      });
      continue;
    }
    // Withheld: prior asserted a type but no evidence found, or truly empty.
    withheld.push({
      ...base,
      individuation_type: null,
      aliasesResolved: false,
      from_prior: !!r.fromPrior,
      withheld_because: !r.fromPrior
        ? "no evidence for this referent"
        : !type
          ? `prior asserted no individuation type${asserted ? ` ("${asserted}" is not one of ${INDIVIDUATION_TYPES.join("/")})` : ""}`
          : "no evidence for any unscoped surface — every surface that occurs in this document is scope-restricted, or none occurs at all",
    });
  }

  // A referent the prior NAMED but that could not be shown is the important
  // one: it means the prior and the text disagree — a mistyped surface, or a
  // prior written against another edition — and it is a bug in witness
  // knowledge that someone can fix. A candidate the text merely proposed is
  // routine, and there are hundreds. Ordering the former first keeps `limit`
  // from burying the actionable case behind the noise.
  withheld.sort((a, b) => (b.from_prior === true) - (a.from_prior === true) || b.mentions - a.mentions);

  const priorUnmatched = withheld.filter((r) => r.from_prior);
  if (priorUnmatched.length) {
    gaps.push(typedGap(
      "prior.referents",
      `the prior asserts ${priorUnmatched.length} referent(s) this document cannot show — ${priorUnmatched.map((r) => `"${r.name}"`).join(", ")}. The prior and the text disagree: check the surfaces against this edition's spelling.`,
      "model",
    ));
  }

  if (withheld.length) {
    gaps.push(typedGap(
      "referents",
      `${withheld.length} referent(s) withheld: no evidence found for these candidates.`,
      "engine",
    ));
  }
  gaps.push(typedGap("referents[].mass", "the Ground→Figure gate (referents/individuation.js) is not wired to a document — it needs mass, coupling and their Born-null samples, which the host facade does not compute. mentions/frames are the observables that do exist.", "engine"));
  gaps.push(typedGap("frame", "frame kind, coupling-dispersion and subject-re-entry are outputs of the same unwired gate", "engine"));
  gaps.push(typedGap("units.passages", "per-passage register/surprise/below-null is not computed for an ingested text", "engine"));
  gaps.push(typedGap("motifs", "recurrence families are not projected by this endpoint", "host"));

  const text = corpusFacade.documentText(session, doc.id);
  const divisions = divisionsFor(session, doc.id, gaps, zThreshold);

  return {
    fold_version: FOLD_PROJECTION_VERSION,
    sourceId: doc.id,
    source: {
      id: doc.id,
      path: doc.path,
      name: doc.base,
      medium: "Text",
      words: text ? (text.text.match(/\S+/g) || []).length : null,
      chunks: text ? text.chunks : null,
      publisher: null,
    },
    prior: {
      snapshot: prior.priorId,
      path: prior.priorPath,
      referents_asserted: (prior.referents || []).length,
    },
    // `referents` is never truncated — it is bounded by what the prior
    // asserted, which is small. `withheld` is the audit trail and can run to
    // hundreds, so `limit` applies there; the true count travels beside it so a
    // truncated list cannot be mistaken for the whole one.
    referents,
    withheld: withheld.slice(0, limit),
    withheld_total: withheld.length,
    withheld_truncated: withheld.length > limit,
    sightings: roundCount(sightings),
    survivors: referents.length,
    divisions,
    coverage: coverageFor(session, doc, text ? text.chunks : 0, gaps),
    anchor_policy: {
      anchors_per_referent: anchorsPerReferent,
      surfaces_per_referent: "all unscoped surfaces, round-robined",
      scanned: "this document's admitted pieces only — never pool-wide retrieval",
      searched: "referents a prior typed; withheld candidates are not scanned",
      note: "anchor lists are capped; an empty list means no unscoped surface occurs in this document, not that the referent is absent",
    },
    gaps,
  };
}
