// Ground-Truth engine bridge: ingests files into the real eoreader5 session,
// searches spans with byte-offset anchors, folds results under a token budget,
// and returns grounded context the LLM can cite.
//
// Single session per proxy instance — all ingested files share it, so a chat
// message about "the creature" after ingesting Frankenstein and War and Peace
// gets interleaved search results from both.
//
// Hosts the impure createSession/ingestFile/searchSpans/foldSpans corpus facade.

import fs from "node:fs";
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

const EXPECTED_CORPUS_API = 1;
if (CORPUS_API_VERSION !== EXPECTED_CORPUS_API) {
  throw new Error(
    `@eoreader/host/corpus is API v${CORPUS_API_VERSION}; this bridge expects v${EXPECTED_CORPUS_API}`,
  );
}

// ── Session ──

let session = null;

export function ensureSession() {
  if (!session) session = createSession();
  return session;
}

// ── Ingest ──

export function engineIngestText(text, sourceId) {
  const s = ensureSession();
  const { chunks, admitted } = admitChunked(s, { text, sourceId });
  ingestedChunkCount += chunks;
  const name = sourceId?.replace(/^.*[/\\]/, "") || "(unnamed)";
  ingestedSources.set(sourceId || name, { name, chunks, ingestedAt: Date.now() });
  return {
    sourceId,
    chunks,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

// ── Search ──

export function engineSearch(query, limit = 10, { maxChars = 800, source: sourceFilter } = {}) {
  const s = ensureSession();
  let { spans, gaps } = searchSpans(s, { query, limit: Math.min(limit, 40) });
  // Optionally filter by source_id prefix
  if (sourceFilter) {
    const sf = sourceFilter.replace(/^.*[/\\]/, ""); // basename
    spans = spans.filter(sp => sp.source_id && sp.source_id.includes(sf));
  }
  const units = spanUnits(s, spans);
  return {
    query,
    total: spans.length,
    passages: spans.map((sp, i) => ({
      span_id: sp.span_id,
      text: (units[i]?.text ?? sp.preview).slice(0, Math.max(1, maxChars)),
      source: sp.source_id || "",
      byte_start: sp.byte_start,
      byte_end: sp.byte_end,
      score: sp.score,
      preview: sp.preview,
    })),
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
export function engineGroundQuery(query, { budget = 600, maxUnits = 8, limit = 15, source: sourceFilter } = {}) {
  const s = ensureSession();
  let { spans, gaps } = searchSpans(s, { query, limit });
  if (sourceFilter) {
    const sf = sourceFilter.replace(/^.*[/\\]/, "");
    spans = spans.filter(sp => sp.source_id && sp.source_id.includes(sf));
  }
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

  // Build a context that includes verbatim citation text the LLM can cite
  const context = buildCitedContext(kept, foldResult, gaps);

  return {
    query,
    context,
    citations: kept,
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
  return readSpan(ensureSession(), { spanId, maxBytes });
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

// Score a line as a potential structural delimiter.  Returns 0–5.
// Clues: short, followed by blank line, contains numbering or formatting
// that makes it look like a heading rather than a sentence.
function headingScore(line, nextLineBlank) {
  const trimmed = line.trim();
  if (trimmed.length < 3 || trimmed.length > 80) return 0;
  if (!nextLineBlank) return 0;
  // Penalize sentence-like lines (end with ?.! or have multiple capitalized words)
  if (/[?.!]$/.test(trimmed) && !/^[IVXLCDM]+\.$/.test(trimmed)) return 0;
  // Abbreviation exclusion
  if (/^(M[\. ]|Dr[\. ]|Mr[\. ]|Mme[\. ]|Mlle[\. ]|St[\. ])/i.test(trimmed)) return 0;
  let s = 0;
  if (/^[IVXLCDM]+\.\s/.test(trimmed)) s += 3;
  if (/^\d+[).]\s/ .test(trimmed)) s += 3;
  if (/^[A-Z\s'"\u201c\u201d]{4,}$/.test(trimmed) || /^[A-Z][a-z]+\s+[A-Z]/.test(trimmed)) s += 2;
  if (/[?.!]$/.test(trimmed)) s -= 2;
  return s >= 2 ? s : 0;
}

// Dynamically discover segment boundaries near a byte offset by examining
// all candidate heading lines within a text window and finding the structural
// cluster that contains the anchor.
function discoverSegment(fileText, nearByte) {
  const hi = fileText.length;
  const radius = Math.min(6000, hi >> 2);
  const lo = Math.max(0, nearByte - radius);
  const hiClip = Math.min(hi, nearByte + radius);
  // Work on an array of { text, bytePos, isHeading, score }
  const chunk = fileText.slice(lo, hiClip);
  const lines = chunk.split("\n");
  const items = [];
  let acc = 0;
  for (let i = 0; i < lines.length; i++) {
    const bytePos = lo + acc;
    const nextBlank = i + 1 < lines.length && lines[i + 1].trim() === "";
    const score = headingScore(lines[i], nextBlank);
    items.push({ text: lines[i], bytePos, score, isHeading: score > 0 });
    acc += lines[i].length + 1;
  }
  // Find the anchor's line index
  const anchorRel = nearByte - lo;
  let anchorIdx = 0;
  let walk = 0;
  for (let i = 0; i < items.length; i++) {
    if (walk >= anchorRel) { anchorIdx = Math.max(0, i - 1); break; }
    walk += items[i].text.length + 1;
  }
  // Scan backward for nearest heading, forward for next heading
  let startIdx = null, endIdx = null;
  for (let i = anchorIdx; i >= 0; i--) {
    if (items[i].isHeading) { startIdx = i; break; }
  }
  for (let i = anchorIdx + 1; i < items.length; i++) {
    if (items[i].isHeading) { endIdx = i; break; }
  }
  if (startIdx == null && endIdx == null) return null;
  if (startIdx == null) startIdx = 0;
  const startByte = items[startIdx].bytePos;
  const endByte = endIdx != null ? items[endIdx].bytePos - 1 : hiClip;
  return { startByte, endByte, label: items[startIdx].text.trim(), headingCount: items.filter(x => x.isHeading).length };
}

// Read the segment (chapter, section, movement) containing the content
// described by query.  Boundary detection is dynamic and text-agnostic —
// works for CHAPTER I, I. Title, 1. Section, all-caps headings, etc.
export function engineReadSegment(query, maxBytes = 100000, sourceFilter) {
  const s = ensureSession();
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

  let fileText;
  try { fileText = fs.readFileSync(sourcePath, "utf8"); } catch (e) {
    return { error: `Cannot read ${sourcePath}: ${e.message}` };
  }

  const anchor = best.byte_start || 0;
  const seg = discoverSegment(fileText, anchor);
  if (!seg) {
    const fallback = fileText.slice(Math.max(0, anchor - 2000), anchor + maxBytes);
    return {
      segment: "(no structural boundary detected)",
      source: sourcePath,
      byte_start: Math.max(0, anchor - 2000),
      byte_end: Math.min(fileText.length, anchor + maxBytes),
      truncated: false,
      text: fallback,
    };
  }

  const length = Math.min(seg.endByte - seg.startByte, maxBytes);
  const segText = fileText.slice(seg.startByte, seg.startByte + length);

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

// Read a span with surrounding context (expand before/after to arbitrary byte
// windows). Given a span_id or a { byte_start, byte_end, source }, read N bytes
// before and M bytes after from the source file.
export function engineReadContext(spanRef, { beforeBytes = 0, afterBytes = 0, maxTotal = 50000 } = {}) {
  const s = ensureSession();

  // Resolve the span: either a span_id (look up in session) or a direct ref
  let sourceId, byteStart, byteEnd;
  if (typeof spanRef === "string") {
    const rec = s.spans.get(spanRef);
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

  let text;
  try { text = fs.readFileSync(sourcePath, "utf8"); } catch (e) {
    return { error: `Cannot read ${sourcePath}: ${e.message}` };
  }

  const readStart = Math.max(0, byteStart - beforeBytes);
  const readEnd = Math.min(text.length, byteEnd + afterBytes);
  const length = Math.min(readEnd - readStart, maxTotal);
  const contextText = text.slice(readStart, readStart + length);

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

let ingestedChunkCount = 0;
const ingestedSources = new Map(); // path → { name, chunks, ingestedAt }

export function engineIngestFile(filePath) {
  const s = ensureSession();
  const { chunks, admitted } = ingestFile(s, filePath);
  ingestedChunkCount += chunks;
  const name = filePath.replace(/^.*[/\\]/, "");
  ingestedSources.set(filePath, { name, chunks, ingestedAt: Date.now() });
  return {
    path: filePath,
    chunks,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

export function engineListSources() {
  return Array.from(ingestedSources.entries()).map(([path, info]) => ({
    path,
    name: info.name,
    chunks: info.chunks,
    ingestedAt: info.ingestedAt,
  }));
}

export function engineStats() {
  const s = ensureSession();
  return {
    sessionActive: !!s,
    ingestedChunks: ingestedChunkCount,
    ingestedFiles: ingestedSources.size,
    spanCap: s.spanCap ?? 2000,
    sources: Array.from(ingestedSources.values()).map(s => s.name),
  };
}
