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

export function engineIngestFile(filePath) {
  const s = ensureSession();
  const { chunks, admitted } = ingestFile(s, filePath);
  ingestedChunkCount += chunks;
  return {
    path: filePath,
    chunks,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

export function engineIngestText(text, sourceId) {
  const s = ensureSession();
  const { chunks, admitted } = admitChunked(s, { text, sourceId });
  ingestedChunkCount += chunks;
  return {
    sourceId,
    chunks,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

// ── Search ──

export function engineSearch(query, limit = 10, { maxChars = 800 } = {}) {
  const s = ensureSession();
  const { spans, gaps } = searchSpans(s, { query, limit: Math.min(limit, 40) });
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
export function engineGroundQuery(query, { budget = 600, maxUnits = 8, limit = 15 } = {}) {
  const s = ensureSession();
  const { spans, gaps } = searchSpans(s, { query, limit });
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

// ── Context snipping: read chapters and arbitrary windows ──

// Pattern for chapter/section markers in Project Gutenberg texts.
// Covers: CHAPTER I, CHAPTER VII, CHAPITRE PREMIER, CHAPITRE II,
// Roman-numeral headings like "II. Le violon enchanté", "III. ..."
const CHAPTER_PATTERN = /^(?:CHAPTER\s+(?:[A-Z]+|\d+)|CHAPITRE\s+(?:PREMIER|I[XV]?|V?I{0,3}|[A-Z]+)|[IVXLCDM]+\..+)$/m;

function resolveSourcePath(sourceId) {
  // source_id is like "source:/path/to/file.txt:chunk-N" or "source:/path/to/file.txt"
  const match = sourceId?.match(/^source:(.+?)(?::chunk-\d+)?$/);
  return match ? match[1] : null;
}

// Find chapter boundaries in a source file given a chapter query.
// Returns { startByte, endByte, chapterLabel } or null.
function findChapterBoundary(session, chapterQuery) {
  // Search for the chapter marker
  const { spans } = searchSpans(session, { query: chapterQuery, limit: 5 });
  if (!spans.length) return null;

  // Use the best-scoring result to identify the source file
  const best = spans[0];
  const sourcePath = resolveSourcePath(best.source_id);
  if (!sourcePath) return null;

  let text;
  try { text = fs.readFileSync(sourcePath, "utf8"); } catch { return null; }

  // Find the chapter marker byte offset in the file
  const searchText = chapterQuery.toLowerCase();
  const idx = text.toLowerCase().indexOf(searchText, Math.max(0, (best.byte_start || 0) - 500));
  if (idx < 0) return null;

  // Extract chapter label from the line
  const lineStart = text.lastIndexOf("\n", idx) + 1;
  const lineEnd = text.indexOf("\n", idx);
  const chapterLabel = text.slice(lineStart, lineEnd >= 0 ? lineEnd : idx + 80).trim();

  // Find next chapter boundary
  const afterStart = idx + searchText.length;
  const nextChapter = text.slice(afterStart).search(CHAPTER_PATTERN);
  const endByte = nextChapter >= 0 ? afterStart + nextChapter : text.length;

  return {
    startByte: lineStart,
    endByte,
    chapterLabel,
    sourcePath,
    byte_start: best.byte_start,
    byte_end: best.byte_end,
  };
}

// Read the full text of a chapter found by query.
// "chapter 2", "CHAPTER VII", "II. Le violon enchanté", etc.
export function engineReadChapter(chapterQuery, maxBytes = 50000) {
  const s = ensureSession();
  const boundary = findChapterBoundary(s, chapterQuery);
  if (!boundary) return { error: `Chapter not found: "${chapterQuery}"` };

  let text;
  try { text = fs.readFileSync(boundary.sourcePath, "utf8"); } catch (e) {
    return { error: `Cannot read ${boundary.sourcePath}: ${e.message}` };
  }

  const length = Math.min(boundary.endByte - boundary.startByte, maxBytes);
  const chapterText = text.slice(boundary.startByte, boundary.startByte + length);

  return {
    chapter: boundary.chapterLabel,
    source: boundary.sourcePath,
    byte_start: boundary.startByte,
    byte_end: boundary.startByte + length,
    truncated: length < (boundary.endByte - boundary.startByte),
    text: chapterText,
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

// ── Status ──

let ingestedChunkCount = 0;

export function engineStats() {
  const s = ensureSession();
  return {
    sessionActive: !!s,
    ingestedChunks: ingestedChunkCount,
    spanCap: s.spanCap ?? 2000,
  };
}
