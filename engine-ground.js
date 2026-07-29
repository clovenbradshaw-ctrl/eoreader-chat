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
  return {
    path: filePath,
    chunks,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

export function engineIngestText(text, sourceId) {
  const s = ensureSession();
  const { chunks, admitted } = admitChunked(s, { text, sourceId });
  return {
    sourceId,
    chunks,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

// ── Search ──

export function engineSearch(query, limit = 10) {
  const s = ensureSession();
  const { spans, gaps } = searchSpans(s, { query, limit: Math.min(limit, 40) });
  const units = spanUnits(s, spans);
  return {
    query,
    total: spans.length,
    passages: spans.map((sp, i) => ({
      span_id: sp.span_id,
      text: (units[i]?.text ?? sp.preview).slice(0, 800),
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

// Search + fold into a single compact context block the LLM can consume.
// Returns the folded summary and the underlying evidence passages so the UI
// can display citations.
export function engineGroundQuery(query, { budget = 600, maxUnits = 8, limit = 15 } = {}) {
  const s = ensureSession();
  const { spans, gaps } = searchSpans(s, { query, limit });
  const units = spanUnits(s, spans);
  const foldResult = foldSpans(s, { units, query, tokenBudget: budget, maxUnits });

  // Pick the passages that were kept by the fold (by span_id match)
  const kept = (foldResult.selected || []).map((u) => {
    const rec = units.find(r => r.meta?.span_id === u.meta?.span_id);
    return {
      text: u.text?.slice(0, 800),
      source: u.meta?.source_id ?? u.meta?.source ?? "unknown",
      score: u.meta?.score ?? 0,
      span_id: u.meta?.span_id,
    };
  });

  // Build a cited, grounded context block
  let context = "";
  if (foldResult.summary) {
    context = foldResult.summary;
  } else if (units.length) {
    // No fold summary (units too short for compression) — use raw units
    context = units.map((u, i) => `[${i + 1}] ${u.text.slice(0, 500)}`).join("\n\n");
  }

  // Include gap information for the LLM — a typed gap is honest, a silent miss is not
  if (gaps?.length) {
    context += `\n\n[Engine gaps: ${gaps.map(g => g.reason || g).join("; ")}]`;
  }

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

// ── Status ──

export function engineStats() {
  const s = ensureSession();
  return {
    sessionActive: !!s,
    ingestedChunks: s.spans?.size ?? 0,
    spanCap: s.spanCap ?? 2000,
  };
}
