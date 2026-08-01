#!/usr/bin/env node
// Affordance conformance harness — one check per success criterion in
// UX-DESIGN.md, so "does the basic thing work" has an answer that is measured
// rather than argued. Numbering follows the doc's Affordance Catalog.
//
// Usage: node eoreader-chat/scripts/test-affordances.mjs [proxyUrl]

import fs from "node:fs";

const PROXY = process.argv[2] || "http://localhost:11435";
const results = [];

function record(af, name, ok, detail, criterion) {
  results.push({ af, name, ok, detail, criterion });
  const mark = ok === true ? "PASS" : ok === false ? "FAIL" : "WARN";
  console.log(`${mark.padEnd(4)} [${af}] ${name}\n       ${detail}`);
}

const get = async (p, ms = 60000) =>
  fetch(PROXY + p, { signal: AbortSignal.timeout(ms) });

// Drive one chat turn, collecting the SSE events and their arrival times.
async function chat(content, { session = "affordance-test", webSearch = false } = {}) {
  const t0 = performance.now();
  const res = await fetch(PROXY + "/api/chat/tools", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content }],
      session, webSearch,
      groundBudget: 3000, groundMaxUnits: 8, groundLimit: 16,
    }),
    signal: AbortSignal.timeout(300000),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", event = null;
  const events = [];
  let answer = "", groundingAt = null, grounding = null, model = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("event: ")) { event = line.slice(7).trim(); continue; }
      if (!line.startsWith("data: ")) continue;
      let d; try { d = JSON.parse(line.slice(6)); } catch { continue; }
      events.push({ event, at: performance.now() - t0 });
      if (event === "grounding" || d.sourceCount !== undefined) {
        grounding = d; groundingAt ??= performance.now() - t0;
      }
      if (d.model) model = d.model;
      if (d.content) answer = d.content;
      if (event === "done" && d.content) answer = d.content;
    }
  }
  return { answer, grounding, groundingAt, events, model, totalMs: performance.now() - t0 };
}

console.log(`\n=== EOChat affordance conformance — ${PROXY} ===\n`);

// ── 3. Document Ingest ──
const sources = await (await get("/api/sources?pool=corpus")).json();
record(3, "sources are ingested and listed",
  sources.length >= 3,
  `${sources.length} source(s): ${sources.map((s) => `${s.name} (${s.chunks} chunks)`).join(", ")}`,
  "user can drop a book and search it");

// Duplicate detection
const dupRes = await fetch(PROXY + "/api/ingest", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ path: sources[0]?.path }),
  signal: AbortSignal.timeout(120000),
});
const dupBody = await dupRes.text();
record(3, "duplicate ingest is rejected",
  /duplicate|already/i.test(dupBody) || dupRes.status >= 400,
  `re-ingesting ${sources[0]?.name} → ${dupRes.status} ${dupBody.slice(0, 120)}`,
  "duplicate files detected (SHA hash) and rejected");

// ── 6. Verbatim Search ──
const vT0 = performance.now();
const vRes = await (await get("/api/verbatim?q=" + encodeURIComponent("dreary night of November") + "&limit=3")).json();
const vMs = performance.now() - vT0;
const top = vRes.passages?.[0];
record(6, "verbatim search finds the right source",
  !!top && /pg84/.test(top.source || ""),
  `top hit: ${(top?.source || "none").replace(/^.*\//, "")} in ${Math.round(vMs)}ms`,
  "search returns exact text from source");
record(6, "search latency < 3s",
  vMs < 3000, `${Math.round(vMs)}ms`, "search latency < 3s");

// Byte offsets must actually address the file.
let offsetOk = null, offsetDetail = "no byte offsets returned";
if (top && top.byte_start != null) {
  const srcPath = sources.find((s) => top.source.includes(s.name))?.path;
  if (srcPath && fs.existsSync(srcPath)) {
    const fd = fs.openSync(srcPath, "r");
    const len = top.byte_end - top.byte_start;
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, top.byte_start);
    fs.closeSync(fd);
    const fromFile = buf.toString("utf8").replace(/\s+/g, " ").trim();
    const fromEngine = String(top.text || "").replace(/\s+/g, " ").trim();
    offsetOk = fromFile === fromEngine;
    offsetDetail = offsetOk
      ? `bytes ${top.byte_start}–${top.byte_end} of ${srcPath.replace(/^.*\//, "")} match the engine's text exactly`
      : `MISMATCH at bytes ${top.byte_start}–${top.byte_end}\n         file:   ${fromFile.slice(0, 90)}\n         engine: ${fromEngine.slice(0, 90)}`;
  }
}
record(6, "byte offsets address the real file", offsetOk, offsetDetail,
  "citation verifiability: 100% of citations link to exact source");

// ── 4. Grounded Citations + 8. Streaming ──
const q1 = await chat("What does Victor see when the creature first opens its eye?");
record(8, "grounding event arrives before the model answers",
  q1.groundingAt != null && q1.groundingAt < 2000,
  `grounding at ${Math.round(q1.groundingAt ?? -1)}ms, full answer at ${Math.round(q1.totalMs)}ms (model ${q1.model})`,
  "grounding event arrives first; first token < 1s");
record(4, "answer carries inline citations",
  /\[\d+\]/.test(q1.answer),
  q1.answer ? `${(q1.answer.match(/\[\d+\]/g) || []).length} citation(s); answer: ${q1.answer.replace(/\s+/g, " ").slice(0, 150)}` : "(empty answer)",
  "every factual claim has a citation");
record(4, "answer is not a refusal while holding evidence",
  !/do(es)? not contain|cannot provide|no information|not mentioned/i.test(q1.answer || ""),
  `${q1.grounding?.sourceCount ?? 0} retrieved / ${q1.grounding?.foldedCount ?? 0} offered to model`,
  "gap reporting: missing evidence produces typed gaps, not fake answers");
record(4, "grounded response latency < 5s",
  q1.totalMs < 5000, `${Math.round(q1.totalMs)}ms`, "response latency < 5s for grounded answers");

// Every citation the model emitted must resolve to a real span.
const citNums = [...new Set((q1.answer?.match(/\[(\d+)\]/g) || []).map((s) => s.replace(/\D/g, "")))];
const table = q1.grounding?.citations || [];
const unresolvable = citNums.filter((n) => !table.some((c) => String(c.index) === n));
record(4, "no citation points at a passage that does not exist",
  citNums.length === 0 ? null : unresolvable.length === 0,
  citNums.length === 0 ? "answer emitted no citations to check"
    : `cited ${citNums.map((n) => "[" + n + "]").join(" ")} against a table of ${table.length}; unresolvable: ${unresolvable.length ? unresolvable.join(",") : "none"}`,
  "citations are mechanically verified (not model-generated)");

// ── 5. Cross-Text Reasoning ──
const q2 = await chat("Compare how creation is described in Frankenstein and in Genesis.");
const q2Sources = new Set((q2.grounding?.citations || []).map((c) => String(c.source_id).replace(/^.*\//, "").replace(/:chunk.*$/, "")));
record(5, "cross-text query grounds in more than one source",
  q2Sources.size >= 2,
  `citations drawn from: ${[...q2Sources].join(", ") || "none"}`,
  "query returns passages from all relevant sources");

// ── 1. Natural Conversation / 2. Persistent Memory ──
const memSession = "affordance-memory-" + Date.now();
await chat("Hello, my name is Alice and I am studying Gothic novels.", { session: memSession });
const recall = await chat("What is my name?", { session: memSession });
record(1, "recalls a fact from earlier in the same session",
  /alice/i.test(recall.answer || ""),
  `answer: ${(recall.answer || "(empty)").replace(/\s+/g, " ").slice(0, 150)}`,
  'ask "What is my name?" → should respond "Alice"');

const stats = await (await get("/api/discourse/stats?session=" + memSession)).json();
record(2, "discourse records the conversation",
  (stats.messageCount || 0) > 0,
  `messageCount ${stats.messageCount}, tokens ${stats.tokens}, ${stats.usagePercent}% of ${stats.contextWindow}`,
  "message count matches");

// ── 9. Source Control ──
const victim = sources.find((s) => /pg84/.test(s.name));
let sourceControl = null, scDetail = "no source to test with";
if (victim) {
  await fetch(PROXY + "/api/sources?source=" + encodeURIComponent(victim.name), { method: "DELETE", signal: AbortSignal.timeout(60000) });
  const afterDel = await (await get("/api/verbatim?q=" + encodeURIComponent("dreary night of November") + "&limit=5")).json();
  const stillThere = (afterDel.passages || []).some((p) => /pg84/.test(p.source || ""));
  const bin = await (await get("/api/recycle-bin")).json();
  const inBin = Array.isArray(bin) && bin.some((b) => /pg84/.test(b.name || b.source || JSON.stringify(b)));

  const restoreRes = await fetch(PROXY + "/api/recycle-bin/restore", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: victim.name }),
    signal: AbortSignal.timeout(180000),
  });
  const afterRestore = await (await get("/api/verbatim?q=" + encodeURIComponent("dreary night of November") + "&limit=5")).json();
  const back = (afterRestore.passages || []).some((p) => /pg84/.test(p.source || ""));

  sourceControl = !stillThere && inBin && back;
  scDetail = `after delete: ${stillThere ? "STILL SEARCHABLE" : "gone from search"}; in recycle bin: ${inBin}; restore ${restoreRes.status} → ${back ? "searchable again" : "NOT SEARCHABLE"}`;
}
record(9, "delete → recycle bin → restore round-trips", sourceControl, scDetail,
  "deleted source no longer appears; restored source is searchable again");

// ── 10. Priors ──
const priors = await (await get("/api/priors")).json();
record(10, "priors are listed and pooled separately",
  (priors.count || 0) > 0 && priors.pool === "priors",
  `${priors.count} prior(s) in pool "${priors.pool}"${priors.gaps?.length ? `; ${priors.gaps.length} declared gap(s)` : ""}`,
  "user can browse what priors are loaded");
const corpusCitedPriors = (q1.grounding?.citations || []).filter((c) => /eoPriors|priors\//.test(String(c.source_id)));
record(10, "priors never leak into corpus citations",
  corpusCitedPriors.length === 0,
  corpusCitedPriors.length ? `LEAKED: ${corpusCitedPriors.map((c) => c.source_id).join(", ")}` : "no prior appeared as a corpus citation",
  "priors don't leak into corpus grounding");

// ── 12. Model Routing ──
const router = await (await get("/v1/router")).json();
const greet = await chat("Hi");
record(12, 'a greeting is answered quickly',
  greet.totalMs < 3000,
  `"Hi" → ${Math.round(greet.totalMs)}ms on ${greet.model}; router mode: ${router.mode}`,
  "'Hello' routes to tiny model (< 1s response)");

// ── Scorecard ──
const pass = results.filter((r) => r.ok === true).length;
const fail = results.filter((r) => r.ok === false).length;
const warn = results.filter((r) => r.ok === null).length;
console.log(`\n=== ${pass} pass · ${fail} fail · ${warn} inconclusive (of ${results.length}) ===`);
if (fail) {
  console.log("\nFailing criteria:");
  for (const r of results.filter((r) => r.ok === false)) {
    console.log(`  [${r.af}] ${r.name}\n        criterion: ${r.criterion}\n        measured:  ${r.detail.split("\n")[0]}`);
  }
}
