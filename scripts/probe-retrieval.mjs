#!/usr/bin/env node
// Retrieval probe — every query below is a VERBATIM phrase (or an unambiguous
// paraphrase) from exactly one of the ingested sources, so the correct source
// is known without a hand-labelled relevance judgement. Scores top-1 and top-3
// source accuracy plus wall time, so a ranking change can be measured rather
// than eyeballed.
//
// Usage: node eoreader-chat/scripts/probe-retrieval.mjs [proxyUrl]

const PROXY = process.argv[2] || "http://localhost:11435";

const PROBES = [
  // Frankenstein — pg84.txt
  ["dreary night of November", "pg84"],
  ["I beheld the wretch whom I had created", "pg84"],
  ["his yellow skin scarcely covered the work of muscles", "pg84"],
  ["Victor Frankenstein created the creature", "pg84"],
  ["the ice raft on which the creature departed", "pg84"],
  ["Elizabeth Lavenza letter to Victor", "pg84"],
  // War and Peace — pg2600.txt
  ["Well, Prince, so Genoa and Lucca are now just family estates", "pg2600"],
  ["Natasha danced at her first ball", "pg2600"],
  ["the battle of Borodino", "pg2600"],
  ["Pierre Bezukhov inherited his father's fortune", "pg2600"],
  // KJV Bible — pg10.txt
  ["In the beginning God created the heaven and the earth", "pg10"],
  ["The LORD is my shepherd I shall not want", "pg10"],
  ["Jesus wept", "pg10"],
  // weather.csv
  ["temperature precipitation daily reading", "weather"],
];

const norm = (s) => (s || "").replace(/^source:/, "").replace(/:chunk-\d+$/, "").replace(/^.*\//, "");

// Only probe sources actually ingested right now. A probe for an absent source
// measures nothing about ranking, and silently counts as a miss — which is how
// a stale `weather.csv` left over from an earlier proxy session made the score
// look one worse than it was.
const ingested = await (await fetch(`${PROXY}/api/sources?pool=corpus`)).json();
const present = new Set(ingested.map((s) => s.name.replace(/\.[^.]+$/, "")));
const active = PROBES.filter(([, want]) => present.has(want));
for (const [q, want] of PROBES) {
  if (!present.has(want)) console.log(`- SKIP (source "${want}" not ingested): "${q}"`);
}

let top1 = 0, top3 = 0, totalMs = 0, missing = 0;
const rows = [];

for (const [q, want] of active) {
  const t0 = performance.now();
  let passages = [];
  try {
    const res = await fetch(`${PROXY}/api/verbatim?q=${encodeURIComponent(q)}&limit=3`, {
      signal: AbortSignal.timeout(60000),
    });
    passages = (await res.json()).passages || [];
  } catch (err) {
    rows.push([q, want, "ERROR: " + err.message, 0]);
    missing++;
    continue;
  }
  const ms = performance.now() - t0;
  totalMs += ms;

  const got = passages.map((p) => norm(p.source));
  if (got[0]?.startsWith(want)) top1++;
  if (got.some((g) => g.startsWith(want))) top3++;
  if (!got.length) missing++;
  rows.push([q, want, got.join(", ") || "(none)", ms]);
}

for (const [q, want, got, ms] of rows) {
  const hit = got.split(", ")[0]?.startsWith(want) ? "✓" : got.includes(want) ? "~" : "✗";
  console.log(`${hit} ${Math.round(ms).toString().padStart(5)}ms  want=${want.padEnd(8)} got=${got}`);
  console.log(`         "${q}"`);
}

const n = active.length;
console.log(`\ntop-1 ${top1}/${n}   top-3 ${top3}/${n}   empty ${missing}/${n}   mean ${Math.round(totalMs / n)}ms`);
