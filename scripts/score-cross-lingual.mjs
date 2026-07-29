#!/usr/bin/env node
/**
 * Score cross-lingual / same-language verbatim retrieval.
 *
 * Reads /tmp/cross-lingual-golden.json, ingests the source text,
 * runs engineSearch for each test question, and reports recall@K.
 *
 * Usage:
 *   node scripts/score-cross-lingual.mjs [--k 5] [--max-chars 800]
 */

import { readFileSync } from "fs";
import { ensureSession, engineIngestFile, engineSearch } from "../engine-ground.js";

const golden = JSON.parse(readFileSync("/tmp/cross-lingual-golden.json", "utf8"));

const K = parseInt(process.argv.find(a => a.startsWith("--k="))?.split("=")[1] || "5", 10);
const maxChars = parseInt(process.argv.find(a => a.startsWith("--max-chars="))?.split("=")[1] || "800", 10);

console.error(`Ingesting ${golden.local_path}...`);
ensureSession();
engineIngestFile(golden.local_path);

let hits = 0;
let total = 0;
const details = [];

for (const test of golden.tests) {
  total++;
  const res = engineSearch(test.question, K, { maxChars });
  const topK = res.passages;
  const found = topK.some(p =>
    test.terms.some(term => p.text.toLowerCase().includes(term.toLowerCase()))
  );
  if (found) hits++;
  details.push({
    question: test.question,
    terms: test.terms,
    found,
    topScore: topK[0]?.score.toFixed(3) ?? "—",
    totalResults: res.total,
    resultTerms: topK.map(p =>
      test.terms.filter(t => p.text.toLowerCase().includes(t.toLowerCase())).join(",") || "—"
    ),
  });
}

console.log(JSON.stringify({
  golden: golden.name,
  language: golden.language,
  K,
  maxChars,
  total,
  hits,
  recall: total > 0 ? (hits / total).toFixed(3) : "0.000",
  details,
}, null, 2));
