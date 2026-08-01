#!/usr/bin/env node
/**
 * score-holonic-correction-golden.mjs — validates that the holonic-task
 * correction loop actually corrects, not just that citation matching works.
 *
 * Two layers, both deterministic (no real model call):
 *   1. _scoreGrounding() rates a deliberately off-topic first draft below
 *      threshold and a hand-written faithful paraphrase above it, strictly
 *      higher — the information-theoretic "does this draft, used as a
 *      prior, reduce surprise on the real source" measure actually works,
 *      not just on verbatim copies but on genuine restatement.
 *   2. The full correction loop (executeSubtask, with a scripted mock model
 *      that returns the golden's draft1 in response to the correction
 *      prompt built from draft0) detects low grounding, corrects, and
 *      stops — exactly once, not indefinitely.
 *
 * Usage: node scripts/score-holonic-correction-golden.mjs
 */

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { HolonicTask } from "../holonic-task.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const GOLDEN_PATH = join(ROOT, "..", "test-fixtures", "holonic-correction-golden.json");

let golden;
try {
  golden = JSON.parse(readFileSync(GOLDEN_PATH, "utf8"));
} catch (e) {
  console.log(`SKIP: golden not found at ${GOLDEN_PATH}`);
  process.exit(0);
}

const results = { pass: 0, fail: 0 };
function record(id, passed, detail = "") {
  if (passed) {
    results.pass++;
    console.log(`PASS ${id}`);
  } else {
    results.fail++;
    console.log(`FAIL ${id}: ${detail}`);
  }
}

const task = new HolonicTask({ task: golden.task, model: "golden-mock" });
const surf = golden.surf;

// ── 1. Scoring function assertions ──

console.log("\n=== Scoring: does the info-theoretic measure reward comprehension? ===");

const score0 = task._scoreGrounding(golden.draft0, surf, []);
const score1 = task._scoreGrounding(golden.draft1, surf, []);

record("SCORE.draft0_below_threshold",
  score0.groundingScore < task.groundingThreshold,
  `groundingScore=${score0.groundingScore.toFixed(4)} threshold=${task.groundingThreshold}`);

record("SCORE.draft1_above_threshold",
  score1.groundingScore >= task.groundingThreshold,
  `groundingScore=${score1.groundingScore.toFixed(4)} threshold=${task.groundingThreshold}`);

record("SCORE.draft1_improves_on_draft0",
  score1.groundingScore > score0.groundingScore,
  `draft0=${score0.groundingScore.toFixed(4)} draft1=${score1.groundingScore.toFixed(4)}`);

const notes0 = task._diagnoseGrounding(surf, score0.coverage, score0.driftFraction, golden.draft0);
const expectedUnder = new Set(golden.expected.underCoveredSurfIndexOnDraft0 || []);
const actualUnder = new Set(notes0.underCoveredPassages.map((u) => u.surfIndex));
const underMatches = [...expectedUnder].every((i) => actualUnder.has(i));
record("SCORE.diagnosis_flags_expected_passages",
  underMatches,
  `expected=${[...expectedUnder]} actual=${[...actualUnder]}`);

// ── 2. Full correction-loop mechanics, with a scripted mock model ──

console.log("\n=== Loop: does low grounding actually trigger a correction, and stop once fixed? ===");

let calls = 0;
task._call = async (messages) => {
  calls++;
  const userMsg = messages.find((m) => m.role === "user")?.content || "";
  const isCorrection = userMsg.includes("Here is your previous draft");
  return isCorrection ? golden.draft1 : golden.draft0;
};

const result = await task.executeSubtask(golden.subTask, { surf, priors: [] });

record("LOOP.detects_low_grounding_on_iteration_0",
  result.iterations[0].groundingScore < task.groundingThreshold,
  `iteration0=${result.iterations[0].groundingScore.toFixed(4)}`);

record("LOOP.converges_by_iteration_1",
  result.iterations.length >= 2 && result.iterations[1].groundingScore >= task.groundingThreshold,
  `iterations=${result.iterations.map((it) => it.groundingScore.toFixed(4)).join(",")}`);

record("LOOP.stops_once_converged",
  result.iterations.length === 2,
  `expected exactly 2 iterations (low draft0, then converged draft1), got ${result.iterations.length}`);

record("LOOP.final_content_is_draft1",
  result.content === golden.draft1,
  `final content did not match the expected converged draft`);

record("LOOP.model_called_expected_number_of_times",
  calls === 2,
  `expected 2 model calls (initial + 1 correction), got ${calls}`);

console.log(`\n=== RESULTS: ${results.pass} pass, ${results.fail} fail ===`);
process.exit(results.fail > 0 ? 1 : 0);
