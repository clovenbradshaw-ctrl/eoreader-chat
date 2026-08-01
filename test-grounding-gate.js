#!/usr/bin/env node
/**
 * Test script: the grounding gate (grounding-gate.js).
 *
 *   node test-grounding-gate.js
 *
 * What this guards is the app's grounding contract: every served answer is
 * grounded in the reader's sources, never the model's own knowledge. The gate
 * has three moving parts and one invariant:
 *
 *   - buildGroundedSystemMessage: with evidence, the model is offered a
 *     numbered SOURCE MATERIAL table and told to cite [N]; without evidence,
 *     it is told it MUST NOT answer from its own knowledge.
 *   - validateCitations: a fabricated [N] is masked, never served.
 *   - citedNumbers: only brackets landing on an OFFERED passage count as
 *     grounding; anything else is zero.
 *
 * The invariant (the gate itself, enforced by the server after the turn):
 * an answer that cites none of the offered passages — or was offered none —
 * is MODEL-tier and must be voided, not served.
 */

import assert from "node:assert";
import { validateCitations, citedNumbers, buildGroundedSystemMessage, voidedAnswer } from "./grounding-gate.js";

const results = [];
const record = (name, fn) => {
  try { fn(); results.push({ name, pass: true }); console.log(`PASS  ${name}`); }
  catch (err) { results.push({ name, pass: false }); console.log(`FAIL  ${name}\n      ${err.message}`); }
};

// ── citedNumbers: grounding is only a real, offered passage ──
record("citedNumbers: counts in-range brackets", () => {
  assert.deepStrictEqual(citedNumbers("As the source says [1] and [3].", 4), [1, 3]);
});
record("citedNumbers: deduplicates", () => {
  assert.deepStrictEqual(citedNumbers("[2] then [2] again", 4), [2]);
});
record("citedNumbers: out-of-range bracket grounds nothing", () => {
  assert.deepStrictEqual(citedNumbers("See [7] for the truth.", 4), [], "only [1..maxCitation] may ground");
});
record("citedNumbers: zero maxCitation means nothing can ground", () => {
  assert.deepStrictEqual(citedNumbers("Claims [1], [2], [3].", 0), []);
});
record("citedNumbers: empty content is ungrounded", () => {
  assert.deepStrictEqual(citedNumbers("", 5), []);
});

// ── validateCitations: fabricated numbers are masked, real ones survive ──
record("validateCitations: keeps offered citations verbatim", () => {
  assert.strictEqual(validateCitations("It rained [1] that night.", 3), "It rained [1] that night.");
});
record("validateCitations: masks fabricated numbers as gaps", () => {
  assert.strictEqual(validateCitations("It rained [9] that night.", 3), "It rained [⊘ no source 9] that night.");
});
record("validateCitations: no citation table → no masking (nothing to mask)", () => {
  assert.strictEqual(validateCitations("Just prose [4].", 0), "Just prose [4].");
});

// ── buildGroundedSystemMessage: with evidence, cite the numbers ──
const grounded = {
  context: "=== CITED PASSAGES ===\n[1] …\n",
  citations: [{ span_id: "s1" }, { span_id: "s2" }],
  total: 5, folded: 1, tokens: 120,
};
record("buildGroundedSystemMessage: offers a numbered SOURCE MATERIAL table", () => {
  const { message } = buildGroundedSystemMessage(grounded, "q");
  assert.match(message.content, /SOURCE MATERIAL/);
  assert.match(message.content, /\[1\] through \[2\]/);
  assert.match(message.content, /ONLY cite these numbers/);
  assert.strictEqual(message._citationCount, 2);
});
record("buildGroundedSystemMessage: warming=false when evidence exists", () => {
  const built = buildGroundedSystemMessage(grounded, "q");
  assert.strictEqual(built.warming, false);
});

// ── buildGroundedSystemMessage: no evidence → forbidden to free-answer ──
const silent = { gaps: [], context: null };
record("buildGroundedSystemMessage: no evidence → citation count zero", () => {
  const { message } = buildGroundedSystemMessage(silent, "q", false);
  assert.strictEqual(message._citationCount, 0);
});
record("buildGroundedSystemMessage: no evidence → model must NOT answer from own knowledge", () => {
  const { message } = buildGroundedSystemMessage(silent, "q", false);
  assert.match(message.content, /Do NOT answer from your own knowledge/);
  assert.match(message.content, /no numbered SOURCE MATERIAL table/);
});
record("buildGroundedSystemMessage: warming is a distinct fact, reported distinctly", () => {
  const { message, warming } = buildGroundedSystemMessage(silent, "q", true);
  assert.strictEqual(warming, true);
  assert.match(message.content, /index is still warming up/);
});
record("buildGroundedSystemMessage: null grounding result is treated as silent", () => {
  const { message } = buildGroundedSystemMessage(null, "q", false);
  assert.strictEqual(message._citationCount, 0);
});

// ── voidedAnswer: the typed gap served in place of model-only prose ──
record("voidedAnswer: names the void and the reason", () => {
  const text = voidedAnswer("no passage matched");
  assert.match(text, /Answer voided/);
  assert.match(text, /no passage matched/);
});

console.log(`\n${results.filter(r => r.pass).length}/${results.length} passed`);
process.exit(results.every(r => r.pass) ? 0 : 1);
