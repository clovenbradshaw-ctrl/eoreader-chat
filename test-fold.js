#!/usr/bin/env node
/**
 * Test script: the /api/fold projection (engine-ground.js::engineFoldSource).
 *
 *   node test-fold.js [--frankenstein <path>] [--warandpeace <path>]
 *
 * What this guards is not "does it return data" but "does it ever return data
 * it does not have". The projection's whole reason to exist is that eochat's
 * buildEntityMatcher — a regex over capitalized words, top-20 by frequency —
 * confidently returns a cast it invented. Every assertion below is a way of
 * failing that would look like success in the UI:
 *
 *   - a referent shown with a type no prior asserted
 *   - a referent shown with a mass nothing computed
 *   - a byte anchor that does not point at the surface it claims
 *   - a beat "agreed" by derivations that were never emitted
 *   - a cast that changes depending on what ELSE is ingested (the regression
 *     that cost Frankenstein's Creature: see anchorsFor's header)
 *
 * Sources default to the CLAUDE.md corpus paths. Missing files SKIP the cases
 * that need them rather than failing, so this runs on a fresh checkout.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { engineIngestFile, engineFoldSource, FOLD_PROJECTION_VERSION } from "./engine-ground.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function argOf(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

const FRANKENSTEIN = argOf("frankenstein", path.resolve(__dirname, "..", "pg84.txt"));
const WAR_AND_PEACE = argOf("warandpeace", path.join(process.env.HOME ?? "", "Downloads", "pg2600.txt"));

const GATED = ["holon", "emanon", "protogon", "field", "apparatus"];

let passed = 0;
let failed = 0;
let skipped = 0;

function ok(cond, label, detail) {
  if (cond) { passed++; console.log(`  ✔ ${label}`); }
  else { failed++; console.log(`  ✘ ${label}${detail ? `\n      ${detail}` : ""}`); }
}
function skip(label, why) { skipped++; console.log(`  – ${label} (skipped: ${why})`); }

// Every anchor must reproduce its own surface when the raw file is sliced at
// the byte range it reports. This is the mechanical citation contract: an
// anchor that cannot be re-read from the file is a fabricated one.
function checkAnchors(fold, filePath, label) {
  const buf = fs.readFileSync(filePath);
  const bad = [];
  let n = 0;
  for (const ref of fold.referents) {
    for (const a of ref.provenance.anchors) {
      n++;
      const got = buf.subarray(a.byte_start, a.byte_end).toString("utf8");
      if (got !== a.surface) bad.push(`${ref.name} @${a.byte_start}: claims ${JSON.stringify(a.surface)}, file has ${JSON.stringify(got)}`);
    }
  }
  ok(n > 0 && bad.length === 0, `${label}: all ${n} anchors re-read byte-exact from the source file`, bad.join("\n      "));
}

// The invariants that hold for ANY fold, priored or not.
function checkContract(fold, label) {
  ok(fold.fold_version === FOLD_PROJECTION_VERSION, `${label}: carries a fold version`);

  ok(fold.referents.every((r) => GATED.includes(r.individuation_type)),
    `${label}: every shown referent carries a gated individuation_type`,
    fold.referents.filter((r) => !GATED.includes(r.individuation_type)).map((r) => `${r.name}=${r.individuation_type}`).join(", "));

  ok(fold.referents.every((r) => r.mass === null),
    `${label}: no referent carries a synthesized mass`,
    fold.referents.filter((r) => r.mass !== null).map((r) => `${r.name}.mass=${r.mass}`).join(", "));

  ok(fold.referents.every((r) => r.aliasesResolved === true && r.evidence.length > 0),
    `${label}: every shown referent has resolved aliases and at least one anchor`);

  ok(fold.withheld.every((r) => r.aliasesResolved === false && r.individuation_type === null),
    `${label}: every withheld referent is flagged unresolved and untyped`);

  ok(fold.withheld.every((r) => typeof r.withheld_because === "string" && r.withheld_because.length > 0),
    `${label}: every withheld referent says why`);

  ok(fold.gaps.length > 0 && fold.gaps.every((g) => g.field && g.reason && ["engine", "model", "host"].includes(g.tier)),
    `${label}: gaps are typed {field, reason, tier}`);

  // Absent data is null-with-a-gap, never a plausible number.
  ok(fold.coverage.buckets === null && fold.gaps.some((g) => g.field === "coverage.buckets"),
    `${label}: coverage buckets are null with a gap, not invented chrome/dup/nul percentages`);
  ok(fold.gaps.some((g) => g.field === "referents[].mass"),
    `${label}: the unwired individuation gate is reported as a gap`);

  // A truncated audit list must not read as a complete one.
  ok(fold.withheld.length <= fold.withheld_total && (fold.withheld_truncated === (fold.withheld.length < fold.withheld_total)),
    `${label}: withheld truncation is reported honestly`);

  const cuts = (fold.divisions.derivations[0] || {}).cuts || [];
  ok(cuts.every((c) => c >= 0 && c <= 1),
    `${label}: division cuts are 0..1 fractions`, `got ${JSON.stringify(cuts.slice(0, 5))}`);

  // Only derivations we actually have may appear. A "dom" derivation here would
  // make reconcileDivisions report agreement between a real signal and a
  // fabricated one.
  const ids = fold.divisions.derivations.map((d) => d.id);
  ok(ids.every((id) => id === "novelty"),
    `${label}: only the derivation that exists is emitted`, `got ${JSON.stringify(ids)}`);
  ok(fold.gaps.some((g) => g.field === "divisions.derivations[dom]") && fold.gaps.some((g) => g.field === "divisions.derivations[strain]"),
    `${label}: the derivations we cannot produce are reported as gaps`);
}

console.log(`\n=== fold projection (${FOLD_PROJECTION_VERSION}) ===\n`);

// ── Case 1: a text WITH a per-text coref prior ──
console.log("Frankenstein (has eoPriors/priors/coref/pg84-frankenstein.json):");
let frankFold = null;
if (!fs.existsSync(FRANKENSTEIN)) {
  skip("priored source", `${FRANKENSTEIN} not found`);
} else {
  engineIngestFile(FRANKENSTEIN);
  frankFold = engineFoldSource(FRANKENSTEIN);
  ok(!frankFold.error, "folds without error", frankFold.error);
  checkContract(frankFold, "frankenstein");
  checkAnchors(frankFold, FRANKENSTEIN, "frankenstein");

  // The case the regex matcher cannot express at all: an unnamed referent.
  // "the Creature" has no name in the book; capitalization physics cannot find
  // it, and only the prior can type it.
  const creature = frankFold.referents.find((r) => r.individuation_type === "emanon");
  ok(!!creature, "an emanon (unnamed, ambient) referent survives — the case a capitalized-word regex cannot represent");
  ok(!!creature && creature.provenance.tier === "model",
    "the emanon's individuation is attributed to the model tier, not derived");
  ok(!!creature && creature.provenance.prior_snapshot?.identity === "coref/pg84-frankenstein",
    "the referent names the prior that individuated it");

  // Descriptor coref is witness-tier. Candidates the TEXT proposed must not be
  // typed on capitalization physics, however massive they are.
  ok(frankFold.withheld.length > 0 && frankFold.withheld.every((r) => !GATED.includes(String(r.individuation_type))),
    `all ${frankFold.withheld_total} text-proposed candidates are withheld, not typed`);
}

// ── Case 2: a text with NO prior ──
console.log("\nNo-prior source (a fold that must assert no cast at all):");
const noPrior = path.resolve(__dirname, "..", "nexrad-chronicle.txt");
if (!fs.existsSync(noPrior)) {
  skip("unpriored source", `${noPrior} not found`);
} else {
  engineIngestFile(noPrior);
  const fold = engineFoldSource(noPrior);
  ok(!fold.error, "folds without error", fold.error);
  checkContract(fold, "no-prior");
  ok(fold.referents.length === 0,
    "no prior ⇒ no cast: zero referents shown rather than a top-20 by frequency");
  ok(fold.sightings > 0,
    "sightings are still counted, so the demotion is visible as 'N sightings → 0'");
  ok(fold.gaps.some((g) => g.field === "prior" && g.tier === "model"),
    "the missing prior is a typed model-tier gap");
}

// ── Case 3: the cast must not depend on what else is ingested ──
// This is a regression guard with a measured failure behind it: when anchors
// came from pool-wide retrieval, ingesting War and Peace alongside Frankenstein
// pushed the Creature's spans off the result list and silently withheld him.
console.log("\nMulti-source isolation (regression: pool-wide retrieval starved anchors):");
if (!frankFold) {
  skip("isolation", "Frankenstein not available");
} else if (!fs.existsSync(WAR_AND_PEACE)) {
  skip("isolation", `${WAR_AND_PEACE} not found`);
} else {
  engineIngestFile(WAR_AND_PEACE);
  const after = engineFoldSource(FRANKENSTEIN);
  const before = frankFold.referents.map((r) => `${r.individuation_type}:${r.name}`).sort();
  const now = after.referents.map((r) => `${r.individuation_type}:${r.name}`).sort();
  ok(JSON.stringify(before) === JSON.stringify(now),
    "Frankenstein's cast is identical before and after another book is ingested",
    `before=${JSON.stringify(before)}\n      after =${JSON.stringify(now)}`);
  ok(after.referents.every((r) => r.provenance.anchors.every((a) => a.source_id === after.sourceId)),
    "every anchor belongs to the document being folded");

  const wp = engineFoldSource(WAR_AND_PEACE);
  ok(!wp.error, "war and peace folds without error", wp.error);
  checkContract(wp, "warandpeace");
  checkAnchors(wp, WAR_AND_PEACE, "warandpeace");

  // A prior is written against ONE edition and must be checked against its
  // bytes. pg2600 is the Maude translation: it anglicizes given names ("Prince
  // Andrew", never "Andréi") and transliterates surnames with -i and an accent
  // ("Bolkónski", never "Bolkonsky"). The Andrei entry was originally authored
  // for a different translation — every surface on it matched zero times — and
  // the one surface that did occur, "little princess", is Lise, his WIFE,
  // scoped by an anchor ("Prince Andrei felt that") that also occurs zero
  // times. It has been corrected to this edition and he must now show.
  const andrei = wp.referents.find((r) => /andrei-bolkonsky/.test(r.id));
  ok(!!andrei, "a prior corrected to this edition's spelling is shown, not withheld");
  ok(!!andrei && GATED.includes(String(andrei.individuation_type)) && andrei.provenance.anchors.length > 0,
    "the corrected referent is gated and carries byte-verified anchors",
    andrei && `type=${andrei.individuation_type} anchors=${andrei.provenance.anchors.length}`);
  // The teeth of the case: the anchors must land on the Maude forms. A prior
  // regressed to another translation's spelling would still be "shown" if this
  // only counted anchors, because a surface list that matches nothing produces
  // no anchor at all — so assert the surface each anchor actually quotes.
  ok(!!andrei && andrei.provenance.anchors.every((a) => /^(Prince\s+)?Andrew(\s+Bolkónski)?$/.test(a.surface)),
    "every anchor quotes this edition's spelling, not another translation's",
    andrei && JSON.stringify(andrei.provenance.anchors.map((a) => a.surface)));
  ok(!wp.withheld.some((r) => r.from_prior) && !wp.gaps.some((g) => g.field === "prior.referents"),
    "no prior-asserted referent is left unshown, so there is no prior/text disagreement gap",
    JSON.stringify(wp.withheld.filter((r) => r.from_prior).map((r) => r.name)));
  ok(!wp.referents.some((r) => r.provenance.anchors.some((a) => /little princess/i.test(a.surface))),
    "no referent is anchored on a scope-restricted surface");
}

// ── Case 4: source resolution refuses to guess ──
console.log("\nSource resolution:");
{
  const missing = engineFoldSource("definitely-not-a-source");
  ok(!!missing.error && /unknown source/.test(missing.error), "an unknown source is an error, not an empty fold");
  const blank = engineFoldSource("");
  ok(!!blank.error, "a blank source is an error");
}

console.log(`\n${passed} passed, ${failed} failed, ${skipped} skipped\n`);
process.exit(failed ? 1 : 0);
