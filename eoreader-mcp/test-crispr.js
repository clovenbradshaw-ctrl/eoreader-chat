import fs from "fs";
import zlib from "zlib";
import { extractSlots, fillSlot, applySlot } from "./lib/crispr.js";

function gzip(text) {
  return zlib.gzipSync(text || "", { level: 9 }).length;
}

function score(gen, ref) {
  const cr = gzip(ref), cg = gzip(gen);
  if (cr === 0 || cg === 0) return { omit: 1, inv: 1 };
  const omit = Math.max(0, (gzip(ref + gen) - cg) / cr);
  const inv = Math.max(0, (gzip(gen + ref) - cr) / cg);
  return { omit, inv };
}

const golden = fs.readFileSync("/tmp/dolphinzone-golden.html", "utf8");

// Step 1: Extract slots
console.log("=== EXTRACT SLOTS ===");
const slots = extractSlots(golden);
console.log("Found", slots.length, "slots:");
slots.forEach(s => console.log("  " + s.name + ': "' + s.golden.slice(0, 50) + '"'));

// Step 2: Score skeleton vs golden (all slots emptied)
console.log("\n=== SKELETON BASELINE ===");
let skeleton = golden;
for (const slot of slots) {
  skeleton = applySlot(skeleton, slot, "");
}
const skelScore = score(skeleton, golden);
console.log("Skeleton:", skeleton.length, "chars, omission:", skelScore.omit.toFixed(3), "invention:", skelScore.inv.toFixed(3));
fs.writeFileSync("/tmp/crispr-skeleton.html", skeleton, "utf8");

// Step 3: Fill each slot with 3 stochastic candidates, pick best by compression
console.log("\n=== CRISPR FILL (3 candidates/slot) ===");
let current = skeleton;
const log = [];

for (let i = 0; i < slots.length; i++) {
  const slot = slots[i];
  console.log("\n[" + (i + 1) + "/" + slots.length + "] " + slot.name);
  console.log('  golden: "' + slot.golden.slice(0, 60) + '"');

  const candidates = await fillSlot(slot, 3);

  for (const c of candidates) {
    console.log("  #" + c.index + ': "' + c.text.slice(0, 50) + '" omit=" + c.omit.toFixed(3)');
  }

  const best = candidates[0];
  console.log('  -> BEST: "' + best.text + '" (omit=' + best.omit.toFixed(3) + ")");

  current = applySlot(current, slot, best.text);

  const afterScore = score(current, golden);
  log.push({
    slot: slot.name,
    chosen: best.text.slice(0, 30),
    omit: afterScore.omit.toFixed(3),
    inv: afterScore.inv.toFixed(3),
  });
}

// Step 4: Final score
console.log("\n=== FINAL RESULT ===");
const finalScore = score(current, golden);
console.log("Size:", current.length, "chars (golden:", golden.length, "chars)");
console.log("Omission:", finalScore.omit.toFixed(3));
console.log("Invention:", finalScore.inv.toFixed(3));
console.log("\nImprovement from skeleton to filled:");
console.log("  Omission:", skelScore.omit.toFixed(3), "->", finalScore.omit.toFixed(3));
console.log("  Invention:", skelScore.inv.toFixed(3), "->", finalScore.inv.toFixed(3));

fs.writeFileSync("/tmp/crispr-result.html", current, "utf8");
console.log("\nSaved to /tmp/crispr-result.html");

// Step 5: Log summary
console.log("\n=== SLOT LOG ===");
log.forEach(l => console.log("  " + l.slot + ': "' + l.chosen + '" omit=' + l.omit + " inv=" + l.inv));
