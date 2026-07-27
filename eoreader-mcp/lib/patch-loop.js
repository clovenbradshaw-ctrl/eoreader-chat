// Patch loop — compression-guided error correction.
//
// Takes a generated file and a golden target, runs the sweep to identify
// the worst-k windows, generates patches for each, re-sweeps, loops.
// All scoring is mechanical (compression distance). Model calls only
// happen for the actual patching — the steering is free.
//
// This is the meta-organ: it uses the steer + craft tools in a loop
// to converge on the golden without the model ever reading the full target.

import fs from "fs";
import { sweep, worstWindows, uncoveredTarget } from "./sweep.js";

export async function patchLoop(handlers, generatedPath, goldenPath, session, options = {}) {
  const maxRounds = options.maxRounds || 6;
  const targetOmission = options.targetOmission || 0.2;
  const k = options.k || 3;

  let currentPath = generatedPath;
  let history = [];

  for (let round = 0; round < maxRounds; round++) {
    const gen = fs.readFileSync(currentPath, "utf8");
    const golden = fs.readFileSync(goldenPath, "utf8");

    // 1. Sweep — mechanical scoring
    const channels = sweep(gen, golden);
    const worst = worstWindows(gen, golden, 0.1, k);
    const uncovered = uncoveredTarget(gen, golden, k);

    const avgOmission = channels.reduce((a, c) => a + c.omission, 0) / channels.length;
    const avgInvention = channels.reduce((a, c) => a + c.invention, 0) / channels.length;

    history.push({ round, avgOmission, avgInvention, channels, worstWindows: worst.length });

    console.log(`[patch] round ${round}: omission=${avgOmission.toFixed(3)} invention=${avgInvention.toFixed(3)}`);

    if (avgOmission < targetOmission && avgInvention < targetOmission) {
      console.log(`[patch] converged at round ${round}`);
      return { path: currentPath, history, converged: true };
    }

    // 2. Build patch spec from uncovered target chunks + worst window descriptions
    const worstChan = channels.filter(c => c.omission > 0.4).map(c => c.channel);
    const uncoverText = uncovered.map(u => u.preview.slice(0, 200)).join("\n\n").slice(0, 1500);

    const spec = [
      `Patch the HTML file at ${currentPath}.`,
      worstChan.length ? `Channels needing improvement: ${worstChan.join(", ")}.` : "",
      `The golden target has structural patterns your output is missing.`,
      `Inject these missing patterns from the golden target's content:`,
      ``,
      uncoverText,
      ``,
      `Keep all existing structure. Only add what's missing. Do not remove working code.`,
    ].filter(Boolean).join("\n");

    // 3. Generate patch
    const result = await handlers.craft({
      spec,
      canon: `Must be valid HTML. Must preserve existing content. Must add the uncovered golden patterns.`,
      output_path: `/tmp/patched-round-${round}.html`,
      file_type: "html",
      session,
    });

    // 4. Verify improvement
    const patched = fs.readFileSync(`/tmp/patched-round-${round}.html`, "utf8");
    const newChannels = sweep(patched, golden);
    const newOmission = newChannels.reduce((a, c) => a + c.omission, 0) / newChannels.length;

    if (newOmission < avgOmission * 1.1) {
      // Improvement or within 10% — keep the patch
      currentPath = `/tmp/patched-round-${round}.html`;
      console.log(`[patch]  → omission ${avgOmission.toFixed(3)} → ${newOmission.toFixed(3)} ✓`);
    } else {
      console.log(`[patch]  → omission ${avgOmission.toFixed(3)} → ${newOmission.toFixed(3)} ✗ (reverted)`);
      // Revert — keep previous state
    }
  }

  return { path: currentPath, history, converged: false };
}
