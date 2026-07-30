#!/usr/bin/env node
/**
 * test-holonic-task.js — End-to-end test of holonic task decomposition.
 *
 * Tests:
 *   1. Mock engine fallback with surf + prior simulation
 *   2. Real model integration with Ollama
 *   3. Mechanical citation (n-gram overlap) tracking
 *   4. Provenance assembly (surf + priors + citations)
 *
 * Usage:
 *   node test-holonic-task.js [task] [--model <model>] [--output <path>] [--mock-only]
 *
 * Examples:
 *   node test-holonic-task.js --mock-only
 *   node test-holonic-task.js "write a 3-page essay about Frankenstein's creature" --model gemma2:2b
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { HolonicTask } from "./holonic-task.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..");

// ── Mock engine with prior simulation ──

const KNOWN_PRIORS = [
  { id: "coref:creature", text: 'In this text, "the monster", "the creature", "the daemon", "the fiend", and "the being" all refer to Victor Frankenstein\'s creation.' },
  { id: "coref:victor", text: 'In this text, "Frankenstein", "Victor", and "Victor Frankenstein" refer to the scientist who created the Creature. "My creator" also refers to Victor.' },
  { id: "coref:elizabeth", text: 'In this text, "Elizabeth", "Elizabeth Lavenza", "my cousin" refer to Victor\'s adopted sister and later wife.' },
  { id: "coref:walton", text: 'In this text, "Walton", "Robert Walton", "the stranger" and "the captain" refer to the Arctic explorer who tells the frame story.' },
  { id: "corpus:book", text: "Frankenstein is a novel by Mary Shelley, first published in 1818. It is a Gothic novel exploring themes of creation, ambition, and isolation." },
  { id: "corpus:setting", text: "The story is set across Geneva (Victor's hometown), Ingolstadt (where he creates the Creature), and the Arctic (where the frame story takes place)." },
];

class MockEngine {
  constructor() {
    this.entries = [];
  }

  ingestFile(filePath) {
    try {
      const text = fs.readFileSync(filePath, "utf8");
      const lines = text.split("\n").filter(l => l.trim().length > 40);
      for (let i = 0; i < Math.min(lines.length, 50); i++) {
        this.entries.push({
          text: lines[i].slice(0, 1000),
          meta: { file: path.basename(filePath), line: i },
          ts: Date.now(),
        });
      }
      console.error(`  [mock-engine] Ingested ${Math.min(lines.length, 50)} chunks from ${path.basename(filePath)}`);
      return { file: filePath, chunks: Math.min(lines.length, 50) };
    } catch (err) {
      console.error(`  [mock-engine] Could not read ${filePath}: ${err.message}`);
      return { file: filePath, chunks: 0 };
    }
  }

  search(query, { limit = 5 } = {}) {
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    const scored = this.entries.map(e => {
      const text = e.text.toLowerCase();
      let score = 0;
      for (const w of words) {
        if (text.includes(w)) score += 2;
        const tokens = text.split(/\s+/);
        for (const t of tokens) {
          if (t === w) score += 1;
          else if (t.includes(w) || w.includes(t)) score += 0.3;
        }
      }
      return {
        text: e.text.slice(0, 800),
        source: e.meta.file || "mock",
        score,
        id: e.meta.id || `mock:${e.meta.file}:${e.meta.line}`,
      };
    });
    return scored.filter(s => s.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
  }

  // Return which known priors are activated by the given text
  getPriors(text) {
    if (!text || typeof text !== "string") return [];
    const lower = text.toLowerCase();
    const activated = [];
    for (const p of KNOWN_PRIORS) {
      const triggerWords = p.id.includes("coref:") || p.id.includes("corpus:")
        ? p.text.replace(/["',.;:!?()]/g, "").split(/\s+/).filter(w => w.length > 3)
        : [];
      // Simple check: if the text contains key terms from the prior, activate it
      // More precisely: check if the prior's surface forms appear in the text
      const surfaceForms = p.text.match(/"([^"]+)"/g) || [];
      const hasMatch = surfaceForms.some(sf => {
        const clean = sf.replace(/["]/g, "").toLowerCase();
        return lower.includes(clean);
      });
      if (hasMatch) {
        activated.push({ id: p.id, text: p.text });
      }
    }
    return activated;
  }
}

// ── CLI ──

function parseArgs() {
  const args = process.argv.slice(2);
  const flags = {};
  const positional = [];

  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--")) {
      const key = args[i].slice(2);
      const val = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : true;
      flags[key] = val;
    } else {
      positional.push(args[i]);
    }
  }

  return {
    task: positional.join(" ") || "write a 5-page essay about sea turtles: their biology, ecological importance, threats, and conservation efforts.",
    model: flags.model || "gemma2:2b",
    output: flags.output || path.join(PROJECT_ROOT, "holonic-task-output.md"),
    mockOnly: flags["mock-only"] || false,
    sourceFile: flags.source || path.join(PROJECT_ROOT, "pg84.txt"),
  };
}

// ── Plan fallback ──

function fallbackPlan(task) {
  const topic = task.replace(/^(write|create|generate|produce|research)\s+(a\s+|an\s+)?/i, "").slice(0, 100).trim();
  return {
    subTasks: [
      { id: "intro", label: "Introduction", description: `Introduce the topic "${topic}", its significance, and what the document will cover.`, type: "introduction" },
      { id: "body1", label: "Core Analysis", description: `Present the main analysis of "${topic}".`, type: "section" },
      { id: "body2", label: "Evidence & Examples", description: `Support the analysis of "${topic}" with evidence and examples from the source material.`, type: "section" },
      { id: "body3", label: "Broader Context", description: `Place "${topic}" in broader context.`, type: "section" },
      { id: "conclusion", label: "Conclusion", description: `Synthesize key themes about "${topic}" and provide closing thoughts.`, type: "conclusion" },
    ],
  };
}

// ── Main ──

async function runTest() {
  const config = parseArgs();

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║     Holonic Task Decomposition — End-to-End Test        ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
  console.log("");
  console.log(`Task:   ${config.task}`);
  console.log(`Model:  ${config.model}`);
  console.log(`Output: ${config.output}`);
  if (config.mockOnly) console.log("Mode:   MOCK ONLY (no Ollama calls)");
  console.log("");

  // Set up engine
  const engine = new MockEngine();

  if (!config.mockOnly) {
    if (config.sourceFile && fs.existsSync(config.sourceFile)) {
      console.error(`Ingesting source: ${config.sourceFile}`);
      engine.ingestFile(config.sourceFile);
    } else if (config.sourceFile) {
      console.error(`Source file not found: ${config.sourceFile} (continuing without engine)`);
    }
  }

  const holonic = new HolonicTask({
    task: config.task,
    model: config.model,
    engine: config.mockOnly ? null : engine,
    outputPath: config.output,
  });

  let planData;

  if (config.mockOnly) {
    planData = fallbackPlan(config.task);
    holonic.planResult = planData;
    holonic.subTaskResults = planData.subTasks.map(st => ({
      id: st.id,
      label: st.label,
      content: `[MOCK] ${st.label} — content would be generated here by the model.`,
      surf: [],
      priors: [],
      citations: [],
    }));
  } else {
    // Phase 1: Plan
    console.log("");
    console.log("─── Phase 1: Planning ───");
    try {
      planData = await holonic.plan();
      holonic.planResult = planData;
      console.log(`Generated plan: ${planData.subTasks.length} sub-tasks`);
      for (const st of planData.subTasks) {
        console.log(`  ${st.id}: ${st.label} (${st.type})`);
      }
    } catch (err) {
      console.error(`Plan failed: ${err.message}`);
      planData = fallbackPlan(config.task);
      holonic.planResult = planData;
      console.log(`Using fallback plan: ${planData.subTasks.length} sub-tasks`);
    }
    console.log("");

    // Phase 2: Execute each sub-task
    console.log("─── Phase 2: Execution ───");
    let draft = "";
    for (let i = 0; i < planData.subTasks.length; i++) {
      const st = planData.subTasks[i];
      console.log(`\n[${i + 1}/${planData.subTasks.length}] ${st.label}`);

      // Research
      console.log(`  Researching...`);
      const { surf, priors } = await holonic.researchSubtask(st);
      if (surf.length > 0) {
        console.log(`  Surf: ${surf.length} passages (top score: ${surf[0].score.toFixed(1)})`);
      } else {
        console.log(`  No engine surf found`);
      }
      if (priors.length > 0) {
        console.log(`  Priors: ${priors.length} activated`);
      }

      // Generate
      console.log(`  Generating...`);
      const result = await holonic.executeSubtask(st, { surf, priors, previousSections: draft });
      holonic.subTaskResults.push(result);
      draft += `\n\n## ${result.label}\n\n${result.content}`;

      const elapsed = ((Date.now() - i * 1000) / 1000).toFixed(1);
      console.log(`  Done: ${result.content.length} chars, ${result.citations.length} mechanical citations`);
      if (result.citations.length > 0) {
        for (const c of result.citations.slice(0, 3)) {
          const s = result.surf[c.surfIndex];
          console.log(`    surf[${c.surfIndex}] jaccard=${c.evidence.jaccard}: "${(s ? s.text.slice(0, 80) : '?')}..."`);
        }
        if (result.citations.length > 3) console.log(`    ... and ${result.citations.length - 3} more`);
      }
    }
    console.log("");
  }

  // Phase 3: Assemble
  console.log("─── Phase 3: Assembly ───");
  const output = await holonic.assemble();

  // Print summary
  console.log("");
  console.log("─── Results ───");
  console.log(`Task:     ${config.task}`);
  console.log(`Sections: ${holonic.subTaskResults.length}`);
  console.log(`Output:   ${output.length} chars (≈${Math.round(output.length / 3000)} pages)`);

  if (!config.mockOnly) {
    console.log(`Plan time:    ${(holonic.metrics.planTime / 1000).toFixed(1)}s`);
    console.log(`Execute time: ${(holonic.metrics.executeTime / 1000).toFixed(1)}s`);
    console.log(`Assemble time:${(holonic.metrics.assembleTime / 1000).toFixed(1)}s`);
    console.log(`Total time:   ${(holonic.metrics.totalTime / 1000).toFixed(1)}s`);
    console.log(`Total tokens: ${holonic.metrics.totalTokens}`);
  }

  if (holonic.outputPath) {
    console.log(`Saved to:     ${holonic.outputPath}`);
  }

  // ── Provenance summary ──
  console.log("");
  console.log("─── Provenance ───");
  for (const r of holonic.subTaskResults) {
    console.log(`  ${r.label}: ${r.surf.length} surf passages, ${r.priors.length} priors, ${r.citations.length} mechanical citations`);
  }

  // Count total mechanical citations
  const totalMc = holonic.subTaskResults.reduce((a, r) => a + r.citations.length, 0);
  const totalSurf = holonic.subTaskResults.reduce((a, r) => a + r.surf.length, 0);
  console.log(`  Total: ${totalMc} mechanical citations across ${totalSurf} surf passages`);

  // ── Gap report ──
  if (holonic.gaps.length > 0) {
    console.log(`\n─── Gaps ───`);
    console.log(`${holonic.gaps.length} engine gap(s) recorded`);
    for (const g of holonic.gaps.slice(0, 3)) {
      console.log(`  [${g.subTask}] ${g.reason}: "${g.text.slice(0, 80)}"`);
    }
  }

  // Output preview
  console.log("");
  console.log("─── Output Preview (first 25 lines) ───");
  const previewLines = output.split("\n").slice(0, 25);
  for (const line of previewLines) {
    console.log(line);
  }
  if (output.split("\n").length > 25) {
    console.log("... (truncated)");
  }

  console.log("\n=== Test Complete ===");
}

runTest().catch(err => {
  console.error(`\nFATAL: ${err.message}`);
  console.error(err.stack);
  process.exit(1);
});
