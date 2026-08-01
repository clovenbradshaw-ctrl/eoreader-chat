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

// ── Mock model: deliberately low-grounding first draft, corrects on retry ──
//
// This exercises the real correction loop in HolonicTask.executeSubtask()
// end-to-end without needing Ollama: the first draft ignores the source
// passages (low grounding, forces a correction pass); subsequent drafts echo
// the passages the correction prompt flagged as missing (grounding should
// clear the threshold and the loop should stop).

function makeMockModelCall() {
  return async function mockCall(messages) {
    const userMsg = messages.find(m => m.role === "user")?.content || "";
    const isCorrection = userMsg.includes("Here is your previous draft");
    if (!isCorrection) {
      return "This section touches on broadly relevant ideas in a general way, without citing anything specific from any particular source.";
    }
    // Only the "Incorporate it" section holds actual source passages — the
    // block before it is the (deliberately ungrounded) previous draft, which
    // must NOT be echoed back or the mock would never converge.
    const afterMarker = userMsg.split("Incorporate it:")[1] || "";
    const passageBlocks = [...afterMarker.matchAll(/---\n([\s\S]*?)\n---/g)].map(m => m[1]).filter(Boolean);
    return passageBlocks.length > 0
      ? passageBlocks.slice(0, 3).join(" ")
      : "Revised section incorporating the source material more closely.";
  };
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
    tree: flags.tree || false,
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

// ── Tree test (mock model, 2-level tree) ──
//
// Builds a small holonic tree manually and exercises recursive execution
// (branch synthesis + leaf execution) with the mock correction loop.
// Verifies that surplus scoring fires on cross-passage synthesis and
// that branch nodes correctly synthesize their children.

async function runTreeTest(config) {
  const { HolonNode } = await import("./holonic-task.js");

  // Seed a mock engine so researchSubtask returns surf passages
  const engine = new MockEngine();

  const holonic = new HolonicTask({
    task: config.task,
    model: config.model,
    engine: config.mockOnly ? engine : null,
    outputPath: null,
    surplusThreshold: 0.01,
    maxDepth: 2,
  });
  holonic._call = makeMockModelCall();

  // Fallback surf so every leaf gets at least 2 passages (surplus needs ≥2)
  const origResearch = holonic.researchSubtask.bind(holonic);
  holonic.researchSubtask = async (st) => {
    const r = await origResearch(st);
    if (r.surf.length === 0) {
      r.surf = [
        { text: `Detailed source material about ${st.label}: covering specific facts and terminology relevant to this aspect.`, source: "mock", score: 1, spanId: null, byteStart: null, byteEnd: null },
        { text: `Additional context about ${st.label}: supporting details and broader background information.`, source: "mock", score: 0.8, spanId: null, byteStart: null, byteEnd: null },
      ];
    }
    return r;
  };

  // Build a 2-level tree manually
  const root = new HolonNode({
    id: "root", label: "Sea Turtles", description: config.task, type: "root", level: 0,
  });
  root.isLeaf = false;

  const intro = new HolonNode({
    id: "intro", label: "Introduction", description: "Introduce sea turtles", type: "introduction", level: 1, parent: root,
  });
  const biology = new HolonNode({
    id: "biology", label: "Biology", description: "Sea turtle biology and anatomy", type: "section", level: 1, parent: root,
  });
  biology.isLeaf = false;

  const anatomy = new HolonNode({
    id: "anatomy", label: "Anatomy", description: "Physical anatomy of sea turtles", type: "section", level: 2, parent: biology,
  });
  const lifecycle = new HolonNode({
    id: "lifecycle", label: "Lifecycle", description: "Sea turtle reproduction and lifecycle", type: "section", level: 2, parent: biology,
  });
  const conservation = new HolonNode({
    id: "conservation", label: "Conservation", description: "Threats and conservation", type: "section", level: 1, parent: root,
  });

  root.children = [intro, biology, conservation];
  biology.children = [anatomy, lifecycle];
  holonic.treeRoot = root;

  console.log("Tree structure:");
  console.log("  root");
  console.log("    ├── intro (leaf)");
  console.log("    ├── biology (branch)");
  console.log("    │   ├── anatomy (leaf)");
  console.log("    │   └── lifecycle (leaf)");
  console.log("    └── conservation (leaf)");
  console.log("");

  // Execute tree
  console.log("─── Phase 2: Tree Execution ───");
  let draft = "";
  for (const child of root.children) {
    console.log(`\nExecuting: ${child.label}`);
    const result = await holonic._executeNode(child, { draft });
    draft += `\n\n${child.headingMarker} ${child.label}\n\n${result.content}`;
    const trail = result.iterations
      ? result.iterations.map(it => `g=${it.groundingScore.toFixed(2)} s=${(it.surplusScore ?? 0).toFixed(2)}`).join(" → ")
      : "synthesis";
    console.log(`  ${trail}`);
  }

  // Assemble tree
  console.log("\n─── Phase 3: Tree Assembly ───");
  const output = await holonic.assembleTree();
  console.log(`Output: ${output.length} chars`);

  // Results
  console.log("\n─── Results ───");
  const leaves = root.leaves;
  console.log(`Leaves: ${leaves.length}`);
  console.log(`Depth: ${Math.max(...leaves.map(l => l.level), 0) + 1} levels`);
  for (const leaf of leaves) {
    if (leaf.result) {
      console.log(`  ${leaf.label} (${leaf.path}): g=${leaf.groundingScore.toFixed(3)} s=${(leaf.surplusScore ?? 0).toFixed(3)} ${leaf.result.iterations.length} iters`);
    }
  }
  for (const child of root.children) {
    if (!child.isLeaf && child.result) {
      console.log(`  ${child.label} (branch): g=${child.groundingScore.toFixed(3)} s=${(child.surplusScore ?? 0).toFixed(3)}`);
    }
  }

  // Verify surplus is being computed for leaf nodes with multiple surf passages
  const leafWithMultipleSurf = leaves.find(l => l.result && l.result.surf && l.result.surf.length > 1);
  if (leafWithMultipleSurf) {
    console.log(`\n  Surplus test: ${leafWithMultipleSurf.label} had ${leafWithMultipleSurf.result.surf.length} surf passages, surplus=${leafWithMultipleSurf.surplusScore.toFixed(3)}`);
  }

  console.log("\n=== Tree Test Complete ===");

  // Clean up test output
  try { fs.unlinkSync(config.output); } catch {}
}

// ── Live tree test (uses runTree with learning phase, real model) ──

async function runTreeLive(config) {
  const engine = new MockEngine();
  if (config.sourceFile && fs.existsSync(config.sourceFile)) {
    console.error(`Ingesting source: ${config.sourceFile}`);
    engine.ingestFile(config.sourceFile);
  }

  const holonic = new HolonicTask({
    task: config.task,
    model: config.model,
    engine,
    outputPath: config.output,
    surplusThreshold: 0.01,
    maxDepth: 2,
  });

  const result = await holonic.runTree({ onProgress: (phase, msg) => console.error(`[${phase}] ${msg}`) });

  // Print summary
  console.log("");
  console.log("─── Results ───");
  console.log(`Task:     ${config.task}`);
  const leaves = holonic.treeRoot.leaves.length;
  const depth = Math.max(...holonic.treeRoot.leaves.map(l => l.level), 0) + 1;
  console.log(`Leaves:   ${leaves} across ${depth} levels`);
  console.log(`Output:   ${result.output.length} chars`);
  console.log(`Plan time:    ${(holonic.metrics.planTime / 1000).toFixed(1)}s`);
  console.log(`Learn time:   ${(holonic.metrics.learnTime / 1000).toFixed(1)}s`);
  console.log(`Execute time: ${(holonic.metrics.executeTime / 1000).toFixed(1)}s`);

  // Score summary
  console.log("");
  console.log("─── Leaf Scores ───");
  for (const leaf of holonic.treeRoot.leaves) {
    if (leaf.result) {
      const trail = leaf.result.iterations.map(it => `g=${it.groundingScore.toFixed(2)} s=${(it.surplusScore ?? 0).toFixed(2)}`).join(" → ");
      console.log(`  ${leaf.path}: ${trail} (${leaf.result.iterations.length} iters)`);
    }
  }

  // Output preview
  console.log("");
  console.log("─── Output Preview (first 30 lines) ───");
  const previewLines = result.output.split("\n").slice(0, 30);
  for (const line of previewLines) console.log(line);

  console.log("\n=== Tree Live Test Complete ===");
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
  if (config.tree) console.log("Mode:   TREE TEST (2-level holonic tree)");
  console.log("");

  if (config.tree && config.mockOnly) {
    return runTreeTest(config);
  }

  if (config.tree && !config.mockOnly) {
    return runTreeLive(config);
  }

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
    holonic._call = makeMockModelCall();

    console.log("");
    console.log("─── Phase 2: Execution (mock model, real correction loop) ───");
    holonic.subTaskResults = [];
    let draft = "";
    for (const st of planData.subTasks) {
      const surf = [{
        text: `Detailed source material about ${st.label.toLowerCase()}, covering specific facts, terminology, and examples relevant to this section of the document.`,
        source: "mock",
        score: 1,
        spanId: null,
        byteStart: null,
        byteEnd: null,
      }];
      const result = await holonic.executeSubtask(st, { surf, priors: [], previousSections: draft });
      holonic.subTaskResults.push(result);
      draft += `\n\n## ${result.label}\n\n${result.content}`;
      const trail = result.iterations.map(it => it.groundingScore.toFixed(2)).join(" → ");
      console.log(`  ${st.label}: grounding ${trail} (${result.iterations.length} iteration${result.iterations.length > 1 ? "s" : ""})`);
    }
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

      const trail = result.iterations.map(it => it.groundingScore.toFixed(2)).join(" → ");
      console.log(`  Done: ${result.content.length} chars, ${result.citations.length} mechanical citations, grounding ${trail} (${result.iterations.length} iteration${result.iterations.length > 1 ? "s" : ""})`);
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
    const trail = (r.iterations || []).map(it => it.groundingScore.toFixed(2)).join(" → ");
    console.log(`  ${r.label}: ${r.surf.length} surf passages, ${r.priors.length} priors, ${r.citations.length} mechanical citations, grounding ${trail || "n/a"}`);
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
