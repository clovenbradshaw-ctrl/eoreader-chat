#!/usr/bin/env node
/**
 * Code repo test: Load eoreader5 source into engine, test code search
 */

const fs = require('fs');
const path = require('path');

function createEngine() {
  let observations = [];
  let observationIndex = {};

  function hashString(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = ((h << 5) - h) + str.charCodeAt(i);
      h |= 0;
    }
    return h.toString(16).padStart(8, '0');
  }

  return {
    admitObservation(obs) {
      const id = `obs-${hashString(JSON.stringify({ path: obs.file_path, type: obs.type }))}`;
      const entry = { id, timestamp: new Date().toISOString(), obs, content_hash: hashString(JSON.stringify(obs)) };
      observations.push(entry);
      observationIndex[id] = entry;
      return { id, entry };
    },

    search(query, opts = {}) {
      const topK = opts.top_k || 10;
      const queryLower = (query || '').toLowerCase();
      const queryWords = queryLower.split(/\s+/).filter(w => w.length > 1);

      const scored = observations.map(entry => {
        const text = (entry.obs.text || '').toLowerCase();
        let score = 0;
        for (const qw of queryWords) {
          if (text.includes(qw)) score += 2.0;
          const words = text.split(/\s+/);
          for (const w of words) {
            if (w === qw) score += 1.0;
            else if (w.includes(qw) || qw.includes(w)) score += 0.5;
          }
        }
        // Boost for function/class definitions
        const defPatterns = [/function\s+\w+/, /class\s+\w+/, /const\s+\w+\s*=/, /module\.exports/];
        for (const pat of defPatterns) {
          if (pat.test(entry.obs.text)) score += 0.5;
        }
        return { id: entry.id, text: entry.obs.text || '', score, timestamp: entry.timestamp, file_path: entry.obs.file_path || '' };
      });

      return scored.filter(s => s.score > 0).sort((a, b) => b.score - a.score).slice(0, topK);
    },

    getState() { return { observations: observations.length }; },
    getObservations() { return observations; }
  };
}

function loadRepo(repoPath, engine) {
  const files = fs.readdirSync(repoPath, { recursive: true, withFileTypes: true });
  let loaded = 0;

  for (const file of files) {
    if (!file.isFile() || !file.name.endsWith('.js')) continue;
    const fullPath = path.join(file.parentPath || file.path, file.name);
    try {
      const content = fs.readFileSync(fullPath, 'utf8');
      if (content.length < 10) continue;

      // Split into chunks of ~1500 chars (function/module level)
      const chunks = [];
      const lines = content.split('\n');
      let currentChunk = [];
      let currentSize = 0;

      for (const line of lines) {
        currentChunk.push(line);
        currentSize += line.length;
        if (currentSize > 1500 || line.match(/^(module\.exports|exports\.|\/\/---|\/\/ ==)/)) {
          chunks.push(currentChunk.join('\n'));
          currentChunk = [];
          currentSize = 0;
        }
      }
      if (currentChunk.length > 0) chunks.push(currentChunk.join('\n'));

      for (const chunk of chunks) {
        if (chunk.trim().length < 20) continue;
        engine.admitObservation({
          text: chunk,
          type: "source_code",
          file_path: fullPath.replace(repoPath + '/', ''),
          language: "javascript"
        });
        loaded++;
      }
    } catch (e) {
      // skip unreadable files
    }
  }
  return loaded;
}

async function main() {
  console.log("=== Code Repo Test: eoreader5 ===\n");

  const engine = createEngine();
  const repoPath = "/Users/mlacy/Documents/Default Project/eoreader5";

  const startTime = Date.now();
  const loaded = loadRepo(repoPath, engine);
  const loadTime = Date.now() - startTime;

  console.log(`Loaded ${loaded} code chunks in ${loadTime}ms`);
  console.log(`Engine state: ${engine.getState()}\n`);

  // ── Test: Programming queries ──
  const queries = [
    // Core engine operations
    "how to create a new engine state",
    "function that searches observations",
    "validate observation envelope schema",
    "how does the replay state work",
    "what operators are available",
    // Specific patterns
    "hash function canonical json",
    "cube geometry diagonal cells",
    "observation envelope blocks hash",
    "prior snapshot operator epoch",
    "admission observation admit command",
    // Architecture questions
    "how to project referents",
    "discovery advance candidates",
    "reading snapshot projection",
    "coherence drift entropy",
    "ledger head basis id"
  ];

  console.log("--- Code Search Results ---\n");

  for (const query of queries) {
    const results = engine.search(query, { top_k: 3 });
    console.log(`Query: "${query}"`);
    for (const r of results) {
      // Extract first meaningful line
      const firstLine = r.text.split('\n').find(l => l.trim().length > 10) || r.text.slice(0, 80);
      console.log(`  [${r.file_path}] score=${r.score.toFixed(1)}: "${firstLine.trim().slice(0, 80)}"`);
    }
    console.log();
  }

  // ── Test: Would this help you code? ──
  console.log("--- Coding Context Quality ---\n");

  const codingTasks = [
    {
      task: "Add a new validation rule to the observation envelope",
      search: "validate observation envelope fields anchors",
    },
    {
      task: "Implement a function that computes content hashes for blocks",
      search: "block content hash blocks_hash source_content_hash",
    },
    {
      task: "Create a new operator that works in the cube ledger",
      search: "operator cube ledger mode domain grain",
    },
    {
      task: "Understand how search scores observations against queries",
      search: "search observations scoring query units",
    }
  ];

  for (const { task, search } of codingTasks) {
    console.log(`Task: "${task}"`);
    const results = engine.search(search, { top_k: 3 });
    if (results.length > 0) {
      console.log(`  Best match: [${results[0].file_path}]`);
      console.log(`  Score: ${results[0].score.toFixed(1)}`);
      console.log(`  Content preview:`);
      const preview = results[0].text.split('\n').slice(0, 8).join('\n');
      console.log(`    ${preview.split('\n').join('\n    ')}`);
      console.log(`  → ${results.length > 0 ? 'YES - has relevant context' : 'NO - missing context'}\n`);
    } else {
      console.log(`  → NO RESULTS\n`);
    }
  }

  console.log("=== Code Test Complete ===");
}

main().catch(console.error);
