#!/usr/bin/env node
/**
 * Test script: eoreader5 in-browser engine with Frankenstein content
 * Tests fold, search, projection, and conversation sustain
 */

const fs = require('fs');
const path = require('path');

// ── In-browser engine (copied from index.html) ──

function createEngine() {
  let observations = [];
  let events = [];
  let observationIndex = {};
  let coherence = { coherence: 1.0, drift: 0.0, entropy: 0.0 };

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
      const id = `obs-${Date.now()}-${hashString(JSON.stringify(obs))}`;
      const entry = {
        id,
        timestamp: new Date().toISOString(),
        obs: obs,
        content_hash: hashString(JSON.stringify(obs))
      };
      observations.push(entry);
      observationIndex[id] = entry;

      const advanced = this.discoverAdvance();
      return { id, entry, advanced };
    },

    discoverAdvance() {
      if (observations.length === 0) return null;
      const last = observations[observations.length - 1];
      const text = last.obs.text || '';
      const words = text.split(/\s+/).filter(w => w.length > 2);
      const bigrams = [];
      for (let i = 0; i < words.length - 1; i++) {
        bigrams.push(`${words[i]} ${words[i+1]}`);
      }
      return {
        observation_id: last.id,
        word_count: words.length,
        bigrams,
        coherence
      };
    },

    search(query, opts = {}) {
      const topK = opts.top_k || 10;
      const threshold = opts.threshold || 0.0;
      const queryLower = (query || '').toLowerCase();
      const queryWords = queryLower.split(/\s+/).filter(w => w.length > 1);

      const scored = observations.map(entry => {
        const text = (entry.obs.text || '').toLowerCase();
        const obsWords = new Set(text.split(/\s+/).filter(w => w.length > 1));
        let score = 0;
        for (const qw of queryWords) {
          for (const ow of obsWords) {
            if (ow === qw) score += 1.0;
            else if (ow.includes(qw) || qw.includes(ow)) score += 0.5;
          }
        }
        for (const qw of queryWords) {
          if (text.includes(qw)) score += 0.3;
        }
        return { id: entry.id, text: entry.obs.text || '', score, timestamp: entry.timestamp };
      });

      return scored
        .filter(s => s.score > threshold)
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
    },

    project(obsId, opts = {}) {
      const entry = observationIndex[obsId];
      if (!entry) return null;
      const text = entry.obs.text || '';
      const window = opts.window_size || 200;
      const start = Math.max(0, (opts.offset || 0));
      return {
        id: entry.id,
        text: text.slice(start, start + window),
        full_length: text.length,
        coherence
      };
    },

    readingSnapshot(obsId) {
      const entry = observationIndex[obsId];
      if (!entry) return null;
      return {
        reading_id: `reading-${obsId}`,
        observation_id: obsId,
        text: entry.obs.text || '',
        timestamp: entry.timestamp,
        coherence
      };
    },

    getState() {
      return {
        observations: observations.length,
        events: events.length,
        coherence
      };
    },

    getObservations() {
      return observations;
    },

    getCoherence() {
      return coherence;
    }
  };
}

// ── Test harness ──

async function main() {
  console.log("=== EOReader5 Engine Test: Frankenstein ===\n");

  const engine = createEngine();
  const frankensteinPath = path.join(__dirname, 'memory', 'frankenstein.txt');
  const text = fs.readFileSync(frankensteinPath, 'utf8');

  // Split into chapters/paragraphs for realistic ingestion
  const chapters = text.split(/(?:Letter \d+|Chapter \d+)/gi).filter(ch => ch.trim().length > 100);

  console.log(`Frankenstein loaded: ${text.length} chars, ${chapters.length} chapters\n`);

  // ── Test 1: Fold chapters into engine ──
  console.log("--- Test 1: Folding chapters into engine ---");
  const startTime = Date.now();
  for (let i = 0; i < chapters.length; i++) {
    const chapter = chapters[i].trim().slice(0, 2000); // Cap at 2000 chars per chapter
    const result = engine.admitObservation({
      text: chapter,
      type: "frankenstein_chapter",
      chapter_index: i
    });
    console.log(`  Chapter ${i}: admitted (${result.entry.content_hash})`);
  }
  const foldTime = Date.now() - startTime;
  console.log(`\nFolded ${chapters.length} chapters in ${foldTime}ms`);
  console.log(`Engine state:`, engine.getState());

  // ── Test 2: Search for specific themes ──
  console.log("\n--- Test 2: Search for themes ---");

  const queries = [
    "monster creation creature",
    "Victor Frankenstein laboratory",
    "death loss grief suffering",
    "Arctic ice snow cold",
    "loneliness isolation abandoned"
  ];

  for (const query of queries) {
    const results = engine.search(query, { top_k: 3 });
    console.log(`\n  Query: "${query}"`);
    for (const r of results) {
      console.log(`    [${r.id}] score=${r.score.toFixed(2)}: "${r.text.slice(0, 80)}..."`);
    }
  }

  // ── Test 3: Project specific observations ──
  console.log("\n--- Test 3: Project specific observations ---");
  const allObs = engine.getObservations();
  if (allObs.length > 0) {
    const obs = allObs[0];
    const projected = engine.project(obs.id, { window_size: 300 });
    console.log(`  Projected ${obs.id}:`);
    console.log(`    Text: "${projected.text}"`);
    console.log(`    Full length: ${projected.full_length}`);
  }

  // ── Test 4: Reading snapshot ──
  console.log("\n--- Test 4: Reading snapshot ---");
  if (allObs.length > 0) {
    const snapshot = engine.readingSnapshot(allObs[0].id);
    console.log(`  Reading ${snapshot.reading_id}:`);
    console.log(`    Observation: ${snapshot.observation_id}`);
    console.log(`    Timestamp: ${snapshot.timestamp}`);
    console.log(`    Coherence:`, snapshot.coherence);
  }

  // ── Test 5: Conversation sustain simulation ──
  console.log("\n--- Test 5: Conversation sustain ---");

  const conversation = [
    { role: "user", content: "Tell me about Victor Frankenstein's creation" },
    { role: "assistant", content: "Victor Frankenstein created a creature from dead body parts in his laboratory. He was driven by ambition to conquer death." },
    { role: "user", content: "What happened after he created the creature?" },
    { role: "assistant", content: "The creature came to life but Victor was horrified by its appearance. He abandoned it, leading to tragic consequences." },
    { role: "user", content: "How did the story end?" },
  ];

  // Fold conversation into engine
  for (const msg of conversation) {
    engine.admitObservation({ text: msg.content, type: `conversation_${msg.role}` });
  }

  // Search engine for context
  console.log("\n  Simulating conversation context assembly:");
  for (const msg of conversation) {
    const results = engine.search(msg.content, { top_k: 2 });
    console.log(`\n  ${msg.role}: "${msg.content.slice(0, 50)}..."`);
    console.log(`    Found ${results.length} relevant observations`);
    for (const r of results) {
      console.log(`      - [${r.id}] score=${r.score.toFixed(2)}`);
    }
  }

  // ── Test 6: Context window limit ──
  console.log("\n--- Test 6: Context window simulation ---");
  const contextLimit = 3500;
  let totalTokens = 0;
  let contextMessages = [];

  for (const msg of [...conversation].reverse()) {
    const tokens = Math.ceil(msg.content.length / 3.5);
    if (totalTokens + tokens > contextLimit) break;
    totalTokens += tokens;
    contextMessages.unshift(msg);
  }

  console.log(`  Context limit: ${contextLimit} tokens`);
  console.log(`  Messages in context: ${contextMessages.length}`);
  console.log(`  Total tokens: ${totalTokens}`);
  console.log(`  Dropped: ${conversation.length - contextMessages.length} messages`);

  // ── Final stats ──
  console.log("\n--- Final Engine State ---");
  console.log(JSON.stringify(engine.getState(), null, 2));
  console.log("\n=== Test Complete ===");
}

main().catch(console.error);
