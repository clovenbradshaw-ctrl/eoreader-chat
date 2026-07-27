#!/usr/bin/env node
/**
 * Extended conversation test: simulates 10+ turns of dialogue
 * about Frankenstein, testing context sustain and memory retrieval
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
      const id = `obs-${Date.now()}-${hashString(JSON.stringify(obs))}`;
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
          if (text.includes(qw)) score += 1.0;
          const words = text.split(/\s+/);
          for (const w of words) {
            if (w === qw) score += 0.5;
            else if (w.includes(qw) || qw.includes(w)) score += 0.25;
          }
        }
        return { id: entry.id, text: entry.obs.text || '', score, timestamp: entry.timestamp };
      });

      return scored.filter(s => s.score > 0).sort((a, b) => b.score - a.score).slice(0, topK);
    },

    getState() { return { observations: observations.length }; },
    getObservations() { return observations; }
  };
}

function assembleContext(userMessage, engine, recentHistory, memoryFiles, limit = 3500) {
  let context = [];
  let tokens = 0;

  // System prompt with memory file references
  const memRefs = memoryFiles.length > 0
    ? `\n\nAvailable memory files: ${memoryFiles.join(", ")}\nUse FETCH:<filename> to retrieve a memory file's content.`
    : "";
  const systemPrompt = `You are a helpful assistant with access to a memory store.${memRefs}`;
  tokens += Math.ceil(systemPrompt.length / 3.5);
  context.push({ role: "system", content: systemPrompt });

  // Search engine for relevant context
  const results = engine.search(userMessage, { top_k: 3 });
  if (results.length > 0) {
    const engineContext = "\n\n[Engine context]\n" + results.map(r => r.text.slice(0, 300)).join("\n---\n");
    const ctxTokens = Math.ceil(engineContext.length / 3.5);
    if (tokens + ctxTokens < limit) {
      tokens += ctxTokens;
      context.push({ role: "system", content: engineContext });
    }
  }

  // Recent conversation
  for (const msg of recentHistory.slice(-6)) {
    const msgTokens = Math.ceil(msg.content.length / 3.5);
    if (tokens + msgTokens > limit) break;
    tokens += msgTokens;
    context.push(msg);
  }

  // User message
  context.push({ role: "user", content: userMessage });

  return { context, totalTokens: tokens };
}

async function main() {
  console.log("=== Extended Conversation Test: Frankenstein ===\n");

  const engine = createEngine();
  const frankPath = path.join(__dirname, 'memory', 'frankenstein.txt');
  const text = fs.readFileSync(frankPath, 'utf8');

  // Fold Frankenstein into engine
  const chapters = text.split(/(?:Letter \d+|Chapter \d+)/gi).filter(ch => ch.trim().length > 100);
  for (let i = 0; i < chapters.length; i++) {
    engine.admitObservation({ text: chapters[i].trim().slice(0, 2000), type: "frankenstein_chapter", chapter_index: i });
  }
  console.log(`Folded ${chapters.length} chapters into engine\n`);

  // Simulated conversation turns
  const turns = [
    { role: "user", content: "What inspired Victor Frankenstein to create the creature?" },
    { role: "assistant", content: "Victor was inspired by ancient alchemists like Cornelius Agrippa, Paracelsus, and Albertus Magnus. His study of natural philosophy and chemistry at the University of Ingolstadt, combined with the death of his mother, drove him to conquer death itself." },
    { role: "user", content: "Tell me more about the creature's first experiences after being brought to life." },
    { role: "assistant", content: "When the creature opened its eyes, Victor was horrified by its appearance and fled. The creature was abandoned, confused, and alone. It wandered through the wilderness, experiencing hunger, cold, and fear without understanding what was happening to it." },
    { role: "user", content: "How did the creature learn to speak and read?" },
    { role: "assistant", content: "The creature observed the De Lacey family through a crack in their cottage wall. It learned language by listening to them speak, and learned to read by finding Milton's Paradise Lost, Plutarch's Lives, and Goethe's Sorrows of Young Werther in the woods." },
    { role: "user", content: "What happened when the creature tried to communicate with the blind old man?" },
    { role: "assistant", content: "The creature spoke to old De Lacey, who was blind and couldn't see its monstrous form. They had a pleasant conversation. But when the rest of the family returned and saw the creature, they attacked it and drove it away. This was the creature's first attempt at human connection, and it ended in rejection." },
    { role: "user", content: "Why did the creature kill Victor's brother William?" },
    { role: "assistant", content: "The creature killed William out of revenge against Victor for abandoning it. After being rejected by the De Lacey family, the creature burned their cottage and decided to make Victor suffer as it had suffered. When it encountered young William, it recognized him as Victor's relative and strangled him." },
    { role: "user", content: "Tell me about the creature's demand for a mate." },
    { role: "assistant", content: "The creature demanded that Victor create a female companion for it, promising to disappear into the wilderness forever if he did. Victor initially agreed but then destroyed the female creature mid-creation, fearing they might breed and threaten humanity. This betrayal drove the creature to its final murderous rampage." },
    { role: "user", content: "How does the story end?" },
    { role: "assistant", content: "Victor dies aboard Robert Walton's ship, telling his story. The creature appears, mourning Victor's death. It expresses guilt and sorrow for its actions, then disappears into the darkness of the Arctic, planning to end its own life on a funeral pyre." },
    { role: "user", content: "What themes does Shelley explore in the novel?" },
    { role: "assistant", content: "Shelley explores several key themes: the danger of unchecked ambition, the consequences of playing God, the nature of humanity and monstrosity, isolation and loneliness, the importance of companionship, and the responsibility creators have toward their creations." },
    { role: "user", content: "How does the novel relate to scientific ethics?" },
  ];

  // Process each turn
  let recentHistory = [];
  let memoryFiles = ["frankenstein.txt"];
  let totalTokens = 0;

  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    engine.admitObservation({ text: turn.content, type: `conversation_${turn.role}` });

    if (turn.role === "user") {
      const { context, totalTokens: tokens } = assembleContext(turn.content, engine, recentHistory, memoryFiles);
      totalTokens += tokens;
      console.log(`Turn ${i + 1} [${turn.role}]: "${turn.content.slice(0, 60)}..."`);
      console.log(`  Context tokens: ${tokens}, Running total: ${totalTokens}`);
      console.log(`  Context messages: ${context.length}`);

      const results = engine.search(turn.content, { top_k: 2 });
      console.log(`  Engine search results: ${results.length}`);
      for (const r of results) {
        console.log(`    - score=${r.score.toFixed(2)}: "${r.text.slice(0, 50)}..."`);
      }
    } else {
      console.log(`Turn ${i + 1} [${turn.role}]: "${turn.content.slice(0, 60)}..."`);
    }

    recentHistory.push(turn);
  }

  // Final summary
  console.log("\n--- Summary ---");
  console.log(`Total turns: ${turns.length}`);
  console.log(`Engine observations: ${engine.getState().observations}`);
  console.log(`Total tokens accumulated: ${totalTokens}`);
  console.log(`Average tokens per user turn: ${Math.round(totalTokens / turns.filter(t => t.role === 'user').length)}`);

  // Test that engine can still find relevant context after many turns
  console.log("\n--- Final context retrieval test ---");
  const finalQuery = "What happened to the creature at the end?";
  const finalResults = engine.search(finalQuery, { top_k: 3 });
  console.log(`Query: "${finalQuery}"`);
  for (const r of finalResults) {
    console.log(`  [${r.id}] score=${r.score.toFixed(2)}: "${r.text.slice(0, 80)}..."`);
  }

  console.log("\n=== Extended Test Complete ===");
}

main().catch(console.error);
