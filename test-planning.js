#!/usr/bin/env node
/**
 * Multi-prompt planning test: uses Ollama + eoreader5 engine
 * to plan a coding task across multiple generation steps
 */

const fs = require('fs');
const path = require('path');

// ── Engine (same as before) ──

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
      const id = `obs-${hashString(JSON.stringify({ path: obs.file_path, ts: Date.now() }))}`;
      const entry = { id, timestamp: new Date().toISOString(), obs, content_hash: hashString(JSON.stringify(obs)) };
      observations.push(entry);
      observationIndex[id] = entry;
      return { id, entry };
    },

    search(query, opts = {}) {
      const topK = opts.top_k || 5;
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
        return { id: entry.id, text: entry.obs.text || '', score, file_path: entry.obs.file_path || '' };
      });

      return scored.filter(s => s.score > 0).sort((a, b) => b.score - a.score).slice(0, topK);
    },

    getState() { return { observations: observations.length }; },
    getObservations() { return observations; }
  };
}

// ── Load repo ──

function loadRepo(repoPath, engine) {
  const files = fs.readdirSync(repoPath, { recursive: true, withFileTypes: true });
  let loaded = 0;

  for (const file of files) {
    if (!file.isFile() || !file.name.endsWith('.js')) continue;
    if (file.name.includes('node_modules')) continue;
    const fullPath = path.join(file.parentPath || file.path, file.name);
    try {
      const content = fs.readFileSync(fullPath, 'utf8');
      if (content.length < 20) continue;

      const lines = content.split('\n');
      let currentChunk = [];
      let currentSize = 0;

      for (const line of lines) {
        currentChunk.push(line);
        currentSize += line.length;
        if (currentSize > 1200 || line.match(/^(module\.exports|exports\.|\/\/---|\/\/ ==)/)) {
          const chunk = currentChunk.join('\n');
          if (chunk.trim().length > 30) {
            engine.admitObservation({
              text: chunk,
              type: "source_code",
              file_path: fullPath.replace(repoPath + '/', ''),
              language: "javascript"
            });
            loaded++;
          }
          currentChunk = [];
          currentSize = 0;
        }
      }
      if (currentChunk.length > 0) {
        const chunk = currentChunk.join('\n');
        if (chunk.trim().length > 30) {
          engine.admitObservation({
            text: chunk,
            type: "source_code",
            file_path: fullPath.replace(repoPath + '/', ''),
            language: "javascript"
          });
          loaded++;
        }
      }
    } catch {}
  }
  return loaded;
}

// ── Ollama call ──

async function callOllama(model, messages) {
  const resp = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream: false })
  });
  const data = await resp.json();
  return data.message?.content || "";
}

// ── Context assembly ──

function assembleContext(engine, userMessage, history, limit = 3500) {
  let context = [];
  let tokens = 0;

  const systemPrompt = "You are an expert software engineer. You help plan and implement code changes. Be specific and reference actual file paths and function names from the codebase context provided.";
  tokens += Math.ceil(systemPrompt.length / 3.5);
  context.push({ role: "system", content: systemPrompt });

  // Search engine for relevant code
  const results = engine.search(userMessage, { top_k: 4 });
  if (results.length > 0) {
    const codeContext = "\n\n[Relevant code from codebase]\n" +
      results.map(r => `--- ${r.file_path} ---\n${r.text.slice(0, 600)}`).join("\n\n");
    const ctxTokens = Math.ceil(codeContext.length / 3.5);
    if (tokens + ctxTokens < limit) {
      tokens += ctxTokens;
      context.push({ role: "system", content: codeContext });
    }
  }

  // History
  for (const msg of history.slice(-8)) {
    const msgTokens = Math.ceil(msg.content.length / 3.5);
    if (tokens + msgTokens > limit) break;
    tokens += msgTokens;
    context.push(msg);
  }

  context.push({ role: "user", content: userMessage });
  return { context, tokens };
}

// ── Planning test ──

async function main() {
  const MODEL = "gemma2:2b";
  console.log(`=== Multi-Prompt Planning Test: ${MODEL} ===\n`);

  const engine = createEngine();
  const repoPath = "/Users/mlacy/Documents/Default Project/eoreader5";
  const loaded = loadRepo(repoPath, engine);
  console.log(`Loaded ${loaded} code chunks into engine\n`);

  // The coding task: add a new validation function
  const codingTask = "Add a new function called `validateObservationBlock` to the observation-index.js file. It should validate that a block has required fields: schema, block_id, value_type, shape, and values. Return { valid: true } or { valid: false, errors: [...] }.";

  // Multi-prompt planning steps
  const planningSteps = [
    {
      prompt: `I need to plan a code change. Here's the task:\n\n${codingTask}\n\nFirst, analyze the existing codebase. What does observation-index.js currently look like? What validation patterns already exist? Search the codebase and tell me the key structures I need to work with.`,
      label: "Step 1: Analyze existing code"
    },
    {
      prompt: `Based on the existing code, outline the implementation plan. What should the function signature look like? What validation logic do I need? Where exactly should I add it in the file? Give me a step-by-step plan.`,
      label: "Step 2: Plan implementation"
    },
    {
      prompt: `Now write the actual implementation of validateObservationBlock. Include the full function code with proper error handling. Make sure it follows the existing code patterns in the repo.`,
      label: "Step 3: Write the code"
    },
    {
      prompt: `Finally, write the test for this function. What edge cases should I test? Write the complete test code.`,
      label: "Step 4: Write tests"
    }
  ];

  let history = [];

  for (let i = 0; i < planningSteps.length; i++) {
    const step = planningSteps[i];
    console.log(`\n--- ${step.label} ---`);
    console.log(`Prompt: "${step.prompt.slice(0, 80)}..."\n`);

    // Fold the prompt into engine
    engine.admitObservation({ text: step.prompt, type: "planning_prompt", step: i });

    // Assemble context
    const { context, tokens } = assembleContext(engine, step.prompt, history);
    console.log(`Context: ${tokens} tokens, ${context.length} messages`);

    // Search results
    const results = engine.search(step.prompt, { top_k: 3 });
    console.log(`Engine search: ${results.length} results`);
    for (const r of results.slice(0, 2)) {
      console.log(`  [${r.file_path}] score=${r.score.toFixed(1)}`);
    }

    // Call model
    console.log(`\nGenerating with ${MODEL}...`);
    const startTime = Date.now();
    let response;
    try {
      response = await callOllama(MODEL, context);
    } catch (e) {
      console.log(`Error: ${e.message}`);
      continue;
    }
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`Generated in ${elapsed}s`);
    console.log(`Response length: ${response.length} chars`);
    console.log(`Response preview:\n${response.slice(0, 500)}${response.length > 500 ? '...' : ''}\n`);

    // Add to history
    history.push({ role: "user", content: step.prompt });
    history.push({ role: "assistant", content: response });

    // Fold response into engine
    engine.admitObservation({ text: response, type: "planning_response", step: i });
  }

  // Final state
  console.log("\n--- Final State ---");
  console.log(`Engine observations: ${engine.getState().observations}`);
  console.log(`History length: ${history.length} messages`);
  console.log("\n=== Planning Test Complete ===");
}

main().catch(console.error);
