// Stochastic multi-candidate generation.
//
// The model is a diversity engine, not an intelligence engine.
// Generate N candidates with temperature > 0, return all of them.
// Selection happens later, mechanically, via compression score.
//
// This is the "block of stone" — we generate a big block,
// then carve away what doesn't match.

import { callModel } from "./model-bridge.js";

// Generate N candidates from the same prompt.
// Each call uses temperature > 0 for diversity.
// Returns array of { text, index, model, promptHash }.
export async function generate(model, messages, n = 5, options = {}) {
  const maxTokens = options.maxTokens || 4096;
  const temperature = options.temperature || 0.9;

  const candidates = [];
  const errors = [];

  for (let i = 0; i < n; i++) {
    try {
      const text = await callModel(model, messages, maxTokens, { temperature });
      candidates.push({
        text,
        index: i,
        model,
        size: text.length,
      });
    } catch (err) {
      errors.push({ index: i, error: err.message });
    }
  }

  return { candidates, errors };
}

// Generate candidates for a specific section of an HTML page.
// Takes the full page and a section selector (CSS selector or tag name),
// generates N variations of just that section.
export async function generateSection(model, fullPage, sectionSelector, n = 5, options = {}) {
  const messages = [
    {
      role: "system",
      content: "You are a precise HTML generator. Output ONLY the raw HTML for the requested section. No markdown fences. No explanation. No doctype. Just the element.",
    },
    {
      role: "user",
      content: `Generate ${n} different variations of the ${sectionSelector} section for a Reddit-style page about dolphins. Each variation should have different content but the same structure. Output each variation separated by ===VARIATION===.\n\nHere is the full page for context:\n${fullPage.slice(0, 3000)}`,
    },
  ];

  const result = await callModel(model, messages, options.maxTokens || 4096, {
    temperature: options.temperature || 0.9,
  });

  // Split on variation separator
  const parts = result.split(/===VARIATION===/).map(s => s.trim()).filter(Boolean);
  return parts.map((text, i) => ({ text, index: i, model, size: text.length }));
}

// Generate a full page from a spec, with stochastic variation.
// Returns N complete page candidates.
export async function generateCandidates(model, spec, n = 5, options = {}) {
  const messages = [
    {
      role: "system",
      content: "You are a precise HTML generator. Output ONLY raw HTML. No markdown fences. No explanation. The page must be complete, valid HTML with embedded CSS. Include real content, not placeholder comments.",
    },
    {
      role: "user",
      content: spec,
    },
  ];

  return generate(model, messages, n, options);
}
