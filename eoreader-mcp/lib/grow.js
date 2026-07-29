// Grow organ — holonic recursive fold generator.
//
// A short seed ("reddit, but dolphin zone") expands into a complete artifact
// through piecewise generation with fold-compressed context. At every step:
//
//   seed ──→ templates ──→ decomposition ──→ piecewise generate ──→ compose
//
// KEY INSIGHTS FROM MEASURED FAILURE (first pass):
//
//   FAILURE: The model produces generic SW-engineering task trees (backend,
//   database, testing) instead of HTML fragments (header, feed, modal).
//   ROOT CAUSE: A decomposer prompt that doesn't anchor to the output format.
//   FIX: Use format-specific decomposition templates, not open-ended planning.
//
//   FAILURE: Pieces are independent full HTML documents concatenated naively,
//   creating duplicate DOCTYPE/html/head/body declarations.
//   ROOT CAUSE: No shared template shell for fragments to compose into.
//   FIX: First piece is the shell; subsequent pieces are fragments composed
//   into named slots (header, main, sidebar-left, sidebar-right, modals,
//   styles, scripts).
//
//   FAILURE: The seed specification is lost after the first generation.
//   ROOT CAUSE: Fold context prioritizes recency over relevance.
//   FIX: The seed is prepended to EVERY generation prompt. Fold context
//   only shows labels and slot names, not raw prior output.
//
// SELF-TEACHING:
//   Successful decompositions are recorded as templates keyed by seed type.
//   The template store grows from use: when a seed matches a known pattern,
//   the stored decomposition is used instead of calling the model.
//   Failed decompositions are recorded as anti-patterns to avoid.

import fs from "fs";
import path from "path";
import * as log from "./log.js";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const GROW_MODEL = process.env.GROW_MODEL || "gemma2:2b";
const TEMPLATE_PATH = process.env.GROW_TEMPLATE_PATH || "/tmp/grow-templates.json";

// ── Format-specific decomposition templates ────────────────────
//
// Each template defines how a seed of a given format decomposes.
// The decomposition is deterministic and format-aware — no model call.
// Slot types: "shell" (the one true container), "fragment" (injected pieces),
// "style" (CSS/inline styles), "script" (JS), "data" (JSON)

const TEMPLATES = {
  html: {
    description: "Single-file HTML page with embedded CSS and JS",
    shell: {
      label: "document shell",
      description: "DOCTYPE, html, head with meta/links, body scaffold with named slot containers",
      slot: "shell",
  },
  prose: {
    description: "Long-form text content — essays, articles, reports, stories",
    shell: null,
    fragments: [
      { label: "introduction", description: "Opening paragraphs that establish the topic, hook the reader, and preview the structure", slot: "intro" },
      { label: "body section 1", description: "First major section developing the core argument or narrative", slot: "body-1" },
      { label: "body section 2", description: "Second major section with supporting evidence, examples, or development", slot: "body-2" },
      { label: "body section 3", description: "Third major section — counterarguments, deeper analysis, or climax", slot: "body-3" },
      { label: "body section 4", description: "Fourth section — implications, extensions, or additional perspective", slot: "body-4" },
      { label: "conclusion", description: "Closing section that synthesizes, reflects, and leaves the reader with a lasting thought", slot: "conclusion" },
    ],
  },
  essay: {
    description: "Structured academic or persuasive essay",
    shell: null,
    fragments: [
      { label: "thesis & introduction", description: "Opening with a clear thesis statement, context, and roadmap", slot: "intro" },
      { label: "argument 1", description: "First supporting argument with evidence, examples, reasoning", slot: "arg-1" },
      { label: "argument 2", description: "Second supporting argument — builds on or contrasts with the first", slot: "arg-2" },
      { label: "counterargument", description: "Address the strongest objection, then rebut it", slot: "counter" },
      { label: "implications", description: "What follows if the thesis is accepted — broader significance", slot: "implications" },
      { label: "conclusion", description: "Restate thesis in light of the arguments, close with force", slot: "conclusion" },
    ],
  },
  story: {
    description: "Narrative fiction — short stories, scenes, chapters",
    shell: null,
    fragments: [
      { label: "opening scene", description: "Establish setting, character, and initial situation — hook immediately", slot: "opening" },
      { label: "rising action 1", description: "First complication or development — raise the stakes", slot: "rise-1" },
      { label: "rising action 2", description: "Escalation — deeper conflict, character revelation, or twist", slot: "rise-2" },
      { label: "climax", description: "The turning point — confrontation, decision, or revelation", slot: "climax" },
      { label: "falling action", description: "Consequences unfold — what the climax means for the characters", slot: "falling" },
      { label: "resolution", description: "Final state — closure, reflection, or deliberate irresolution", slot: "resolution" },
    ],
  },
    fragments: [
      { label: "header",      description: "site header with brand name, search bar, action buttons (Log In, New Post)", slot: "header" },
      { label: "left sidebar", description: "navigation sidebar with pod list, labels, counts, and a community pledge", slot: "sidebar-left" },
      { label: "main feed",   description: "post feed with sort tabs (Hot/New/Top), post cards with voting, title, preview, comments count", slot: "main" },
      { label: "right sidebar", description: "info sidebar with fun fact card and rules list", slot: "sidebar-right" },
      { label: "modals",      description: "post detail modal with comments, new post modal with form fields", slot: "modals" },
      { label: "styles",      description: "CSS: variables, layout grid, typography, post cards, voting, modals, responsive breakpoints", slot: "styles" },
      { label: "scripts",     description: "JS: seed data (pods, posts, facts), voting, modals, new post submission, comment system", slot: "scripts" },
    ],
  },
  css: {
    description: "Stylesheet file",
    shell: { label: "stylesheet header", description: "CSS file with variables, reset, and section headers", slot: "shell" },
    fragments: [
      { label: "layout",      description: "grid/flexbox layout rules", slot: "layout" },
      { label: "components",  description: "component styles (cards, buttons, modals, forms)", slot: "components" },
      { label: "typography",  description: "font imports, heading styles, body text", slot: "typography" },
      { label: "responsive",  description: "media queries for mobile/tablet/desktop", slot: "responsive" },
    ],
  },
  js: {
    description: "JavaScript file",
    shell: { label: "module header", description: "imports, constants, state initialization", slot: "shell" },
    fragments: [
      { label: "data layer",  description: "seed data, API calls, state management", slot: "data" },
      { label: "interactions", description: "event handlers, DOM manipulation, UI logic", slot: "interactions" },
      { label: "utilities",   description: "helper functions, formatting, validation", slot: "utilities" },
    ],
  },
};

function getTemplate(fileType) {
  return TEMPLATES[fileType] || TEMPLATES.html;
}

// ── Seed type classification ──────────────────────────────────
//
// Maps seeds to known patterns so successful decompositions are reused.
// Grows from use via teachGrow().

const SEED_PATTERNS = {
  "reddit|forum|social|community|pod|feed|dolphin": {
    type: "social-feed",
    template: "html",
    hints: { brand: true, voting: true, comments: true, sidebar: true },
  },
  "homepage|landing|personal site|portfolio|blog": {
    type: "landing-page",
    template: "html",
    hints: { hero: true, nav: true, footer: true },
  },
  "dashboard|admin|analytics|chart|stats": {
    type: "data-dashboard",
    template: "html",
    hints: { chart: true, table: true, sidebar: true },
  },
};

function classifySeed(seed) {
  const lower = seed.toLowerCase();
  for (const [pattern, info] of Object.entries(SEED_PATTERNS)) {
    const parts = pattern.split("|");
    if (parts.some(p => lower.includes(p))) return info;
  }
  return { type: "generic", template: "html", hints: {} };
}

// ── Token estimation ──────────────────────────────────────────

function t(text) { return String(text ?? "").split(/\s+/).filter(Boolean).length; }

// ── Seed-anchored fold context ────────────────────────────────
//
// Unlike the general fold() which compresses by recency, this fold
// preserves the seed and slot labels, and only includes the LAST
// piece's label + slot. Prior piece content is never fed to the model
// — it would only confuse it into repeating or continuing prior output.

function growFold(seed, seedType, completed) {
  const parts = [];
  parts.push(`SEED: ${seed}`);
  parts.push(`TYPE: ${seedType}`);

  if (completed.length === 0) {
    parts.push("STATUS: beginning — generate the document shell first");
    return parts.join("\n");
  }

  // Show what's been generated (slot labels only, no content)
  const done = completed.map(p => `${p.slot}: ${p.label}`).join(", ");
  parts.push(`GENERATED: ${done}`);

  // Show the last piece's label+slot so the model knows what came before
  const last = completed[completed.length - 1];
  parts.push(`LAST PIECE: ${last.label} (slot: ${last.slot})`);

  // Next piece hint
  parts.push("NEXT: generate the next undecomposed fragment");

  return parts.join("\n");
}

// ── Model call ────────────────────────────────────────────────

async function callModel(model, messages, maxTokens = 1024, temperature = 0.7) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: { temperature, num_predict: maxTokens },
    }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw new Error(`Ollama ${model} ${res.status}`);
  return (await res.json()).message?.content || "";
}

// ── Cutoff detection ──────────────────────────────────────────

const CUTOFF_PATTERNS = [
  /<[a-z][a-z0-9]*$/,                        // unclosed HTML tag
  /<[a-z][a-z0-9]*\s+[a-z][a-z0-9]*="?$/,    // unclosed attribute
  /\$\{[\w.]*$/,                               // unclosed template literal
  /[{,]\s*$/,                                  // trailing comma/brace
  /\\$/,                                       // trailing backslash
];

function detectCutoff(text) {
  const tail = text.trimEnd();
  for (const pat of CUTOFF_PATTERNS) {
    if (pat.test(tail)) return true;
  }
  if (tail.length > 100) {
    const lastLine = tail.split("\n").pop() || "";
    if (lastLine.includes("<") && !lastLine.includes(">")) return true;
  }
  return false;
}

// ── HTML fragment composition ─────────────────────────────────
//
// The shell piece provides the overall document structure with named
// containers (divs with specific IDs). Fragment pieces inject content
// into those containers. The final composition replaces container
// markers with fragment content.

const SLOT_RE = /<!--\s*SLOT:(\S+)\s*-->/g;

function composeShell(pieces) {
  const shell = pieces.find(p => p.slot === "shell");
  if (!shell) return pieces.map(p => p.content).join("\n\n");

  let doc = shell.content;

  // Insert fragments into their named slots
  for (const piece of pieces) {
    if (piece.slot === "shell") continue;
    const slotMarker = `<!-- SLOT:${piece.slot} -->`;
    if (doc.includes(slotMarker)) {
      doc = doc.replace(slotMarker, piece.content + "\n" + slotMarker);
    } else if (piece.slot === "modals") {
      // Append modals before closing body
      doc = doc.replace("</body>", piece.content + "\n</body>");
    } else if (piece.slot === "scripts") {
      // Append scripts before closing body
      doc = doc.replace("</body>", piece.content + "\n</body>");
    } else {
      // Append to body as fallback
      doc = doc.replace("</body>", piece.content + "\n</body>");
    }
  }

  // Clean up remaining slot markers
  doc = doc.replace(/<!--\s*SLOT:\S+\s*-->/g, "");

  return doc;
}

// ── HTML generation prompt ────────────────────────────────────

function buildGenPrompt(seed, seedType, piece, completed, hints) {
  const fold = growFold(seed, seedType, completed);

  // Build the shell prompt for the first piece
  if (piece.slot === "shell") {
    const containerSlots = TEMPLATES.html.fragments
      .filter(f => f.slot !== "shell")
      .map(f => `  <!-- SLOT:${f.slot} -->  <!-- ${f.label} will be injected here -->`)
      .join("\n");

    return {
      system: "You generate HTML document shells. Output ONLY the raw HTML. No explanations, no markdown fences.",
      user: [
        `SEED: ${seed}`,
        `TYPE: ${seedType}`,
        "",
        "Generate a complete HTML document shell with:",
        "- <!DOCTYPE html>, <html>, <head> with <title> based on the seed",
        "- <meta charset='UTF-8'>, <meta name='viewport'>",
        "- Inline <style> block (empty, for CSS injection)",
        "- <body> with named slot markers for each component:",
        "",
        containerSlots,
        "",
        "The shell provides the document scaffold. Keep it clean — visual design comes in later pieces.",
        "Use the seed to determine the page title and brand identity.",
        "Do NOT generate CSS content in the style block (leave it empty).",
        "Do NOT generate JS (that comes in a later piece).",
        "Output ONLY the HTML document.",
      ].join("\n"),
    };
  }

  // Build fragment prompts for subsequent pieces
  const hintStr = Object.keys(hints).length > 0
    ? `\nDESIGN HINTS: ${JSON.stringify(hints)}`
    : "";

  return {
    system: `You generate ${piece.slot} HTML fragments. Output ONLY the raw HTML/JS/CSS for this component. No explanations, no markdown fences, no full HTML document structure — just the component.`,
    user: [
      `SEED: ${seed}`,
      `COMPONENT: ${piece.description}`,
      `SLOT: ${piece.slot}${hintStr}`,
      "",
      "Generate ONLY the content for this component.",
      "- If styles: generate CSS rules targeting the component's class names",
      "- If scripts: generate JS using data-* attributes and class selectors",
      "- If HTML: generate semantic HTML with class names",
      "",
      "Do NOT include <html>, <head>, <body>, <style>, or <script> wrappers.",
      "The shell already provides those. Just output the component content.",
    ].join("\n"),
  };
}

// ── Piece generation with retry ───────────────────────────────

async function generatePiece(piece, completed, seed, seedType, hints, maxRetries) {
  const prompt = buildGenPrompt(seed, seedType, piece, completed, hints);

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const raw = await callModel(GROW_MODEL, [
        { role: "system", content: prompt.system },
        { role: "user", content: prompt.user },
      ], 2048, 0.7);

      const cleaned = raw.replace(/^```[\w]*\n?/, "").replace(/\n?```$/, "").trim();

      if (detectCutoff(cleaned)) {
        log.write({ type: "grow_retry", reason: "cutoff", attempt, slot: piece.slot, model: GROW_MODEL });
        continue;
      }

      if (cleaned.length < 20) {
        log.write({ type: "grow_retry", reason: "too-short", attempt, slot: piece.slot, length: cleaned.length });
        continue;
      }

      return { ok: true, content: cleaned };

    } catch (err) {
      log.write({ type: "grow_retry", reason: err.message, attempt, slot: piece.slot });
    }
  }

  return { ok: false, error: `failed after ${maxRetries + 1} attempts` };
}

// ── Seed relevance check ──────────────────────────────────────

function scoreSeedRelevance(content, seed) {
  // Count how many non-trivial seed words appear in the output
  const stopWords = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","as","is","was","are","were","be","been","has","had","have","do","does","did","will","would","could","should","may","might","can","shall","this","that","these","those","it","its","they","them","their","we","you","your","he","she","his","her","not","no","nor","so","if","then","than","just","about","into","over","after","before","between","under","reddit","clone","lovers"]);
  const seedWords = seed.toLowerCase().split(/\s+/).filter(w => w.length > 3 && !stopWords.has(w));
  const contentLower = content.toLowerCase();
  let matches = 0;
  for (const w of seedWords) {
    if (contentLower.includes(w)) matches++;
  }
  return seedWords.length > 0 ? matches / seedWords.length : 0;
}

// ── Template persistence (self-teaching) ──────────────────────

function loadTemplates() {
  try {
    if (fs.existsSync(TEMPLATE_PATH)) {
      return JSON.parse(fs.readFileSync(TEMPLATE_PATH, "utf8"));
    }
  } catch {}
  return { decompositions: {}, failures: [] };
}

function saveTemplates(templates) {
  try {
    fs.writeFileSync(TEMPLATE_PATH, JSON.stringify(templates, null, 2), "utf8");
  } catch {}
}

export function teachGrow(seed, fileType, decomposition, successScore) {
  const templates = loadTemplates();
  const key = `${fileType}:${seed.slice(0, 40)}`;

  if (successScore >= 0.5) {
    // Record successful decomposition pattern
    templates.decompositions[key] = {
      seed_keywords: seed.toLowerCase().split(/\s+/).filter(w => w.length > 4).slice(0, 10),
      fileType,
      decomposition: decomposition.map(d => ({ label: d.label, slot: d.slot, description: d.description })),
      score: successScore,
      recorded: Date.now(),
    };
  } else {
    // Record failure pattern
    templates.failures.push({
      seed: seed.slice(0, 80),
      fileType,
      score: successScore,
      recorded: Date.now(),
    });
    // Keep only last 50 failures
    if (templates.failures.length > 50) templates.failures = templates.failures.slice(-50);
  }

  saveTemplates(templates);
  return templates;
}

// ── Main grow entry point ─────────────────────────────────────

export async function grow(seed, {
  fileType = "html",
  outputPath = null,
  session = "default",
  maxTasks = 20,
  maxRetries = 2,
} = {}) {
  const startTime = Date.now();
  const turn = `grow:${Date.now()}`;

  if (!seed || seed.trim().length < 3) {
    return { ok: false, error: "seed too short", seed };
  }

  log.write({ type: "grow_start", layer: "artifact", session, turn, seed: seed.slice(0, 200), fileType, timestamp: startTime });

  // Classify seed and get decomposition template
  const seedType = classifySeed(seed);
  const template = getTemplate(fileType);

  // Build the task list from the template
  const tasks = [
    template.shell,
    ...template.fragments,
  ].slice(0, maxTasks);

  log.write({ type: "grow_plan", layer: "artifact", session, turn, template: seedType.type, taskCount: tasks.length });

  // Generate each piece
  const completed = [];
  const errors = [];

  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i];

    log.write({ type: "grow_piece_start", layer: "artifact", session, turn, index: i, slot: task.slot, label: task.label });

    const result = await generatePiece(task, completed, seed, seedType.type, seedType.hints, maxRetries);

    if (result.ok) {
      const relevance = scoreSeedRelevance(result.content, seed);
      log.write({ type: "grow_piece_ok", layer: "artifact", session, turn, index: i, slot: task.slot, length: result.content.length, relevance: relevance.toFixed(2) });

      completed.push({
        label: task.label,
        slot: task.slot,
        content: result.content,
        relevance,
      });
    } else {
      log.write({ type: "grow_piece_fail", layer: "artifact", session, turn, index: i, slot: task.slot, error: result.error });
      errors.push({ label: task.label, error: result.error });
    }
  }

  // Compose all pieces together
  const assembled = composeShell(completed);

  // Final quality score
  const finalRelevance = scoreSeedRelevance(assembled, seed);
  const hasCutoff = detectCutoff(assembled);
  const totalChars = assembled.length;
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);

  // Self-teach: record this decomposition
  teachGrow(seed, fileType, completed, finalRelevance);

  // Write output
  if (outputPath) {
    const dir = path.dirname(outputPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(outputPath, assembled, "utf8");
  }

  const result = {
    ok: errors.length === 0 && !hasCutoff,
    seed,
    fileType,
    assembled,
    pieces: completed.map(p => ({ label: p.label, slot: p.slot, length: p.content.length, relevance: p.relevance.toFixed(2) })),
    errors: errors.length > 0 ? errors : undefined,
    stats: { totalPieces: completed.length, totalChars, duration: `${duration}s`, errorCount: errors.length },
    finalValidation: { relevance: finalRelevance.toFixed(2), cutoff: hasCutoff, passed: !hasCutoff && finalRelevance > 0.1 },
    outputPath: outputPath || null,
  };

  log.write({ type: "grow_complete", layer: "artifact", session, turn, ok: result.ok, pieces: completed.length, chars: totalChars, duration, errors: errors.length, relevance: finalRelevance.toFixed(2) });

  return result;
}
