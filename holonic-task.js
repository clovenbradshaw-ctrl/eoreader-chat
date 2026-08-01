// holonic-task.js — Generic holonic task decomposition and execution.
//
// Takes an arbitrary task description and:
//   1. PLANS: decomposes into sub-tasks using the model
//   2. For each sub-task:
//      a. RESEARCHES: searches engine → surf (passages) + activated priors
//      b. EXECUTES: model writes content (no citation knowledge)
//      c. CITES: mechanical n-gram matching links content → surf
//   3. ASSEMBLES: combines all output with provenance (surf + priors + citations)
//
// Citations are MECHANICAL (post-hoc), never model-generated. The model never
// sees citation numbers or is asked to cite. Missing evidence → typed gap.
//
// Provenance per sub-task:
//   - surf: what the engine returned
//   - priors: what coref/corpus priors were activated on the surf
//   - citations: mechanically matched EOT against content

const OLLAMA_URL = "http://localhost:11434";

// Ceiling on passages per sub-task surf, including prior-driven expansion.
// Named because it bounds what a prior is allowed to contribute, and hitting
// it is reported as a gap rather than passing silently.
const SURF_CAP = 8;

// How many alternate surface forms per activated referent get their own
// expansion search. Bounded because each one is a full retrieval round trip.
const EXPANSION_SURFACES_PER_REFERENT = 2;

// ── HolonNode: a node in the task-decomposition tree ──
//
// Each node is Janus-faced (Koestler, 1967): simultaneously a complete whole
// with respect to its children and a subordinate part with respect to its
// parent. Atomicity is fixed at plan time (Whitehead's epochal principle):
// a leaf is indivisible at its level. Completion is a live recursive test
// at every level: a fold is done when its precision-weighted residual bottoms
// out (grounding) AND the output shows cross-passage synthesis (surplus).

class HolonNode {
  constructor({ id, label, description, type, level = 0, parent = null } = {}) {
    this.id = id;
    this.label = label;
    this.description = description;
    this.type = type;
    this.level = level;
    this.parent = parent;
    this.children = [];
    this.result = null;
    this.surf = [];
    this.priors = [];
    this.groundingScore = 0;
    this.surplusScore = 0;
    this.isLeaf = true;
    this.phase = null;
    this.groundingMode = "required"; // "required" | "optional" | "none"
  }

  get path() {
    const parts = [];
    let node = this;
    while (node.parent) {
      parts.unshift(node.id);
      node = node.parent;
    }
    return parts.join('.');
  }

  get headingMarker() {
    if (this.level === 0) return '';
    return '#'.repeat(Math.min(this.level + 1, 6));
  }

  get leaves() {
    if (this.isLeaf) return [this];
    const result = [];
    for (const c of this.children) result.push(...c.leaves);
    return result;
  }

  nodesAtLevel(targetLevel) {
    if (this.level === targetLevel) return [this];
    if (this.isLeaf) return [];
    const result = [];
    for (const c of this.children) result.push(...c.nodesAtLevel(targetLevel));
    return result;
  }

  toJSON() {
    return {
      id: this.id, label: this.label, type: this.type,
      level: this.level, path: this.path, isLeaf: this.isLeaf,
      phase: this.phase, children: this.children.map(c => c.toJSON()),
      groundingScore: this.groundingScore, surplusScore: this.surplusScore,
      hasResult: this.result !== null,
      surfCount: this.surf.length, priorCount: this.priors.length,
    };
  }

  // Create a flat array from the tree (breadth-first), for
  // backward compatibility with code that iterates subTaskResults.
  flatten() {
    const all = [];
    const queue = [this];
    while (queue.length > 0) {
      const n = queue.shift();
      if (n.isLeaf) all.push(n);
      for (const c of n.children) queue.push(c);
    }
    return all;
  }
}

function estimateTokens(text) {
  return Math.ceil((text || "").length / 3.5);
}

// Two search hits are the same passage when they anchor to the same bytes of
// the same source — NOT when they share a span_id. The engine mints a fresh
// span_id per retrieval, so the same chunk returned by two different queries
// carries two different ids; deduping on span_id let prior-driven expansion
// re-add passages already in the surf. That both overstated what the prior
// contributed and spent the surf cap on duplicates. The byte anchor is the
// passage's identity; the id is just a handle to this retrieval of it.
function samePassage(a, b) {
  if (a.source && a.source === b.source && a.byteStart != null && b.byteStart != null) {
    return a.byteStart === b.byteStart && a.byteEnd === b.byteEnd;
  }
  if (a.spanId && b.spanId && a.spanId === b.spanId) return true;
  return a.text === b.text;
}

function charTrigrams(text) {
  const set = new Set();
  for (let i = 0; i <= text.length - 3; i++) {
    set.add(text.slice(i, i + 3));
  }
  return set;
}

// ── Information-theoretic grounding: char-trigram frequency models ──
//
// The mechanical CITATION record (_mechanicalCite, below) stays exactly as
// it was — Jaccard over trigram sets, strictly verbatim, never touched by
// any of this. What changes is the CONTROL SIGNAL that drives the
// correction loop: not "do these two texts share substrings" but "does
// using this draft as a predictive prior make the real source text less
// surprising than a same-shaped but scrambled prior would." That is the
// actual comprehension test — a draft that merely reshuffles vocabulary
// without capturing structure gets no credit; a draft that captures the
// source's specific, low-probability content gets real credit, in bits.

function charTrigramCounts(text) {
  const counts = new Map();
  const s = String(text || "").toLowerCase();
  for (let i = 0; i <= s.length - 3; i++) {
    const tri = s.slice(i, i + 3);
    counts.set(tri, (counts.get(tri) ?? 0) + 1);
  }
  return counts;
}

function charTrigramList(text) {
  const s = String(text || "").toLowerCase();
  const list = [];
  for (let i = 0; i <= s.length - 3; i++) list.push(s.slice(i, i + 3));
  return list;
}

// Average bits-per-trigram to encode `targetText` under `modelCounts`
// (add-1 smoothed over the model's own vocabulary — the model IS the prior).
function surprisalUnder(targetText, modelCounts) {
  const list = charTrigramList(targetText);
  if (list.length === 0) return 0;
  const total = [...modelCounts.values()].reduce((a, b) => a + b, 0);
  const vocab = modelCounts.size || 1;
  let bits = 0;
  for (const tri of list) {
    const count = modelCounts.get(tri) ?? 0;
    const p = (count + 1) / (total + vocab);
    bits += -Math.log2(p);
  }
  return bits / list.length;
}

// A CONDITIONAL null: same characters as modelText (unigram frequency
// preserved exactly), order scrambled — destroys trigram-level structure,
// which is the specific axis being tested, without changing a fixed/global
// baseline. (Permuting COUNTS across the same key set, tried first, doesn't
// work: for short/sparse text almost every trigram is already unique, so
// the "shuffled" vocabulary is identical to the real one and the null is
// statistically indistinguishable from real — the null must vary structure,
// per the project's "an unconditional null is a units change" rule.)
function shuffledText(text) {
  const chars = text.split("");
  for (let i = chars.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [chars[i], chars[j]] = [chars[j], chars[i]];
  }
  return chars.join("");
}

// Averaged over a few trials to damp shuffle-to-shuffle noise.
function nullSurprisalFor(targetText, modelText, trials = 3) {
  let total = 0;
  for (let t = 0; t < trials; t++) {
    total += surprisalUnder(targetText, charTrigramCounts(shuffledText(modelText)));
  }
  return total / trials;
}

// Fractional bits saved (clipped to [0,1]) predicting `targetText` using
// `modelText` as prior, vs. that same text's characters scrambled. 0 = the
// model's specific content carries no predictive value for the target;
// 1 = perfect.
function infoGain(targetText, modelText) {
  const modelCounts = charTrigramCounts(modelText);
  if (modelCounts.size === 0) return 0;
  const real = surprisalUnder(targetText, modelCounts);
  const nullSurprisal = nullSurprisalFor(targetText, modelText);
  if (nullSurprisal <= 0) return 0;
  return Math.max(0, Math.min(1, (nullSurprisal - real) / nullSurprisal));
}

export { HolonNode };
export class HolonicTask {
  constructor({
    task,
    model = "gemma2:2b",
    engine = null,
    outputPath = null,
    perSubTaskBudget = 2000,
    maxSubTasks = 8,
    ollamaUrl = OLLAMA_URL,
    apiTimeout = 600000,
    // Calibrated empirically against real Frankenstein passages (~800 chars):
    // identical text scores ~0.09 combined, a genuine paraphrase ~0.03,
    // unrelated text ~0.02 (see holonic-task-provenance.md). Bits-saved-as-
    // fraction-of-null is naturally compressed well below 1 at this scale —
    // add-1 smoothing over a ~150-trigram vocabulary caps how much any
    // single passage's model can distinguish real from null. This is a
    // first calibration; expect to retune against real model output.
    groundingThreshold = 0.03,
    maxCorrectionIterations = 3,
    driftPenalty = 0.5,
    // Bounded re-planning: at most this many replan passes per run, no
    // matter how many sub-tasks stay unresolved or redundant — this is a
    // correction pass, not a planner that loops until satisfied.
    maxReplans = 1,
    redundancyOverlapThreshold = 0.6,
    // Surplus threshold: minimum cross-passage synthesis score for
    // completion. Surplus is the second channel — independent of error-
    // closure — measuring whether the output captures structure that
    // EMERGES from combining passages, not just restating each one.
    surplusThreshold = 0.1,
    // Maximum holonic depth. Root = level 0, first decomposition = 1,
    // etc. 0 means no decomposition (flat only). Acts as the Whitehead-
    // epochal atomicity safeguard: no fold can decompose finer than this,
    // preventing infinite regress independent of whatever live residual
    // test decides completion within a grain.
    maxDepth = 3,
    // Phase strategy: "auto" (planner decides per node), "unified" (all
    // nodes use the same error-closure objective), or "split" (early
    // nodes are epistemic-value-dominant, late nodes minimize reader
    // surprise). "auto" is the default and defers to the planner.
    phaseStrategy = "auto",
  } = {}) {
    if (!task || typeof task !== "string") throw new TypeError("HolonicTask requires a { task } string");

    this.task = task;
    this.model = model;
    this.engine = engine;
    this.outputPath = outputPath;
    this.perSubTaskBudget = perSubTaskBudget;
    this.maxSubTasks = maxSubTasks;
    this.ollamaUrl = ollamaUrl;
    this.apiTimeout = apiTimeout;
    this.groundingThreshold = groundingThreshold;
    this.maxCorrectionIterations = maxCorrectionIterations;
    this.driftPenalty = driftPenalty;
    this.maxReplans = maxReplans;
    this.redundancyOverlapThreshold = redundancyOverlapThreshold;
    this.surplusThreshold = surplusThreshold;
    this.maxDepth = maxDepth;
    this.phaseStrategy = phaseStrategy;

    this.planResult = null;
    this.subTaskResults = [];
    this.provenance = [];
    this.gaps = [];
    this.replanCount = 0;
    this.replanHistory = [];
    this.treeRoot = null;
    this.learningGuide = null;
    this.metrics = { planTime: 0, executeTime: 0, assembleTime: 0, learnTime: 0, totalTokens: 0 };
  }

  async _call(messages, maxTokens = 1024) {
    const totalPrompt = estimateTokens(messages.map(m => m.content).join(" "));
    const resp = await fetch(`${this.ollamaUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        messages,
        stream: false,
        options: { temperature: 0.7, num_predict: Math.max(256, maxTokens) },
      }),
      signal: AbortSignal.timeout(this.apiTimeout),
    });
    if (!resp.ok) throw new Error(`Ollama ${resp.status}: ${await resp.text()}`);
    const data = await resp.json();
    const content = data.message?.content || "";
    this.metrics.totalTokens += totalPrompt + estimateTokens(content);
    return content;
  }

  _parseJSON(text) {
    const braceMatch = text.match(/\[[\s\S]*?\]/);
    if (braceMatch) {
      try { return JSON.parse(braceMatch[0]); } catch {}
    }
    const objectMatch = text.match(/\{[\s\S]*\}/);
    if (objectMatch) {
      try { return JSON.parse(objectMatch[0]); } catch {}
    }
    return null;
  }

  // Depth-aware JSON extraction (respecting string literals) — unlike
  // _parseJSON's lazy regex, this correctly handles a payload with NESTED
  // arrays (e.g. replan()'s {"changes": [{"resultingSubTasks": [...]}]}),
  // where a non-greedy [.*?] would truncate at the first "]" it meets,
  // which may belong to an inner array rather than the outer one.
  _extractBalancedJSON(text) {
    const start = text.search(/[{[]/);
    if (start === -1) return null;
    const open = text[start];
    const close = open === "{" ? "}" : "]";
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (escape) escape = false;
        else if (ch === "\\") escape = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') { inString = true; continue; }
      if (ch === open) depth++;
      else if (ch === close) {
        depth--;
        if (depth === 0) {
          try { return JSON.parse(text.slice(start, i + 1)); } catch { return null; }
        }
      }
    }
    return null;
  }

  // ── Learning phase — a series of sub-tasks that learn how to do the task ──
  //
  // Before planning the content, the system learns what format/genre is
  // needed, researches its conventions, and produces a guide. This guide
  // is then injected into the planner and executor prompts so every level
  // of the tree understands the expected output format.
  //
  // The learning itself is a holonic tree: identify format → research
  // conventions → produce example → compile guide. Each sub-task is a
  // model call. The output is a textual guide stored in this.learningGuide.

  async _learnSeries() {
    const start = Date.now();
    const learnRoot = new HolonNode({
      id: 'learn', label: `Learn how to do: "${this.task.slice(0, 60)}"`,
      description: `Learn what format, genre, and conventions are needed to complete this task: ${this.task}`,
      type: 'root', level: 0,
    });

    // Step 1: Identify the format/genre
    const formatResp = await this._call([{
      role: "system", content: "You identify the format, genre, and output conventions required by a task description. Be specific.",
    }, {
      role: "user", content: `What format, genre, and output conventions does this task require?\n\nTASK: ${this.task}\n\nRespond with a concise paragraph identifying the format (e.g., screenplay, poem, essay, story, letter, report) and its key conventions. If the format has specific structural rules (e.g., INT./EXT. headings for screenplays, rhyme schemes for sonnets), list them.`,
    }], 600);
    learnRoot.children.push(new HolonNode({
      id: 'learn-format', label: 'Format Identification',
      description: formatResp.trim(), type: 'research', level: 1, parent: learnRoot,
    }));

    // Step 2: Research conventions
    const conventionResp = await this._call([{
      role: "system", content: "You are a writing coach who explains format conventions clearly and specifically.",
    }, {
      role: "user", content: `Based on this format identification, explain the specific conventions and structural rules in detail:\n\nFORMAT: ${formatResp.trim().slice(0, 500)}\n\nWhat are the rules for structure, formatting, tone, and style? Give concrete examples.`,
    }], 800);
    learnRoot.children.push(new HolonNode({
      id: 'learn-conventions', label: 'Conventions',
      description: conventionResp.trim(), type: 'research', level: 1, parent: learnRoot,
    }));

    // Step 3: Produce a structural outline
    const outlineResp = await this._call([{
      role: "system", content: "You produce structural outlines that show the expected format of a document.",
    }, {
      role: "user", content: `Given these conventions, produce a brief structural outline showing how the output should be organized:\n\nCONVENTIONS: ${conventionResp.trim().slice(0, 800)}\n\nTASK: ${this.task}\n\nShow the skeleton structure (e.g., for a screenplay: ACT ONE → Scene 1 → Scene 2 → ACT TWO...). Just the skeleton, no content.`,
    }], 600);
    learnRoot.children.push(new HolonNode({
      id: 'learn-outline', label: 'Structural Outline',
      description: outlineResp.trim(), type: 'example', level: 1, parent: learnRoot,
    }));

    // Step 4: Compile the learning guide
    const guideResp = await this._call([{
      role: "system", content: "You compile research into a concise, actionable guide for a writer. Be specific and practical.",
    }, {
      role: "user", content: `Compile this research into a concise learning guide for a writer who will produce the output. Include format, conventions, and structure. Be specific and actionable:\n\nFORMAT: ${formatResp.trim().slice(0, 400)}\n\nCONVENTIONS: ${conventionResp.trim().slice(0, 600)}\n\nSTRUCTURE: ${outlineResp.trim().slice(0, 400)}`,
    }], 800);

    this.learningGuide = guideResp.trim();
    this.metrics.learnTime = Date.now() - start;
    console.error(`[learn] Guide produced: ${this.learningGuide.length} chars across ${learnRoot.children.length} sub-tasks`);
    return this.learningGuide;
  }

  // ── Format detection ──
  //
  // The planner and executor use genre-specific prompts when the task
  // names a known output format. A screenplay gets acts and scenes;
  // a poem gets stanzas; an essay gets sections. Null = generic.

  _detectFormat(task) {
    const lower = task.toLowerCase();
    if (/\b(screenplay|script|screen play|teleplay)\b/.test(lower)) return 'screenplay';
    if (/\b(poem|poetry|sonnet|haiku|verse|ballad)\b/.test(lower)) return 'poem';
    if (/\b(story|short story|fiction|narrative|tale)\b/.test(lower)) return 'story';
    if (/\b(essay|article|paper|report|analysis|memo)\b/.test(lower)) return 'essay';
    if (/\b(letter|correspondence|email)\b/.test(lower)) return 'letter';
    return null;
  }

  // Format guidance for the planner: uses the learning guide if available,
  // falls back to hardcoded format detection otherwise. The learning guide
  // is richer — it was produced by the model itself through a series of
  // research sub-tasks — so it takes priority.
  _formatPlanInstruction(task) {
    if (this.learningGuide) {
      return `\n\nLEARNING GUIDE (produced by prior research — follow these conventions):\n${this.learningGuide.slice(0, 1500)}\n\nUse the structure and format described above. Decompose the task into appropriate sub-tasks for this format (e.g., acts/scenes for a screenplay, stanzas for a poem, sections for an essay).`;
    }
    const fmt = this._detectFormat(task);
    const instructions = {
      screenplay: '\nThis is a SCREENPLAY. Sub-tasks should be ACTS and SCENES (e.g. "Act One — Meeting", "Act Two — Crisis"). Use types "scene" or "act".',
      poem: '\nThis is a POEM. Sub-tasks should be STANZAS or sections. Use type "stanza" or "section".',
      story: '\nThis is a SHORT STORY. Sub-tasks should be CHAPTERS or narrative sections. Use types "chapter" or "section".',
      essay: '\nSub-tasks should be LOGICAL SECTIONS (introduction, body, conclusion). Use types "introduction", "section", "analysis", "conclusion".',
      letter: '\nThis is a LETTER. Sub-tasks should be PARTS (salutation, body, closing). Use types "salutation", "body", "closing".',
    };
    return instructions[fmt] || '';
  }

  // System prompt for the executor: uses the learning guide when available.
  _formatSystemPrompt(task) {
    if (this.learningGuide) {
      return `Follow these format conventions (produced by prior research):\n${this.learningGuide.slice(0, 2000)}`;
    }
    const fmt = this._detectFormat(task);
    const prompts = {
      screenplay: 'Write screenplay content using proper format: INT./EXT. — LOCATION — TIME headings, ALL CAPS character names before dialogue, parentheticals, action lines in present tense. Never use markdown headings.',
      poem: 'Write poetic content with attention to meter, imagery, and line breaks appropriate to the form.',
      story: 'Write narrative prose with vivid description, dialogue, and pacing.',
      letter: 'Write correspondence with salutation, body paragraphs, and closing.',
    };
    return prompts[fmt] || 'Write clear, coherent content that restates and reorganizes source passages.';
  }

  // Default grounding mode by node type: creative types get "none"
  // (scenes, dialogue, stanzas, acts — written freely), analytical
  // types get "required" (analysis, section, evidence — needs sources).
  _defaultGroundingMode(type) {
    const creative = new Set(["scene", "act", "dialogue", "stanza", "poem", "monologue", "description", "introspection", "dream", "flashback", "montage"]);
    if (creative.has(type)) return "none";
    if (type === "introduction" || type === "conclusion") return "optional";
    return "required";
  }

  _formatAssembleLabel(task) {
    if (this.learningGuide) {
      const guideLower = this.learningGuide.toLowerCase();
      if (/\bscene\b/.test(guideLower)) return 'scenes';
      if (/\bstanza\b/.test(guideLower)) return 'stanzas';
      if (/\bchapter\b/.test(guideLower)) return 'chapters';
      if (/\bact\b/.test(guideLower)) return 'acts';
    }
    const fmt = this._detectFormat(task);
    const labels = {
      screenplay: 'scenes', poem: 'stanzas', story: 'chapters',
      essay: 'sections', letter: 'parts',
    };
    return labels[fmt] || 'sections';
  }

  // ── Phase 1: Plan ──

  async plan() {
    const start = Date.now();

    const formatInst = this._formatPlanInstruction(this.task);
    const formatTypes = formatInst ? '' : ' Use types "introduction", "section", "analysis", or "conclusion".';
    const prompt = `You are a precise task planner. Decompose the following task into ${Math.min(this.maxSubTasks, 4)}-${this.maxSubTasks} focused sub-tasks. Each must be self-contained and achievable by a small language model writing 400-600 words.${formatTypes}

TASK: ${this.task}
${formatInst}
Respond with a JSON array of objects. Each object:
  - "id": short unique identifier (e.g. "intro", "biology")
  - "label": human-readable heading
  - "description": what this sub-task covers (2-3 sentences)
  - "type": one of the suggested types above

Return ONLY the JSON array. Example:
[{"id":"first","label":"First Section","description":"Start the piece.","type":"introduction"}]`;

    const system = "You are a precise task planner. Always respond with valid JSON.";
    const response = await this._call([
      { role: "system", content: system },
      { role: "user", content: prompt },
    ], 1500);

    const subTasks = this._parseJSON(response);
    if (!Array.isArray(subTasks) || subTasks.length < 2) {
      throw new Error(
        `Plan parsing failed. Model returned:\n${response.slice(0, 400)}\n\n` +
        `Expected a JSON array with 2-${this.maxSubTasks} sub-tasks.`
      );
    }

    this.planResult = { subTasks: subTasks.slice(0, this.maxSubTasks) };
    this.metrics.planTime = Date.now() - start;
    return this.planResult;
  }

  // ── Recursive tree planner (planTree) ──
  //
  // Decomposes tasks recursively to arbitrary depth N. Each level produces
  // sub-tasks; the planner decides per sub-task whether it needs further
  // decomposition. Atomicity is fixed at plan time (Whitehead's epochal
  // principle): a leaf is declared atomic and never subdivided at runtime.
  // The atomicity safeguard (maxDepth) prevents infinite regress independent
  // of whatever residual test decides completion within a grain.

  async planTree(rootDesc = null) {
    const start = Date.now();
    this.treeRoot = new HolonNode({
      id: 'root',
      label: rootDesc || this.task.slice(0, 80),
      description: this.task,
      type: 'root',
      level: 0,
    });
    await this._planNode(this.treeRoot);
    this.planResult = this.planResult || {};
    this.planResult.root = this.treeRoot;
    this.planResult.subTasks = this.treeRoot.leaves.map(n => ({
      id: n.id, label: n.label, description: n.description, type: n.type,
    }));
    this.metrics.planTime = Date.now() - start;
    return this.planResult;
  }

  async _planNode(node) {
    if (node.level >= this.maxDepth) return;

    const usePhasePrompt = this.phaseStrategy !== "unified";
    const formatInst = this._formatPlanInstruction(this.task);
    const prompt = `You are decomposing a task into sub-tasks. Each must be self-contained and achievable by a small language model writing 400-600 words.

CURRENT TASK: ${node.description}
Level: ${node.level} (deeper = finer grain)
${formatInst}
Respond with a JSON object:
{"subTasks": [
  {
    "id": "short-id",
    "label": "human-readable heading",
    "description": "what this covers (1-2 sentences)",
    "type": "the appropriate type for this format",
    "needsDecomposition": true|false,
    "groundingMode": "required"|"optional"|"none"
  }
]}

For each sub-task:
- needsDecomposition: true if it needs finer sub-tasks, false if atomic (narrow enough for 400-600 words)
- groundingMode: "required" (must cite sources), "none" (creative writing — scenes, dialogue, stanzas), or "optional" (intro/conclusion)${usePhasePrompt ? `\n\nAlso, for each sub-task, set "phase" to "exploratory" (seeking confusing/uncertain areas — appropriate early in the task or when the topic is poorly understood) or "expository" (conveying clear understanding to a reader — appropriate late in the task or when the topic is well-understood).` : ''}

Return ONLY the JSON object. Example:
{"subTasks":[{"id":"first","label":"First","description":"Start the piece.","type":"section","needsDecomposition":false,"groundingMode":"optional","phase":"expository"}]}`;

    const system = "You are a precise task planner. Always respond with valid JSON.";
    const response = await this._call([
      { role: "system", content: system },
      { role: "user", content: prompt },
    ], 1500);

    const parsed = this._extractBalancedJSON(response);
    const subTasks = parsed?.subTasks;
    if (!Array.isArray(subTasks) || subTasks.length < 1) return;

    node.isLeaf = false;
    for (const st of subTasks.slice(0, this.maxSubTasks)) {
      const child = new HolonNode({
        id: st.id || `node-${node.path}-${node.children.length}`,
        label: st.label || st.id,
        description: st.description || `${st.label} aspect of ${node.label}`,
        type: st.type || 'section',
        level: node.level + 1,
        parent: node,
      });
      child.phase = st.phase || (usePhasePrompt ? 'expository' : null);
      child.groundingMode = st.groundingMode || this._defaultGroundingMode(st.type || 'section');
      node.children.push(child);
      if (st.needsDecomposition && child.level < this.maxDepth) {
        await this._planNode(child);
      }
    }
  }

  // ── Phase 2a: Research (returns { surf, priors }) ──

  _mapSearchResults(results) {
    return results.map((r) => ({
      text: (r.text || r.preview || "").slice(0, 800),
      source: r.source || r.file_path || r.source_id || "?",
      score: r.score || 0,
      spanId: r.span_id || r.id || null,
      byteStart: r.byte_start ?? null,
      byteEnd: r.byte_end ?? null,
    })).filter((r) => r.text.length > 20);
  }

  async researchSubtask(subTask) {
    if (!this.engine || typeof this.engine.search !== "function") {
      return { surf: [], priors: [] };
    }

    const taskTopic = this.task.replace(/^(write|create|generate|produce|research)\s+(a\s+|an\s+)?/i, "").slice(0, 120);
    const query = `${taskTopic} — ${subTask.label}: ${subTask.description}`;
    let results;
    try {
      results = this.engine.search(query, { limit: 5 });
    } catch {
      return { surf: [], priors: [] };
    }

    if (!results || !Array.isArray(results)) return { surf: [], priors: [] };

    let surf = this._mapSearchResults(results);

    // Activate priors on the retrieved text — steering only, never narrated
    // (see executeSubtask). Two response shapes are supported: a legacy
    // array of prior-description strings (inert here by design — only
    // structured priors carry grounding credit), or the real bridge's
    // { activated: [{referentId, display, matchedSurfaces, expansionSurfaces}], gap }.
    let priors = [];
    // What the prior added to this surf, and which artifact it came from —
    // carried out to the UI so the user can audit the widening.
    const expansions = [];
    let priorId = null;
    if (this.engine.getPriors && typeof this.engine.getPriors === "function") {
      const surfText = surf.map((s) => s.text).join(" ");
      const activation = this.engine.getPriors(surfText, surf[0]?.source);

      if (Array.isArray(activation)) {
        priors = activation;
      } else if (activation && Array.isArray(activation.activated)) {
        priors = activation.activated;
        priorId = activation.priorId ?? null;
        if (activation.gap) {
          this.gaps.push({ subTask: subTask.id, reason: "prior_gap", text: activation.gap });
        }

        // Retrieval expansion: re-search using each referent's OTHER surface
        // forms, so the model gets exposed to more of the source under
        // varied phrasing — it absorbs the coref through evidence, never
        // through a rule stated to it.
        //
        // Each passage admitted this way is stamped with `viaPrior`: which
        // prior, which referent, and which surface form reached it. Without
        // that stamp a prior-widened passage is indistinguishable from base
        // retrieval, and the user cannot see what the prior actually did to
        // their surf — which is the whole point of showing priors at all.
        // `unsearched` counts SURFACE FORMS the cap prevented us from trying —
        // not passages. Comparing surfaces against passages under-reports,
        // because one surface search can admit several passages.
        const unsearched = [];
        for (const p of priors) {
          for (const altSurface of (p.expansionSurfaces || []).slice(0, EXPANSION_SURFACES_PER_REFERENT)) {
            if (surf.length >= SURF_CAP) { unsearched.push(altSurface); continue; }
            let more;
            try {
              more = this.engine.search(`${taskTopic} — ${altSurface}`, { limit: 3 });
            } catch {
              continue;
            }
            if (!Array.isArray(more)) continue;
            for (const mapped of this._mapSearchResults(more)) {
              if (surf.length >= SURF_CAP) break;
              if (surf.some((s) => samePassage(s, mapped))) continue;
              mapped.viaPrior = {
                priorId: p.priorId ?? activation.priorId ?? null,
                referentId: p.referentId,
                display: p.display,
                surface: altSurface,
              };
              surf.push(mapped);
              expansions.push(mapped.viaPrior);
            }
          }
        }

        // The cap is a real limit on what the prior was allowed to contribute.
        // Reporting it keeps "3 passages added" from reading as "the prior had
        // nothing more to offer" (AGENTS.md: no silent caps).
        if (unsearched.length > 0) {
          this.gaps.push({
            subTask: subTask.id,
            reason: "prior_expansion_capped",
            text: `surf hit the ${SURF_CAP}-passage cap; ${unsearched.length} prior surface form(s) not searched: ${unsearched.join(", ")}`,
          });
        }
      }
    }

    return { surf, priors, expansions, priorId: priorId ?? null };
  }

  // ── Phase 2b: Execute — a grounding-driven correction loop ──
  //
  // Priors NEVER appear as narrated text in a model-facing prompt (nameless-
  // referent principle: identity lives in the referent, never a string told
  // to the model). Instead they steer _scoreGrounding()'s cross-surface
  // coverage credit (see §3 in specs/holonic-task-provenance.md) — invisibly.
  //
  // Loop: generate a draft, mechanically score how much of the surf material
  // it actually reflects (and how much of it is unsupported drift), and if
  // that score is too low, show the model a diagnosis (which source material
  // it missed, which of its own sentences aren't grounded) and have it
  // revise. Repeat until the score clears the threshold or the iteration
  // budget runs out. The full trace is returned as `iterations[]` — the
  // auditable record of whether the model actually converged toward the
  // source, not just a single unexamined guess.

  _executeSystemPrompt() {
    const fmt = this._detectFormat(this.task);
    const base = fmt
      ? this._formatSystemPrompt(this.task)
      : "You rewrite source passages into a coherent section of a larger document. Keep the original key terms and specific language from the passages. Restate and reorganize — do not add new information. Do not use citation markers or reference numbers.";
    return base + " Do not use markdown headings or title lines — just write the content.";
  }

  _buildInitialExecutePrompt(subTask, context) {
    const { surf = [] } = context;
    const fmt = this._detectFormat(this.task);
    const typeLabel = fmt || "section";
    let prompt = `Write the ${typeLabel} "${subTask.label}".\n\n`;
    prompt += `OVERALL TASK: ${this.task}\n\n`;
    prompt += `${typeLabel.toUpperCase()} AIM: ${subTask.description}\n\n`;

    if (surf.length > 0) {
      prompt += `SOURCE PASSAGES:\n`;
      for (let i = 0; i < surf.length; i++) {
        prompt += `---\n${surf[i].text}\n`;
      }
      prompt += "\n";
    } else {
      prompt += "(No source passages. Write what you know.)\n\n";
    }

    if (context.previousSections) {
      prompt += `EARLIER CONTENT (for continuity):\n${context.previousSections.slice(0, 800)}\n\n`;
    }

    prompt += `Now write the ${typeLabel} "${subTask.label}". Draw on the source passages above. Do not use markdown headings. Do not use citation markers. This is a first draft.`;

    return { system: this._executeSystemPrompt(), prompt };
  }

  _buildCorrectionExecutePrompt(subTask, context, prevDraft, correctionNotes) {
    const { surf = [] } = context;
    const fmt = this._detectFormat(this.task);
    const typeLabel = fmt || "section";
    let prompt = `Here is your previous draft of the "${subTask.label}" ${typeLabel}:\n\n---\n${prevDraft}\n---\n\n`;

    const under = correctionNotes?.underCoveredPassages || [];
    if (under.length > 0) {
      prompt += `The following source material was NOT reflected in your draft. Incorporate it:\n\n`;
      for (const u of under) {
        const s = surf[u.surfIndex];
        if (s) prompt += `---\n${s.text}\n---\n`;
      }
      prompt += "\n";
    }

    const ungrounded = correctionNotes?.ungroundedSentences || [];
    if (ungrounded.length > 0) {
      prompt += `The following sentences in your draft are NOT supported by any source passage. Revise or remove them:\n\n`;
      for (const sent of ungrounded) {
        prompt += `- "${sent.trim()}"\n`;
      }
      prompt += "\n";
    }

    prompt += `OVERALL TASK: ${this.task}\n\n`;
    prompt += `${typeLabel.toUpperCase()} AIM: ${subTask.description}\n\n`;
    prompt += `Revise the "${subTask.label}" ${typeLabel} to more faithfully reflect the source passages above. Keep the parts of your draft that already work. Do not use markdown headings. Do not use citation markers or reference numbers.`;

    return { system: this._executeSystemPrompt(), prompt };
  }

  // Diagnose what a draft got wrong relative to surf: which passages it
  // barely touched (by the same info-gain measure as _scoreGrounding), and
  // which of its own sentences the source can't explain at all.
  _diagnoseGrounding(surf, coverage, driftFraction, content, activatedPriors = []) {
    // The aggregate check gates on meanCoverage * (1 - driftPenalty*drift),
    // not raw coverage — so the per-passage floor here must be scaled by
    // the same factor, or a passage can clear this check while the
    // drift-scaled aggregate still fails, leaving nothing to flag.
    const scale = Math.max(1e-6, 1 - this.driftPenalty * driftFraction);
    const floor = this.groundingThreshold / scale;

    // Prior-aware tiebreak: among under-covered passages, prefer ones that
    // cover an activated referent under several surface forms — richer
    // evidence of the same thing to re-show the model during correction.
    const richness = surf.map((s) => {
      const lower = s.text.toLowerCase();
      let count = 0;
      for (const p of activatedPriors) {
        const forms = [...(p.matchedSurfaces || []), ...(p.expansionSurfaces || [])];
        if (forms.some((f) => f && lower.includes(String(f).toLowerCase()))) count++;
      }
      return count;
    });

    const underCoveredPassages = surf
      .map((s, i) => ({ surfIndex: i, coverage: coverage[i] ?? 0, richness: richness[i] }))
      .filter((u) => u.coverage < floor)
      .sort((a, b) => (a.coverage - a.richness * 0.01) - (b.coverage - b.richness * 0.01))
      .slice(0, 3)
      .map(({ surfIndex, coverage }) => ({ surfIndex, coverage }));

    const combinedSourceText = surf.map((s) => s.text).join(" ");
    const sentences = content.split(/(?<=[.!?])\s+/).filter((s) => s.trim().length > 15);
    const ungroundedSentences = sentences.filter((sent) => infoGain(sent, combinedSourceText) < floor).slice(0, 3);

    return { underCoveredPassages, ungroundedSentences };
  }

  // Cross-surface variants of a passage under a prior's other surface forms —
  // used only to compute grounding credit, never shown to the model or
  // substituted into the citation record (which stays strictly verbatim).
  _priorSurfaceVariants(text, prior) {
    const variants = [];
    const from = prior?.matchedSurfaces || [];
    const to = prior?.expansionSurfaces || [];
    for (const a of from) {
      const aLower = String(a).toLowerCase();
      if (!aLower || !text.includes(aLower)) continue;
      for (const b of to) {
        const bLower = String(b).toLowerCase();
        if (!bLower || bLower === aLower) continue;
        variants.push(text.split(aLower).join(bLower));
      }
    }
    return variants;
  }

  // The grounding score IS the error signal that drives correction — not a
  // post-hoc report. activatedPriors (structured { matchedSurfaces,
  // expansionSurfaces } objects, wired by the priors bridge) let a passage
  // using one referent surface still credit a draft using a different one,
  // without either ever being narrated to the model.
  //
  // The measure itself: treat the draft as a predictive PRIOR. `coverage[i]`
  // is how much using this draft as that prior reduces surprise when
  // predicting passage i, relative to a conditional null (the same draft
  // model with its trigram identities scrambled — same shape, no
  // correspondence). This is "does understanding the draft make the source
  // less surprising," not "do these two strings share substrings." Drift is
  // the mirror question: does the source explain the draft, or did the
  // model invent content the source can't account for?
  _scoreGrounding(content, surf, activatedPriors = []) {
    if (surf.length === 0 || content.trim().length === 0) {
      return { groundingScore: 0, driftFraction: 1, coverage: surf.map(() => 0) };
    }

    const coverage = surf.map((s) => {
      let best = infoGain(s.text, content);
      for (const prior of activatedPriors) {
        for (const variant of this._priorSurfaceVariants(s.text.toLowerCase(), prior)) {
          const alt = infoGain(variant, content);
          if (alt > best) best = alt;
        }
      }
      return best;
    });
    const meanCoverage = coverage.reduce((a, b) => a + b, 0) / coverage.length;

    const combinedSourceText = surf.map((s) => s.text).join(" ");
    const draftExplainedBySource = infoGain(content, combinedSourceText);
    const driftFraction = Math.max(0, Math.min(1, 1 - draftExplainedBySource));

    const groundingScore = Math.max(0, Math.min(1, meanCoverage * (1 - this.driftPenalty * driftFraction)));

    return { groundingScore, driftFraction, coverage };
  }

  // ── Surplus: the second reward channel ──
  //
  // Measures cross-passage synthesis: how much MORE of the output is
  // explained by the COMBINATION of all surf passages vs. the best single
  // one. This is structurally different from error-closure: a paraphrase
  // that perfectly restates each passage individually gets high grounding
  // but ZERO surplus (each sentence maps to exactly one source passage).
  // A genuine synthesis that connects passages gets positive surplus
  // because the combined source explains more than any single passage does.
  //
  // normalized in [0, 1]: 0 = pure restatement, >0 = cross-passage synthesis
  _scoreSurplus(content, surf) {
    if (surf.length < 2 || content.trim().length === 0) return 0;
    const combinedSource = surf.map(s => s.text).join(" ");
    const combinedInfoGain = infoGain(content, combinedSource);
    if (combinedInfoGain <= 0) return 0;
    const perPassage = surf.map(s => infoGain(content, s.text));
    const maxSingle = Math.max(...perPassage, 0);
    const rawSurplus = combinedInfoGain - maxSingle;
    return Math.max(0, Math.min(1, rawSurplus / combinedInfoGain));
  }

  // High-precision escalation check: the fold converged on precision
  // (low drift, faithful to what it covers) but some passages remain
  // under-covered AND the deficit isn't just noise (it's the same
  // passages iteration after iteration). When true, the fold escalates
  // to its parent — it cannot resolve at its current grain.
  _detectEscalation(iterations, surf) {
    if (iterations.length < 2) return false;
    const last = iterations[iterations.length - 1];
    // High precision: drift below half of threshold
    const highPrecision = last.driftFraction < this.groundingThreshold / 2;
    if (!highPrecision) return false;
    // Partial coverage: some passages still under threshold
    const floor = this.groundingThreshold / Math.max(1e-6, 1 - this.driftPenalty * last.driftFraction);
    const underCount = (last.coverage || []).filter(c => c < floor).length;
    if (underCount === 0) return false;
    // Persistent: count didn't improve in last 2 iterations
    if (iterations.length >= 3) {
      const prev = iterations[iterations.length - 2];
      const prevUnder = (prev.coverage || []).filter(c => c < floor).length;
      if (prevUnder < underCount) return false;
    }
    return underCount > 0;
  }

  async executeSubtask(subTask, context, { onIteration = null } = {}) {
    const { surf = [], priors = [] } = context;
    const groundingMode = subTask.groundingMode || "required";
    // Only structured priors (from the real priors bridge) carry grounding
    // credit — plain strings (e.g. an old mock) are inert here by design.
    const activatedPriors = (priors || []).filter(
      (p) => p && typeof p === "object" && (p.matchedSurfaces || p.expansionSurfaces)
    );

    const noSourceMaterial = surf.length === 0;
    const maxIterations = groundingMode === "none"
      ? 1  // no correction loop for ungrounded nodes
      : Math.max(1, this.maxCorrectionIterations);

    const iterations = [];
    let content = "";
    let raw = "";
    let correctionNotes = null;

    for (let i = 0; i < maxIterations; i++) {
      const { system, prompt } = i === 0
        ? this._buildInitialExecutePrompt(subTask, context)
        : this._buildCorrectionExecutePrompt(subTask, context, content, correctionNotes);

      const maxTokens = Math.min(this.perSubTaskBudget, Math.max(400, this.perSubTaskBudget - estimateTokens(system) - estimateTokens(prompt)));

      raw = await this._call([
        { role: "system", content: system },
        { role: "user", content: prompt },
      ], maxTokens);
      content = this._stripLocalRefs(raw);

      const { groundingScore, driftFraction, coverage } = this._scoreGrounding(content, surf, activatedPriors);
      const surplusScore = this._scoreSurplus(content, surf);
      const groundingMet = noSourceMaterial || groundingMode === "none" || groundingScore >= this.groundingThreshold;
      const surplusMet = noSourceMaterial || groundingMode === "none" || surplusScore >= this.surplusThreshold;
      const converged = groundingMet && surplusMet;
      const hasMoreIterations = i < maxIterations - 1;

      correctionNotes = (!converged && hasMoreIterations && groundingMode !== "optional")
        ? this._diagnoseGrounding(surf, coverage, driftFraction, content, activatedPriors)
        : null;

      iterations.push({ iteration: i, groundingScore, surplusScore, driftFraction, correctionNotes });
      if (onIteration) onIteration({ id: subTask.id, iteration: i, groundingScore, surplusScore, driftFraction, converged });

      if (converged || !hasMoreIterations || groundingMode === "optional") break;
    }

    // Mechanical citation: match content against surf via n-gram overlap.
    // Strictly verbatim/mechanical — never swapped for the prior-adjusted
    // scoring above, which is an internal steering signal only.
    // When no surf passage matches (creative/ungrounded mode), cite as VOID.
    const citations = this._mechanicalCite(content, surf);
    if (citations.length === 0 && (surf.length === 0 || groundingMode === "none")) {
      citations.push({ surfIndex: -1, evidence: { jaccard: 0, source: "VOID" } });
    }

    // Track gaps: passages from unknown sources
    this.gaps.push(...surf.filter(s => s.source === "?").map(s => ({
      subTask: subTask.id,
      reason: "no_engine_source",
      text: s.text.slice(0, 100),
    })));

    return {
      id: subTask.id,
      label: subTask.label,
      content,
      raw,
      surf,
      priors,
      citations,
      iterations,
      groundingScore: iterations[iterations.length - 1].groundingScore,
      surplusScore: iterations[iterations.length - 1].surplusScore,
      escalated: this._detectEscalation(iterations, surf),
    };
  }

  // ── Recursive tree execution ──

  async executeTree(root = null) {
    root = root || this.treeRoot;
    if (!root) throw new Error("No tree to execute. Call planTree() first.");
    const start = Date.now();
    const progress = this._progress || ((phase, msg) => console.error(`[holonic] ${msg}`));
    progress("execute_tree", `Executing tree: ${root.leaves.length} leaf nodes across ${root.children.length} top-level branches`);
    let draft = "";
    const results = [];
    for (const child of root.children) {
      const res = await this._executeNode(child, { draft });
      results.push(res);
      draft += `\n\n${child.headingMarker} ${child.label}\n\n${res.content}`;
      this.subTaskResults.push(res);
    }
    this.metrics.executeTime += Date.now() - start;
    return results;
  }

  async _executeNode(node, context) {
    if (node.isLeaf) {
      const result = await this._executeOneLeaf(node, context);
      node.result = result;
      node.surf = result.surf;
      node.priors = result.priors;
      node.groundingScore = result.groundingScore;
      node.surplusScore = result.surplusScore;
      return result;
    }
    // Branch: execute children, then synthesize
    let childDraft = context.draft || "";
    const childResults = [];
    for (const child of node.children) {
      const res = await this._executeNode(child, { draft: childDraft });
      childResults.push(res);
      childDraft += `\n\n${child.headingMarker} ${child.label}\n\n${res.content}`;
    }
    const result = await this._synthesizeBranch(node, childResults, context);
    node.result = result;
    node.groundingScore = result.groundingScore;
    node.surplusScore = result.surplusScore;
    return result;
  }

  async _executeOneLeaf(node, context) {
    const st = { id: node.id, label: node.label, description: node.description, type: node.type, groundingMode: node.groundingMode };
    // Skip research for ungrounded nodes (creative writing doesn't need surf)
    let surf = [];
    let priors = [];
    let expansions = [];
    let priorId = null;
    if (node.groundingMode !== "none") {
      const research = await this.researchSubtask(st);
      surf = research.surf;
      priors = research.priors;
      expansions = research.expansions || [];
      priorId = research.priorId || null;
    }
    node.surf = surf;
    node.priors = priors;

    const onIteration = (data) => {
      if (this._progress) this._progress("subtask_iteration", `${node.label} — iter ${data.iteration}: g=${data.groundingScore.toFixed(3)} s=${(data.surplusScore ?? 0).toFixed(3)}`, data);
    };

    const result = await this.executeSubtask(st, { surf, priors, previousSections: context.draft || "" }, { onIteration });
    result.expansions = expansions;
    result.priorId = priorId;

    const trail = result.iterations.map(it => `g=${it.groundingScore.toFixed(2)} s=${(it.surplusScore ?? 0).toFixed(2)}`).join(" → ");
    console.error(`  Leaf ${node.path}: ${result.content.length} chars, ${result.citations.length} citations, ${trail}`);
    return result;
  }

  // Branch synthesis: write a section overview that connects children's
  // content. The synthesis prompt shows the children's content and asks
  // the model to write a coherent overview. Branch-level surplus is
  // measured identically — cross-child connections not present in any
  // single child's output.
  async _synthesizeBranch(node, childResults, context) {
    const childTexts = childResults.map(r => r.content).join("\n\n");
    const prompt = `Write a coherent overview for "${node.label}".
The following sub-sections have been written. Synthesize the key points into a brief overview that connects them:

${childTexts.slice(0, 2000)}

OVERVIEW AIM: ${node.description}

Keep your overview to 3-5 sentences. Do not use citation markers.`;

    const system = "You write clear, connective overviews that synthesize sub-sections into a coherent whole.";
    const raw = await this._call([
      { role: "system", content: system },
      { role: "user", content: prompt },
    ], 500);

    const content = this._stripLocalRefs(raw);
    const surf = childResults.map(r => ({ text: r.content, source: r.id, score: 1 }));

    const { groundingScore, driftFraction, coverage } = this._scoreGrounding(content, surf);
    const surplusScore = this._scoreSurplus(content, surf);
    const citations = this._mechanicalCite(content, surf);

    console.error(`  Branch ${node.path}: ${content.length} chars, surplus=${surplusScore.toFixed(3)}, grounding=${groundingScore.toFixed(3)}`);
    return { id: node.id, label: node.label, content, raw, surf, priors: [], citations, iterations: [], groundingScore, surplusScore, escalated: false };
  }

  _stripLocalRefs(text) {
    const blocks = [
      /\n\*{2}References\*{2}[\s\S]*/i,
      /\n\*{2}Word Count:.*/i,
      /\n\*{2}Works Cited\*{2}[\s\S]*/i,
      /\n\*{2}Bibliography\*{2}[\s\S]*/i,
      /\n#+ References[\s\S]*/i,
    ];
    let cleaned = text;
    for (const pat of blocks) {
      cleaned = cleaned.replace(pat, "");
    }
    cleaned = cleaned.replace(/(\n\[\d+\].*?(?:\n\[\d+\].*?)*)\s*$/, "");
    return cleaned.trim();
  }

  // ── Mechanical citation: n-gram overlap between content and surf ──

  _mechanicalCite(content, surf) {
    const contentLower = content.toLowerCase();
    const contentTri = charTrigrams(contentLower);

    const results = [];
    for (let si = 0; si < surf.length; si++) {
      const passageLower = surf[si].text.toLowerCase();
      const passageTri = charTrigrams(passageLower);

      const intersection = new Set();
      for (const tri of contentTri) {
        if (passageTri.has(tri)) {
          intersection.add(tri);
        }
      }

      if (intersection.size > 0) {
        const union = new Set(contentTri);
        for (const tri of passageTri) union.add(tri);
        const jaccard = intersection.size / union.size;

        const sampleMatches = [...intersection].slice(0, 5);

        results.push({
          surfIndex: si,
          evidence: {
            jaccard: Math.round(jaccard * 10000) / 10000,
            matchedChars: sampleMatches,
          },
        });
      }
    }

    return results.sort((a, b) => b.evidence.jaccard - a.evidence.jaccard);
  }

  // ── Phase 3: Assemble ──

  async assemble() {
    const start = Date.now();

    const titleMatch = this.task.match(/(?:about|on|of)\s+(.+?)(?:\s+(?:with|using|-|$))/i);
    const title = titleMatch
      ? titleMatch[1].trim()
      : this.task.replace(/^(write|create|generate|produce)\s+/i, "").slice(0, 80);

    const lines = [];
    const fmtLabel = this._formatAssembleLabel(this.task);
    const fmt = this._detectFormat(this.task);
    lines.push(`# ${title.charAt(0).toUpperCase() + title.slice(1)}`);
    lines.push("");
    lines.push(`*Generated via holonic task decomposition — ${this.subTaskResults.length} ${fmtLabel}*`);
    lines.push("");
    lines.push("---");
    lines.push("");

    // Content sections
    for (const st of this.subTaskResults) {
      if (fmt !== 'screenplay') {
        lines.push(`## ${st.label}`);
        lines.push("");
      }
      lines.push(st.content.trim());
      lines.push("");
      lines.push("---");
      lines.push("");
    }

    // Provenance section
    lines.push("## Provenance");
    lines.push("");

    if (this.replanHistory.length > 0) {
      lines.push(`**Re-planning (${this.replanHistory.length} decision${this.replanHistory.length > 1 ? "s" : ""}):**`);
      lines.push("");
      for (const r of this.replanHistory) {
        lines.push(`- ${r.action}: ${r.justification}`);
      }
      lines.push("");
    }

    for (const st of this.subTaskResults) {
      lines.push(`### ${st.label}`);
      lines.push("");

      // Surf. Passages a prior widened the search to reach carry via-prior
      // attribution, so the reader can tell base retrieval from prior-driven
      // retrieval at a glance instead of taking the passage list on faith.
      if (st.surf && st.surf.length > 0) {
        const viaCount = st.surf.filter((s) => s.viaPrior).length;
        lines.push(`**Surf (${st.surf.length} passages${viaCount ? `, ${viaCount} via prior` : ""}):**`);
        lines.push("");
        for (let i = 0; i < st.surf.length; i++) {
          const s = st.surf[i];
          const via = s.viaPrior
            ? ` via-prior="${s.viaPrior.priorId ?? "?"}" via-referent="${s.viaPrior.display ?? s.viaPrior.referentId}" via-surface="${s.viaPrior.surface}"`
            : "";
          lines.push(`<surf idx="${i}" source="${s.source}" score="${s.score.toFixed(1)}"${via}>`);
          lines.push(s.text.slice(0, 300));
          lines.push(`</surf>`);
          lines.push("");
        }
      }

      // Priors. These entries are structured records from priors-bridge, not
      // strings — interpolating one directly rendered "[object Object]" and
      // silently destroyed the only user-visible account of what the prior did.
      if (st.priors && st.priors.length > 0) {
        lines.push(`**Activated priors (${st.priors.length}):**`);
        lines.push("");
        for (const p of st.priors) {
          if (typeof p === "string") { lines.push(`- ${p}`); continue; }
          const matched = (p.matchedSurfaces || []).join(", ") || "—";
          const expansion = (p.expansionSurfaces || []).join(", ") || "—";
          lines.push(`- **${p.display ?? p.referentId}** (\`${p.referentId}\`${p.priorId ? ` from \`${p.priorId}\`` : ""})`);
          lines.push(`  - matched in surf: ${matched}`);
          lines.push(`  - expansion forms: ${expansion}`);
        }
        lines.push("");
      }

      // What the prior actually changed about this surf. Grouped by the search
      // that ran, since one surface form can admit several passages — listing
      // them per-passage repeats the same line and buries how many distinct
      // searches the prior actually caused.
      if (st.expansions && st.expansions.length > 0) {
        lines.push(`**Prior-driven retrieval (${st.expansions.length} passage${st.expansions.length > 1 ? "s" : ""} added):**`);
        lines.push("");
        // Grouped on a Map whose VALUES carry the fields, so no delimiter is
        // needed. An earlier version packed them into a delimited string key,
        // which meant picking a separator that could not occur in a surface
        // form or display name -- and writing raw NUL bytes into the source to
        // get one. That made the file read as binary to grep and diff.
        const bySearch = new Map();
        for (const e of st.expansions) {
          const priorId = e.priorId ?? "?";
          const display = e.display ?? e.referentId;
          const key = JSON.stringify([priorId, display, e.surface]);
          const hit = bySearch.get(key);
          if (hit) hit.count++;
          else bySearch.set(key, { priorId, display, surface: e.surface, count: 1 });
        }
        for (const g of bySearch.values()) {
          lines.push(`- \`${g.priorId}\` → **${g.display}** → searched "${g.surface}" → ${g.count} passage${g.count > 1 ? "s" : ""}`);
        }
        lines.push("");
      }

      // Correction-loop trace — the auditable "did it learn" evidence
      if (st.iterations && st.iterations.length > 0) {
        lines.push(`**Grounding trace (${st.iterations.length} iteration${st.iterations.length > 1 ? "s" : ""}):**`);
        lines.push("");
        for (const it of st.iterations) {
          lines.push(`- iteration ${it.iteration}: grounding ${it.groundingScore.toFixed(3)}, drift ${it.driftFraction.toFixed(3)}` +
            (it.correctionNotes ? " — correcting" : ""));
        }
        lines.push("");
      }

      // Mechanical citations
      if (st.citations && st.citations.length > 0) {
        lines.push(`**Mechanical citations (${st.citations.length}):**`);
        lines.push("");
        for (const c of st.citations) {
          const s = st.surf[c.surfIndex];
          const charList = c.evidence.matchedChars.join(", ");
          lines.push(`- surf[${c.surfIndex}] (jaccard: ${c.evidence.jaccard}): "${s ? s.text.slice(0, 120) : '?'}..."`);
          lines.push(`  char-trigrams: "${charList}"`);
        }
        lines.push("");
      } else if (st.surf && st.surf.length > 0) {
        lines.push("_No mechanical citation match found (content does not overlap with surf passages)._");
        lines.push("");
      }

      lines.push("---");
      lines.push("");
    }

    const output = lines.join("\n");
    this.metrics.assembleTime = Date.now() - start;

    if (this.outputPath) {
      const fs = await import("fs");
      fs.writeFileSync(this.outputPath, output, "utf8");
    }

    return output;
  }

  // ── Tree assembler: hierarchical markdown output ──
  //
  // Recursively walks the tree and renders each node at its heading level.
  // Leaf nodes produce content (from executeSubtask); branch nodes produce
  // their synthesis content. The heading level reflects the node's depth.
  // Provenance (surf/priors/citations) is appended as a flat section at the
  // bottom, since per-level provenance would clutter the hierarchy.

  async assembleTree(root = null) {
    const start = Date.now();
    root = root || this.treeRoot;
    if (!root) return await this.assemble();

    const titleMatch = this.task.match(/(?:about|on|of)\s+(.+?)(?:\s+(?:with|using|-|$))/i);
    const title = titleMatch
      ? titleMatch[1].trim()
      : this.task.replace(/^(write|create|generate|produce)\s+/i, "").slice(0, 80);
    const fmtLabel = this._formatAssembleLabel(this.task);
    const totalLeaves = root.leaves.length;

    const lines = [];
    lines.push(`# ${title.charAt(0).toUpperCase() + title.slice(1)}`);
    lines.push("");
    lines.push(`*Generated via holonic task decomposition — ${totalLeaves} ${fmtLabel} across ${root.children.length} top-level branches*`);
    lines.push("");
    lines.push("---");
    lines.push("");

    for (const child of root.children) {
      this._renderNode(child, lines);
    }

    // Provenance section (flat, at the bottom)
    lines.push("## Provenance");
    lines.push("");

    if (this.replanHistory.length > 0) {
      lines.push(`**Re-planning (${this.replanHistory.length} decision${this.replanHistory.length > 1 ? "s" : ""}):**`);
      lines.push("");
      for (const r of this.replanHistory) lines.push(`- ${r.action}: ${r.justification}`);
      lines.push("");
    }

    const allLeaves = root.leaves;
    for (const node of allLeaves) {
      if (!node.result) continue;
      const st = node.result;
      lines.push(`### ${st.label} (\`${node.path}\`)`);
      lines.push("");

      // Surf
      if (st.surf && st.surf.length > 0) {
        const viaCount = st.surf.filter((s) => s.viaPrior).length;
        lines.push(`**Surf (${st.surf.length} passages${viaCount ? `, ${viaCount} via prior` : ""}):**`);
        lines.push("");
        for (let i = 0; i < st.surf.length; i++) {
          const s = st.surf[i];
          const via = s.viaPrior ? ` via-prior="${s.viaPrior.priorId ?? "?"}"` : "";
          lines.push(`<surf idx="${i}" source="${s.source}" score="${s.score.toFixed(1)}"${via}>`);
          lines.push(s.text.slice(0, 300));
          lines.push(`</surf>`);
          lines.push("");
        }
      }

      // Priors
      if (st.priors && st.priors.length > 0) {
        lines.push(`**Activated priors (${st.priors.length}):**`);
        lines.push("");
        for (const p of st.priors) {
          if (typeof p === "string") { lines.push(`- ${p}`); continue; }
          const matched = (p.matchedSurfaces || []).join(", ") || "—";
          const expansion = (p.expansionSurfaces || []).join(", ") || "—";
          lines.push(`- **${p.display ?? p.referentId}** (\`${p.referentId}\`${p.priorId ? ` from \`${p.priorId}\`` : ""})`);
          lines.push(`  - matched in surf: ${matched}`);
          lines.push(`  - expansion forms: ${expansion}`);
        }
        lines.push("");
      }

      // Score summary
      if (st.iterations && st.iterations.length > 0) {
        const last = st.iterations[st.iterations.length - 1];
        lines.push(`**Scores: grounding ${last.groundingScore.toFixed(3)}, surplus ${(last.surplusScore ?? 0).toFixed(3)}**`);
        lines.push("");
        lines.push(`**Grounding trace (${st.iterations.length} iteration${st.iterations.length > 1 ? "s" : ""}):**`);
        lines.push("");
        for (const it of st.iterations) {
          lines.push(`- iter ${it.iteration}: g=${it.groundingScore.toFixed(3)} s=${(it.surplusScore ?? 0).toFixed(3)} d=${it.driftFraction.toFixed(3)}${it.correctionNotes ? " — correcting" : ""}`);
        }
        lines.push("");
      }

      // Mechanical citations
      if (st.citations && st.citations.length > 0) {
        lines.push(`**Mechanical citations (${st.citations.length}):**`);
        lines.push("");
        for (const c of st.citations) {
          const s = st.surf[c.surfIndex];
          lines.push(`- surf[${c.surfIndex}] (jaccard: ${c.evidence.jaccard}): "${s ? s.text.slice(0, 120) : '?'}..."`);
        }
        lines.push("");
      } else if (st.surf && st.surf.length > 0) {
        lines.push("_No mechanical citation match found._");
        lines.push("");
      }

      lines.push("---");
      lines.push("");
    }

    const output = lines.join("\n");
    this.metrics.assembleTime = Date.now() - start;

    if (this.outputPath) {
      const fs = await import("fs");
      fs.writeFileSync(this.outputPath, output, "utf8");
    }

    return output;
  }

  _renderNode(node, lines) {
    if (!node.result) return;
    const hm = node.headingMarker;
    if (hm) {
      lines.push(`${hm} ${node.label}`);
      lines.push("");
    }
    lines.push(node.result.content.trim());
    lines.push("");
    // If branch, render children inline
    if (!node.isLeaf) {
      for (const child of node.children) {
        this._renderNode(child, lines);
      }
    }
    if (hm) {
      lines.push("---");
      lines.push("");
    }
  }

  // ── Bounded re-planning ──
  //
  // The PLAN phase is not fixed. If a sub-task never converges even after
  // the correction loop, or two sub-tasks turn out to share most of their
  // cited material, the decomposition itself may have been wrong — earned
  // via the same two holon-level tests the engine uses elsewhere
  // (existence-dependency, possibility-constraint), never by noticing
  // overlap and reshuffling on a hunch. Bounded to `maxReplans` (default 1)
  // per run: this is a correction pass, not a planner that loops until
  // satisfied.

  _citedSpanIds(result) {
    const ids = new Set();
    for (const c of result.citations || []) {
      const s = result.surf?.[c.surfIndex];
      if (s?.spanId) ids.add(s.spanId);
    }
    return ids;
  }

  // Which sub-tasks never converged, and which pairs share most of their
  // cited source material (candidates for merge — possibly peers, possibly
  // one nested in the other; replan() decides which, it isn't assumed here).
  _detectReplanTriggers() {
    const unresolvedIds = this.subTaskResults
      .filter((r) => r.groundingScore < this.groundingThreshold)
      .map((r) => r.id);

    const redundantPairs = [];
    for (let i = 0; i < this.subTaskResults.length; i++) {
      for (let j = i + 1; j < this.subTaskResults.length; j++) {
        const a = this.subTaskResults[i];
        const b = this.subTaskResults[j];
        const idsA = this._citedSpanIds(a);
        const idsB = this._citedSpanIds(b);
        if (idsA.size === 0 || idsB.size === 0) continue;
        let shared = 0;
        for (const id of idsA) if (idsB.has(id)) shared++;
        const overlap = shared / Math.min(idsA.size, idsB.size);
        if (overlap >= this.redundancyOverlapThreshold) {
          redundantPairs.push({ aId: a.id, bId: b.id, overlap });
        }
      }
    }

    return { unresolvedIds, redundantPairs };
  }

  // Ask the model to justify any change via the two holon-level tests
  // before proposing one. "Peer, no change" is a valid, expected outcome —
  // it must come back recorded, not treated as a non-answer.
  async replan(unresolvedIds, redundantPairs) {
    const currentSubTasks = this.planResult.subTasks;
    const flaggedIds = new Set([...unresolvedIds, ...redundantPairs.flatMap((p) => [p.aId, p.bId])]);
    if (flaggedIds.size === 0) return null;

    const listing = currentSubTasks
      .filter((st) => flaggedIds.has(st.id))
      .map((st) => `- ${st.id}: ${st.label} — ${st.description}`)
      .join("\n");

    let prompt = `You are revising a document's sub-task decomposition. These sections are flagged for review:\n\n${listing}\n\n`;

    if (unresolvedIds.length > 0) {
      prompt += `SECTIONS THAT NEVER GROUNDED WELL after revision (may be scoped wrong, or lack real source material):\n`;
      for (const id of unresolvedIds) prompt += `  - "${id}"\n`;
      prompt += "\n";
    }
    if (redundantPairs.length > 0) {
      prompt += `SECTION PAIRS SHARING MOST OF THE SAME CITED SOURCE MATERIAL:\n`;
      for (const p of redundantPairs) prompt += `  - "${p.aId}" and "${p.bId}" (${Math.round(p.overlap * 100)}% shared)\n`;
      prompt += "\n";
    }

    prompt += `For each flagged section or pair, answer two tests before proposing any change:
1. EXISTENCE-DEPENDENCY: could one section's content exist without the other's already existing?
2. POSSIBILITY-CONSTRAINT: does one section's scope constrain what the other can contain (nested, not a peer), or are they genuinely unconstrained by each other (peer)?

Based on your answers, propose what to do with ONLY the flagged sections above — merge, split, rescope, or explicitly conclude "peer, no change" if they are legitimately independent. Do not repeat unaffected sections; they are not part of this decision.

Respond with ONLY a JSON object of this shape:
{"changes": [{"action": "merge"|"split"|"rescope"|"peer", "justification": "one sentence citing which test decided it", "resultingSubTasks": [{"id": "...", "label": "...", "description": "...", "type": "section"}]}]}
For "peer" (no change), "resultingSubTasks" should be the original flagged sections unchanged.`;

    const system = "You are a precise structural editor. Always respond with valid JSON.";
    const response = await this._call([
      { role: "system", content: system },
      { role: "user", content: prompt },
    ], 1200);

    const parsed = this._extractBalancedJSON(response);
    if (!parsed || !Array.isArray(parsed.changes)) return null;

    return { flaggedIds, changes: parsed.changes };
  }

  // ── Run: full pipeline ──

  async _executeOne(st, draft, progress, label) {
    const execStart = Date.now();
    progress("subtask_start", `Executing ${label}: ${st.label}`, { id: st.id, label: st.label });

    const { surf, priors, expansions, priorId } = await this.researchSubtask(st);
    if (surf.length > 0) {
      console.error(`  Found ${surf.length} surf passages (scores: ${surf.map(s => s.score.toFixed(1)).join(", ")})`);
    } else {
      console.error(`  No engine sources found — relying on model knowledge`);
    }
    if (priors.length > 0) {
      console.error(`  Activated ${priors.length} priors${priorId ? ` from ${priorId}` : ""}` +
        (expansions.length ? `, widening surf by ${expansions.length} passage(s)` : ""));
    }

    // The user-facing account of prior influence on this surf, emitted before
    // generation so the UI can show what shaped the answer while it is still
    // being written rather than only in the final provenance block.
    progress("subtask_priors", `${st.label}: ${priors.length} prior(s) active`, {
      id: st.id,
      label: st.label,
      priorId,
      priors: priors.map((p) => (typeof p === "string" ? { display: p } : {
        referentId: p.referentId,
        display: p.display,
        priorId: p.priorId ?? priorId ?? null,
        matchedSurfaces: p.matchedSurfaces ?? [],
        expansionSurfaces: p.expansionSurfaces ?? [],
      })),
      expansions,
      surfTotal: surf.length,
      surfViaPrior: surf.filter((s) => s.viaPrior).length,
    });

    const onIteration = (data) => progress(
      "subtask_iteration",
      `${st.label} — iteration ${data.iteration}: grounding ${data.groundingScore.toFixed(3)}`,
      data
    );

    const result = await this.executeSubtask(st, { surf, priors, previousSections: draft }, { onIteration });
    result.expansions = expansions;
    result.priorId = priorId;
    const elapsed = ((Date.now() - execStart) / 1000).toFixed(1);
    const scoreTrail = result.iterations.map(it => it.groundingScore.toFixed(2)).join(" → ");
    console.error(`  Done in ${elapsed}s: ${result.content.length} chars, ${result.citations.length} mechanical citations, grounding ${scoreTrail}`);

    progress("subtask_done", `${st.label}: grounding ${scoreTrail}`, {
      id: st.id,
      label: st.label,
      groundingScore: result.groundingScore,
      citationsCount: result.citations.length,
      iterationsCount: result.iterations.length,
    });

    return result;
  }

  async run({ onProgress = null } = {}) {
    const startTotal = Date.now();

    const progress = (phase, msg, data = {}) => {
      if (onProgress) onProgress(phase, msg, data);
      console.error(`[holonic] ${msg}`);
    };

    // Phase 1: Plan
    progress("plan", `Planning decomposition for: "${this.task.slice(0, 100)}..."`);
    await this.plan();
    progress("plan", `Plan: ${this.planResult.subTasks.length} sub-tasks`, { subTasks: this.planResult.subTasks });
    for (const st of this.planResult.subTasks) {
      console.error(`  - ${st.id}: ${st.label}`);
    }

    // Phase 2: Execute each sub-task
    let draft = "";
    for (let i = 0; i < this.planResult.subTasks.length; i++) {
      const st = this.planResult.subTasks[i];
      const result = await this._executeOne(st, draft, progress, `${i + 1}/${this.planResult.subTasks.length}`);
      this.subTaskResults.push(result);
      draft += `\n\n## ${result.label}\n\n${result.content}`;
    }

    // Phase 2.5: Bounded re-plan — only if a sub-task never converged, or
    // two sub-tasks turned out redundant. Capped at maxReplans; "peer, no
    // change" is a valid outcome and is recorded, not silently dropped.
    if (this.replanCount < this.maxReplans) {
      const { unresolvedIds, redundantPairs } = this._detectReplanTriggers();
      if (unresolvedIds.length > 0 || redundantPairs.length > 0) {
        progress("replan_start", `Replanning: ${unresolvedIds.length} unresolved, ${redundantPairs.length} redundant pair(s)`,
          { unresolvedIds, redundantPairs });

        let replanResult = null;
        try {
          replanResult = await this.replan(unresolvedIds, redundantPairs);
        } catch (err) {
          console.error(`  Replan failed: ${err.message}`);
        }
        this.replanCount++;

        if (replanResult && replanResult.changes.length > 0) {
          const { flaggedIds, changes } = replanResult;
          this.replanHistory = changes.map((c) => ({ action: c.action, justification: c.justification }));

          // "Peer, no change" preserves the original sub-task and its
          // already-converged result untouched — no re-execution, no
          // wasted iteration budget on a decision that changed nothing.
          const peerIds = new Set();
          const actionableChanges = [];
          for (const c of changes) {
            if (c.action === "peer") {
              for (const nt of c.resultingSubTasks || []) peerIds.add(nt.id);
            } else {
              actionableChanges.push(c);
            }
          }

          const keptResults = this.subTaskResults.filter((r) => !flaggedIds.has(r.id) || peerIds.has(r.id));
          const keptSubTasks = this.planResult.subTasks.filter((st) => !flaggedIds.has(st.id) || peerIds.has(st.id));

          const usedIds = new Set(keptSubTasks.map((st) => st.id));
          const newSubTasks = [];
          for (const c of actionableChanges) {
            for (const nt of c.resultingSubTasks || []) {
              let id = nt.id || `replanned-${newSubTasks.length}`;
              let n = 1;
              while (usedIds.has(id)) id = `${nt.id || "replanned"}-${++n}`;
              usedIds.add(id);
              newSubTasks.push({ ...nt, id });
            }
          }

          let replanDraft = keptResults.map((r) => `\n\n## ${r.label}\n\n${r.content}`).join("");
          const newResults = [];
          for (let i = 0; i < newSubTasks.length; i++) {
            const result = await this._executeOne(newSubTasks[i], replanDraft, progress, `replan ${i + 1}/${newSubTasks.length}`);
            newResults.push(result);
            replanDraft += `\n\n## ${result.label}\n\n${result.content}`;
          }

          this.planResult.subTasks = [...keptSubTasks, ...newSubTasks];
          this.subTaskResults = [...keptResults, ...newResults];
          progress("replan_done", `Replan complete: ${changes.length} change(s) recorded, ${newSubTasks.length} section(s) re-executed`,
            { changes: this.replanHistory });
        } else {
          progress("replan_done", "Replan produced no changes — keeping original decomposition", { changes: [] });
        }
      }
    }

    // Phase 3: Assemble
    progress("assemble", "Assembling final output with provenance");
    const output = await this.assemble();

    this.metrics.executeTime = Date.now() - startTotal - this.metrics.planTime - this.metrics.assembleTime;
    this.metrics.totalTime = Date.now() - startTotal;

    const totalCitations = this.subTaskResults.reduce((a, r) => a + r.citations.length, 0);
    progress("done",
      `Done: ${output.length} chars, ${this.subTaskResults.length} sections, ` +
      `${totalCitations} mechanical citations, ` +
      `${this.gaps.length} gaps, ` +
      `${(this.metrics.totalTime / 1000).toFixed(1)}s total`,
      {
        sections: this.subTaskResults.length,
        chars: output.length,
        mechanicalCitations: totalCitations,
        gaps: this.gaps.length,
        totalTimeMs: this.metrics.totalTime,
      }
    );

    return {
      task: this.task,
      model: this.model,
      plan: this.planResult,
      results: this.subTaskResults,
      gaps: this.gaps,
      replanHistory: this.replanHistory,
      output,
      path: this.outputPath,
      metrics: this.metrics,
    };
  }

  // ── runTree: full pipeline with recursive tree decomposition ──
  //
  // Phase split: early/exploratory folds are epistemic-value-dominant
  // (seek confusion, don't route around it); late/expository folds
  // minimize reader surprise. The planner assigns phase per node.
  // Two-channel completion: BOTH grounding (error-closure, gives a
  // stopping point) AND surplus (cross-passage synthesis, prevents
  // dark-room collapse) must clear their thresholds.

  async runTree({ onProgress = null } = {}) {
    const startTotal = Date.now();
    this._progress = (phase, msg, data = {}) => {
      if (onProgress) onProgress(phase, msg, data);
      console.error(`[holonic] ${msg}`);
    };
    const progress = this._progress;

    // Phase 0: Learn — a series of sub-tasks that research format, genre,
    // and conventions before planning the content. The output (learningGuide)
    // is injected into planner and executor prompts at every level.
    progress("learn", `Learning phase: researching format and conventions for: "${this.task.slice(0, 80)}..."`);
    await this._learnSeries();
    progress("learn", `Learning guide produced: ${this.learningGuide.length} chars`, { guidePreview: this.learningGuide.slice(0, 120) });

    // Phase 1: Recursive tree planning
    progress("plan_tree", `Planning tree decomposition for: "${this.task.slice(0, 100)}..."`);
    await this.planTree();
    const leaves = this.treeRoot.leaves;
    const depth = Math.max(...leaves.map(l => l.level), 0);
    progress("plan_tree", `Tree: ${leaves.length} leaf nodes across ${depth + 1} levels`, {
      nodes: this.treeRoot.children.length,
      leaves: leaves.length,
      depth,
      phaseStrategy: this.phaseStrategy,
    });

    // Phase 2: Execute the tree (depth-first)
    progress("execute_tree", `Executing ${leaves.length} leaf nodes across ${this.treeRoot.children.length} top-level branches`);
    this.subTaskResults = [];
    let draft = "";
    for (let i = 0; i < this.treeRoot.children.length; i++) {
      const child = this.treeRoot.children[i];
      progress("subtask_start", `[${i + 1}/${this.treeRoot.children.length}] ${child.label} (level ${child.level})`, { id: child.id, label: child.label });
      const result = await this._executeNode(child, { draft });
      this.subTaskResults.push(result);
      draft += `\n\n${child.headingMarker} ${child.label}\n\n${result.content}`;
    }
    this.subTaskResults = this.treeRoot.leaves.map(n => n.result).filter(Boolean).flat();

    // Phase 3: Assemble tree
    progress("assemble_tree", "Assembling hierarchical output");
    const output = await this.assembleTree();

    this.metrics.executeTime = Date.now() - startTotal - this.metrics.planTime - this.metrics.assembleTime;
    this.metrics.totalTime = Date.now() - startTotal;

    const totalCitations = this.subTaskResults.reduce((a, r) => a + r.citations.length, 0);
    const leafCount = this.treeRoot.leaves.length;
    progress("done",
      `Done: ${output.length} chars, ${leafCount} leaves across ${depth + 1} levels, ` +
      `${totalCitations} citations, ${this.gaps.length} gaps, ` +
      `${(this.metrics.totalTime / 1000).toFixed(1)}s total`,
      { leafCount, levels: depth + 1, chars: output.length, citations: totalCitations, gaps: this.gaps.length }
    );

    return {
      task: this.task,
      model: this.model,
      tree: this.treeRoot.toJSON(),
      plan: this.planResult,
      results: this.subTaskResults,
      gaps: this.gaps,
      replanHistory: this.replanHistory,
      output,
      path: this.outputPath,
      metrics: this.metrics,
    };
  }
}
