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

function estimateTokens(text) {
  return Math.ceil((text || "").length / 3.5);
}

function charTrigrams(text) {
  const set = new Set();
  for (let i = 0; i <= text.length - 3; i++) {
    set.add(text.slice(i, i + 3));
  }
  return set;
}

export class HolonicTask {
  constructor({
    task,
    model = "gemma2:2b",
    engine = null,
    outputPath = null,
    perSubTaskBudget = 4000,
    maxSubTasks = 8,
    ollamaUrl = OLLAMA_URL,
  } = {}) {
    if (!task || typeof task !== "string") throw new TypeError("HolonicTask requires a { task } string");

    this.task = task;
    this.model = model;
    this.engine = engine;
    this.outputPath = outputPath;
    this.perSubTaskBudget = perSubTaskBudget;
    this.maxSubTasks = maxSubTasks;
    this.ollamaUrl = ollamaUrl;

    this.planResult = null;
    this.subTaskResults = [];
    this.provenance = [];
    this.gaps = [];
    this.metrics = { planTime: 0, executeTime: 0, assembleTime: 0, totalTokens: 0 };
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
      signal: AbortSignal.timeout(300000),
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

  // ── Phase 1: Plan ──

  async plan() {
    const start = Date.now();

    const prompt = `You are a precise task planner. Decompose the following task into ${Math.min(this.maxSubTasks, 4)}-${this.maxSubTasks} focused sub-tasks. Each must be self-contained and achievable by a small language model writing 400-600 words.

TASK: ${this.task}

Respond with a JSON array of objects. Each object:
  - "id": short unique identifier (e.g. "intro", "biology")
  - "label": human-readable heading
  - "description": what this sub-task covers (2-3 sentences)
  - "type": "introduction", "section", "analysis", or "conclusion"

Return ONLY the JSON array. Example:
[{"id":"intro","label":"Introduction","description":"Introduce the topic and its significance.","type":"introduction"}]`;

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

  // ── Phase 2a: Research (returns { surf, priors }) ──

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

    const surf = results.map((r) => ({
      text: (r.text || r.preview || "").slice(0, 800),
      source: r.source || r.file_path || r.source_id || "?",
      score: r.score || 0,
      spanId: r.span_id || r.id || null,
      byteStart: r.byte_start ?? null,
      byteEnd: r.byte_end ?? null,
    })).filter(r => r.text.length > 20);

    // Get activated priors from engine (or empty if engine doesn't support it)
    let priors = [];
    if (this.engine.getPriors && typeof this.engine.getPriors === "function") {
      const surfText = surf.map(s => s.text).join(" ");
      priors = this.engine.getPriors(surfText);
    }

    return { surf, priors };
  }

  // ── Phase 2b: Execute (model writes without citation knowledge) ──

  async executeSubtask(subTask, context) {
    const { surf = [], priors = [] } = context;

    // Build prompt with priors and source passages — NO citation numbering
    const system = [
      "You are a research writer. Write ONE focused section of a larger document.",
      "Draw key terms, specific phrasing, and evidence from the SOURCE TEXT passages below.",
      "Write naturally — do not use citation markers or reference numbers.",
    ].join(" ");

    let prompt = `Write the "${subTask.label}" section.\n\n`;
    prompt += `OVERALL TASK: ${this.task}\n\n`;
    prompt += `SECTION CONTEXT: ${subTask.description}\n\n`;

    // Inject activated priors (coref/corpus context for the model)
    if (priors.length > 0) {
      prompt += `TEXT PRIORS:\n`;
      for (const p of priors) {
        prompt += `- ${p}\n`;
      }
      prompt += "\n";
    }

    // Provide source passages for the model to draw from
    if (surf.length > 0) {
      prompt += `SOURCE TEXT (draw your key terms, specific phrasing, and evidence from these passages):\n`;
      for (let i = 0; i < surf.length; i++) {
        prompt += `---\n${surf[i].text}\n`;
      }
      prompt += "\n";
    } else {
      prompt += "(No specific source text available. Rely on general knowledge.)\n\n";
    }

    if (context.previousSections) {
      prompt += `PREVIOUS SECTIONS (for continuity):\n${context.previousSections.slice(0, 800)}\n\n`;
    }

    prompt += `Write the "${subTask.label}" section now. Use specific language from the SOURCE TEXT. Do not use citation markers or reference numbers. Stay on topic.`;

    const maxTokens = Math.min(this.perSubTaskBudget, Math.max(400, this.perSubTaskBudget - estimateTokens(system) - estimateTokens(prompt)));

    const content = await this._call([
      { role: "system", content: system },
      { role: "user", content: prompt },
    ], maxTokens);

    const cleaned = this._stripLocalRefs(content);

    // Mechanical citation: match content against surf via n-gram overlap
    const citations = this._mechanicalCite(cleaned, surf);

    // Track gaps: passages from unknown sources
    this.gaps.push(...surf.filter(s => s.source === "?").map(s => ({
      subTask: subTask.id,
      reason: "no_engine_source",
      text: s.text.slice(0, 100),
    })));

    return {
      id: subTask.id,
      label: subTask.label,
      content: cleaned,
      raw: content,
      surf,
      priors,
      citations,
    };
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
    lines.push(`# ${title.charAt(0).toUpperCase() + title.slice(1)}`);
    lines.push("");
    lines.push(`*Generated via holonic task decomposition — ${this.subTaskResults.length} sections*`);
    lines.push("");
    lines.push("---");
    lines.push("");

    // Content sections
    for (const st of this.subTaskResults) {
      lines.push(`## ${st.label}`);
      lines.push("");
      lines.push(st.content.trim());
      lines.push("");
      lines.push("---");
      lines.push("");
    }

    // Provenance section
    lines.push("## Provenance");
    lines.push("");

    for (const st of this.subTaskResults) {
      lines.push(`### ${st.label}`);
      lines.push("");

      // Surf
      if (st.surf && st.surf.length > 0) {
        lines.push(`**Surf (${st.surf.length} passages):**`);
        lines.push("");
        for (let i = 0; i < st.surf.length; i++) {
          const s = st.surf[i];
          lines.push(`<surf idx="${i}" source="${s.source}" score="${s.score.toFixed(1)}">`);
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
          lines.push(`- ${p}`);
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

  // ── Run: full pipeline ──

  async run({ onProgress = null } = {}) {
    const startTotal = Date.now();

    const progress = (phase, msg) => {
      if (onProgress) onProgress(phase, msg);
      console.error(`[holonic] ${msg}`);
    };

    // Phase 1: Plan
    progress("plan", `Planning decomposition for: "${this.task.slice(0, 100)}..."`);
    await this.plan();
    progress("plan", `Plan: ${this.planResult.subTasks.length} sub-tasks`);
    for (const st of this.planResult.subTasks) {
      console.error(`  - ${st.id}: ${st.label}`);
    }

    // Phase 2: Execute each sub-task
    let draft = "";
    for (let i = 0; i < this.planResult.subTasks.length; i++) {
      const st = this.planResult.subTasks[i];
      const execStart = Date.now();

      progress("execute", `Executing ${i + 1}/${this.planResult.subTasks.length}: ${st.label}`);

      const { surf, priors } = await this.researchSubtask(st);
      if (surf.length > 0) {
        console.error(`  Found ${surf.length} surf passages (scores: ${surf.map(s => s.score.toFixed(1)).join(", ")})`);
      } else {
        console.error(`  No engine sources found — relying on model knowledge`);
      }
      if (priors.length > 0) {
        console.error(`  Activated ${priors.length} priors`);
      }

      const result = await this.executeSubtask(st, { surf, priors, previousSections: draft });
      this.subTaskResults.push(result);
      draft += `\n\n## ${result.label}\n\n${result.content}`;

      const elapsed = ((Date.now() - execStart) / 1000).toFixed(1);
      console.error(`  Done in ${elapsed}s: ${result.content.length} chars, ${result.citations.length} mechanical citations`);
    }

    // Phase 3: Assemble
    progress("assemble", "Assembling final output with provenance");
    const output = await this.assemble();

    this.metrics.executeTime = Date.now() - startTotal - this.metrics.planTime - this.metrics.assembleTime;
    this.metrics.totalTime = Date.now() - startTotal;

    progress("done",
      `Done: ${output.length} chars, ${this.subTaskResults.length} sections, ` +
      `${this.subTaskResults.reduce((a, r) => a + r.citations.length, 0)} mechanical citations, ` +
      `${this.gaps.length} gaps, ` +
      `${(this.metrics.totalTime / 1000).toFixed(1)}s total`
    );

    return {
      task: this.task,
      model: this.model,
      plan: this.planResult,
      results: this.subTaskResults,
      gaps: this.gaps,
      output,
      path: this.outputPath,
      metrics: this.metrics,
    };
  }
}
