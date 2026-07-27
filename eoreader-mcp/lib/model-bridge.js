const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const THINK_MODEL = process.env.THINK_MODEL || "gemma2:2b";
const SPEAK_MODEL = process.env.SPEAK_MODEL || "llama3.2";
const PLAN_MODEL = process.env.PLAN_MODEL || SPEAK_MODEL;

export async function callModel(model, messages, maxTokens = 512) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: { temperature: 0.7, num_predict: maxTokens },
    }),
    signal: AbortSignal.timeout(300000),
  });
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.message?.content || "";
}

// ── Think: verify sufficiency, suggest reformulation ──

export async function think(query, summary, round = 1) {
  if (!summary || summary.length < 50) {
    return { sufficient: false, evidence: null, gap: "no source material", reformulation: query };
  }
  const result = await callModel(THINK_MODEL, [
    {
      role: "system",
      content: "GROUND: source material below\nFIGURE: the question\nPATTERN: decide if GROUND contains specific evidence to answer FIGURE\n\nIF YES: respond: THINK: YES — <specific evidence found>\nIF NO: respond: THINK: NO — <gap> — SEARCH: <reformulated query>\n\nExample: THINK: YES — source describes yellow skin and dull yellow eye\nExample: THINK: NO — no physical description — SEARCH: dull yellow eye yellow skin teeth",
    },
    {
      role: "user",
      content: `GROUND (round ${round}):\n${summary.slice(0, 1200)}\n\nFIGURE: ${query}\n\nDoes GROUND contain the specific evidence needed?`,
    },
  ], 40);
  const trimmed = result.trim();
  const sufficient = trimmed.toUpperCase().includes("THINK: YES");
  const reformulateMatch = trimmed.match(/SEARCH:\s*(.+)/i);
  return {
    sufficient,
    evidence: sufficient ? trimmed.replace(/^THINK:\s*YES\s*—?\s*/i, "").trim() : null,
    gap: sufficient ? null : trimmed,
    reformulation: reformulateMatch ? reformulateMatch[1].trim() : query,
    raw: trimmed,
  };
}

// ── Speak: generate answer from verified fold ──

export async function speak(query, summary, instruction) {
  const messages = [
    {
      role: "system",
      content: "You are a precise research assistant. The following source material contains verified evidence for the question. Answer directly from it using specific details and quotations. Do NOT cite file paths, line numbers, or source IDs — just answer naturally from the material.",
    },
    {
      role: "system",
      content: `--- Source material ---\n${summary || "(no relevant material found)"}`,
    },
  ];
  if (instruction) {
    messages[0].content += `\n\nFormat: ${instruction}`;
  }
  messages.push({ role: "user", content: query });
  return callModel(SPEAK_MODEL, messages, 1024);
}

// ── Plan: predict next step — given current state and goal, what reduces error most? ──

export async function plan(sessionId, log, query, priorPlan) {
  const recent = log.read({ session: sessionId }).slice(-20);
  const artifacts = log.read({ type: "craft", session: sessionId });
  const summary = [
    `Goal: ${query}`,
    `Completed steps: ${recent.filter(e => e.canon_satisfied).length}`,
    `Failed steps: ${recent.filter(e => e.canon_satisfied === false).length}`,
    `Artifacts produced: ${artifacts.length}`,
    ``,
    `Recent activity:`,
    ...recent.slice(-10).map(e => `  [${e.type}] satisfied=${e.canon_satisfied} issues=${(e.canon_issues||[]).join(";")}`),
    ``,
    priorPlan ? `Prior plan context:\n${priorPlan.slice(0, 500)}` : "No prior plan.",
  ].join("\n");

  const messages = [
    {
      role: "system",
      content: "You are a predictive next-step engine. Given a GOAL and the CURRENT STATE (recent work log), predict the single most impactful thing to do next. Respond with:\n\nNEXT_STEP: <one clear action> — what to build, decide, or generate next\nCANON: <what this output must satisfy> — the constraint that defines success for this step\nRATIONALE: <why this step reduces the gap between current state and goal>\n\nThis is not a plan — it's a prediction of the next error-reducing step. Be specific and actionable.",
    },
    {
      role: "user",
      content: `GOAL: ${query}\n\nCURRENT STATE:\n${summary.slice(0, 2500)}\n\nWhat is the single most impactful thing to do next?`,
    },
  ];
  return callModel(PLAN_MODEL, messages, 512);
}

// ── Evaluate: test an answer against its provenance ──

export async function evaluate(answer, evidence) {
  const messages = [
    {
      role: "system",
      content: "You are a forensic auditor. Given an answer and the evidence it claims to be based on, identify:\n1. FABRICATION — does the answer contain claims NOT supported by the evidence?\n2. POLARITY FLIP — does the answer reverse the meaning of the evidence?\n3. THESIS INJECTION — does the answer introduce claims from outside the evidence?\n4. CONSTRAINT VIOLATION — does the answer assert something the evidence doesn't establish?\n\nRespond with:\nPASS: YES / NO\nFINDINGS: <list of specific issues, or \"none\">\nDETAIL: <explanation>",
    },
    {
      role: "user",
      content: `EVIDENCE:\n${(evidence || "(no evidence)").slice(0, 2000)}\n\nANSWER:\n${(answer || "").slice(0, 2000)}\n\nDoes the answer follow from the evidence?`,
    },
  ];
  const result = await callModel(PLAN_MODEL, messages, 512);
  const passes = result.trim().toUpperCase().includes("PASS: YES");
  return { passes, findings: result.slice(0, 300), detail: result };
}

// ── Revise: restructure frame when evaluation breaks ──

export async function revise(sessionId, log, evalEntry, planText) {
  const recent = log.read({ session: sessionId }).slice(-20);
  const summary = recent.map(e => `[${e.type}] ${JSON.stringify(e).slice(0, 200)}`).join("\n");

  const messages = [
    {
      role: "system",
      content: "You are a structural revision engine. An evaluation has found a problem with the system's output. Your job is to propose what must change.\n\nEvaluate whether the failure is:\n- MATERIAL: the source doesn't contain what was needed (→ change search strategy, ingest more)\n- STRUCTURAL: the evidence was sufficient but was misinterpreted (→ change folding or thinking approach)\n- FRAME: the thesis or plan is wrong (→ restructure the plan)\n- EXPRESSIVE: the answer misrepresented correct evidence (→ change speak instructions)\n\nRespond with:\nFAILURE_TYPE: <material|structural|frame|expressive>\nREVISE_PLAN: <changes to the plan>\nREVISE_THESIS: <if the thesis/thesis must change>\nREVISE_QUERY: <if search must change>\nREVISE_PROMPT: <if model prompts must change>",
    },
    {
      role: "user",
      content: `Current plan:\n${(planText || "(none)").slice(0, 1000)}\n\nEvaluation findings:\n${(evalEntry.detail || evalEntry.findings || "(none)").slice(0, 1000)}\n\nRecent session:\n${summary.slice(0, 2000)}\n\nWhat must change to prevent this failure from recurring?`,
    },
  ];
  return callModel(PLAN_MODEL, messages, 512);
}
