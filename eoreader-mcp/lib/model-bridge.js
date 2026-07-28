const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const THINK_MODEL = process.env.THINK_MODEL || "gemma2:2b";
const SPEAK_MODEL = process.env.SPEAK_MODEL || "llama3.2";
const PLAN_MODEL = process.env.PLAN_MODEL || SPEAK_MODEL;

export async function callModel(model, messages, maxTokens = 512, opts = {}) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: { temperature: opts.temperature ?? 0.7, num_predict: maxTokens },
    }),
    signal: AbortSignal.timeout(300000),
  });
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.message?.content || "";
}

/**
 * callModelStream — async generator yielding tokens from Ollama's streaming API.
 * Each yield is a string token. The caller iterates with `for await`.
 */
export async function* callModelStream(model, messages, maxTokens = 512, opts = {}) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
      options: { temperature: opts.temperature ?? 0.7, num_predict: maxTokens },
    }),
    signal: AbortSignal.timeout(300000),
  });
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.message?.content) yield obj.message.content;
      } catch {}
    }
  }
  if (buf.trim()) {
    try {
      const obj = JSON.parse(buf);
      if (obj.message?.content) yield obj.message.content;
    } catch {}
  }
}

/**
 * Build a conversation context prefix from chat history.
 * Returns null if no context, or a system message to prepend.
 */
function conversationPrefix(chatContext) {
  if (!chatContext || chatContext.length === 0) return null;
  const lines = chatContext.map(m => {
    const label = m.role === "user" ? "User" : m.role === "assistant" ? "Assistant" : "System";
    return `${label}: ${m.content.slice(0, 400)}`;
  });
  return {
    role: "system",
    content: `[Conversation so far]\n${lines.join("\n")}`,
  };
}

// ── Think: verify sufficiency, suggest reformulation ──

export async function think(query, summary, round = 1, chatContext = null) {
  if (!summary || summary.length < 50) {
    return { sufficient: false, evidence: null, gap: "no source material", reformulation: query };
  }
  const messages = [
    {
      role: "system",
      content: "GROUND: source material below\nFIGURE: the question\nPATTERN: decide if GROUND contains specific evidence to answer FIGURE\n\nIF YES: respond: THINK: YES — <specific evidence found>\nIF NO: respond: THINK: NO — <gap> — SEARCH: <reformulated query>\n\nExample: THINK: YES — source describes yellow skin and dull yellow eye\nExample: THINK: NO — no physical description — SEARCH: dull yellow eye yellow skin teeth",
    },
  ];

  // Inject conversation context so think understands what the user is actually after
  const prefix = conversationPrefix(chatContext);
  if (prefix) messages.push(prefix);

  messages.push({
    role: "user",
    content: `GROUND (round ${round}):\n${summary.slice(0, 1200)}\n\nFIGURE: ${query}\n\nDoes GROUND contain the specific evidence needed?`,
  });

  const result = await callModel(THINK_MODEL, messages, 40);
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
// opts.onToken(token) — called for each token when streaming is desired.
// When onToken is provided, uses streaming and returns the full accumulated text.

export async function speak(query, summary, instruction, chatContext = null, opts = {}) {
  const messages = [
    {
      role: "system",
      content: "You are a precise research assistant. The following source material contains verified evidence for the question. Answer directly from it using specific details and quotations. Do NOT cite file paths, line numbers, or source IDs — just answer naturally from the material. If there is conversation context, maintain continuity with what was previously discussed.",
    },
    {
      role: "system",
      content: `--- Source material ---\n${summary || "(no relevant material found)"}`,
    },
  ];

  // Inject conversation context for continuity
  const prefix = conversationPrefix(chatContext);
  if (prefix) messages.push(prefix);

  if (instruction) {
    messages[0].content += `\n\nFormat: ${instruction}`;
  }
  messages.push({ role: "user", content: query });

  if (opts.onToken) {
    let full = "";
    for await (const token of callModelStream(SPEAK_MODEL, messages, 1024, opts)) {
      full += token;
      opts.onToken(token);
    }
    return full;
  }
  return callModel(SPEAK_MODEL, messages, 1024, opts);
}

// ── Plan: predict next step — given current state and goal, what reduces error most? ──

export async function plan(sessionId, log, query, priorPlan, chatContext = null) {
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
  ];

  const prefix = conversationPrefix(chatContext);
  if (prefix) messages.push(prefix);

  messages.push({
    role: "user",
    content: `GOAL: ${query}\n\nCURRENT STATE:\n${summary.slice(0, 2500)}\n\nWhat is the single most impactful thing to do next?`,
  });

  return callModel(PLAN_MODEL, messages, 512);
}

// ── Evaluate: test an answer against its provenance ──

export async function evaluate(answer, evidence, chatContext = null) {
  const messages = [
    {
      role: "system",
      content: "You are a forensic auditor. Given an answer and the evidence it claims to be based on, identify:\n1. FABRICATION — does the answer contain claims NOT supported by the evidence?\n2. POLARITY FLIP — does the answer reverse the meaning of the evidence?\n3. THESIS INJECTION — does the answer introduce claims from outside the evidence?\n4. CONSTRAINT VIOLATION — does the answer assert something the evidence doesn't establish?\n\nRespond with:\nPASS: YES / NO\nFINDINGS: <list of specific issues, or \"none\">\nDETAIL: <explanation>",
    },
  ];

  const prefix = conversationPrefix(chatContext);
  if (prefix) messages.push(prefix);

  messages.push({
    role: "user",
    content: `EVIDENCE:\n${(evidence || "(no evidence)").slice(0, 2000)}\n\nANSWER:\n${(answer || "").slice(0, 2000)}\n\nDoes the answer follow from the evidence?`,
  });

  const result = await callModel(PLAN_MODEL, messages, 512);
  const passes = result.trim().toUpperCase().includes("PASS: YES");
  return { passes, findings: result.slice(0, 300), detail: result };
}

// ── Revise: restructure frame when evaluation breaks ──

export async function revise(sessionId, log, evalEntry, planText, chatContext = null) {
  const recent = log.read({ session: sessionId }).slice(-20);
  const summary = recent.map(e => `[${e.type}] ${JSON.stringify(e).slice(0, 200)}`).join("\n");

  const messages = [
    {
      role: "system",
      content: "You are a structural revision engine. An evaluation has found a problem with the system's output. Your job is to propose what must change.\n\nEvaluate whether the failure is:\n- MATERIAL: the source doesn't contain what was needed (→ change search strategy, ingest more)\n- STRUCTURAL: the evidence was sufficient but was misinterpreted (→ change folding or thinking approach)\n- FRAME: the thesis or plan is wrong (→ restructure the plan)\n- EXPRESSIVE: the answer misrepresented correct evidence (→ change speak instructions)\n\nRespond with:\nFAILURE_TYPE: <material|structural|frame|expressive>\nREVISE_PLAN: <changes to the plan>\nREVISE_THESIS: <if the thesis/thesis must change>\nREVISE_QUERY: <if search must change>\nREVISE_PROMPT: <if model prompts must change>",
    },
  ];

  const prefix = conversationPrefix(chatContext);
  if (prefix) messages.push(prefix);

  messages.push({
    role: "user",
    content: `Current plan:\n${(planText || "(none)").slice(0, 1000)}\n\nEvaluation findings:\n${(evalEntry.detail || evalEntry.findings || "(none)").slice(0, 1000)}\n\nRecent session:\n${summary.slice(0, 2000)}\n\nWhat must change to prevent this failure from recurring?`,
  });

  return callModel(PLAN_MODEL, messages, 512);
}

// ── Chat: full conversational turn with rolling context ──

const CHAT_SYSTEM_PROMPT = `You are EO, a focused research and analysis assistant. You answer questions directly from source material when available, and reason clearly when it is not. You do not fabricate citations or invent facts. When you lack information, say so. Keep responses concise unless detail is specifically requested.`;

/**
 * Handle a conversational turn with rolling context.
 * Records the exchange in chat history and returns the response.
 *
 * When sourceContext is provided, the model answers from actual ingested
 * material, not from training data. The proxy is the grounding layer.
 *
 * @param {string} sessionId
 * @param {string} userMessage
 * @param {object} chatHistory - from ./chat-history.js
 * @param {object} [opts] - { onToken(token), sourceContext(string) }
 * @returns {{ response: string, stats: object, grounded: boolean, citations: Array }}
 */
export async function chatTurn(sessionId, userMessage, chatHistory, opts = {}) {
  // Record user message (may trigger fold)
  const afterUser = chatHistory.addMessage(sessionId, "user", userMessage);

  // Get conversation context for the model
  const context = chatHistory.getContext(sessionId, CHAT_SYSTEM_PROMPT);

  // If the proxy found relevant source material, inject it so the model
  // answers from actual content, not from training data.
  if (opts.sourceContext) {
    context.push({
      role: "system",
      content: `The following source material was found in memory for this question. Answer ONLY from this material. If it does not contain the answer, say so. Do not add information beyond what is stated below.\n\n--- Source material ---\n${opts.sourceContext}`,
    });
  }

  context.push({ role: "user", content: userMessage });

  let response;
  if (opts.onToken) {
    response = "";
    for await (const token of callModelStream(SPEAK_MODEL, context, 1024, opts)) {
      response += token;
      opts.onToken(token);
    }
  } else {
    response = await callModel(SPEAK_MODEL, context, 1024);
  }

  // Record assistant response (may trigger fold)
  const afterAssistant = chatHistory.addMessage(sessionId, "assistant", response);

  return {
    response,
    stats: {
      ...afterAssistant,
      ...chatHistory.getSessionStats(sessionId),
    },
  };
}
