import * as log from "./log.js";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const CANON_CHECKER = process.env.CANON_CHECKER || "gemma2:2b";

async function checkCanon(output, canon) {
  if (!canon || canon.trim().length === 0) {
    return { satisfied: true, issues: [] };
  }
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: CANON_CHECKER,
      messages: [
        {
          role: "system",
          content: "You are a canon enforcer. Check if the output satisfies the canon. Respond:\nCANON: YES — confirmation\nCANON: NO — specific issue\n\nNothing else.",
        },
        {
          role: "user",
          content: `CANON:\n${canon}\n\nOUTPUT:\n${(output || "").slice(0, 3000)}\n\nDoes the output satisfy the canon?`,
        },
      ],
      stream: false,
      options: { temperature: 0.1, num_predict: 40 },
    }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) return { satisfied: false, issues: ["canon checker unavailable"] };
  const data = await res.json();
  const verdict = (data.message?.content || "").trim();
  const satisfied = verdict.toUpperCase().includes("CANON: YES");
  const issueMatch = verdict.match(/CANON:\s*(NO|YES)\s*—\s*(.+)/i);
  const issues = satisfied ? [] : [issueMatch ? issueMatch[2] : verdict];
  return { satisfied, issues };
}

export function withCanon(handler, maxRetries = 2) {
  return async (args) => {
    const canon = args.canon || "";
    const session = args.session || "default";
    const turn = `turn:${Date.now()}`;

    // Log the canon constraint — system addressable
    const canonEntry = log.write({
      type: "canon", layer: "meta", session, turn,
      tool: handler.name || "anonymous",
      canon_text: canon.slice(0, 500),
      max_retries: maxRetries,
    });

    let currentArgs = { ...args };
    let lastIssues = [];
    let firstResult = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      // Log the attempt
      log.write({
        type: "canon_attempt", layer: "meta", session, turn,
        parent: canonEntry.id,
        attempt: attempt + 1,
        spec_preview: (currentArgs.spec || currentArgs.query || "").slice(0, 200),
      });

      const result = await handler(currentArgs);
      if (attempt === 0) firstResult = result;

      // Extract the output to check — prefer 'preview' from parsed JSON, fall back to raw text
      const rawText = result.content?.[0]?.text || "";
      let outputToCheck = rawText;
      try {
        const parsed = JSON.parse(rawText);
        if (parsed.preview) outputToCheck = parsed.preview;
      } catch {}

      const check = await checkCanon(outputToCheck, canon);
      lastIssues = check.issues;

      // Log the check result — system addressable
      log.write({
        type: "canon_check", layer: "meta", session, turn,
        parent: canonEntry.id,
        attempt: attempt + 1,
        passed: check.satisfied,
        issues: check.issues,
      });

      result.canon_id = canonEntry.id;
      result.canon = canon;
      result.canon_satisfied = check.satisfied;
      result.canon_issues = check.issues;
      result.canon_attempts = attempt + 1;

      if (check.satisfied) return result;

      // Retry: append canon failure to the spec for self-correction
      const specKey = Object.keys(currentArgs).find(k =>
        k === "spec" || k === "query" || k === "summary" || k === "ask"
      );
      if (specKey) {
        currentArgs[specKey] +=
          `\n\n[CANON REVIEW: ${check.issues.join("; ")} — revise to satisfy the canon]`;
      }
    }

    // All retries exhausted — log the failure
    log.write({
      type: "canon_exhausted", layer: "meta", session, turn,
      parent: canonEntry.id,
      issues: lastIssues,
      max_retries: maxRetries,
    });

    if (firstResult) {
      firstResult.canon_id = canonEntry.id;
      firstResult.canon = canon;
      firstResult.canon_satisfied = false;
      firstResult.canon_issues = lastIssues;
      firstResult.canon_attempts = maxRetries + 1;
      return firstResult;
    }

    return {
      content: [{ type: "text", text: lastIssues.join("; ") || "canon check failed after retries" }],
      canon_id: canonEntry.id,
      canon,
      canon_satisfied: false,
      canon_issues: lastIssues,
      canon_attempts: maxRetries + 1,
    };
  };
}

export const NO_CANON = new Set(["ingest", "scout", "cite", "compile", "fold"]);
