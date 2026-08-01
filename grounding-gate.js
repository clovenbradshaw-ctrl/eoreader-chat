// Grounding gate — the app's mechanical contract for "every answer is
// grounded in the reader's sources, never the model's own knowledge".
//
// Host-side (app) concern: it builds the prompt the model answers from and
// decides, from the model's own output, whether that answer is grounded
// enough to serve. It changes no engine reading.
//
// Three functions, one invariant:
//   - buildGroundedSystemMessage: the system message a grounding result
//     becomes. Two shapes, never a third:
//       * a numbered SOURCE MATERIAL table → every factual claim must cite [N]
//       * no retrievable passage → a typed gap; the model is FORBIDDEN to
//         answer from its own knowledge (void-if-just-model starts here, by
//         never inviting the answer in the first place)
//   - validateCitations: mask out-of-range [N] so the reader never sees a
//     fabricated citation number.
//   - citedNumbers: count how many of the OFFERED citations the content
//     actually uses. A bracket that does not land on an available passage
//     grounds nothing — it is noise, and the gate counts it as zero.
//
// The gate itself: if maxCitation === 0 there was nothing to cite, so any
// model prose is MODEL-tier by construction and is voided. If citations were
// offered and the answer cites none of them, the model answered from its own
// knowledge — also voided. Either way the reader is served the typed gap,
// never the model's ungrounded prose.

// Mask fabricated or out-of-range [N] markers with a visible gap marker so a
// number the reader can never verify does not masquerade as a citation.
export function validateCitations(content, maxCitation) {
  if (!content || maxCitation <= 0) return content;
  return content.replace(/\[(\d+)\]/g, (match, numStr) => {
    const num = parseInt(numStr, 10);
    if (num >= 1 && num <= maxCitation) return match;
    return `[⊘ no source ${numStr}]`;
  });
}

// Which of the offered citation numbers does `content` actually use? Sorted,
// deduplicated, strictly within [1, maxCitation].
export function citedNumbers(content, maxCitation) {
  if (maxCitation <= 0) return [];
  const used = new Set();
  const re = /\[(\d+)\]/g;
  let m;
  while ((m = re.exec(content || "")) !== null) {
    const n = parseInt(m[1], 10);
    if (n >= 1 && n <= maxCitation) used.add(n);
  }
  return [...used].sort((a, b) => a - b);
}

// The system message a grounding result becomes. `warming` distinguishes the
// two no-evidence facts: index still loading, versus sources genuinely silent.
// `_citationCount` on the returned message is the gate's handle: the server
// counts how many of those the model actually cited after the turn.
export function buildGroundedSystemMessage(groundResult, query, warming = false) {
  if (!groundResult || !groundResult.context) {
    const content = warming
      ? `The document index is still loading, so no source passages are available for this question yet. ` +
        `Tell the reader the index is still warming up and offer to answer again in a moment. ` +
        `Do NOT answer from your own knowledge, and do NOT use bracketed citations like [1] — there are no passages to cite.`
      : `No passage in the reader's sources matches this question. You may use your tools to search for source passages, or fetch web pages that can be ingested into the reader and cited. ` +
        `If no citable passage can be found, say plainly in one sentence that nothing in the reader's sources answers this question. ` +
        `Do NOT answer from your own knowledge, and do NOT use bracketed citations like [1], [2] — there is no numbered SOURCE MATERIAL table.`;
    return { message: { role: "system", content, _citationCount: 0 }, warming };
  }
  const citationRange = groundResult.citations.length > 0
    ? `You have ${groundResult.citations.length} source passage(s) numbered [1] through [${groundResult.citations.length}]. ` +
      `ONLY cite these numbers. NEVER cite [${groundResult.citations.length + 1}] or higher — those do not exist. `
    : "";
  const content =
    `You are answering a question grounded in SOURCE MATERIAL below. ` +
    citationRange +
    `Cite specific passages using bracketed numbers like [1], [2], etc. ` +
    `Do NOT invent facts beyond what the sources contain. If the sources ` +
    `do not contain the answer, say so honestly.\n\n` +
    `IMPORTANT: You have access to tools. If the source material above is ` +
    `insufficient, use verbatim_search to find more exact passages from ` +
    `ingested documents, or search_memory for relevant context. Do NOT say ` +
    `"no information" without first trying these tools.\n\n` +
    `--- Source material (${groundResult.total} passages found, ${groundResult.folded} folded, ${groundResult.tokens} tokens) ---\n` +
    `${groundResult.context}`;
  return { message: { role: "system", content, _citationCount: groundResult.citations.length }, warming: false };
}

// The typed gap served in place of a voided, model-only answer.
export const voidedAnswer = (reason) =>
  `⊘ Answer voided — not grounded in your sources.\n\n${reason}`;
