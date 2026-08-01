#!/usr/bin/env node
// Which local model should answer a grounded reading question?
//
// Routing currently optimizes for "did the tool loop finish cleanly", which a
// code model wins without ever being a good reading companion. This measures
// the thing we actually want: given real engine passages, does the model answer
// the question and cite the passages it was handed — and how fast.

const OLLAMA = process.env.OLLAMA || "http://localhost:11434";
const PROXY = process.env.PROXY || "http://localhost:11435";
const MODELS = process.argv.slice(2);
if (!MODELS.length) {
  console.error("usage: node probe-models.mjs <model> [model...]");
  process.exit(1);
}

const QUESTIONS = [
  "What does Victor see when the creature first opens its eye?",
  "How does Pierre Bezukhov come into his inheritance?",
  "What happens on the dreary night of November?",
];

// Grounding comes from the engine, identical for every model, so the only
// variable is the model's use of it.
async function ground(q) {
  const res = await fetch(
    `${PROXY}/api/verbatim?q=${encodeURIComponent(q)}&limit=8`,
    { signal: AbortSignal.timeout(60000) }
  );
  const passages = (await res.json()).passages || [];
  const block = passages
    .map((p, i) => `[${i + 1}] (${(p.source || "").replace(/^.*\//, "")})\n${(p.text || "").slice(0, 700)}`)
    .join("\n\n");
  return { block, count: passages.length };
}

for (const model of MODELS) {
  let totalMs = 0, cited = 0, refused = 0, answered = 0;
  const samples = [];

  for (const q of QUESTIONS) {
    const { block, count } = await ground(q);
    const system =
      `You are answering a question grounded in SOURCE MATERIAL below. ` +
      `You have ${count} source passage(s) numbered [1] through [${count}]. ONLY cite these numbers. ` +
      `Cite specific passages using bracketed numbers like [1], [2]. ` +
      `Do NOT invent facts beyond what the sources contain.\n\n--- Source material ---\n${block}`;

    const t0 = performance.now();
    let text = "";
    try {
      const res = await fetch(`${OLLAMA}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          stream: false,
          messages: [{ role: "system", content: system }, { role: "user", content: q }],
        }),
        signal: AbortSignal.timeout(180000),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      text = data.message?.content || "";
    } catch (err) {
      console.log(`  ${model}: ERROR ${err.message}`);
      continue;
    }
    const ms = performance.now() - t0;
    totalMs += ms;
    answered++;

    const hasCitation = /\[\d+\]/.test(text);
    // The failure we are hunting: a refusal issued while holding the evidence.
    const isRefusal = /do(es)? not contain|cannot provide|no information|not mentioned|unable to answer/i.test(text);
    if (hasCitation) cited++;
    if (isRefusal) refused++;
    samples.push(`      ${hasCitation ? "cite" : "----"} ${isRefusal ? "REFUSED" : "answered"} ${Math.round(ms)}ms :: ${text.replace(/\s+/g, " ").slice(0, 110)}`);
  }

  console.log(`\n${model}`);
  console.log(`  cited ${cited}/${answered}   refused ${refused}/${answered}   mean ${Math.round(totalMs / Math.max(1, answered))}ms`);
  samples.forEach((s) => console.log(s));
}
