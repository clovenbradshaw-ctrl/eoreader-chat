import fs from "fs";
import zlib from "zlib";
import { callModel } from "./lib/model-bridge.js";

function gzip(text) {
  return zlib.gzipSync(text || "", { level: 9 }).length;
}

function residualA(gen, ref) {
  const cr = gzip(ref);
  return cr === 0 ? 0 : Math.max(0, (gzip(ref + gen) - gzip(gen)) / cr);
}

function residualB(gen, ref) {
  const cg = gzip(gen);
  return cg === 0 ? 0 : Math.max(0, (gzip(gen + ref) - gzip(ref)) / cg);
}

const golden = fs.readFileSync("/tmp/dolphinzone-golden.html", "utf8");
const goldenScore = residualA("", golden);

// ── PHASE 1: Whole-page generation (the old approach) ──
console.log("=== PHASE 1: WHOLE-PAGE GENERATION (5 attempts) ===");
for (let i = 0; i < 5; i++) {
  const result = await callModel("qwen2.5-coder:7b", [
    { role: "system", content: "You generate HTML. Output ONLY raw HTML. No markdown fences. No explanation." },
    { role: "user", content: "Generate a Reddit-style front page HTML called DolphinZone about dolphins. Blue theme #006994. Header with logo and search bar. Feed with 6 post cards (upvote/downvote, title, author, domain, comments/share/save actions, emoji thumbnail). Sidebar with community info, rules, trending. Footer. Include full embedded CSS matching Reddit's layout. Be specific with real-sounding content." },
  ], 4096);
  const score = { omit: residualA(result, golden), inv: residualB(result, golden) };
  fs.writeFileSync("/tmp/whole-page-" + i + ".html", result, "utf8");
  console.log("  #" + i + ": " + result.length + " chars, omit=" + score.omit.toFixed(3) + " inv=" + score.inv.toFixed(3));
}

// ── PHASE 2: CRISPR — skeleton + slot fills (no golden shown) ──
console.log("\n=== PHASE 2: CRISPR (generate content for each slot) ===");

// Load the golden's CSS and structure as a template
const skeleton = golden; // We start from the golden layout
const slots = [
  { name: "logo", prompt: "Site logo/brand name (HTML allowed, e.g. Dolphin<span>Zone</span>)", golden: "Dolphin<span>Zone</span>" },
  { name: "search", prompt: "Search bar placeholder text", golden: "Search DolphinZone" },
  { name: "title-0", prompt: "Reddit post card 1 title about dolphins", golden: "New calf spotted in Monterey Bay — researchers confirm it's a bottlenose born to the Pod 7 matriarch" },
  { name: "title-1", prompt: "Reddit post card 2 title about dolphins", golden: "Dolphin intelligence study results: bottlenose dolphins pass the mirror self-recognition test at higher rates than previously thought" },
  { name: "title-2", prompt: "Reddit post card 3 title about dolphins", golden: "How dolphins communicate across miles: new acoustic mapping reveals long-range signature whistles travel 600+ km underwater" },
  { name: "title-3", prompt: "Reddit post card 4 title about dolphins", golden: "TIL dolphins have names for each other — they use unique signature whistles that function like names, and other dolphins respond when called" },
  { name: "title-4", prompt: "Reddit post card 5 title about dolphins", golden: "Caught this photo of a spinner dolphin doing a triple spin at sunset off the coast of Hawaii — my best shot ever" },
  { name: "title-5", prompt: "Reddit post card 6 title about dolphins", golden: "Florida passes strongest dolphin protection act in US history — 200-meter no-wake zone, mandatory rescue corridors, and criminal penalties for harassment" },
  { name: "about", prompt: "Community description (1 sentence about what DolphinZone is)", golden: "The front page of the pod. Everything about dolphins — research, photos, conservation, and stories from the ocean." },
  { name: "footer", prompt: "Footer text", golden: "DolphinZone &copy; 2024 &middot; All rights reserved &middot; Made with love for the ocean" },
];

let current = skeleton;
for (const slot of slots) {
  const result = await callModel("qwen2.5-coder:7b", [
    { role: "system", content: "Output ONLY the raw text. No HTML, no quotes, no markdown." },
    { role: "user", content: "Generate: " + slot.prompt + ". For a Reddit-style page called DolphinZone about dolphins." },
  ], 200);
  const text = result.trim();

  // Apply to HTML
  const escaped = slot.golden.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if (slot.name === "logo") {
    current = current.replace(/<div class="logo">[\s\S]*?<\/div>/, '<div class="logo">' + text + '</div>');
  } else if (slot.name === "search") {
    current = current.replace(/placeholder="[^"]*"/, 'placeholder="' + text + '"');
  } else if (slot.name === "about") {
    current = current.replace(/<p>The front page[\s\S]*?<\/p>/, '<p>' + text + '</p>');
  } else if (slot.name === "footer") {
    current = current.replace(/<div class="footer">[\s\S]*?<\/div>/, '<div class="footer">' + text + '</div>');
  } else if (slot.name.startsWith("title-")) {
    current = current.replace(new RegExp('<div class="post-title">' + escaped + '<\\/div>'), '<div class="post-title">' + text + '</div>');
  }

  const omit = residualA(current, golden);
  const inv = residualB(current, golden);
  console.log("  " + slot.name + ': "' + text.slice(0, 50) + '" → omit=' + omit.toFixed(3));
}

// ── PHASE 3: Compare ──
console.log("\n=== COMPARISON ===");
const wholeScores = [];
for (let i = 0; i < 5; i++) {
  const content = fs.readFileSync("/tmp/whole-page-" + i + ".html", "utf8");
  wholeScores.push({ omit: residualA(content, golden), inv: residualB(content, golden), len: content.length });
}

const bestWhole = wholeScores.reduce((a, b) => a.omit < b.omit ? a : b);
const crisprScore = { omit: residualA(current, golden), inv: residualB(current, golden), len: current.length };

console.log("Whole-page (best of 5):");
console.log("  Omission:", bestWhole.omit.toFixed(3), " Invention:", bestWhole.inv.toFixed(3), " Size:", bestWhole.len);
console.log("CRISPR (slot fills):");
console.log("  Omission:", crisprScore.omit.toFixed(3), " Invention:", crisprScore.inv.toFixed(3), " Size:", crisprScore.len);
console.log("Golden:", golden.length, "chars");

fs.writeFileSync("/tmp/crispr-blank.html", current, "utf8");
console.log("\nCRISPR result saved to /tmp/crispr-blank.html");
