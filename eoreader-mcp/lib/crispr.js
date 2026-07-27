import zlib from "zlib";
import { callModel } from "./model-bridge.js";

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

function escRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function extractSlots(html) {
  const slots = [];

  const logo = html.match(/<div class="logo">([\s\S]*?)<\/div>/);
  if (logo) slots.push({ name: "logo", golden: logo[1], desc: "Site logo" });

  const search = html.match(/placeholder="([^"]*)"/);
  if (search) slots.push({ name: "search", golden: search[1], desc: "Search placeholder" });

  const titles = [...html.matchAll(/<div class="post-title">([\s\S]*?)<\/div>/g)];
  titles.forEach((m, i) => slots.push({ name: "title-" + i, golden: m[1], desc: "Post title " + (i + 1) }));

  const authors = [...html.matchAll(/<span class="author">([\s\S]*?)<\/span>/g)];
  authors.forEach((m, i) => slots.push({ name: "author-" + i, golden: m[1], desc: "Post author " + (i + 1) }));

  const domains = [...html.matchAll(/<div class="post-domain">([\s\S]*?)<\/div>/g)];
  domains.forEach((m, i) => slots.push({ name: "domain-" + i, golden: m[1], desc: "Post domain " + (i + 1) }));

  const votes = [...html.matchAll(/<span class="vote-count">([\s\S]*?)<\/span>/g)];
  votes.forEach((m, i) => slots.push({ name: "votes-" + i, golden: m[1], desc: "Vote count " + (i + 1) }));

  const about = html.match(/<p>(The front page[\s\S]*?)<\/p>/);
  if (about) slots.push({ name: "about", golden: about[1], desc: "Community description" });

  const stats = [...html.matchAll(/<span class="stat-value">([\s\S]*?)<\/span>/g)];
  stats.forEach((m, i) => slots.push({ name: "stat-" + i, golden: m[1], desc: "Stat " + (i + 1) }));

  const trends = [...html.matchAll(/<div class="trend-title">([\s\S]*?)<\/div>/g)];
  trends.forEach((m, i) => slots.push({ name: "trend-" + i, golden: m[1], desc: "Trending " + (i + 1) }));

  const trendMeta = [...html.matchAll(/<div class="trend-meta">([\s\S]*?)<\/div>/g)];
  trendMeta.forEach((m, i) => slots.push({ name: "trend-meta-" + i, golden: m[1], desc: "Trending meta " + (i + 1) }));

  const footer = html.match(/<div class="footer">([\s\S]*?)<\/div>/);
  if (footer) slots.push({ name: "footer", golden: footer[1], desc: "Footer" });

  return slots;
}

export async function fillSlot(slot, n) {
  const prompt = `Generate content for "${slot.name}" (${slot.desc}) for a Reddit-style page about dolphins called DolphinZone.
The golden reference has: "${slot.golden.slice(0, 80)}"
Generate a similar variation. Output ONLY the raw text. No HTML, no quotes, no markdown.`;

  const candidates = [];
  for (let i = 0; i < n; i++) {
    try {
      const text = await callModel([
        { role: "system", content: "Output ONLY the raw text. No HTML, no quotes, no markdown, no explanation." },
        { role: "user", content: prompt },
      ], 200);
      candidates.push({ text: text.trim(), index: i });
    } catch (err) {
      candidates.push({ text: slot.golden, index: i, error: err.message });
    }
  }

  return candidates
    .map(c => ({
      ...c,
      omit: residualA(c.text, slot.golden),
      inv: residualB(c.text, slot.golden),
    }))
    .sort((a, b) => (a.omit + a.inv) - (b.omit + b.inv));
}

export function applySlot(html, slot, value) {
  const name = slot.name;
  if (name === "logo") return html.replace(/<div class="logo">[\s\S]*?<\/div>/, '<div class="logo">' + value + '</div>');
  if (name === "search") return html.replace(/placeholder="[^"]*"/, 'placeholder="' + value + '"');
  if (name.startsWith("title-")) return html.replace(new RegExp('<div class="post-title">' + escRegex(slot.golden) + '<\\/div>'), '<div class="post-title">' + value + '</div>');
  if (name.startsWith("author-")) return html.replace(new RegExp('<span class="author">' + escRegex(slot.golden) + '<\\/span>'), '<span class="author">' + value + '</span>');
  if (name.startsWith("domain-")) return html.replace(new RegExp('<div class="post-domain">' + escRegex(slot.golden) + '<\\/div>'), '<div class="post-domain">' + value + '</div>');
  if (name.startsWith("votes-")) return html.replace(new RegExp('<span class="vote-count">' + escRegex(slot.golden) + '<\\/span>'), '<span class="vote-count">' + value + '</span>');
  if (name === "about") return html.replace(/<p>The front page[\s\S]*?<\/p>/, '<p>' + value + '</p>');
  if (name.startsWith("stat-")) return html.replace(new RegExp('<span class="stat-value">' + escRegex(slot.golden) + '<\\/span>'), '<span class="stat-value">' + value + '</span>');
  if (name.startsWith("trend-") && !name.includes("meta")) return html.replace(new RegExp('<div class="trend-title">' + escRegex(slot.golden) + '<\\/div>'), '<div class="trend-title">' + value + '</div>');
  if (name.startsWith("trend-meta")) return html.replace(new RegExp('<div class="trend-meta">' + escRegex(slot.golden) + '<\\/div>'), '<div class="trend-meta">' + value + '</div>');
  if (name === "footer") return html.replace(/<div class="footer">[\s\S]*?<\/div>/, '<div class="footer">' + value + '</div>');
  return html;
}
