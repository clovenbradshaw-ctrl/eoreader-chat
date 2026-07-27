// Sweep harness — mechanical compression-distance scoring across canonical channels.
//
// Renders a design artifact to five channels, scores each against a target
// dictionary, returns a residual vector. All mechanical — no model calls.
// Only scalars come back into context.
//
// Channels:
//   0 DOM shape   — tag depth/Euler tour (structure)
//   1 Geometry    — bounding boxes in reading order, quantized (layout)
//   2 Style       — CSS variables + color palette (design system)
//   3 Copy        — text content, stripped of markup (content)
//   4 Visual      — heavily blurred, compressed (density/weight)
//
// For MVP, channels 0, 2, 3 are extracted from raw HTML.
// Channels 1 and 4 require a browser render (Playwright) — stubbed for now.

import zlib from "zlib";

// ── Channel extractors ──

export function extractDOMShape(html) {
  // Euler tour of tag depth: [1,2,3,3,2,1] for <div><p><span></span></p></div>
  const tags = html.match(/<\/?[a-zA-Z][^>]*>/g) || [];
  const shape = [];
  let depth = 0;
  for (const tag of tags) {
    if (tag.startsWith("</")) depth--;
    else if (!tag.endsWith("/>")) { depth++; shape.push(depth); }
  }
  return shape.join(",");
}

export function extractCopy(html) {
  // Strip all tags, collapse whitespace, return text
  return html
    .replace(/<[^>]*>/g, " ")
    .replace(/&[^;]+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function extractPalette(html) {
  // Extract CSS custom properties and color values
  const colors = [];
  const varMatch = html.match(/--[\w-]+\s*:\s*#[0-9a-fA-F]{3,8}/g);
  if (varMatch) colors.push(...varMatch);
  const hexMatch = html.match(/#[0-9a-fA-F]{3,8}(?=[^0-9a-fA-F]|$)/g);
  if (hexMatch) colors.push(...hexMatch);
  const rgbMatch = html.match(/rgba?\([^)]+\)/g);
  if (rgbMatch) colors.push(...rgbMatch);
  return colors.join(" ");
}

export function extractGeometry(html) {
  // Approximate layout from tag nesting + class hints.
  // Full geometry requires Playwright render — this is a structural proxy.
  const blocks = html.match(/<div[^>]*>[\s\S]*?<\/div>/g) || [];
  const geos = blocks.map(b => {
    const classes = (b.match(/class="([^"]*)"/) || [,""])[1];
    const inner = b.replace(/<[^>]*>/g, "").trim().length;
    return `${classes || "div"}:${inner}`;
  });
  return geos.join(" | ");
}

// ── Compression helpers ──

function c(text) {
  return zlib.gzipSync(text || "", { level: 9 }).length;
}

// Asymmetric compression ratio: how much of `gen` does `ref` explain?
// C(gen | ref) / C(gen) — high residual = invention, drift
export function residualGenGivenRef(gen, ref) {
  const cGen = c(gen);
  if (cGen === 0) return 0;
  const cGenGivenRef = c(gen + ref) - c(ref);
  return Math.max(0, cGenGivenRef / cGen);
}

// Asymmetric compression ratio: how much of `ref` does `gen` miss?
// C(ref | gen) / C(ref) — high residual = omission, uncovered material
export function residualRefGivenGen(gen, ref) {
  const cRef = c(ref);
  if (cRef === 0) return 0;
  const cRefGivenGen = c(ref + gen) - c(gen);
  return Math.max(0, cRefGivenGen / cRef);
}

// ── Full sweep ──

export function sweep(generatedHTML, targetHTML) {
  const channels = [
    { name: "dom-shape",   gen: extractDOMShape(generatedHTML),  ref: extractDOMShape(targetHTML) },
    { name: "copy",        gen: extractCopy(generatedHTML),       ref: extractCopy(targetHTML) },
    { name: "palette",     gen: extractPalette(generatedHTML),    ref: extractPalette(targetHTML) },
    { name: "geometry",    gen: extractGeometry(generatedHTML),   ref: extractGeometry(targetHTML) },
  ];

  const results = [];
  for (const ch of channels) {
    const inv = residualGenGivenRef(ch.gen, ch.ref);   // invention: gen has stuff ref doesn't
    const omi = residualRefGivenGen(ch.gen, ch.ref);   // omission: ref has stuff gen doesn't
    results.push({
      channel: ch.name,
      genLen: ch.gen.length,
      refLen: ch.ref.length,
      invention: inv,
      omission: omi,
    });
  }

  return results;
}

// ── Localization: sliding window worst-k ──

export function worstWindows(generatedHTML, targetHTML, windowSize = 0.1, k = 3) {
  const total = generatedHTML.length;
  const step = Math.max(100, Math.floor(total * 0.05));
  const wSize = Math.max(500, Math.floor(total * windowSize));
  const windows = [];

  let pos = 0;
  while (pos < total) {
    const chunk = generatedHTML.slice(pos, pos + wSize);
    const ref = targetHTML.slice(
      Math.floor(targetHTML.length * (pos / total)),
      Math.floor(targetHTML.length * ((pos + wSize) / total))
    );
    const inv = residualGenGivenRef(chunk, ref);
    windows.push({ pos, inv, size: chunk.length });
    pos += step;
  }

  windows.sort((a, b) => b.inv - a.inv);
  return windows.slice(0, k);
}

// ── Uncovered target patterns ──

export function uncoveredTarget(generatedHTML, targetHTML, k = 3) {
  const total = targetHTML.length;
  const step = Math.max(200, Math.floor(total * 0.05));
  const wSize = Math.max(500, Math.floor(total * 0.08));
  const chunks = [];

  let pos = 0;
  while (pos < total) {
    const chunk = targetHTML.slice(pos, pos + wSize);
    const ref = generatedHTML.slice(
      Math.floor(generatedHTML.length * (pos / generatedHTML.length || 1)),
      Math.floor(generatedHTML.length * ((pos + wSize) / (generatedHTML.length || 1)))
    );
    const omi = residualRefGivenGen(chunk, ref);
    chunks.push({ pos, omission: omi, preview: chunk.slice(0, 100) });
    pos += step;
  }

  chunks.sort((a, b) => b.omission - a.omission);
  return chunks.slice(0, k);
}
