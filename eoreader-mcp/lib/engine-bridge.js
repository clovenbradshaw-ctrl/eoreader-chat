import { createState, applyCommand } from "../../../eoreader5/packages/engine/replay/index.js";
import { search } from "../../../eoreader5/packages/engine/search/index.js";
import { fold as compressFold } from "../../../eoreader5/packages/engine/emergence/fold/index.js";
import { blockContentHash } from "../../../eoreader5/packages/engine/observation-index.js";
import { canonicalHashSync } from "../../../eoreader5/packages/spec/canonical-json/index.js";
import { CURRENT_OPERATOR_EPOCH } from "../../../eoreader5/packages/spec/operators/epoch.js";
import fs from "fs";
import { execSync } from "child_process";
import * as log from "./log.js";

// ── Binary content detection and extraction ──

const AUDIO_EXTS = new Set(["mp3","wav","ogg","flac","aac","m4a","wma","opus"]);
const VIDEO_EXTS = new Set(["mp4","mkv","avi","mov","webm","flv","wmv","m4v"]);
const IMAGE_EXTS = new Set(["jpg","jpeg","png","gif","bmp","tiff","webp","heic"]);
const BINARY_EXTS = new Set([...AUDIO_EXTS, ...VIDEO_EXTS, ...IMAGE_EXTS]);

function extOf(filePath) {
  return filePath.split(".").pop().toLowerCase();
}

function isBinary(filePath) {
  return BINARY_EXTS.has(extOf(filePath));
}

function runFfprobe(filePath) {
  try {
    const out = execSync(
      `ffprobe -v quiet -print_format json -show_format -show_streams "${filePath}"`,
      { timeout: 15000, encoding: "utf8" }
    );
    return JSON.parse(out);
  } catch { return null; }
}

function runFfmpegExtract(filePath, outPath, opts = {}) {
  try {
    const args = opts.args || `-i "${filePath}" ${opts.extra || ""} "${outPath}"`;
    execSync(`ffmpeg -y -v quiet ${args}`, { timeout: 60000 });
    return fs.existsSync(outPath);
  } catch { return false; }
}

function extractAudioSignal(filePath, sessionId) {
  const meta = runFfprobe(filePath);
  if (!meta) return null;

  const fmt = meta.format || {};
  const stream = (meta.streams || []).find(s => s.codec_type === "audio") || {};
  const lines = [
    `Audio file: ${filePath.split("/").pop()}`,
    `Duration: ${fmt.duration ? parseFloat(fmt.duration).toFixed(1) + "s" : "unknown"}`,
    `Format: ${fmt.format_long_name || fmt.format_name || "unknown"}`,
    `Codec: ${stream.codec_name || "unknown"}`,
    `Sample rate: ${stream.sample_rate || "unknown"} Hz`,
    `Channels: ${stream.channels || "unknown"}`,
    `Bit rate: ${fmt.bit_rate ? (parseInt(fmt.bit_rate) / 1000).toFixed(0) + " kbps" : "unknown"}`,
  ];

  // Extract subtitle/speech text if available
  const subStreams = (meta.streams || []).filter(s => s.codec_type === "subtitle");
  if (subStreams.length > 0) {
    const subOut = `/tmp/eo-sub-${Date.now()}.srt`;
    if (runFfmpegExtract(filePath, subOut, { extra: "-map 0:s:0" })) {
      const subText = fs.readFileSync(subOut, "utf8").replace(/\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n/g, "");
      if (subText.trim()) lines.push(`Subtitles: ${subText.trim().slice(0, 2000)}`);
      try { fs.unlinkSync(subOut); } catch {}
    }
  }

  // Generate waveform stats for audio fingerprinting
  try {
    const stats = execSync(
      `ffmpeg -i "${filePath}" -af "astats=metadata=1:reset=1" -f null - 2>&1 | grep -E "RMS level|Peak level|Crest factor" | head -6`,
      { timeout: 30000, encoding: "utf8" }
    );
    if (stats.trim()) lines.push(`Audio stats:\n${stats.trim()}`);
  } catch {}

  return lines.join("\n");
}

function extractVideoSignal(filePath, sessionId) {
  const meta = runFfprobe(filePath);
  if (!meta) return null;

  const fmt = meta.format || {};
  const videoStream = (meta.streams || []).find(s => s.codec_type === "video") || {};
  const audioStream = (meta.streams || []).find(s => s.codec_type === "audio") || {};
  const lines = [
    `Video file: ${filePath.split("/").pop()}`,
    `Duration: ${fmt.duration ? parseFloat(fmt.duration).toFixed(1) + "s" : "unknown"}`,
    `Format: ${fmt.format_long_name || fmt.format_name || "unknown"}`,
    `Video: ${videoStream.codec_name || "unknown"}, ${videoStream.width || "?"}x${videoStream.height || "?"}`,
    `Frame rate: ${videoStream.r_frame_rate || "unknown"}`,
    `Audio: ${audioStream.codec_name || "none"}, ${audioStream.sample_rate || "?"} Hz, ${audioStream.channels || "?"} ch`,
    `Bit rate: ${fmt.bit_rate ? (parseInt(fmt.bit_rate) / 1000).toFixed(0) + " kbps" : "unknown"}`,
    `Size: ${fmt.size ? (parseInt(fmt.size) / 1048576).toFixed(1) + " MB" : "unknown"}`,
  ];

  // Extract embedded text streams (chapters, metadata)
  const textStreams = (meta.streams || []).filter(s => s.codec_type === "text" || s.codec_type === "subtitle");
  if (textStreams.length > 0) {
    lines.push(`Text streams: ${textStreams.length} (subtitle/text tracks detected)`);
  }

  // Extract chapter metadata
  const chapters = meta.chapters || [];
  if (chapters.length > 0) {
    lines.push(`Chapters: ${chapters.map(c => `${c.tags?.title || "untitled"} (${c.start_time}s-${c.end_time}s)`).join(", ")}`);
  }

  // Extract format-level tags
  const tags = fmt.tags || {};
  const tagStr = Object.entries(tags).map(([k,v]) => `${k}: ${v}`).join(", ");
  if (tagStr) lines.push(`Metadata: ${tagStr}`);

  // Extract audio subtitles (forced narratives)
  if (audioStream.codec_name) {
    try {
      const subOut = `/tmp/eo-vsub-${Date.now()}.srt`;
      if (runFfmpegExtract(filePath, subOut, { extra: `-map 0:s:0? -c:s srt` })) {
        const subText = fs.readFileSync(subOut, "utf8");
        if (subText.trim().length > 20) {
          lines.push(`Embedded subtitles:\n${subText.trim().slice(0, 3000)}`);
        }
        try { fs.unlinkSync(subOut); } catch {}
      }
    } catch {}
  }

  return lines.join("\n");
}

function extractImageSignal(filePath) {
  const meta = runFfprobe(filePath);
  const lines = [
    `Image file: ${filePath.split("/").pop()}`,
  ];

  if (meta) {
    const stream = (meta.streams || []).find(s => s.codec_type === "video") || {};
    if (stream.width) lines.push(`Dimensions: ${stream.width}x${stream.height}`);
    if (stream.codec_name) lines.push(`Codec: ${stream.codec_name}`);
  }

  // Extract EXIF via ffprobe tags
  if (meta?.format?.tags) {
    const tags = meta.format.tags;
    const interesting = ["artist","title","description","comment","date","location","gps","camera"];
    for (const [k, v] of Object.entries(tags)) {
      if (interesting.some(i => k.toLowerCase().includes(i))) {
        lines.push(`${k}: ${v}`);
      }
    }
  }

  // Try mdls for macOS EXIF
  try {
    const mdls = execSync(`mdls -name kMDItemAuthors -name kMDItemTitle -name kMDItemTextContent "${filePath}"`, { timeout: 5000, encoding: "utf8" });
    const textContent = mdls.match(/kMDItemTextContent\s*=\s*"(.+?)"/);
    if (textContent) lines.push(`Extracted text: ${textContent[1]}`);
  } catch {}

  return lines.join("\n");
}

export function ingestBinary(filePath, sessionId) {
  const ext = extOf(filePath);
  let signal = null;

  if (AUDIO_EXTS.has(ext)) {
    signal = extractAudioSignal(filePath, sessionId);
  } else if (VIDEO_EXTS.has(ext)) {
    signal = extractVideoSignal(filePath, sessionId);
  } else if (IMAGE_EXTS.has(ext)) {
    signal = extractImageSignal(filePath);
  }

  if (!signal) {
    signal = `Binary file: ${filePath.split("/").pop()} (type: ${ext}, unable to extract signal)`;
  }

  const sourceId = `binary:${filePath}`;
  const result = ingestContent(signal, sourceId, sessionId);
  log.write({ type: "ingest_binary", layer: 1, session: sessionId, path: filePath, ext, chunks: result.chunks, signal_length: signal.length });
  return { chunks: result.chunks, signal_length: signal.length, ext };
}

let state = null;

export function ensureState() {
  if (state) return state;
  const priorSnapshot = {
    schema_version: "PriorSnapshot@1",
    prior_id: "prior:sha256:" + "0".repeat(64),
    operator_epoch: CURRENT_OPERATOR_EPOCH,
    ledger_head: "head:empty",
    basis_id: "basis:none",
    content_hash: "sha256:" + "1".repeat(64),
  };
  state = createState({
    engineVersion: "0.1.0",
    operatorEpoch: CURRENT_OPERATOR_EPOCH,
    priorSnapshot,
  });
  return state;
}

export function ingestFile(filePath, sessionId) {
  if (isBinary(filePath)) {
    return ingestBinary(filePath, sessionId);
  }

  const raw = fs.readFileSync(filePath, "utf8");
  const entries = [];
  const lines = raw.split("\n");
  let chunk = [], size = 0, c = 0;
  const CHUNK_SIZE = 2000;
  for (const line of lines) {
    chunk.push(line);
    size += line.length;
    if (size > CHUNK_SIZE) {
      const body = chunk.join("\n").trim();
      if (body.length > 50) {
        addChunk(body, filePath, c, sessionId);
        entries.push({ id: `chunk-${c}`, size: body.length });
        c++;
      }
      chunk = [];
      size = 0;
    }
  }
  if (chunk.length > 0) {
    const body = chunk.join("\n").trim();
    if (body.length > 50) {
      addChunk(body, filePath, c, sessionId);
      entries.push({ id: `chunk-${c}`, size: body.length });
      c++;
    }
  }
  return { chunks: c, entries };
}

export function ingestDir(dirPath, sessionId, extensions) {
  let total = 0;
  const exts = extensions || ["js","ts","jsx","tsx","mjs","cjs","json","md","py","rs","go","rb","java","kt","swift","c","cpp","h","hpp","mp3","wav","ogg","flac","aac","m4a","mp4","mkv","avi","mov","jpg","jpeg","png","gif","txt","csv","xml","yaml","yml","toml"];
  const extSet = new Set(exts.map(e => e.startsWith(".") ? e : "." + e));
  function walk(dir) {
    let dirents;
    try { dirents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const d of dirents) {
      if (d.name.startsWith(".") || d.name === "node_modules") continue;
      const full = `${dir}/${d.name}`;
      if (d.isDirectory()) walk(full);
      else if (d.isFile() && extSet.has("." + d.name.split(".").pop())) {
        try {
          const result = ingestFile(full, sessionId);
          total += result.chunks;
        } catch {}
      }
    }
  }
  walk(dirPath);
  return { chunks: total };
}

function addChunk(text, source, index, sessionId) {
  const s = ensureState();
  const sourceId = `source:${source}:chunk-${index}`;
  const block = {
    schema: "ObservationBlock@1",
    block_id: `block:${canonicalHashSync({ source: sourceId, values: [text] })}`,
    value_type: "string",
    shape: [1],
    axis_order: ["paragraph"],
    values: [text],
    selectors: [{ byte_start: 0, byte_end: Buffer.byteLength(text, "utf8") }],
    loss: [{ kind: "none" }],
  };
  block.content_hash = blockContentHash(block);
  const blocks = [block];
  const blocks_hash = canonicalHashSync(blocks.map(b => ({ block_id: b.block_id, content_hash: b.content_hash })));
  const envelope = {
    schema: "ObservationEnvelope@1",
    source_id: sourceId,
    source_media_type: "text/plain",
    decoder: { id: "plain-text", version: "1", loss: [{ kind: "none" }] },
    axes: [{ axis_id: "paragraph", topology: "ordered", unit: "paragraph" }],
    fields: [{ field_id: "paragraph:text", value_type: "string", block_id: block.block_id, axes: ["paragraph"] }],
    anchors: { scheme: "byte", selectors: { "paragraph:text": block.selectors } },
    source_content_hash: canonicalHashSync({ bytes: Buffer.from(text, "utf8").toString("base64") }),
    blocks_hash,
  };
  state = applyCommand(state, { type: "observation.admit", payload: { envelope, blocks } });
  log.write({
    type: "source", layer: 1, session: sessionId,
    source: sourceId, path: source, text,
    size: text.length, chunk_index: index,
  });
}

export function ingestContent(text, sourceId, sessionId) {
  ensureState();
  const lines = text.split("\n");
  let chunk = [], size = 0, c = 0;
  const CHUNK_SIZE = 2000;
  for (const line of lines) {
    chunk.push(line);
    size += line.length;
    if (size > CHUNK_SIZE) {
      const body = chunk.join("\n").trim();
      if (body.length > 50) {
        addChunk(body, sourceId, c, sessionId);
        c++;
      }
      chunk = [];
      size = 0;
    }
  }
  if (chunk.length > 0) {
    const body = chunk.join("\n").trim();
    if (body.length > 50) {
      addChunk(body, sourceId, c, sessionId);
      c++;
    }
  }
  return { chunks: c };
}

export function searchQuery(query, limit = 10) {
  const s = ensureState();
  const result = search(s, { query, limit: Math.min(limit, 40) });
  const passages = (result.passages || []).map(p => ({
    text: (p.anchors?.exact_text || []).join(" "),
    source: p.source_id || "",
    score: p.score,
    signalScore: p.signalScore,
    keywordScore: p.keywordScore,
  }));
  return { query, total: result.passages ? result.passages.length : 0, passages, gaps: result.gaps || [] };
}

export function foldUnits(units, query, budget = 600, maxUnits = 8) {
  const result = compressFold(
    { units: units.map(u => ({ text: u.text, coord: null, meta: { source: u.source } })), query },
    { tokenBudget: budget, maxUnits }
  );
  return {
    summary: result.summary,
    selected: result.selected.length,
    tokens: result.totalTokens,
    budget: result.budget,
    dropped: result.dropped,
  };
}
