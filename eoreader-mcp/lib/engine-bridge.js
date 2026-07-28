// Engine access goes through the declared package surface. This file used to
// import six modules by relative filesystem path and hand-roll its own
// ObservationBlock@1 construction — a fork that wrote `byte_start: 0` on every
// chunk, so no quote from this server could be checked against its source
// file. Admission, chunking, byte offsets, search, spans, and folding now come
// from @eoreader/host/corpus. The media extraction below is genuine host work
// and stays here.
import {
  CORPUS_API_VERSION,
  createSession,
  admitChunked,
  ingestFile as corpusIngestFile,
  searchSpans,
  readSpan,
  spanUnits,
  foldSpans,
} from "@eoreader/host/corpus";
import fs from "fs";
import { execSync } from "child_process";
import * as log from "./log.js";

const EXPECTED_CORPUS_API = 1;
if (CORPUS_API_VERSION !== EXPECTED_CORPUS_API) {
  throw new Error(
    `@eoreader/host/corpus is API v${CORPUS_API_VERSION}; this bridge expects v${EXPECTED_CORPUS_API}`,
  );
}

// ── CV Model Configuration ──
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const VISION_MODEL = process.env.VISION_MODEL || "llava:13b";

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

async function extractImageUnderstanding(filePath, model) {
  const imageBuffer = fs.readFileSync(filePath);
  const base64Image = imageBuffer.toString("base64");
  const使用的模型 = model || VISION_MODEL;

  try {
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: 使用的模型,
        prompt: "Describe this image in detail. Include: main subjects, scene/environment, colors, any visible text, mood, notable features, and spatial relationships. Be specific and descriptive for searchability.",
        images: [base64Image],
        stream: false,
      }),
      signal: AbortSignal.timeout(120000), // 2 min timeout for large images
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const result = await response.json();
    return {
      caption: result.response,
      model: 使用的模型,
      eval_count: result.eval_count,
    };
  } catch (err) {
    // Return structured error so caller can fallback gracefully
    return {
      caption: null,
      model: 使用的模型,
      error: err.message,
    };
  }
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

async function extractImageSignal(filePath, opts = {}) {
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

  // NEW: Call CV model for visual understanding
  if (opts.useCVModel !== false) {
    const cvResult = await extractImageUnderstanding(filePath, opts.model);
    if (cvResult.caption) {
      lines.push(`\nVisual Description:\n${cvResult.caption}`);
      lines.push(`\n[Extracted via ${cvResult.model}]`);
    } else if (cvResult.error) {
      lines.push(`\n[CV model unavailable: ${cvResult.error}]`);
    }
  }

  return lines.join("\n");
}

export async function ingestBinary(filePath, sessionId, opts = {}) {
  const ext = extOf(filePath);
  let signal = null;

  if (AUDIO_EXTS.has(ext)) {
    signal = extractAudioSignal(filePath, sessionId);
  } else if (VIDEO_EXTS.has(ext)) {
    signal = extractVideoSignal(filePath, sessionId);
  } else if (IMAGE_EXTS.has(ext)) {
    signal = await extractImageSignal(filePath, {
      useCVModel: !opts.skipCV,
      model: opts.model,
    });
  }

  if (!signal) {
    signal = `Binary file: ${filePath.split("/").pop()} (type: ${ext}, unable to extract signal)`;
  }

  // Append user-provided caption if given
  if (opts.caption) {
    signal = signal + `\n\nUser caption: ${opts.caption}`;
  }

  const sourceId = `binary:${filePath}`;
  const result = ingestContent(signal, sourceId, sessionId);
  log.write({ type: "ingest_binary", layer: 1, session: sessionId, path: filePath, ext, chunks: result.chunks, signal_length: signal.length, has_caption: !!opts.caption, cv_used: !opts.skipCV });
  return { chunks: result.chunks, signal_length: signal.length, ext, preview: signal.slice(0, 500) };
}

let session = null;

export function ensureState() {
  if (!session) session = createSession();
  return session;
}

// One record per admitted chunk, now carrying real byte ranges instead of the
// byte_start: 0 every record used to get.
function logAdmitted(admitted, sourcePath, sessionId) {
  admitted.forEach((entry, index) => {
    log.write({
      type: "source", layer: 1, session: sessionId,
      source: entry.sourceId, path: sourcePath,
      byte_start: entry.byteStart, byte_end: entry.byteEnd,
      size: entry.byteEnd - entry.byteStart, chunk_index: index,
    });
  });
}

export async function ingestFile(filePath, sessionId, opts = {}) {
  if (isBinary(filePath)) {
    return ingestBinary(filePath, sessionId, opts);
  }
  const s = ensureState();
  const { chunks, admitted } = corpusIngestFile(s, filePath);
  logAdmitted(admitted, filePath, sessionId);
  return {
    chunks,
    entries: admitted.map((a, i) => ({ id: `chunk-${i}`, size: a.byteEnd - a.byteStart })),
  };
}

export async function ingestDir(dirPath, sessionId, extensions) {
  // Not corpus.ingestDir: this server also ingests audio/video/image, which
  // needs the CV + ffprobe path above rather than a UTF-8 read. The walk stays
  // here so binaries route through ingestBinary; text files go to the facade.
  const exts = extensions || ["js","ts","jsx","tsx","mjs","cjs","json","md","py","rs","go","rb","java","kt","swift","c","cpp","h","hpp","mp3","wav","ogg","flac","aac","m4a","mp4","mkv","avi","mov","jpg","jpeg","png","gif","txt","csv","xml","yaml","yml","toml"];
  const extSet = new Set(exts.map(e => e.startsWith(".") ? e : "." + e));
  let total = 0;

  async function walk(dir) {
    let dirents;
    try { dirents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const d of dirents) {
      if (d.name.startsWith(".") || d.name === "node_modules") continue;
      const full = `${dir}/${d.name}`;
      if (d.isDirectory()) {
        await walk(full);
      } else if (d.isFile() && extSet.has("." + d.name.split(".").pop())) {
        try {
          const result = await ingestFile(full, sessionId);
          total += result.chunks;
        } catch {}
      }
    }
  }

  await walk(dirPath);
  return { chunks: total };
}

// Admit content already in memory — an upload, a paste, or the extracted
// signal from a binary file.
export function ingestContent(text, sourceId, sessionId) {
  const s = ensureState();
  const { chunks, admitted } = admitChunked(s, { text, sourceId });
  logAdmitted(admitted, sourceId, sessionId);
  return { chunks };
}

export function searchQuery(query, limit = 10) {
  const s = ensureState();
  const { spans, gaps } = searchSpans(s, { query, limit: Math.min(limit, 40) });
  const units = spanUnits(s, spans);
  return {
    query,
    total: spans.length,
    passages: spans.map((sp, i) => ({
      span_id: sp.span_id,
      // Previously exact_text.join(" ") — a reconstruction inserting separators
      // absent from the source. This is the verbatim admitted value, and
      // byte_start/byte_end address it in the source file, so a citation drawn
      // from it is checkable rather than taken on trust.
      text: units[i]?.text ?? "",
      source: sp.source_id || "",
      byte_start: sp.byte_start,
      byte_end: sp.byte_end,
      score: sp.score,
      preview: sp.preview,
    })),
    gaps,
  };
}

// Verbatim bytes for a span returned by searchQuery.
export function spanText(spanId, maxBytes) {
  return readSpan(ensureState(), { spanId, maxBytes });
}

export function foldUnits(units, query, budget = 600, maxUnits = 8) {
  const s = ensureState();
  const resolved = units?.length && units[0]?.span_id
    ? spanUnits(s, units)
    : (units || []).map(u => ({ text: u.text, coord: null, meta: { source: u.source } }));
  const result = foldSpans(s, { units: resolved, query, tokenBudget: budget, maxUnits });
  return {
    summary: result.summary,
    selected: result.selectedCount,
    tokens: result.tokens,
    budget: result.budget,
    dropped: result.dropped,
  };
}
