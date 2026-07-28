/**
 * Chat history — session-scoped message store with rolling fold.
 *
 * Problem: when opencode uses the MCP tools across many turns, the
 * model calls (think, speak, plan) each start fresh with no memory
 * of the prior conversation. The model loses the thread.
 *
 * Solution: maintain a per-session rolling history that compresses
 * old turns into summaries when the context budget fills up.
 * Model calls can then receive the conversation context.
 *
 * This is the MCP-side equivalent of chat.html's foldConversation().
 * The host (opencode) manages its own context; this module gives
 * the engine tools access to a compressed conversation record.
 *
 * Persistence: every message is appended to ~/.eoreader/chat/{sessionId}.jsonl
 * on write, and loaded back on first access. This gives a full audit trail
 * per opencode tab / session.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const CONTEXT_WINDOW = 32768; // conservative default for small models
const FOLD_THRESHOLD = 0.55;
const RECENT_KEEP = 8; // keep last N messages verbatim

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CHAT_DIR = path.join(__dirname, "..", "..", "chats");

function chatPath(sessionId) {
  return path.join(CHAT_DIR, `${sessionId}.jsonl`);
}

function ensureDir() {
  if (!fs.existsSync(CHAT_DIR)) fs.mkdirSync(CHAT_DIR, { recursive: true });
}

export function appendToDisk(sessionId, entry) {
  try {
    ensureDir();
    fs.appendFileSync(chatPath(sessionId), JSON.stringify(entry) + "\n");
  } catch {}
}

function loadFromDisk(sessionId) {
  const p = chatPath(sessionId);
  if (!fs.existsSync(p)) return [];
  try {
    return fs.readFileSync(p, "utf8")
      .trim()
      .split("\n")
      .filter(Boolean)
      .map(l => JSON.parse(l));
  } catch { return []; }
}

// Per-session stores
const sessions = new Map();

function getSession(sessionId) {
  if (!sessions.has(sessionId)) {
    // Hydrate from disk if a prior log exists
    const persisted = loadFromDisk(sessionId);
    sessions.set(sessionId, {
      messages: persisted, // { role, content, timestamp }
      priorSummary: null,
      foldCount: 0,
      totalTokens: 0,
    });
    if (persisted.length > 0) {
      sessions.get(sessionId).totalTokens = sessionTokens(sessions.get(sessionId));
    }
  }
  return sessions.get(sessionId);
}

function estimateTokens(text) {
  return Math.ceil(String(text).length / 3.5);
}

function sessionTokens(session) {
  let total = 0;
  for (const msg of session.messages) {
    total += estimateTokens(msg.content);
  }
  return total + estimateTokens(session.priorSummary || "");
}

/**
 * Add a message to the session history.
 * Returns { folded, foldCount, messageCount, tokens }
 */
export function addMessage(sessionId, role, content) {
  const session = getSession(sessionId);
  const msg = { role, content, timestamp: Date.now() };
  session.messages.push(msg);
  session.totalTokens = sessionTokens(session);

  // Persist every message to disk immediately
  appendToDisk(sessionId, msg);

  let folded = false;
  if (session.totalTokens > CONTEXT_WINDOW * FOLD_THRESHOLD && session.messages.length > RECENT_KEEP + 4) {
    foldSession(session);
    folded = true;
  }

  return {
    folded,
    foldCount: session.foldCount,
    messageCount: session.messages.length,
    tokens: session.totalTokens,
  };
}

/**
 * Fold a session's older messages into a summary.
 * Keeps the most recent RECENT_KEEP messages verbatim.
 */
function foldSession(session) {
  const splitIdx = session.messages.length - RECENT_KEEP;
  const toSummarize = session.messages.slice(0, splitIdx);
  const recent = session.messages.slice(splitIdx);

  // Build a cheap keyword-based summary (no model call — this is mechanical)
  const priorCtx = session.priorSummary
    ? `[Prior summary]\n${session.priorSummary}\n\n`
    : "";

  const dialogue = toSummarize
    .filter(m => m.role === "user" || m.role === "assistant")
    .map(m => {
      const label = m.role === "user" ? "User" : "Assistant";
      return `${label}: ${m.content.slice(0, 300)}`;
    })
    .join("\n");

  // Extract key topics via frequency
  const allText = toSummarize.map(m => m.content).join(" ");
  const topics = extractTopics(allText);

  const summary = [
    priorCtx,
    `Topics discussed: ${topics.join(", ")}`,
    `Exchange count: ${toSummarize.length}`,
    ``,
    `Key exchanges (compressed):`,
    dialogue.slice(0, 2000),
  ].filter(Boolean).join("\n");

  session.priorSummary = summary;
  session.messages = recent;
  session.foldCount++;
  session.totalTokens = sessionTokens(session);
}

function extractTopics(text) {
  const stops = new Set([
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","has","had","have",
    "do","does","did","will","would","could","should","may","might","can",
    "that","this","it","its","i","you","we","they","he","she","me","my",
    "your","our","their","his","her","him","not","no","so","if","then",
    "just","about","like","what","when","where","how","which","who",
  ]);
  const words = text.toLowerCase().split(/\s+/);
  const freq = {};
  for (const w of words) {
    const clean = w.replace(/[^a-z0-9]/g, "");
    if (clean.length > 3 && !stops.has(clean)) {
      freq[clean] = (freq[clean] || 0) + 1;
    }
  }
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([w]) => w);
}

/**
 * Get the current conversation context for a session.
 * Returns the messages array ready for model consumption.
 * Includes system prompt + prior summary + recent messages.
 */
export function getContext(sessionId, systemPrompt) {
  const session = getSession(sessionId);
  const messages = [];

  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }

  if (session.priorSummary) {
    messages.push({
      role: "system",
      content: `[Conversation context — ${session.foldCount} prior folds]\n${session.priorSummary}`,
    });
  }

  // Add recent messages (skip any existing system messages to avoid duplication)
  for (const msg of session.messages) {
    if (msg.role !== "system") {
      messages.push({ role: msg.role, content: msg.content });
    }
  }

  return messages;
}

/**
 * Get session stats for display.
 */
export function getSessionStats(sessionId) {
  const session = getSession(sessionId);
  return {
    messageCount: session.messages.length,
    foldCount: session.foldCount,
    tokens: session.totalTokens,
    contextWindow: CONTEXT_WINDOW,
    usagePercent: Math.round((session.totalTokens / CONTEXT_WINDOW) * 100),
    priorSummaryLength: estimateTokens(session.priorSummary || ""),
  };
}

/**
 * Clear a session's history.
 */
export function clearSession(sessionId) {
  sessions.delete(sessionId);
  // Truncate the on-disk log so a fresh session starts clean
  try { fs.writeFileSync(chatPath(sessionId), ""); } catch {}
}

export { CHAT_DIR, chatPath };
