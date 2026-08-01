// priors-bridge.js — real per-text coref/synonymy prior activation for
// holonic-task.js, via the engine's canonical path (never string matching).
//
// Per CLAUDE.md / eoreader5's AGENTS.md: coref resolution is the #1
// repeatedly-reinvented wheel. The canonical path is
// perceiver/text/presence.js::admitReferent, fed by per-text priors at
// eoPriors/priors/coref/*.json. This module is a thin bridge — it does not
// resolve "is this entity here" itself, it asks the engine.
//
// Priors activated here are NEVER shown to the model as text (see
// holonic-task.js's executeSubtask). They only ever (a) widen retrieval
// queries with alternate surface forms, (b) let the grounding scorer credit
// a passage using one surface form when the draft uses another, and (c)
// bias which under-covered passages get pulled into a correction pass.
// Missing prior ⇒ an explicit typed gap, never a silent empty activation.

import { readFileSync, readdirSync, existsSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { admitReferent } from "@eoreader/engine/perceiver/text/presence";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COREF_DIR = path.resolve(__dirname, "..", "eoPriors", "priors", "coref");

const priorCache = new Map(); // sourceId -> { referents, fullText, gap }
const fullTextCache = new Map(); // absolute file path -> normalized text

function shortIdFor(sourceId) {
  // sourceId is typically "source:/abs/path/pg84.txt:chunk-36" (engine-ground's
  // shape) or a bare basename like "pg84" / "pg84.txt". Extract a short id
  // usable to match a coref prior filename by convention.
  const raw = String(sourceId || "");
  const pathMatch = raw.match(/([a-zA-Z0-9_-]+)\.txt/);
  if (pathMatch) return pathMatch[1];
  const bareMatch = raw.match(/^([a-zA-Z0-9_-]+)$/);
  return bareMatch ? bareMatch[1] : raw;
}

function corefFilenameFor(sourceId) {
  if (!existsSync(COREF_DIR)) return null;
  const short = shortIdFor(sourceId);
  if (!short) return null;
  const files = readdirSync(COREF_DIR).filter((f) => f.endsWith(".json"));
  return files.find((f) => f.toLowerCase().startsWith(short.toLowerCase())) || null;
}

function filePathFromSourceId(sourceId) {
  // "source:/abs/path/pg84.txt:chunk-36" -> "/abs/path/pg84.txt"
  const m = String(sourceId || "").match(/^source:(.+?)(?::chunk-\d+)?$/);
  return m ? m[1] : null;
}

function loadFullText(sourceId) {
  const filePath = filePathFromSourceId(sourceId);
  if (!filePath) return "";
  if (fullTextCache.has(filePath)) return fullTextCache.get(filePath);
  let text = "";
  try {
    text = readFileSync(filePath, "utf8").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  } catch {
    text = "";
  }
  fullTextCache.set(filePath, text);
  return text;
}

// Load the per-text coref prior for a source id. Missing file ⇒ an explicit
// typed gap — never a silent empty activation pretending there's nothing to
// activate (CLAUDE.md: "missing prior ⇒ typed gap, never a silently wrong
// number").
// `priorId` / `priorPath` identify WHICH artifact acted. The model never sees
// them, but the user does: eochat shows the priors affecting a surf, and
// "a prior widened this search" is only auditable if you can open the prior
// that did it. Ids match priors-source.js's catalog ("coref/pg84-frankenstein"),
// so the UI can link an activation straight to the readable card.
export function loadCorefPrior(sourceId) {
  const cached = priorCache.get(sourceId);
  if (cached && !cached.gap) return cached;
  if (cached && cached.gap) {
    // Negative result (gap) is never cached permanently — a prior file might be
    // added while the proxy is running. Re-check the filesystem.
    priorCache.delete(sourceId);
  }

  const filename = corefFilenameFor(sourceId);
  if (!filename) {
    return { referents: [], fullText: "", priorId: null, priorPath: null, gap: `no coref prior for source "${sourceId}"` };
  }

  const priorPath = path.join(COREF_DIR, filename);
  const priorId = `coref/${filename.replace(/\.json$/, "")}`;

  let data;
  try {
    data = JSON.parse(readFileSync(priorPath, "utf8"));
  } catch (err) {
    return { referents: [], fullText: "", priorId, priorPath, gap: `failed to parse coref prior ${filename}: ${err.message}` };
  }

  const result = {
    referents: data.referents || [],
    fullText: loadFullText(sourceId),
    priorId,
    priorPath,
    gap: null,
  };
  priorCache.set(sourceId, result);
  return result;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Word-boundary match, not substring containment — a bare `.includes()`
// would let a single-letter narrator-span surface like "i" match almost any
// English text (any word containing the letter), silently over-crediting
// first-person surfaces everywhere.
// Exported as the MATCHER, not just a boolean, so the caller that needs the
// position of a hit (engine-ground's fold anchors, which turn it into a byte
// offset) and the caller that needs only its existence share one definition of
// what counts as a match — including the word-boundary rule above.
export function surfaceMatcher(surface) {
  const s = String(surface || "").trim();
  return s ? new RegExp(`\\b${escapeRegex(s)}\\b`, "i") : null;
}

function surfaceAppearsIn(lowerText, surface) {
  const re = surfaceMatcher(surface);
  return re ? re.test(lowerText) : false;
}

// Activate a loaded prior's referents against retrieved text. Returns
// structured { activated, gap } — `activated` entries carry matchedSurfaces
// (forms actually present in `text`) and expansionSurfaces (the referent's
// other surface forms) — the two things holonic-task.js needs for retrieval
// expansion and prior-aware grounding credit. Never returns prose describing
// the referent — there is nothing here for a prompt to narrate.
//
// Scoped surfaces (e.g. narrator-span "I", or a surface only valid in one
// stretch of the book) are only trusted when this passage's position can be
// located inside corefPrior.fullText and intersects the scope's resolved
// range — otherwise a first-person surface scoped to one character's
// narration could misattribute to a passage from someone else's. If the
// passage can't be located (e.g. the engine's preview text diverges from
// the raw file), scoped surfaces are skipped rather than guessed; only
// globally-valid (unscoped) surfaces are trusted in that case.
export function activatePriors(text, corefPrior) {
  if (!corefPrior || corefPrior.gap) {
    return {
      activated: [],
      priorId: corefPrior?.priorId ?? null,
      priorPath: corefPrior?.priorPath ?? null,
      gap: corefPrior?.gap ?? "no coref prior loaded",
    };
  }
  const normalized = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lower = normalized.toLowerCase();
  const passageOffset = corefPrior.fullText ? corefPrior.fullText.indexOf(normalized) : -1;
  const activated = [];

  for (const referent of corefPrior.referents) {
    // The canonical path — never text.includes(name)/alias-list matching.
    const admission = admitReferent([], referent, { fullText: corefPrior.fullText });
    const seen = new Set();
    const matchedSurfaces = [];
    const expansionSurfaces = [];

    for (const s of admission.surfaces) {
      if (seen.has(s.surface)) continue;
      seen.add(s.surface);

      const inScope = !s.scope
        ? true
        : passageOffset !== -1 && s.scope.some(({ from, to }) => passageOffset < to && passageOffset + normalized.length > from);
      if (!inScope) continue;

      if (surfaceAppearsIn(lower, s.surface)) matchedSurfaces.push(s.surface);
      else expansionSurfaces.push(s.surface);
    }

    if (matchedSurfaces.length === 0) continue;
    activated.push({
      referentId: admission.referentId,
      display: referent.display || referent.id,
      matchedSurfaces,
      expansionSurfaces,
      priorId: corefPrior.priorId,
    });
  }

  return { activated, priorId: corefPrior.priorId, priorPath: corefPrior.priorPath, gap: null };
}
