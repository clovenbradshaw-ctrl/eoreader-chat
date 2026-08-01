// verb-register.js — the single source of truth for every verb this app may invoke.
//
// Governance: an application is a thin host (CONSTITUTION.md I.4, II.3). This
// register is the host's own pointing: for each verb it declares the target the
// verb aims at, the holonic height of that target, the EO route the verb takes,
// and the dependency order it must respect. It is enforced by
// test-verb-register.mjs.
//
// Three rules, in the register's own terms:
//   1. Every verb is EO-compliant: its `domain` answers the routing tests
//      (II.1 omnimodal, II.2 giver, II.3 host; what remains is engine).
//   2. Dependency order is respected: `requires` names verbs that must have run
//      first; the graph is acyclic; every `consumes` noun is produced by some
//      registered verb.
//   3. Every verb aims at a target at a holonic height: `target.noun` and
//      `target.height` come from the declared tables below.
//
// Heights are DECLARED aims of the host relative to the material currently in
// the encounter — `host`, `within`, `self`, `peer`, `above`. They are not
// engine-discovered levels: holon level is discovered, never assigned
// (holon-level.md), and discovery is the engine's. The host points; the engine
// measures. `peer` is a first-class position with no verb yet — awaiting one.

export const DOMAINS = ["host", "engine", "priors"];

export const HEIGHTS = ["host", "within", "self", "peer", "above"];

export const TARGET_HEIGHTS = Object.freeze({
  shell: "host",
  file: "host",
  path: "host",
  attachment: "host",
  process: "host",
  span: "within",
  module: "within",
  material: "self",
  memory: "self",
  task: "self",
  codebase: "above",
  web: "above",
});

export const POPULATE_NOUNS = Object.freeze(["material"]);

export const MEASUREMENT_NOUNS = Object.freeze([
  "memory-context",
  "span-id",
  "span-text",
  "terrain",
  "research-evidence",
]);

const hostPure = {
  bash: { domain: "host", target: { noun: "shell", height: "host" }, requires: [], consumes: [], produces: ["process-output"], justification: "II.3" },
  read_file: { domain: "host", target: { noun: "file", height: "host" }, requires: [], consumes: [], produces: ["file-content"], justification: "II.3" },
  write_file: { domain: "host", target: { noun: "file", height: "host" }, requires: [], consumes: [], produces: ["file-content"], justification: "II.3" },
  edit_file: { domain: "host", target: { noun: "file", height: "host" }, requires: ["read_file"], consumes: ["file-content"], produces: ["file-content"], justification: "II.3" },
  glob: { domain: "host", target: { noun: "path", height: "host" }, requires: [], consumes: [], produces: ["path-list"], justification: "II.3" },
  grep: { domain: "host", target: { noun: "file", height: "host" }, requires: [], consumes: [], produces: ["search-result"], justification: "II.3" },
  ls: { domain: "host", target: { noun: "path", height: "host" }, requires: [], consumes: [], produces: ["path-list"], justification: "II.3" },
  web_search: { domain: "host", target: { noun: "web", height: "above" }, requires: [], consumes: [], produces: ["search-result"], justification: "II.3" },
  web_fetch: { domain: "host", target: { noun: "web", height: "above" }, requires: [], consumes: [], produces: ["web-content"], suggested_after: ["web_search"], justification: "II.3" },
  fetch_attachment: { domain: "host", target: { noun: "attachment", height: "host" }, requires: [], consumes: [], produces: ["attachment-content"], justification: "II.3" },
  ingest: { domain: "host", target: { noun: "material", height: "self" }, requires: [], consumes: [], produces: ["material"], justification: "II.3" },
  codebase_structure: { domain: "host", target: { noun: "codebase", height: "above" }, requires: [], consumes: [], produces: ["tree"], justification: "II.3, II.4" },
  codebase_find: { domain: "host", target: { noun: "codebase", height: "above" }, requires: [], consumes: [], produces: ["search-result"], justification: "II.3, II.4" },
  codebase_lookup: { domain: "host", target: { noun: "module", height: "within" }, requires: [], consumes: [], produces: ["module-info"], justification: "II.3, II.4" },
  codebase_search: { domain: "host", target: { noun: "codebase", height: "above" }, requires: [], consumes: [], produces: ["search-result"], justification: "II.3, II.4" },
  codebase_entities: { domain: "host", target: { noun: "codebase", height: "above" }, requires: [], consumes: [], produces: ["entity-list"], justification: "II.3, II.4" },
  codebase_summary: { domain: "host", target: { noun: "codebase", height: "above" }, requires: [], consumes: [], produces: ["stats"], justification: "II.3, II.4" },
  holonic_task: { domain: "host", target: { noun: "task", height: "self" }, requires: [], consumes: [], produces: ["final"], composes: ["PLAN", "RESEARCH", "EXECUTE", "CITES", "ASSEMBLE"], justification: "II.3" },
  mcp_star: { domain: "host", target: { noun: "process", height: "host" }, requires: [], consumes: [], produces: ["server-result"], justification: "II.3" },
};

const engineRouted = {
  search_memory: { domain: "engine", target: { noun: "memory", height: "self" }, requires: ["ingest"], consumes: [], produces: ["memory-context"], justification: "I.1, II.4" },
  verbatim_search: { domain: "engine", target: { noun: "span", height: "within" }, requires: ["ingest"], consumes: [], produces: ["span-id"], justification: "I.1, II.4" },
  verbatim_read: { domain: "engine", target: { noun: "span", height: "within" }, requires: ["verbatim_search"], consumes: ["span-id"], produces: ["span-text"], justification: "I.1, II.4" },
  terrain_report: { domain: "engine", target: { noun: "material", height: "self" }, requires: ["ingest"], consumes: [], produces: ["terrain"], justification: "I.1, II.4" },
};

const holonicInternal = {
  PLAN: { domain: "host", target: { noun: "task", height: "self" }, requires: [], consumes: [], produces: ["task-plan"], justification: "II.3" },
  RESEARCH: { domain: "engine", target: { noun: "material", height: "self" }, requires: ["ingest"], consumes: [], produces: ["research-evidence"], justification: "I.1, II.2, III.1" },
  EXECUTE: { domain: "host", target: { noun: "task", height: "self" }, requires: ["RESEARCH"], consumes: ["research-evidence"], produces: ["draft"], justification: "II.3" },
  CITES: { domain: "host", target: { noun: "task", height: "self" }, requires: ["EXECUTE", "RESEARCH"], consumes: ["draft", "research-evidence"], produces: ["citations"], justification: "II.3" },
  ASSEMBLE: { domain: "host", target: { noun: "task", height: "self" }, requires: ["CITES"], consumes: ["draft", "citations"], produces: ["final"], justification: "II.3" },
};

export const VERB_REGISTER = Object.freeze({
  ...hostPure,
  ...engineRouted,
  ...holonicInternal,
});

export const INTERNAL_VERBS = Object.freeze(Object.keys(holonicInternal));

export const MCP_CLASS = "mcp_star";

export const PROXY_TOOL_SOURCE = new URL("./proxy.js", import.meta.url);
