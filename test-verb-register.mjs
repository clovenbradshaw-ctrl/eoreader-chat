// test-verb-register.mjs — audit of the app's verb register.
//
// Every verb the app can invoke must be registered with its EO route, its
// target at a holonic height, and its dependency order. An unregistered verb
// is unwired; a register entry that names no verb is fiction; a dependency
// edge that does not resolve or cycles is a broken order.
//
// Run: node --test test-verb-register.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  VERB_REGISTER,
  INTERNAL_VERBS,
  MCP_CLASS,
  DOMAINS,
  HEIGHTS,
  TARGET_HEIGHTS,
  POPULATE_NOUNS,
  MEASUREMENT_NOUNS,
  PROXY_TOOL_SOURCE,
} from "./verb-register.js";

const proxySrc = readFileSync(PROXY_TOOL_SOURCE, "utf8");
const defStart = proxySrc.indexOf("const TOOL_DEFINITIONS = [");
assert.ok(defStart >= 0, "proxy.js must still declare TOOL_DEFINITIONS");
const defEnd = proxySrc.indexOf("\n];", defStart);
const defSlice = proxySrc.slice(defStart, defEnd);
const proxyTools = [...defSlice.matchAll(/\bname: "([^"]+)"/g)].map((m) => m[1]);

const holonicSrc = readFileSync(new URL("./holonic-task.js", import.meta.url), "utf8");

const allKnown = new Set([...Object.keys(VERB_REGISTER), ...proxyTools]);

const producedBy = (verb) => new Set(VERB_REGISTER[verb]?.produces ?? []);
const allProduces = Object.fromEntries(
  Object.keys(VERB_REGISTER).map((v) => [v, producedBy(v)]),
);

test("every tool in TOOL_DEFINITIONS is registered", () => {
  for (const name of proxyTools) {
    const ok = VERB_REGISTER[name] !== undefined || name === MCP_CLASS || name.startsWith("mcp_");
    assert.ok(ok, `unregistered verb: ${name}`);
  }
});

test("every register entry names a real verb", () => {
  for (const [name] of Object.entries(VERB_REGISTER)) {
    const isInternal = INTERNAL_VERBS.includes(name);
    const isProxyTool = proxyTools.includes(name);
    const isClass = name === MCP_CLASS;
    assert.ok(isInternal || isProxyTool || isClass, `register fiction: ${name}`);
  }
});

test("domains are well-formed", () => {
  for (const [name, entry] of Object.entries(VERB_REGISTER)) {
    assert.ok(DOMAINS.includes(entry.domain), `${name}: bad domain ${entry.domain}`);
  }
});

test("every verb is aimed at a target at a holonic height", () => {
  for (const [name, entry] of Object.entries(VERB_REGISTER)) {
    const { noun, height } = entry.target;
    assert.ok(noun in TARGET_HEIGHTS, `${name}: unknown target noun ${noun}`);
    assert.equal(height, TARGET_HEIGHTS[noun], `${name}: ${noun} misaimed at ${height}`);
    assert.ok(HEIGHTS.includes(height), `${name}: bad height ${height}`);
  }
});

test("every verdict cites the articles that produced it (IV.4)", () => {
  for (const [name, entry] of Object.entries(VERB_REGISTER)) {
    assert.match(entry.justification, /\b(?:I|II|III|IV)\.\d\b/, `${name}: missing article citation`);
  }
});

test("dependency order resolves", () => {
  for (const [name, entry] of Object.entries(VERB_REGISTER)) {
    for (const dep of [...(entry.requires ?? []), ...(entry.suggested_after ?? [])]) {
      assert.ok(VERB_REGISTER[dep] !== undefined, `${name}: requires unknown verb ${dep}`);
    }
    for (const composed of entry.composes ?? []) {
      assert.ok(VERB_REGISTER[composed] !== undefined, `${name}: composes unknown verb ${composed}`);
    }
  }
});

test("dependency order is acyclic", () => {
  const byName = new Map(Object.keys(VERB_REGISTER).map((n) => [n, n]));
  const visiting = new Set();
  const visited = new Set();
  const visit = (name, stack) => {
    if (visited.has(name)) return;
    assert.ok(!visiting.has(name), `dependency cycle: ${[...stack, name].join(" -> ")}`);
    visiting.add(name);
    for (const dep of VERB_REGISTER[name].requires ?? []) {
      assert.ok(byName.has(dep), `${name}: requires unknown verb ${dep}`);
      visit(dep, [...stack, name]);
    }
    visiting.delete(name);
    visited.add(name);
  };
  for (const name of Object.keys(VERB_REGISTER)) visit(name, []);
});

test("every consumed noun is produced by some registered verb", () => {
  for (const [name, entry] of Object.entries(VERB_REGISTER)) {
    for (const noun of entry.consumes ?? []) {
      const producers = Object.keys(allProduces).filter((v) => allProduces[v].has(noun));
      assert.ok(producers.length > 0, `${name}: consumes noun with no producer: ${noun}`);
    }
  }
});

test("host verbs never claim engine measurement", () => {
  for (const [name, entry] of Object.entries(VERB_REGISTER)) {
    if (entry.domain !== "host") continue;
    for (const noun of entry.produces ?? []) {
      assert.ok(!MEASUREMENT_NOUNS.includes(noun), `${name}: host verb producing measurement noun ${noun}`);
    }
  }
});

test("engine verbs only surface measurement or populated material", () => {
  for (const [name, entry] of Object.entries(VERB_REGISTER)) {
    if (entry.domain !== "engine") continue;
    for (const noun of entry.produces ?? []) {
      assert.ok(
        MEASUREMENT_NOUNS.includes(noun) || POPULATE_NOUNS.includes(noun),
        `${name}: engine verb producing host noun ${noun}`,
      );
    }
  }
});

test("priors steer retrieval; they are never model context", () => {
  const priorsRouted = Object.entries(VERB_REGISTER).filter(([, e]) => e.domain === "priors");
  assert.deepEqual(priorsRouted, [], "no verb may route to priors as model context; priors steer via the engine (III.1)");
  const research = VERB_REGISTER.RESEARCH;
  assert.ok(research, "RESEARCH must exist to carry priors activation");
  assert.match(research.justification, /II\.2/, "RESEARCH must cite the giver test (II.2)");
  assert.match(research.justification, /III\.1/, "RESEARCH must cite derive-vs-receive (III.1)");
});

test("holonic-task.js documents its internal verbs", () => {
  const phases = ["PLANS", "RESEARCHES", "EXECUTES", "CITES", "ASSEMBLES"];
  for (const phase of phases) {
    assert.ok(holonicSrc.includes(phase), `holonic-task.js no longer documents ${phase}`);
  }
});

test("audit summary", () => {
  const toolCount = proxyTools.length;
  const registeredCount = Object.keys(VERB_REGISTER).length;
  const internalCount = INTERNAL_VERBS.length;
  const acyclic = true;
  console.error(`[verb-register] ${toolCount} proxy tools, ${registeredCount} registered (${internalCount} holonic internal), acyclic=${acyclic}`);
});
