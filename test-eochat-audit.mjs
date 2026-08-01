#!/usr/bin/env node
/**
 * EOChat Full Audit Test
 *
 * Starts proxy.js, imports multiple large documents, chats with them,
 * and audits EVERYTHING — every API call, every SSE event, every engine
 * operation, every grounding decision, every citation.
 *
 * HOW it does things is as important as WHAT it does.
 *
 * Run:
 *   node eoreader-chat/test-eochat-audit.mjs
 *   node eoreader-chat/test-eochat-audit.mjs --skip-llm    (API-only, no model calls)
 *   node eoreader-chat/test-eochat-audit.mjs --quick        (fewer docs, faster)
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PROXY_PATH = path.join(__dirname, "proxy.js");
const AUDIT_DIR = path.join(__dirname, "audit-output");

const SKIP_LLM = process.argv.includes("--skip-llm");
const QUICK = process.argv.includes("--quick");
const FRESH = process.argv.includes("--fresh");

const DOCS = QUICK
  ? [
      { path: path.join(ROOT, "pg84.txt"), label: "Frankenstein" },
    ]
  : [
      { path: path.join(ROOT, "pg84.txt"), label: "Frankenstein" },
      { path: path.join(process.env.HOME || "/Users/mlacy", "Downloads", "pg2600.txt"), label: "War and Peace" },
      { path: path.join(process.env.HOME || "/Users/mlacy", "Downloads", "pg10.txt"), label: "KJV Bible" },
      { path: path.join(process.env.HOME || "/Users/mlacy", "Downloads", "pg1342.txt"), label: "Pride and Prejudice" },
      { path: path.join(process.env.HOME || "/Users/mlacy", "Downloads", "pg5200.txt"), label: "Metamorphosis" },
    ];

const QUESTIONS = QUICK
  ? [
      "How does Victor Frankenstein react when the creature comes to life?",
      "What does the creature say to Victor about a companion?",
    ]
  : [
      "How does Victor Frankenstein react when the creature comes to life?",
      "What does Natasha Rostova feel at her first ball?",
      "Who is Mr. Darcy and how does Elizabeth first meet him?",
      "What transformation does Gregor Samsa undergo?",
      "Compare how Frankenstein's creature and Kafka's Gregor both experience isolation.",
      "What themes of creation and responsibility appear across Frankenstein and the Bible?",
      "Describe the social dynamics at the ball where Natasha meets Andrei Bolkonsky.",
    ];

// ── Audit trail ──

class AuditTrail {
  #entries = [];
  #startTime;

  constructor() {
    this.#startTime = Date.now();
  }

  log(phase, detail) {
    const entry = {
      ts: Date.now(),
      elapsed: Date.now() - this.#startTime,
      phase,
      ...detail,
    };
    this.#entries.push(entry);
    const prefix = `[+${entry.elapsed}ms]`;
    const summary = detail.summary || detail.error || detail.status || detail.name || phase;
    console.log(`${prefix} ${phase}: ${typeof summary === "string" ? summary.slice(0, 120) : JSON.stringify(summary).slice(0, 120)}`);
    return entry;
  }

  toJSON() {
    return {
      startTime: this.#startTime,
      endTime: Date.now(),
      duration: Date.now() - this.#startTime,
      entries: this.#entries,
      summary: {
        totalEvents: this.#entries.length,
        phases: [...new Set(this.#entries.map((e) => e.phase))],
        errors: this.#entries.filter((e) => e.error).length,
      },
    };
  }
}

const audit = new AuditTrail();

// ── HTTP helpers ──

const PROXY_URL = "http://localhost:11435";

async function checkProxyAlive() {
  try {
    const resp = await fetch(`${PROXY_URL}/health`, { signal: AbortSignal.timeout(3000) });
    return resp.ok;
  } catch {
    return false;
  }
}

async function apiGet(path, timeoutMs = 60000) {
  const t0 = Date.now();
  try {
    const resp = await fetch(`${PROXY_URL}${path}`, { signal: AbortSignal.timeout(timeoutMs) });
    const body = await resp.text();
    let json;
    try { json = JSON.parse(body); } catch { json = null; }
    audit.log("api_get", {
      path,
      status: resp.status,
      duration: Date.now() - t0,
      sizeBytes: body.length,
      summary: `${path} → ${resp.status} (${body.length}B, ${Date.now() - t0}ms)`,
    });
    return { status: resp.status, json, text: body };
  } catch (err) {
    audit.log("api_get_error", { path, error: err.message, duration: Date.now() - t0 });
    throw err;
  }
}

async function apiPost(path, body) {
  const t0 = Date.now();
  try {
    const resp = await fetch(`${PROXY_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000),
    });
    const text = await resp.text();
    let json;
    try { json = JSON.parse(text); } catch { json = null; }
    audit.log("api_post", {
      path,
      status: resp.status,
      duration: Date.now() - t0,
      sizeBytes: text.length,
      requestSize: JSON.stringify(body).length,
      summary: `POST ${path} → ${resp.status} (${text.length}B, ${Date.now() - t0}ms)`,
    });
    return { status: resp.status, json, text };
  } catch (err) {
    audit.log("api_post_error", { path, error: err.message, duration: Date.now() - t0 });
    throw err;
  }
}

// ── SSE stream parser ──

async function parseSSEStream(resp, onEvent) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const events = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentEvent = null;
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ") && currentEvent) {
        try {
          const data = JSON.parse(line.slice(6));
          const entry = { type: currentEvent, data, ts: Date.now() };
          events.push(entry);
          if (onEvent) onEvent(entry);
        } catch {}
        currentEvent = null;
      }
    }
  }
  return events;
}

// ── Chat with full SSE audit ──

async function chatWithAudit(question, sessionId = "audit-test") {
  const t0 = Date.now();
  audit.log("chat_start", { question: question.slice(0, 100), sessionId });

  let resp;
  try {
    resp = await fetch(`${PROXY_URL}/api/chat/tools`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: question }],
        session: sessionId,
        webSearch: false,
        groundBudget: 2400,
        groundMaxUnits: 16,
        groundLimit: 30,
      }),
      signal: AbortSignal.timeout(180000),
    });
  } catch (err) {
    audit.log("chat_error", { error: err.message, duration: Date.now() - t0 });
    return { error: err.message, events: [] };
  }

  const sseEvents = [];
  let groundingData = null;
  let toolCalls = [];
  let toolResults = [];
  let finalContent = "";
  let modelUsed = null;
  let llmRounds = 0;

  const events = await parseSSEStream(resp, (evt) => {
    sseEvents.push(evt);

    switch (evt.type) {
      case "grounding":
        groundingData = evt.data;
        audit.log("grounding", {
          sourceCount: evt.data.sourceCount,
          foldedCount: evt.data.foldedCount,
          tokens: evt.data.tokens,
          citationCount: evt.data.citations?.length || 0,
          gaps: evt.data.gaps?.length || 0,
          empty: evt.data.empty || false,
          summary: `Grounding: ${evt.data.sourceCount} sources, ${evt.data.foldedCount} folded, ${evt.data.tokens} tokens, ${evt.data.citations?.length || 0} citations`,
        });
        break;

      case "tools_available":
        audit.log("tools_available", { count: evt.data.count });
        break;

      case "llm_call":
        llmRounds++;
        modelUsed = evt.data.model;
        audit.log("llm_call", {
          round: evt.data.round,
          model: evt.data.model,
          toolCount: evt.data.tools,
          summary: `LLM call round ${evt.data.round}: model=${evt.data.model}, tools=${evt.data.tools}`,
        });
        break;

      case "tool_calls":
        for (const c of evt.data.calls) {
          toolCalls.push(c);
          audit.log("tool_call", {
            name: c.name,
            args: c.args?.slice(0, 200),
            summary: `Tool call: ${c.name}(${(c.args || "").slice(0, 80)})`,
          });
        }
        break;

      case "tool_result":
        toolResults.push({ name: evt.data.name, result: evt.data.result });
        audit.log("tool_result", {
          name: evt.data.name,
          resultPreview: evt.data.result?.slice(0, 200),
          summary: `Tool result: ${evt.data.name} → ${(evt.data.result || "").slice(0, 100)}`,
        });
        break;

      case "tool_call_salvaged":
        audit.log("tool_salvaged", {
          count: evt.data.count,
          names: evt.data.names,
          summary: `Salvaged ${evt.data.count} tool calls from prose: ${evt.data.names?.join(", ")}`,
        });
        break;

      case "source_added":
        audit.log("source_added", {
          name: evt.data.name,
          url: evt.data.url,
          size: evt.data.size,
        });
        break;

      case "gap":
        audit.log("gap", {
          type: evt.data.type,
          reason: evt.data.reason || evt.data.url,
          summary: `Gap: ${evt.data.type} — ${evt.data.reason || evt.data.url}`,
        });
        break;

      case "response":
        finalContent = evt.data.content;
        modelUsed = evt.data.model || modelUsed;
        audit.log("response", {
          model: evt.data.model,
          contentLength: evt.data.content?.length || 0,
          contentPreview: evt.data.content?.slice(0, 200),
          summary: `Response (${evt.data.model}): ${(evt.data.content || "").slice(0, 100)}`,
        });
        break;

      case "done":
        finalContent = evt.data.content || finalContent;
        break;

      case "error":
        audit.log("sse_error", { message: evt.data.message });
        break;
    }
  });

  const duration = Date.now() - t0;

  const chatAudit = {
    question,
    sessionId,
    duration,
    modelUsed,
    llmRounds,
    grounding: groundingData ? {
      sourceCount: groundingData.sourceCount,
      foldedCount: groundingData.foldedCount,
      tokens: groundingData.tokens,
      citations: (groundingData.citations || []).map((c) => ({
        index: c.index,
        source: c.source_id,
        byteRange: `${c.byte_start}-${c.byte_end}`,
        score: c.score,
        textPreview: c.text?.slice(0, 100),
      })),
      gaps: groundingData.gaps,
      systemContextPreview: groundingData.systemContext?.slice(0, 300),
    } : null,
    toolCalls: toolCalls.map((c) => ({ name: c.name, args: c.args })),
    toolResults: toolResults.map((r) => ({ name: r.name, resultPreview: r.result?.slice(0, 300) })),
    finalContent,
    finalContentLength: finalContent?.length || 0,
    sseEventTypes: [...new Set(sseEvents.map((e) => e.type))],
    sseEventCount: sseEvents.length,
    citationsInResponse: (finalContent || "").match(/\[\d+\]/g) || [],
  };

  audit.log("chat_complete", {
    question: question.slice(0, 80),
    duration,
    model: modelUsed,
    llmRounds,
    toolCallCount: toolCalls.length,
    groundingCitations: groundingData?.citations?.length || 0,
    citationsUsed: chatAudit.citationsInResponse.length,
    sseEvents: sseEvents.length,
    summary: `Chat done in ${duration}ms: model=${modelUsed}, rounds=${llmRounds}, tools=${toolCalls.length}, grounding citations=${groundingData?.citations?.length || 0}, response citations=${chatAudit.citationsInResponse.length}`,
  });

  return chatAudit;
}

// ── Start proxy ──

async function startProxy() {
  audit.log("proxy_start", { summary: "Starting proxy.js..." });

  if (!FRESH) {
    const existing = await fetch(`${PROXY_URL}/health`, { signal: AbortSignal.timeout(2000) }).catch(() => null);
    if (existing?.ok) {
      audit.log("proxy_start", { summary: "Proxy already running, reusing" });
      return null;
    }
  } else {
    // Kill any existing proxy
    const { execSync } = await import("node:child_process");
    try { execSync("pkill -f 'node.*proxy.js'", { stdio: "ignore" }); } catch {}
    await new Promise((r) => setTimeout(r, 1500));
    audit.log("proxy_start", { summary: "Killed existing proxy (--fresh)" });
  }

  const proc = spawn("node", [PROXY_PATH, "--port=11435", `--repo=${ROOT}`], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env },
  });

  let stderr = "";
  proc.stderr.on("data", (d) => { stderr += d.toString(); });
  proc.stdout.on("data", (d) => {
    const line = d.toString().trim();
    if (line.includes("EO_PROXY_READY")) {
      audit.log("proxy_ready", { summary: `Proxy ready: ${line}` });
    }
  });

  // Wait for ready
  for (let i = 0; i < 120; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    try {
      const resp = await fetch(`${PROXY_URL}/health`, { signal: AbortSignal.timeout(2000) });
      if (resp.ok) {
        const data = await resp.json();
        audit.log("proxy_ready", {
          storeSize: data.store_size,
          uptime: data.uptime,
          summary: `Proxy ready: store=${data.store_size}, uptime=${data.uptime}s`,
        });
        return proc;
      }
    } catch {}
  }

  audit.log("proxy_start_failed", { stderr: stderr.slice(-500) });
  throw new Error("Proxy did not start within 120s");
}

// ── Main test ──

async function main() {
  console.log("═".repeat(70));
  console.log("  EOChat Full Audit Test");
  console.log(`  Mode: ${SKIP_LLM ? "API-only (no LLM)" : "Full (with LLM)"}`);
  console.log(`  Documents: ${DOCS.length}`);
  console.log(`  Questions: ${QUESTIONS.length}`);
  console.log("═".repeat(70));

  await fsp.mkdir(AUDIT_DIR, { recursive: true });

  let proxyProc = null;
  try {
    // 1. Start proxy
    proxyProc = await startProxy();

    // 2. Health check
    audit.log("phase", { summary: "── Phase 1: Health & Stats ──" });
    const health = await apiGet("/health");
    const stats = await apiGet("/stats");
    const models = await apiGet("/v1/models");

    // 3. Import documents
    audit.log("phase", { summary: "── Phase 2: Document Import ──" });
    const importResults = [];
    for (const doc of DOCS) {
      const t0 = Date.now();
      const exists = fs.existsSync(doc.path);
      if (!exists) {
        audit.log("import_skip", { label: doc.label, path: doc.path, error: "File not found" });
        importResults.push({ label: doc.label, skipped: true, reason: "not found" });
        continue;
      }
      const fileSize = fs.statSync(doc.path).size;
      audit.log("import_start", {
        label: doc.label,
        path: doc.path,
        sizeBytes: fileSize,
        summary: `Importing ${doc.label} (${(fileSize / 1024).toFixed(0)}KB)...`,
      });

      try {
        const result = await apiPost("/api/ingest", { path: doc.path });
        const duration = Date.now() - t0;
        importResults.push({
          label: doc.label,
          ...result.json,
          fileSize,
          duration,
        });
        audit.log("import_done", {
          label: doc.label,
          chunks: result.json?.chunks,
          pool: result.json?.pool,
          duration,
          summary: `Imported ${doc.label}: ${result.json?.chunks} chunks in ${duration}ms`,
        });
      } catch (err) {
        audit.log("import_error", { label: doc.label, error: err.message });
        importResults.push({ label: doc.label, error: err.message });
      }
    }

    // 4. Check engine state after imports
    audit.log("phase", { summary: "── Phase 3: Engine State After Import ──" });
    const engineStats = await apiGet("/api/grounded/stats");
    const sources = await apiGet("/api/sources");
    const discourseStats = await apiGet("/api/discourse/stats?session=audit-test");

    audit.log("engine_state", {
      ingestedChunks: engineStats.json?.ingestedChunks,
      ingestedFiles: engineStats.json?.ingestedFiles,
      sourceCount: sources.json?.length,
      pools: engineStats.json?.pools,
      summary: `Engine: ${engineStats.json?.ingestedChunks} chunks, ${engineStats.json?.ingestedFiles} files, ${sources.json?.length} sources`,
    });

    // 5. Test verbatim search (no LLM needed)
    audit.log("phase", { summary: "── Phase 4: Verbatim Search (engine-only) ──" });
    if (!(await checkProxyAlive())) {
      audit.log("proxy_dead", { summary: "Proxy crashed before Phase 4, restarting..." });
      proxyProc = await startProxy();
    }
    const searchQueries = [
      "Victor Frankenstein creature monster",
      "Natasha ball dance",
      "Elizabeth Darcy pride",
      "Gregor Samsa transformation insect",
      "In the beginning God created",
    ];

    const searchResults = [];
    for (const q of searchQueries) {
      if (!(await checkProxyAlive())) {
        audit.log("proxy_dead_during_search", { summary: `Proxy died during search phase, skipping remaining searches` });
        break;
      }
      const t0 = Date.now();
      try {
        const result = await apiGet(`/api/verbatim?q=${encodeURIComponent(q)}&limit=3`, 120000);
        const duration = Date.now() - t0;
        searchResults.push({
          query: q,
          total: result.json?.total,
          passages: result.json?.passages?.length,
          pool: result.json?.pool,
          gaps: result.json?.gaps,
          duration,
        });
        audit.log("verbatim_search", {
          query: q,
          total: result.json?.total,
          passages: result.json?.passages?.length,
          duration,
          summary: `Search "${q}": ${result.json?.total} total, ${result.json?.passages?.length} passages in ${duration}ms`,
        });
      } catch (err) {
        audit.log("verbatim_search_error", { query: q, error: err.message });
        searchResults.push({ query: q, error: err.message });
      }
    }

    // 6. Test segment reading
    audit.log("phase", { summary: "── Phase 5: Segment Reading ──" });
    try {
      const segResult = await apiGet(`/api/verbatim/segment?q=${encodeURIComponent("Victor Frankenstein laboratory")}&max=5000`, 120000);
      audit.log("segment_read", {
        segment: segResult.json?.segment,
        source: segResult.json?.source,
        byteRange: `${segResult.json?.byte_start}-${segResult.json?.byte_end}`,
        textLength: segResult.json?.text?.length,
        summary: `Segment: "${segResult.json?.segment}" from ${segResult.json?.source}`,
      });
    } catch (err) {
      audit.log("segment_error", { error: err.message });
    }

    // 7. Test discourse persistence
    audit.log("phase", { summary: "── Phase 6: Discourse Persistence ──" });
    await apiPost("/api/discourse/message", {
      sessionId: "audit-test",
      role: "user",
      content: "This is a test message for discourse persistence.",
    });
    await apiPost("/api/discourse/message", {
      sessionId: "audit-test",
      role: "assistant",
      content: "This is a test response stored in the discourse store.",
    });
    const discourseAfter = await apiGet("/api/discourse/stats?session=audit-test");
    const discourseSession = await apiGet("/api/discourse?session=audit-test");
    audit.log("discourse_test", {
      messageCount: discourseAfter.json?.messageCount,
      tokens: discourseAfter.json?.tokens,
      usagePercent: discourseAfter.json?.usagePercent,
      summary: `Discourse: ${discourseAfter.json?.messageCount} messages, ${discourseAfter.json?.tokens} tokens (${discourseAfter.json?.usagePercent}%)`,
    });

    // 8. Test priors
    audit.log("phase", { summary: "── Phase 7: Priors ──" });
    try {
      const priors = await apiGet("/api/priors");
      audit.log("priors_catalog", {
        count: priors.json?.count,
        pool: priors.json?.pool,
        gaps: priors.json?.gaps,
        priorsList: priors.json?.priors?.map((p) => p.id || p.name),
        summary: `Priors: ${priors.json?.count} in pool "${priors.json?.pool}"`,
      });

      if (priors.json?.priors?.length > 0) {
        const firstPrior = priors.json.priors[0];
        const priorId = firstPrior.id || firstPrior.name;
        if (priorId) {
          const priorRead = await apiGet(`/api/priors/read?id=${encodeURIComponent(priorId)}`);
          audit.log("priors_read", {
            id: priorId,
            status: priorRead.status,
            summary: `Read prior "${priorId}" → ${priorRead.status}`,
          });
        }

        const priorSearch = await apiGet(`/api/priors/search?q=${encodeURIComponent("entity")}&limit=3`);
        audit.log("priors_search", {
          query: "entity",
          results: priorSearch.json?.results?.length || 0,
          summary: `Prior search "entity": ${priorSearch.json?.results?.length || 0} results`,
        });
      }
    } catch (err) {
      audit.log("priors_error", { error: err.message });
    }

    // 9. Chat with documents (requires LLM)
    if (!SKIP_LLM) {
      audit.log("phase", { summary: "── Phase 8: Chat with Documents (LLM) ──" });
      if (!(await checkProxyAlive())) {
        audit.log("proxy_dead", { summary: "Proxy crashed before chat phase, restarting..." });
        proxyProc = await startProxy();
      }
      const chatResults = [];
      for (const q of QUESTIONS) {
        const result = await chatWithAudit(q, "audit-test");
        chatResults.push(result);

        // Check citation grounding
        if (result.grounding?.citations?.length > 0) {
          const citedNums = result.citationsInResponse;
          const availableNums = result.grounding.citations.map((c) => `[${c.index}]`);
          audit.log("citation_audit", {
            question: q.slice(0, 60),
            available: availableNums,
            used: citedNums,
            grounded: citedNums.every((c) => availableNums.includes(c)),
            summary: `Citations: available=${availableNums.join(",")}, used=${citedNums.join(",") || "none"}`,
          });
        }
      }

      // 10. Check discourse after chat
      audit.log("phase", { summary: "── Phase 9: Post-Chat Discourse State ──" });
      const finalDiscourse = await apiGet("/api/discourse/stats?session=audit-test");
      const finalSession = await apiGet("/api/discourse?session=audit-test");
      audit.log("final_discourse", {
        messageCount: finalDiscourse.json?.messageCount,
        tokens: finalDiscourse.json?.tokens,
        usagePercent: finalDiscourse.json?.usagePercent,
        foldCount: finalDiscourse.json?.foldCount,
        attachmentCount: finalDiscourse.json?.attachmentCount,
        summary: `Final discourse: ${finalDiscourse.json?.messageCount} msgs, ${finalDiscourse.json?.tokens} tokens, ${finalDiscourse.json?.foldCount} folds`,
      });
    } else {
      audit.log("phase", { summary: "── Phase 8: SKIPPED (LLM disabled) ──" });
    }

    // 11. Final engine stats
    audit.log("phase", { summary: "── Phase 10: Final State ──" });
    const finalEngineStats = await apiGet("/api/grounded/stats");
    const finalSources = await apiGet("/api/sources");
    const finalHealth = await apiGet("/health");

    audit.log("final_state", {
      engineChunks: finalEngineStats.json?.ingestedChunks,
      engineFiles: finalEngineStats.json?.ingestedFiles,
      sourceCount: finalSources.json?.length,
      storeSize: finalHealth.json?.store_size,
      summary: `Final: ${finalEngineStats.json?.ingestedChunks} chunks, ${finalSources.json?.length} sources, store=${finalHealth.json?.store_size}`,
    });

    // ── Write audit report ──
    const auditData = audit.toJSON();
    const reportPath = path.join(AUDIT_DIR, `audit-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.json`);
    await fsp.writeFile(reportPath, JSON.stringify(auditData, null, 2));

    const humanReport = generateHumanReport(auditData, {
      importResults,
      searchResults,
      engineStats: engineStats.json,
      finalEngineStats: finalEngineStats.json,
    });
    const reportMdPath = reportPath.replace(/\.json$/, ".md");
    await fsp.writeFile(reportMdPath, humanReport);

    console.log("\n" + "═".repeat(70));
    console.log("  AUDIT COMPLETE");
    console.log("═".repeat(70));
    console.log(`  JSON: ${reportPath}`);
    console.log(`  Report: ${reportMdPath}`);
    console.log(`  Duration: ${auditData.duration}ms`);
    console.log(`  Events: ${auditData.summary.totalEvents}`);
    console.log(`  Errors: ${auditData.summary.errors}`);
    console.log("═".repeat(70));

    // Cleanup
    if (proxyProc) {
      proxyProc.kill("SIGTERM");
      await new Promise((r) => setTimeout(r, 2000));
    }
  } catch (err) {
    audit.log("fatal", { error: err.message, stack: err.stack });
    console.error(`\nFATAL: ${err.message}`);
    console.error(err.stack);

    const auditData = audit.toJSON();
    const reportPath = path.join(AUDIT_DIR, `audit-FATAL-${Date.now()}.json`);
    await fsp.writeFile(reportPath, JSON.stringify(auditData, null, 2));
    console.log(`Audit saved: ${reportPath}`);

    if (proxyProc) proxyProc.kill("SIGTERM");
    process.exit(1);
  }
}

// ── Human-readable report ──

function generateHumanReport(audit, context) {
  const lines = [];
  const p = (s) => lines.push(s);

  p("# EOChat Full Audit Report");
  p("");
  p(`**Date:** ${new Date(audit.startTime).toISOString()}`);
  p(`**Duration:** ${(audit.duration / 1000).toFixed(1)}s`);
  p(`**Total Events:** ${audit.summary.totalEvents}`);
  p(`**Errors:** ${audit.summary.errors}`);
  p(`**Mode:** ${SKIP_LLM ? "API-only (no LLM)" : "Full (with LLM)"}`);
  p(`**Documents:** ${DOCS.length}`);
  p(`**Questions:** ${QUESTIONS.length}`);
  p("");

  // Timeline
  p("## Timeline");
  p("");
  p("| Elapsed | Phase | Detail |");
  p("|---------|-------|--------|");
  for (const e of audit.entries) {
    const detail = e.summary || e.error || e.status || "";
    p(`| ${e.elapsed}ms | ${e.phase} | ${String(detail).slice(0, 100)} |`);
  }
  p("");

  // Document imports
  p("## Document Imports");
  p("");
  for (const r of context.importResults) {
    if (r.skipped) {
      p(`- **${r.label}**: SKIPPED (${r.reason})`);
    } else if (r.error) {
      p(`- **${r.label}**: ERROR — ${r.error}`);
    } else {
      p(`- **${r.label}**: ${r.chunks} chunks, ${(r.fileSize / 1024).toFixed(0)}KB, ${r.duration}ms`);
    }
  }
  p("");

  // Engine state
  p("## Engine State");
  p("");
  p(`- Ingested chunks: ${context.finalEngineStats?.ingestedChunks || context.engineStats?.ingestedChunks}`);
  p(`- Ingested files: ${context.finalEngineStats?.ingestedFiles || context.engineStats?.ingestedFiles}`);
  p(`- Pools: ${JSON.stringify(context.engineStats?.pools)}`);
  p("");

  // Verbatim searches
  p("## Verbatim Searches (Engine-Only, No LLM)");
  p("");
  for (const s of context.searchResults) {
    if (s.error) {
      p(`- "${s.query}": ERROR — ${s.error}`);
    } else {
      p(`- "${s.query}": ${s.total} total, ${s.passages} passages, ${s.duration}ms`);
    }
  }
  p("");

  // Chat results
  if (!SKIP_LLM) {
    p("## Chat Results (with LLM)");
    p("");
    const chatEntries = audit.entries.filter((e) => e.phase === "chat_complete");
    for (const e of chatEntries) {
      p(`### Q: ${e.question || "(unknown)"}`);
      const chatEvt = audit.entries.find((x) => x.phase === "chat_start" && x.question === e.question);
      p("");
      p(`- **Duration:** ${e.duration}ms`);
      p(`- **Model:** ${e.model}`);
      p(`- **LLM Rounds:** ${e.llmRounds}`);
      p(`- **Tool Calls:** ${e.toolCallCount}`);
      p(`- **Grounding Citations:** ${e.groundingCitations}`);
      p(`- **Citations Used in Response:** ${e.citationsUsed}`);
      p(`- **SSE Events:** ${e.sseEvents}`);
      p("");
    }

    // Citation audit
    p("## Citation Audit (HOW citations work)");
    p("");
    const citationAudits = audit.entries.filter((e) => e.phase === "citation_audit");
    for (const ca of citationAudits) {
      p(`- **Q:** "${ca.question}"`);
      p(`  - Available: ${ca.available}`);
      p(`  - Used: ${ca.used || "none"}`);
      p(`  - All grounded: ${ca.grounded ? "YES" : "NO — some citations may be fabricated"}`);
      p("");
    }
  }

  // Tool call breakdown
  p("## Tool Call Breakdown");
  p("");
  const toolCalls = audit.entries.filter((e) => e.phase === "tool_call");
  const toolCounts = {};
  for (const tc of toolCalls) {
    toolCounts[tc.name] = (toolCounts[tc.name] || 0) + 1;
  }
  for (const [name, count] of Object.entries(toolCounts)) {
    p(`- **${name}**: ${count} calls`);
  }
  p("");

  // SSE event types
  p("## SSE Event Types");
  p("");
  const sseTypes = {};
  for (const e of audit.entries) {
    if (["grounding", "llm_call", "tool_call", "tool_result", "response", "gap", "source_added", "tool_salvaged"].includes(e.phase)) {
      sseTypes[e.phase] = (sseTypes[e.phase] || 0) + 1;
    }
  }
  for (const [type, count] of Object.entries(sseTypes)) {
    p(`- **${type}**: ${count}`);
  }
  p("");

  // Gaps and errors
  const gaps = audit.entries.filter((e) => e.phase === "gap" || e.phase === "sse_error");
  if (gaps.length > 0) {
    p("## Gaps & Errors");
    p("");
    for (const g of gaps) {
      p(`- [${g.phase}] ${g.type || ""}: ${g.reason || g.message || g.error}`);
    }
    p("");
  }

  // Errors
  const errors = audit.entries.filter((e) => e.error);
  if (errors.length > 0) {
    p("## All Errors");
    p("");
    for (const e of errors) {
      p(`- [${e.phase}] ${e.error}`);
    }
    p("");
  }

  return lines.join("\n");
}

main().catch((err) => {
  console.error("Unhandled:", err);
  process.exit(1);
});
