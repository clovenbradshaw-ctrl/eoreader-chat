// Routes a chat turn to one of N candidate Ollama models by reusing
// eoreader5's predictive-competency substrate (@eoreader/engine) instead of
// a bespoke heuristic. See eoreader5/docs/predictive-competency-phase0.md.
//
// This file is the IMPURE boundary: it owns the one Math.random() call, the
// logical step counter, and JSONL persistence. Every call into
// @eoreader/engine stays pure — no ambient time/randomness crosses that
// import boundary (eoreader5/docs/invariants.md).
//
// Framing:
//   - One PredictionTask, shared by all candidates: the target is whether
//     this chat turn's tool loop finished cleanly ("success"/"failure"), a
//     categorical outcome scored with log-loss (a proper scoring rule for
//     categorical kinds — packages/engine/prediction/scoring).
//   - One candidate per literal model id actually served — identity is the
//     resolved model string, never a role label like "tiny"/"medium"
//     (the nameless-referent principle: identity lives in the referent).
//   - Each candidate's "prediction", committed before the turn runs, is its
//     own Laplace-smoothed historical success rate: the Bernoulli analogue
//     of the numeric last-value/global-mean baselines in
//     packages/engine/prediction/baselines (which only covers numeric
//     series, hence reimplemented here for a categorical target).
//   - Two honest weak baselines, scored the same way: "uniform" (p=0.5, no
//     information) and "global-mean" (the pooled success rate across every
//     candidate seen so far). Baselines need no leakage guard — they're
//     deterministic functions of already-visible history recomputed at
//     reveal time, never committed ahead of time (same idiom as the numeric
//     baselines module).
//   - Routing samples a candidate proportional to a softmax of its
//     competency gain over baseline:global-mean, with an epsilon floor so
//     the router keeps exploring instead of collapsing to one arm forever.
//     Below a warmup threshold per candidate, it defers to the pre-existing
//     heuristic (selectModel) instead of guessing — a cold start is never a
//     silent coin flip dressed up as a measured decision.
//
// "Before without a clock": the engine reads no ambient time, so ordering is
// a caller-supplied logical step index. Here that's an in-memory counter,
// reconstructed from the persisted ledger on load. A commitment made at step
// t declares reveal_not_before_step = t+1, and this module always reveals at
// exactly t+1, right after the turn finishes — so the leakage guard is real:
// a bug that read the outcome before committing would throw, not silently
// score wrong.

import fs from "fs";
import fsp from "fs/promises";

import { createPredictionTask } from "@eoreader/engine/prediction/tasks";
import { commitPrediction, revealAndScore } from "@eoreader/engine/prediction/commitments";
import { score as scoreOutput } from "@eoreader/engine/prediction/scoring";
import { createLedger, recordStep, competencyGain } from "@eoreader/engine/competency/ledger";

const BASELINE_UNIFORM = "baseline:uniform";
const BASELINE_GLOBAL_MEAN = "baseline:global-mean";
const BASELINE_IDS = [BASELINE_UNIFORM, BASELINE_GLOBAL_MEAN];
const ESTIMATOR_VERSION = "self-frequency-v1";

// ── Pure helpers (no I/O, no randomness — unit-testable in isolation) ──

/** Laplace-smoothed Bernoulli rate: never exactly 0 or 1 from finite data. */
function smoothedRate(successes, attempts) {
  return (successes + 1) / (attempts + 2);
}

function categoricalOf(pSuccess) {
  return Object.freeze({ kind: "categorical", probs: { success: pSuccess, failure: 1 - pSuccess } });
}

/** Softmax over gains with a temperature, blended with a uniform exploration floor. */
export function selectionWeights(gains, { temperature = 1, epsilon = 0.05 } = {}) {
  const n = gains.length;
  const scaled = gains.map((g) => g / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => (1 - epsilon) * (e / sum) + epsilon / n);
}

/** Sample an index from `weights` (summing to ~1) given a uniform draw u in [0,1). */
export function sampleIndex(weights, u) {
  let acc = 0;
  for (let i = 0; i < weights.length; i += 1) {
    acc += weights[i];
    if (u < acc) return i;
  }
  return weights.length - 1;
}

// competencyGain from the ledger is Σ over every observation — correct for
// scoring one fixed candidate over a whole series, but candidates here are
// pulled unevenly by the very selection process this feeds, so summing lets
// whichever arm gets played more accumulate more total gain regardless of
// per-turn quality (a bandit rich-get-richer artifact, caught empirically
// while building this). Dividing by observations turns it into a mean
// advantage per turn.
function meanCompetencyGain(ledger, baselineId) {
  return ledger.observations > 0 ? competencyGain(ledger)[baselineId] / ledger.observations : 0;
}

// log-odds: the natural monotonic transform for combining/ranking Bernoulli
// rates (0.5 -> 0, confidence grows without bound as p -> 0 or 1). p is
// always Laplace-smoothed before this is called, so it's never infinite.
function logit(p) {
  return Math.log(p / (1 - p));
}

// ── The router ──

export function createModelRouter({
  candidates,
  ledgerPath,
  heuristicFallback,
  warmupPerCandidate = 3,
  temperature = 1,
  epsilon = 0.05,
  population = "eoreader-chat:tool-loop-v1",
} = {}) {
  if (!Array.isArray(candidates) || candidates.length < 2) {
    throw new TypeError("createModelRouter: candidates must list at least 2 model ids");
  }
  if (typeof heuristicFallback !== "function") {
    throw new TypeError("createModelRouter: heuristicFallback(messages) is required for cold start");
  }

  const task = createPredictionTask({
    target_type: "chat-tool-loop-outcome",
    horizon: { kind: "immediate" },
    scoring_rule: "log-loss",
    baseline_ids: BASELINE_IDS,
    population,
  });

  const stats = new Map(); // candidate_id -> { successes, attempts }
  const ledgers = new Map(); // candidate_id -> ledger (immutable; replaced on each recordStep)
  for (const id of candidates) {
    stats.set(id, { successes: 0, attempts: 0 });
    ledgers.set(id, createLedger({ task_id: task.id, candidate_id: id, baseline_ids: BASELINE_IDS }));
  }
  const pooled = { successes: 0, attempts: 0 };
  let nextStep = 0;
  let writeQueue = Promise.resolve(); // serializes appends so concurrent reveals don't interleave writes

  function uniformBaselineOutput() {
    return categoricalOf(0.5);
  }

  function globalMeanBaselineOutput() {
    return categoricalOf(smoothedRate(pooled.successes, pooled.attempts));
  }

  function load() {
    if (!fs.existsSync(ledgerPath)) return;
    const lines = fs.readFileSync(ledgerPath, "utf8").split("\n").filter(Boolean);
    for (const line of lines) {
      let entry;
      try { entry = JSON.parse(line); } catch { continue; }
      // A different task_id (config changed) or an unknown candidate (a model
      // that's no longer served) carries no comparable meaning here — skip
      // rather than fold it in and silently mis-attribute history.
      if (entry.task_id !== task.id || !ledgers.has(entry.candidate_id)) continue;
      ledgers.set(entry.candidate_id, recordStep(ledgers.get(entry.candidate_id), {
        candidate_loss: entry.loss,
        baseline_losses: entry.baseline_losses,
        proper: entry.proper,
      }));
      const s = stats.get(entry.candidate_id);
      s.attempts += 1;
      if (entry.observed === "success") s.successes += 1;
      pooled.attempts += 1;
      if (entry.observed === "success") pooled.successes += 1;
      if (Number.isInteger(entry.step) && entry.step >= nextStep) nextStep = entry.step + 1;
    }
  }

  // The routing weight is each candidate's OWN success rate, not its
  // competency gain — competency gain vs a POOLED baseline is contaminated
  // across candidates: a baseline mixing both candidates' history ends up
  // well-calibrated for whichever one dominates the mix and badly
  // miscalibrated for the other, which inflates the worse candidate's
  // apparent gain (it "correctly predicts its own frequent failure" against
  // a baseline that assumed better). Caught empirically: a candidate that
  // failed 70% of the time out-scored one that always succeeded. The ledger
  // machinery still earns its keep as a GATE, not a ranking signal: a
  // candidate's own rate is trusted for routing only once modeling its own
  // history has been scored as beating baseline:uniform (the honest
  // no-information reference) under log-loss — i.e. only once there's
  // measured evidence the rate isn't noise. Until then its contribution
  // falls back to the pooled rate, so an untrusted candidate doesn't get
  // penalized OR rewarded for a rate that hasn't earned trust yet.
  function routingSignal() {
    const belowWarmup = candidates.some((id) => stats.get(id).attempts < warmupPerCandidate);
    const perCandidate = candidates.map((id) => {
      const ledger = ledgers.get(id);
      const ownRate = smoothedRate(stats.get(id).successes, stats.get(id).attempts);
      const trustworthy = meanCompetencyGain(ledger, BASELINE_UNIFORM) > 0;
      const effectiveRate = trustworthy ? ownRate : smoothedRate(pooled.successes, pooled.attempts);
      return {
        candidateId: id,
        ownRate,
        trustworthy,
        effectiveRate,
        meanGainVsUniform: meanCompetencyGain(ledger, BASELINE_UNIFORM),
        meanGainVsGlobalMean: meanCompetencyGain(ledger, BASELINE_GLOBAL_MEAN),
      };
    });
    const weights = belowWarmup ? null : selectionWeights(perCandidate.map((c) => logit(c.effectiveRate)), { temperature, epsilon });
    return { belowWarmup, perCandidate, weights };
  }

  /**
   * Choose a model for this turn and commit its self-predicted success
   * probability before the turn runs. Synchronous and side-effect-free
   * beyond the in-memory step counter — the caller must eventually call
   * reveal(ctx, outcome) once the turn's tool-loop outcome is known.
   */
  function pick(messages) {
    const { belowWarmup, weights } = routingSignal();
    let chosen;
    if (belowWarmup) {
      chosen = heuristicFallback(messages);
      if (!candidates.includes(chosen)) {
        console.error(`[model-router] heuristic returned unknown candidate ${JSON.stringify(chosen)}; falling back to ${candidates[0]}`);
        chosen = candidates[0];
      }
    } else {
      chosen = candidates[sampleIndex(weights, Math.random())];
    }

    const step = nextStep;
    nextStep += 1;
    const s = stats.get(chosen);
    const commitment = commitPrediction({
      task_id: task.id,
      candidate_id: chosen,
      candidate_version_hash: ESTIMATOR_VERSION,
      input_snapshot_hash: `attempts:${s.attempts}/successes:${s.successes}@step:${step}`,
      predictive_output: categoricalOf(smoothedRate(s.successes, s.attempts)),
      committed_at_step: step,
      reveal_not_before_step: step + 1,
    });

    return { model: chosen, ctx: { commitment, step, candidateId: chosen, usedHeuristic: belowWarmup } };
  }

  /** Reveal the turn's outcome, score it, fold it into the ledger, and persist it. */
  async function reveal(ctx, outcome) {
    if (outcome !== "success" && outcome !== "failure") {
      throw new TypeError(`modelRouter.reveal: outcome must be "success" or "failure", got ${JSON.stringify(outcome)}`);
    }
    const candidateScore = revealAndScore({
      commitment: ctx.commitment,
      observed: outcome,
      revealed_at_step: ctx.step + 1,
      scoring_rule: "log-loss",
    });
    const baselineLosses = {
      [BASELINE_UNIFORM]: scoreOutput(uniformBaselineOutput(), outcome, { rule: "log-loss" }).loss,
      [BASELINE_GLOBAL_MEAN]: scoreOutput(globalMeanBaselineOutput(), outcome, { rule: "log-loss" }).loss,
    };

    ledgers.set(ctx.candidateId, recordStep(ledgers.get(ctx.candidateId), {
      candidate_loss: candidateScore.loss,
      baseline_losses: baselineLosses,
      proper: candidateScore.proper,
    }));

    const s = stats.get(ctx.candidateId);
    s.attempts += 1;
    if (outcome === "success") s.successes += 1;
    pooled.attempts += 1;
    if (outcome === "success") pooled.successes += 1;

    const line = JSON.stringify({
      task_id: task.id,
      candidate_id: ctx.candidateId,
      step: ctx.step,
      observed: outcome,
      loss: candidateScore.loss,
      baseline_losses: baselineLosses,
      proper: candidateScore.proper,
      used_heuristic: ctx.usedHeuristic,
      ts: new Date().toISOString(), // informational only; never read back into a pure call
    }) + "\n";
    writeQueue = writeQueue
      .then(() => fsp.appendFile(ledgerPath, line))
      .catch((err) => console.error(`[model-router] failed to persist ledger entry: ${err.message}`));
    await writeQueue;
  }

  /** Read-only snapshot for a debug endpoint — never used to make a routing decision itself. */
  function describe() {
    const { belowWarmup, perCandidate, weights } = routingSignal();
    return {
      task_id: task.id,
      mode: belowWarmup ? "warmup (heuristic fallback)" : "learned (softmax routing)",
      candidates: perCandidate.map((c, i) => ({
        candidate_id: c.candidateId,
        attempts: stats.get(c.candidateId).attempts,
        successes: stats.get(c.candidateId).successes,
        own_rate: c.ownRate,
        trustworthy: c.trustworthy,
        effective_rate: c.effectiveRate,
        mean_competency_gain_vs_uniform: c.meanGainVsUniform,
        mean_competency_gain_vs_global_mean: c.meanGainVsGlobalMean,
        selection_probability: weights ? weights[i] : null,
      })),
      pooled_rate: smoothedRate(pooled.successes, pooled.attempts),
      next_step: nextStep,
    };
  }

  load();

  return Object.freeze({ pick, reveal, describe, task });
}
