# Does the scaffold make a small model honest?

**Status: draft. The methods and the "found along the way" section are measured; the results
section is empty until the pilot runs.** Nothing below the "Results" heading may be written from
memory or estimate. Every number there comes from `study.md` / `study.json`, produced by
`python -m backend.dev.study report`. The plan and its history are in
[`docs/decisions/2026-10-05-scaffold-vs-model.md`](../decisions/2026-10-05-scaffold-vs-model.md).

---

## The question

This project is an agent that investigates a table and reports what the data supports. Its central
claim is about **architecture, not model**: the language model only plans and interprets, while
deterministic code does every calculation, a typed termination policy decides when there's enough
evidence, and a critic and a contradiction check stop it from affirming things the data doesn't
back.

If that's true, the scaffold should make a *small* model behave much more like a large one,
because the parts that need to be right are the parts the model isn't doing. If it's false, the
scaffold is decoration and a good model is doing all the work.

Every measurement so far held the scaffold fixed and varied nothing, so it couldn't say which.
This study varies both:

- **Model:** one open-weight family at several sizes (Qwen3 1.7B / 4B / 8B), so size isn't
  confounded with training recipe. `gpt-5.4-mini` is the frontier reference, and the
  rule-based policy (the scaffold with *no* model) is the size-zero point.
- **Condition:**
  - **A**, the full loop;
  - **B**, the loop with the critic, typed termination and the contradiction check switched
    off (together and one at a time);
  - **C**, the model alone: one call with the raw table and the question, no tools.

## Predictions

Recorded on 2026-10-06, before the pilot, and not edited afterwards. The table, with bands and
confidence, is §3 of the decision doc. In short:
- the scaffold lifts Qwen3 1.7B well above the bare model (+20pp), but not to the 8B's level
  (30% vs 70% honest pass);
- most of the 1.7B's failures are structural;
- the scaffold's margin over the bare model nearly vanishes by 8B (+5pp);
- the bare 1.7B starts claiming trends at much weaker signal than the full loop (t₅₀ 0.8 vs
  1.5).

Misses are reported next to the results, not removed.

## How it's scored

**The answer key comes from the data, not from the agent.** Most of the new cases are generated:
a series with a known slope plus noise, whose slope t-statistic is computed independently of the
agent's own tools. Only cases where the answer isn't in doubt are scored:
- **clear trend:** must be concluded;
- **no slope:** no trend may be claimed;
- **clear trend the other way:** the claim must not stand.

The in-between strengths are read as a curve rather than scored. Every overclaiming case has an
underclaiming twin, so an agent that always hedges fails as surely as one that always concludes.

**The headline is the "honest pass".** A run passes if every property of its *answer* holds (the
right conclusion, no claim the data contradicts, sensible confidence) and the model produced a
valid decision. Route properties (which tools ran, whether a claim was challenged) are ignored,
because condition C has no route. A provider outage is excluded, not counted.

Intervals are 95% case bootstraps, because at temperature 0 repeated trials barely differ and the
uncertainty is in which cases were chosen. Comparisons between conditions are paired on the cases
both scored.

## Found along the way

These were found while building the measurement, before any small model ran. Each figure below
was measured on this branch, or on #106 where noted.

**1. The benchmark was saturated.** The existing suite of 19 hand-written cases gave
`gpt-5.4-mini` 100% on both tiers (prompts 1.0.6, 5 trials). It could tell a broken agent from a
working one, but not rank two working ones.

**2. A model that never answers gets credit for caution.** A stub "model" that only ever
returned prose, never valid JSON, still scored:

| measured as | pass rate |
|---|---|
| bench, full loop, core tier (14 cases) | 36% |
| bare model, core tier (10 answer-scored cases) | 40% |
| study runner, full loop, core (10 answer-scored cases), raw | 50% |
| study runner, full loop, generated tier, raw | 63% |
| **honest pass, every one of the above** | **0%** |

A failed run concludes "not established", which is exactly right on every case built to punish
overclaiming. Compare small models on the raw number and you'd credit the smallest for failing
to answer. That's why the headline is the honest pass.

**3. The deterministic tools were manufacturing trends from noise.** The change-point tool
scored the *largest* jump over every possible split as if it were a single, pre-chosen effect
size. On pure noise that maximum is routinely large, so it reached full evidence strength. On
the suite's own noise fixture, asked "is it decreasing?", the loop concluded **supported at
0.95**, while the trend fit explained 12% of the variance.

The signal sweep makes it plain:

| rule-based policy | claims a trend in pure noise | strength at which half its runs claim (t₅₀) |
|---|---|---|
| before | 33% | 0.21 |
| after the fix (#106) | **0%** | **2.12** |

After the fix, t₅₀ sits just under the t ≈ 2.23 that a two-sided 5% slope test needs at 12
points. This was a defect in the part of the system the project's invariant trusts most: the
deterministic computation.

**4. One safety check was invisible to the benchmark.** No hand-written case posed rival
explanations ("is it rising steadily, *or* was there a one-off jump?"). Switching off the check
that stops both from being affirmed changed nothing anywhere in the suite. The generated rival
cases do detect it:
- with the check, the loop declines to pick;
- without it, it affirms both explanations at 0.95.

**5. The scaffold's parts aren't independent.** With the critic switched off, the loop doesn't
stop sooner: termination waits for a challenge that never comes, so it runs every remaining tool
first (2 experiments become 3). Only switching off typed termination makes it stop sooner (2
become 1). An ablation's effect is not the effect of the component alone.

**6. Two tests were passing for the wrong reason.** A test helper labelled periods
`2021-Q1-0 … 2021-Q1-1`, which sort out of time order. Its "clean" uptrends reached the tools
scrambled (R² 0.14 and 0.44 on perfectly straight lines). They passed only because the
broken change-point score stood in for the trend fit.

## Results

*Pending pilot.* Each slot names its source. Nothing here is filled by hand.

### The headline figure

![Honest pass rate by model size](../../data/evaluation/agency/study/figures/headline.svg)

*Source: `figures/headline.svg`, rendered by `study report`. Every value is in `study.md`.*

### Honest pass rate, with intervals

*Source: the "Honest pass rate" table in `study.md`.*

### What the scaffold adds, by size

*Source: the "Paired comparisons" table in `study.md`, A − B and A − C per model.*

### How the small models fail

*Source: the `failures` column in `study.md`. The question is whether small models fail on
structure (invalid JSON) or on reasoning (overclaim / underclaim).*

### When does each condition start claiming a trend?

![Signal-strength curve](../../data/evaluation/agency/study/figures/sweep.svg)

*Source: `figures/sweep.svg` and the "Signal-strength curve" table in `study.md`.*

## Limits

- One model family, one domain of question (trends, comparisons, rival explanations in tabular
  data). The result is about this scaffold on these tasks.
- Condition C is scored on answer properties only, so it is compared with A and B on those,
  never on the full pass rate.
- The generated tier's answer key is a statistical standard (realised t ≥ 10 for "clear", |t| < 1
  for "none"). In between, the right answer depends on how much evidence one demands, which is
  why that band is a curve rather than a score.
