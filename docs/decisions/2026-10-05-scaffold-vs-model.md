# Direction: how much of honest reasoning is the scaffold, and how much the model?

Date: 2026-10-05 · Status: **accepted**

Becomes the active plan. It replaces the *sequencing* of
[`2026-08-11-showcase-direction.md`](./2026-08-11-showcase-direction.md); that file's hosting and
spend decisions (D1–D8, as amended by
[`2026-08-14-static-replay-showcase.md`](./2026-08-14-static-replay-showcase.md)) still hold. The
README is rebuilt to lead with this result once it exists, not with the showcase runs.

---

## 1. Why this, and not more showcase work

`suite_agency_v2` is already a good evaluation. It scores reasoning from persisted typed state,
with no model judging a model. Cases come in pairs, so always hedging fails as surely as always
overclaiming. Hard cases are admitted only if they defeat the rule-based baseline
([admission rule](../agent/agency-evaluation.md#admission-rule)). Its gaps are elsewhere:

1. **It is saturated.** `gpt-5.4-mini` passes 100% of both tiers at prompts `1.0.6`
   ([scoreboard](../agent/agency-scoreboard.md)). It can no longer rank two competent agents.
2. **It is small.** 19 cases (14 core, 5 hard) is too few for useful confidence intervals.
3. **The central claim is untested.** The project claims the scaffold (deterministic compute,
   typed termination, contradiction checks) is what produces honest conclusions. Every
   measurement so far holds the scaffold fixed, so nothing separates its contribution from the
   model's.

Small local models address (1) and (3) together, at zero spend.

## 2. Research question

**How much of reliable agentic reasoning comes from the scaffold and how much from the model,
and how does that balance shift with model size?**

Each of these outcomes is reportable:

- The scaffold lets a small model reach near-frontier disposition accuracy.
- The scaffold's benefit shrinks with model size, with a crossover point.
- Small models fail mostly on output structure, not on reasoning.

## 3. Predictions — written before any run

Fill these in **before the pilot** (S1). They are not edited afterwards. Misses get reported
next to the results, as the README already does for the lr-3e-3 prediction.

| # | Prediction | Value | Confidence |
|---|---|---|---|
| P1 | Smallest size, condition A, `right_disposition` on hard tier | | |
| P2 | Largest size, condition A, `right_disposition` on hard tier | | |
| P3 | Share of smallest-size failures that are *structural* | | |
| P4 | A − C gap at smallest size (pp) | | |
| P5 | A − C gap at largest size (pp) | | |
| P6 | Of critic / termination / mutual-exclusivity, which ablation costs most | | |
| P7 | Signal level at which the smallest model starts claiming a trend in noise | | |

## 4. Design

| Factor | Levels |
|---|---|
| **Model** | One open-weight family at 3–4 sizes (e.g. Qwen3 1.7B / 4B / 8B), local via Ollama. The `fixture` policy is the size-zero point (scaffold, no model). `gpt-5.4-mini` is the frontier reference. |
| **Condition** | **A** full loop. **B** ablated loop: same tools, with critic, typed termination and the mutual-exclusivity check switched off one at a time, then together. **C** bare model: one call given the table and the question, returning a disposition as JSON. |
| **Cases** | The existing 19, plus about 30 generated (§6). |
| **Trials** | 3 per cell. 5 if compute allows. |

**One model family**, so that size is not confounded with training recipe. This is the same
one-change-at-a-time discipline as the nanoGPT ablations.

## 5. What the code already tells us

These come from reading the loop. They change parts of the original sketch.

1. **The `fixture` policy is a free data point.** It scores **100% core / 0% hard** with no
   model at all. So the core tier cannot measure what the model contributes, because the
   scaffold alone solves it. The question lives in the hard tier and the generated cases.
   Core results are reported, but they are not the headline.
2. **Condition B is cheaper than feared.** The three components are separate call sites in
   [`agentic/agent/loop.py`](../../agentic/agent/loop.py): `TerminationPolicy.decide`,
   `enforce_mutual_exclusivity` and `Critic.challenge`, all built in `__post_init__`. The real
   design choice is what "termination off" means. `decide` also enforces budget and safety
   limits, which stay. The ablation replaces only the sufficiency and contradiction checks
   with *stop at the first supported hypothesis*. That is the rule the comments in
   `components.py` describe as the bug the current policy fixed. Removing the critic also
   removes model calls, so A and B cost columns are not comparable.
3. **Condition C cannot share the pass rate.** `challenges_before`, `path_adapts` and
   `revises_under_contradiction` exist only inside a loop. The cross-condition metric is
   `right_disposition` (with termination type mapped to a disposition). Full pass rate is
   reported for A and B only. C deliberately breaks the "no number from a model" invariant, so
   it lives in `backend/dev/` beside `agency_bench.py`, never in `agentic/`.
4. **Structural failure is not observable today.** A provider error becomes `""` in
   `agentic_model_policy.py`, then `MalformedPolicyResponse` in `policy.py`, then
   `TerminationReason.error`. So "the 1.7B model can't write JSON" and "the laptop timed out"
   are the same record. **S0 fixes this before anything is measured.**
5. **The frontier rows are only partly reusable.** They hold for the 19 existing cases at
   prompts `1.0.6`. The generated cases need a fresh `gpt-5.4-mini` run (a few dollars).
6. **Trials at temperature 0 add little.** The policy sends `temperature=0.0`, and local models
   at temperature 0 are near-deterministic. Most of the variance is between cases, so the
   bootstrap resamples cases, and more cases beat more trials.

## 6. New cases

The cases are parameterised variants of the existing fixtures, with one parameter for signal
strength, running from a clear trend down to pure noise. This gives a calibration curve: the
signal-to-noise level at which each model and condition starts claiming a trend. The existing
rule holds: **every overclaiming case has an underclaiming twin.** Generated cases are a new
tier, so the frozen tiers stay comparable with published rows.

## 7. Metrics

The existing per-property scoring stays. Three things are added:

- **Failure taxonomy.** Each failing run is classified as exactly one of: *transport* (the
  provider never answered), *structural* (invalid JSON or schema), *overclaim*, *underclaim*,
  *other reasoning* (wrong disposition that is neither), or *budget exhaustion*. Transport is
  reported but excluded from model comparisons. It measures the machine, not the model.
- **Paired comparisons with bootstrap CIs.** A vs B vs C within each model, paired by case,
  with the case as the resampling unit.
- **The headline figure.** `right_disposition` against model size, one line per condition. The
  `fixture` point is at size zero and `gpt-5.4-mini` is the reference line.

## 8. Sequence

### S0 — Make failures distinguishable
The policy records *which decision failed* (interpret, generate, select, critique) and *how*
(transport / invalid JSON / schema), as typed state that the scorer can read. This needs no
provider. It is testable offline with the existing responder seam.

### S1 — Local provider + pilot
1. Point `openai_base_url` at Ollama's OpenAI-compatible endpoint.
2. Add zero-price entries, so `_assert_priced` accepts the model honestly rather than being
   bypassed.
3. Run the 19 cases, condition A, 1 trial, smallest and largest size only.

The pilot answers two questions: do structural failures dominate, and does the largest size
already saturate?

- If structural failures dominate: add a native Ollama provider beside `openai_provider.py`
  with schema-constrained output, and treat constraint on/off as a controlled variable.
- If the largest size saturates: the generator (S2) moves ahead of everything, with weaker
  signal.

### S2 — Case generator
Parameterised signal strength, paired twins, a new tier. Pilot results set the signal range.

### S3 — Condition B flags
Loop configuration with three independent switches, defaulting to today's behavior, plus a
test that the defaults reproduce the frozen tiers byte for byte.

### S4 — Condition C harness
`backend/dev/`: fixture → prompt → JSON disposition → the same disposition scorer.

### S5 — Runner and report
`scoreboard.py` gains model × condition dimensions, the taxonomy, and bootstrap CIs. The full
run is ~50 cases × 3 trials × 3 conditions × 3 sizes ≈ 1,350 runs, a few overnight runs
locally, or Kaggle's free GPU running Ollama.

### S6 — Write-up
The extended scoreboard, the headline figure, a Substack post, and the README rebuilt around
the result. The target claim has the form: *"the scaffold lifts a 4B model from X% to Y%
disposition accuracy, within Z of frontier."*

## 9. Risks

| Risk | Response |
|---|---|
| The largest size already saturates | The scaffold's contribution shows only at smaller sizes. That is a finding in itself. Harden by lowering signal. |
| B turns out tangled | Drop B and run A vs C. The core comparison stands. §5.2 suggests this is unlikely. |
| Structural failures dominate | Constrained decoding (S1), reported as its own variable. |
| Transport failures on a laptop pollute results | S0's taxonomy separates them. Re-run transport failures and never count them as model failures. |

## 10. Housekeeping

`.planning/` was tracked history for the GSD-era roadmap. It is now local-only and gitignored.
This directory (`docs/decisions/`) is the only tracked plan.
