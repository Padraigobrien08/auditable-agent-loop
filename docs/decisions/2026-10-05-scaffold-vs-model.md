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
7. **A broken model gets credit for hedging.** A stub that never returned valid JSON (every
   run ended `interpret_goal:invalid_json`) still passed **36% of core** and 33% of
   `right_disposition`. A failed run concludes as not established, and that is the right
   answer on every case built to punish overclaiming. That is correct for a product: it
   must never invent a conclusion. As a measurement, it would score a model that cannot
   produce output as partly well-calibrated, which is exactly the confusion this study
   exists to remove. The headline metric therefore counts a run with a non-`ungrounded`
   policy failure as **failed**, whatever its disposition. The as-scored pass rate is
   reported next to it, and transport failures are re-run rather than counted. Decided
   in S5, but it must be fixed before any small-model number is read.

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

### S0 — Make failures distinguishable · **landed 2026-10-05**
The policy records *which decision failed* (interpret, generate, select, critique) and *how*
(transport / invalid JSON / schema / ungrounded), as typed state that the scorer can read.

- `TerminationDecision.policy_failure` (`PolicyFailure`), persisted through the existing
  `termination_json` column, so no migration is needed. It surfaces as
  `AgencyCaseResult.observed_policy_failure`.
- The backend responder raises `PolicyTransportError` on a provider error instead of
  returning `""`. An empty *model* reply stays `invalid_json`.
- A selector index past the end of the candidate list is recorded as `ungrounded`. The
  loop's behavior is unchanged: it still stops as if the selector had declined. Only the
  record differs, so frozen results stay comparable.
- **Not covered yet:** a critique naming a hypothesis id that does not exist. It doesn't
  end the run, so it doesn't belong on the termination. If the pilot shows small models
  doing it, it needs a per-decision record on the state, which means a column and a
  migration.

### S1 — Local provider + pilot · **config landed 2026-10-05, pilot not yet run**
1. Point `openai_base_url` at Ollama's OpenAI-compatible endpoint.
2. Add zero-price entries, so `_assert_priced` accepts the model honestly rather than being
   bypassed.
3. Run the 19 cases, condition A, 1 trial, smallest and largest size only.

Landed: `scripts/agency-pilot-local [model ...]` does steps 1 and 2 and runs step 3. It
needs `ollama serve` running and the models pulled. Along the way:

- An explicit zero price now means *free*. The "$0.00 spend means the price key missed"
  guard used to abort every local row after its first trial.
- `--max-elapsed-seconds` (the script uses 1800) raises the per-run wall-clock budget, and
  the safety cap above it. At the default 120 s, a laptop-speed 8B would be scored
  `budget_exhausted`, which measures the hardware. The experiment and model-call budgets,
  which don't depend on hardware, are unchanged.
- Output JSON records the endpoint host and the time budget. `models.json` pins each
  model's digest.
- Verified end to end against a stub server. Garbage replies are recorded as
  `interpret_goal:invalid_json`, HTTP 500s as `interpret_goal:transport`, and missing
  models or a stopped Ollama fail before any run starts.

**Decide before the pilot: Qwen3's thinking mode.** Qwen3 thinks by default. That changes
latency by an order of magnitude and may change what reaches `content` under
`json_object`. Pick on or off, hold it fixed across sizes, and record it. Check the first
raw reply by hand before trusting a row.

The pilot answers two questions: do structural failures dominate, and does the largest size
already saturate?

- If structural failures dominate: add a native Ollama provider beside `openai_provider.py`
  with schema-constrained output, and treat constraint on/off as a controlled variable.
- If the largest size saturates: the generator (S2) moves ahead of everything, with weaker
  signal.

### S2 — Case generator · **landed 2026-10-05**
`agentic/evaluation/generated.py`: 30 scored cases in a new `generated` tier (suite id
`suite_agency_generated_v1`, never part of `AGENCY_CASES`), plus a 36-point unscored sweep.
Run with `--tier generated`, or `run_signal_sweep()` for the curve.

- **Ground truth comes from the generating process, not the loop.** The loop calls a trend
  supported at R² ≥ 0.5. Using that as the answer key would make the full loop pass by
  construction. Instead each series has a known slope, and its realised OLS t-statistic is
  computed with numpy, independently of the tools. Scoring only happens where the answer is
  not in doubt:
  - **clear:** |t| ≥ 10 in the stated direction; must conclude.
  - **null:** true slope 0, |t| < 1; no directional claim may stand.
  - **contradicted:** clear the other way; the claim must not stand.

  Levels in between form the sweep. It's read as a curve (the realised t at which a
  configuration starts claiming), never as a pass rate.
- **8 trend families × (clear, null, contradicted)** across 4 domains with different column
  names and time formats, lengths 8–16, both directions.
- **3 rival pairs on step data.** "Is X rising steadily, or was there a one-off jump?" must
  not end with both supported. The twin asks plainly "is X increasing?" over the same data
  and must conclude. A test shows that with a policy that proposes both branches, the full
  loop passes and `-mutual_exclusivity` affirms both at 0.95. This tier can see the
  ablation the hand-written suite couldn't.
- Every case is reproducible from its fixture id (`gen:trend:rainfall:n12:t15:seed3`).
  Seeds come from a fixed search, never at random.
- The "never affirm both rivals" check is scored under the existing
  `revises_under_contradiction` property. A new property can't be exercised in the core
  tier, because the rule-based policy only ever proposes one claim, and the suite's guards
  rightly refuse a property nothing in core measures.

**Rule-based policy on this tier: 23/30**, the same under every ablation.
- The 3 rival cases fail because it proposes a single claim and never weighs the alternative.
- 4 of the 8 null cases fail because of a scaffold defect, below.

**Finding: the change-point tool manufactures trends from noise.** `detect_change_points`
scores the *maximum* of |shift| / pooled SD over every split. That's a selection statistic:
on pure noise the maximum routinely exceeds 0.8, which is labelled Cohen's d and normalised
to evidence strength 1.0. When the noise's largest shift points the way the goal does, the
loop supports the directional claim at 0.95 on data whose trend fit says R² = 0.01. The
sweep shows the rule-based policy claiming a trend at realised t = 0.37. The hand-written
`noise_is_not_a_trend` passes only because its shift points the other way.

This is a defect in the deterministic compute, the part the project's invariant trusts.
It is **not fixed here**: fixing it changes product behaviour and possibly the published
runs, so it needs a decision. A fix would calibrate the score against its null
distribution (a permutation or analytic max-statistic critical value), or stop letting a
non-directional shift support a directional trend claim. Until it's fixed, every condition
inherits the same false-positive source, so A vs B vs C comparisons stay fair. Absolute
null-case scores, though, understate how well a model would do with sound tools.

### S3 — Condition B flags · **landed 2026-10-05**
`LoopAblations` (`agentic/agent/ablations.py`) has three switches, all on by default:
`critic`, `typed_termination`, `mutual_exclusivity`. The bench takes `--ablate <name>`
(repeatable), and the row label names what was removed (`fixture -critic`).

- **"Termination off" means `NaiveTerminationPolicy`:** stop at the first claim that
  clears the confidence bar. Budget, safety and user-stop limits are kept, because they
  bound cost, not reasoning.
- **Defaults reproduce the frozen suite exactly.** A test compares the full report with and
  without explicit defaults.
- **The switches are not independent.** Typed termination accepts a supported claim only
  once it has been challenged or no intent tool is left. So *critic off* alone doesn't stop
  sooner: the loop runs every remaining tool (2 → 3 on the concluding cases). Only
  *termination off* stops sooner (2 → 1). Report single and combined ablations, and don't
  read an ablation's effect as the component's effect in isolation.
- **Rule-based policy under each ablation** (`fixture`, 1 trial):

  | removed | core | hard | what changed |
  |---|---|---|---|
  | — | 100% | 0% | |
  | critic | 93% | 0% | `supported_claim_is_challenged_before_concluding` fails |
  | typed_termination | 93% | 0% | the same case, and `budget_is_respected` now stops on sufficiency before the budget |
  | mutual_exclusivity | 100% | 0% | **nothing**: no case exercises it |
  | all three | 93% | 0% | |

  The rule-based policy almost never needs rescuing, so the scaffold barely registers
  here. That's expected: the prediction (P4/P5) is that the scaffold's value shows up when
  the policy errs, which is what small models will do.
- **`challenges_before_concluding` fails by construction** under *critic off*, because it
  asserts the scaffold's own behaviour. That's one more reason the cross-condition headline
  is `right_disposition`, not the full pass rate.

### S4 — Condition C harness · **landed 2026-10-05**
`backend/dev/bare_model.py`, run with `--condition bare` (or `CONDITION=bare` in the pilot
script). One call per case: the question plus the whole table as CSV, and a JSON reply with a
disposition, the claims weighed (with statuses) and a confidence. It uses the same
responder, temperature 0 and `json_object` mode as the loop.

- **Same checks, not a copy.** The answer-level checks were extracted from `score_case` into
  shared helpers. `score_answer` runs those exact checks, in the same order. A parity test
  scores every loop run's conclusion both ways across all 49 hand-written and generated
  cases and requires identical outcomes. The suite reports are byte-identical before and
  after the extraction.
- **Only answer properties are scored**: `reaches_the_right_disposition`,
  `revises_under_contradiction` and `calibrated_confidence`. Cases that assert only the
  route (tools, critique, budget, termination) are skipped: 5 of the 19 hand-written cases.
  All 30 generated cases are scorable. A C row is comparable with A/B on those properties,
  never on the overall pass rate.
- **Failures are typed like the loop's**: transport, invalid JSON or schema, under
  `PolicyDecisionKind.bare_answer`. A failed call is scored as an empty answer, which is the
  same thing a failed loop run concludes. So §5.7 applies equally: the headline must count
  these as failed. Measured: through the pilot script against a stub that never returns
  JSON, a bare row still scores 40% on core, all of it from overclaim cases an empty answer
  cannot fail.
- **No hints.** A test checks the prompt carries only the question and the table, never the
  case id, description or expectations. The system prompt offers the same dispositions and
  the same permission to decline as the loop's prompts.
- C breaks the "no number from a model" invariant on purpose: its confidence is the model's
  own. That's why it lives in `backend/dev/`, is never persisted, and has its own prompt
  version (`BARE_PROMPT_VERSION`) recorded in the output JSON.

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
