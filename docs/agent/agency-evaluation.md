# Agency Evaluation (`suite_agency_v2`)

The benchmark suites in `edgar_project/evaluation/` check **outputs**: did the pipeline
produce the right artifacts with the right numbers. That is necessary, and it is unchanged.

It says nothing about the question an agentic system actually has to answer well:

> Given evidence, does the loop draw the right conclusion, revise when contradicted, and
> decline when the data cannot support a claim?

`suite_agency_v2` measures that. It is the suite the runner executes. Inside it,
`suite_agency_v1` is the frozen 13 cases kept under their own id because published results
cite them; the **core tier** is those plus any regression case added since, so it grows while
the published measurement does not. `--tier hard` adds the five cases the deterministic
baseline is designed to fail.

```bash
python -m agentic.evaluation
```

```
suite_agency_v2: 14/14 cases passed (100%)

Per-property pass rate:
   100%  avoids_redundant_experiments
   100%  calibrated_confidence
   100%  challenges_before_concluding
   100%  path_adapts_to_goal
   ...
```

Exit code is non-zero when any case fails, so it can gate a change.

This runs the **core** tier: the 13 frozen, published `suite_agency_v1` cases plus any
regression case added since. A core case is one the deterministic baseline passes, so it
guards behaviour rather than measuring headroom — `weakest_entity_goal_ranks_from_the_bottom`
is one, added after a ranking goal was found being answered with the opposite entity.
`suite_agency_v1` stays frozen at 13 and runnable on its own, because a published measurement
reports against it. `--tier hard` runs the cases the deterministic baseline is designed to fail, and
`--tier all` runs both. The default is core precisely so the exit code stays usable as a gate:
the hard tier is expected to fail for the deterministic policy, and a permanently red command
signals nothing when a real regression arrives.

## Two failure modes, weighted equally

The scoring treats opposite errors as equally bad:

- **Overclaiming** — asserting a trend in flat or noisy data; confirming a hypothesis the
  evidence contradicts; cherry-picking the entity that agrees with the goal.
- **Underclaiming** — concluding "insufficient evidence" on an unambiguous signal.

The second is the one most agent evals miss. **An agent that always hedges is never wrong on
adversarial cases**, so a suite made only of traps would score it perfectly. The positive
controls (`clear_rising_is_concluded`, `clear_falling_is_concluded`) exist to fail exactly that
agent, and they are load-bearing rather than filler.

## Properties

| Property | What it means |
|---|---|
| `terminates_for_the_right_reason` | Stops with the typed reason the evidence warrants |
| `reaches_the_right_disposition` | Concludes supported / refuted / insufficient in line with the data |
| `revises_under_contradiction` | A hypothesis the evidence opposes does not end up supported |
| `preserves_contradicting_evidence` | Opposing evidence is retained, not discarded for a tidy story |
| `path_adapts_to_goal` | The experiments chosen — and the question put to them — reflect what was asked |
| `avoids_redundant_experiments` | No tool is run twice for the same question |
| `respects_budget` | Stays within the resource bounds it was given |
| `calibrated_confidence` | Confidence is proportional to the strength of the evidence |
| `challenges_before_concluding` | A claim it accepted was first tested by an independent method it chose to run |

`challenges_before_concluding` exists because the adversarial behaviour degrades *quietly*
without it. `TerminationPolicy` accepts `tested or not unused`, so a loop whose critic never
fires still reaches `sufficient_evidence` at the same confidence and the same disposition — by
exhausting the candidate tools instead of testing the claim. Before this property was added,
disabling the critic entirely left the suite at 100%.

Every check is **deterministic**, read from persisted typed state — no model judging a model.
A verdict is reproducible and a regression is unambiguous. The per-property breakdown is what
makes the suite diagnostic rather than a single number: `calibrated_confidence 40%` points
somewhere specific.

## Fixtures

Seeded or formula-generated, so a case's verdict never depends on the run. A test asserts the
noisy fixture genuinely has no trend and the opposing fixture genuinely opposes — if a fixture
drifted, the case built on it would be silently invalid.

| Fixture | Designed to catch |
|---|---|
| `clear_rising` / `clear_falling` | Hedging on an unambiguous signal |
| `flat` | Manufacturing a trend from nothing |
| `noisy_no_trend` | Reading variation as direction |
| `too_short` | Claiming a trend from two points |
| `opposing_entities` | Cherry-picking the entity that agrees |
| `separated_groups` | Answering a between-group question with a trend experiment |
| `regional_revenue_spread` | Ranking the wrong way round — answering "which is weakest" with the strongest |

## Does the suite actually discriminate?

A suite that only proves the current agent passes is theatre. `tests/agentic/test_agency_suite.py`
runs deliberately bad agents and asserts the suite catches them:

| Agent | Result | Caught by |
|---|---|---|
| Baseline (deterministic policy) | 14/14 | — |
| `HedgingPolicy` — never selects an experiment | 6/14 | Both positive controls; `preserves_contradicting_evidence` and `challenges_before_concluding` 0%, `terminates_for_the_right_reason` 20%, `reaches_the_right_disposition` 33% |
| `AlwaysTrendPolicy` — reads every goal as a trend | 11/14 | `comparison_goal_uses_comparison_tools`, `clear_falling_is_concluded`, `weakest_entity_goal_ranks_from_the_bottom`; `path_adapts_to_goal` 40% |

These discrimination tests are what make the core baseline 14/14 meaningful.

They also mark the limit of that meaning, which is why the hard tier exists.

## The hard tier

`suite_agency_v1` — now the core tier — turned out to be **saturated**: `gpt-5.4-mini` and the
deterministic baseline both scored 100% on every property ([scoreboard](agency-scoreboard.md)).
It separates broken agents from working ones and has no headroom above "working".

### Admission rule

A case belongs in the hard tier when all three hold:

1. **It fails `FixtureAgentPolicy`.** This is the operational definition of "discriminating".
   It is free and deterministic to check, and `tests/agentic/test_agency_tiers.py` enforces it
   per-case, so a case that does not discriminate cannot be added by accident.
2. **It fails on a named `AgencyProperty`.** A case that fails via a crash or a missing tool
   capability is measuring plumbing, not reasoning.
3. **It stands on its own as a fair test.** The rule engine is the yardstick of convenience,
   not the target. If the only argument for a case is that it breaks `FixtureAgentPolicy`, it
   is a trick. This one is a review judgement, not a test.

### The cases

| Case | Judgement it requires |
|---|---|
| `implied_metric_is_selected_over_the_default` | A question about *how long* something takes is answered with the duration metric, though it names no column and two volume metrics are offered first |
| `ranking_goal_is_not_a_trend_goal` | "Which entity is worst" is a question about entities, not time — even though the word *growth* appears in it |
| `grouping_dimension_is_inferred_from_the_question` | Two segments are named, the column holding them is not; the wrong grouping yields a true answer to a question nobody asked |
| `unanswerable_premise_is_declined` | The dataset holds no complaint metric, so the question cannot be answered here — substituting the nearest rising metric produces a confident answer to the wrong question |
| `two_part_goal_resolves_both_clauses` | A two-clause question needs an answer to both; answering only the first reads as a complete reply while silently dropping half the question |

The last is the tier's counterweight. Every other case rewards reaching the right answer; that
one rewards declining, so the tier cannot be cleared by being uniformly more assertive.

`unanswerable_premise_is_declined` asserts more than a low confidence, and the difference is
the point. A policy that substitutes `p99_latency_ms` and then hedges lands inside
`max_confidence` too — a recorded demo did exactly that, passing this case while measuring the
wrong thing, then reporting the hedge as a principled decline. The case now requires the typed
`unanswerable_premise` termination *and* zero experiments, so declining is distinguishable from
getting lucky on a weak proxy.

### What the tier does not cover

The tier defeats **two** of the four model-backed decisions. Four cases target `interpret_goal`
across different judgements — metric selection, intent selection, dimension selection, premise
validity — and `two_part_goal_resolves_both_clauses` targets `generate_hypotheses`.

That last one was impossible until phase 33. The planner parameterised every tool from a single
`interpretation.metric_hint`, so a second hypothesis about a second metric stayed `proposed`
forever *for any policy*; a case built on it would have failed a perfect agent, which is why
phase 32 dropped it. With experiments parameterised per claim and sufficiency waiting for every
claim, it is now winnable and discriminating.

**`select_experiment` remains uncovered**, and the reason is unchanged:
`expected_information_gain` is `0.85 - 0.1 * position` in the intent's tool list, so a
candidate's rank is fixed by the planner. A case where the highest-gain candidate is wrong would
test disagreement with that ordering rather than reasoning.

Breadth is also enforced over the properties the tier exercises — a tier failing on only one or
two properties would be cleared by a single narrow fix.

### The headroom ceiling

`agentic/evaluation/baselines/fixture_floors.json` carries per-property **floors** on the core
tier and a **ceiling** on the hard tier. Floors catch the agent getting worse; the ceiling
catches the suite getting easier. A breach has two possible causes needing opposite fixes — the
deterministic policy genuinely improved (re-baseline deliberately) or the cases went soft
(sharpen them) — so the assertion names both and the ceiling is never raised just to go green.

## Running it against a real model

The suite takes any `AgentPolicy`, so the same cases can be pointed at a model-backed policy:

```python
from agentic.evaluation import run_agency_suite, format_report
from backend.agents.agentic_model_policy import build_agent_policy

print(format_report(run_agency_suite(policy=build_agent_policy(settings))))
```

That is the intended use: does a real model reason at least as well as the deterministic
baseline, and does upgrading it help or hurt? Pairs naturally with
[replay & diff](replay-and-diff.md), which asks the same question about persisted runs.

### The benchmark harness

One pass of a non-deterministic policy is an anecdote, not a measurement, so
`backend/dev/agency_bench.py` runs repeated trials and aggregates them with variance,
cost, and latency:

```bash
python -m backend.dev.agency_bench \
  --policy fixture --policy model --model gpt-5.4-mini \
  --trials 5 --max-cost-usd 2.00 --format both --out scoreboard
```

A case whose verdict changes across trials is listed as **unstable** rather than averaged into
a number that reads as "mostly fine". Cost and latency come from the `InvestigationEnded`
events the loop already emits, so quality and spend are measured on the same run.

The harness refuses a model row when no provider is configured, or when the model has no
configured price — the first would publish a fixture result under a model's name, the second
would leave `--max-cost-usd` summing a quantity that is always zero. `--allow-unpriced` opts
into a quality-only run.

Each row also lists **policy failures**: runs that a policy decision ended, counted as
`decision:kind` (for example `critique:schema` or `interpret_goal:transport`). A failing row
means different things depending on what is listed: a model that reasons badly, one that
cannot produce valid output, or a provider that never answered.

**Local models.** `scripts/agency-pilot-local` runs model rows against Ollama's
OpenAI-compatible endpoint. It prices each model at an explicit `0.0`, which the harness
reads as "free" rather than "unpriced", and it records each model's digest beside the
results, because a tag like `qwen3:4b` can be re-published. It also raises the
per-investigation wall-clock budget with `--max-elapsed-seconds`. Without that, slow
hardware would be scored as the model exhausting its budget.

It lives in `backend/dev` because assembling a model-backed policy needs settings, a provider,
and the prompt registry, none of which `agentic/` may import. `python -m agentic.evaluation`
stays offline, free, and deterministic.

### Results

See **[the agency scoreboard](agency-scoreboard.md)** for the current measurement.

Short version as of 2026-08-07: `gpt-5.4-mini` and the deterministic baseline both score 100%
on all nine properties over five trials, so **the suite is currently saturated and cannot rank
competent policies**. It does still catch broken ones — it found a prompt defect that cost 38
points — but hardening `AGENCY_CASES` is the open follow-up before any ranking claim.

## Scope

`suite_agency_v1` is deliberately **not** registered in
`edgar_project/evaluation/catalog.py`. That catalog's `BenchmarkInput` / `ExpectedArtifacts`
shapes are built around tickers, fixture paths, and artifact assertions; agency cases are about
reasoning over a generic frame. Forcing one into the other would distort both. The offline
default suite (`suite_fixtures_v1`) and the CLI entry points are untouched.

**Follow-up:** exposing agency runs through the evaluation control-plane API (so they persist
`EvaluationRun` rows like the other suites) is not implemented; the suite is CLI- and
library-level today.
