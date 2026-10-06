# Agency Scoreboard — `suite_agency_v2`

**Measured:** 2026-08-26 · **Prompts:** `1.0.6` · **Model snapshot:** `gpt-5.4-mini` · **Trials:** 5 per policy per tier

> **Measured before `detect_change_points` 1.1** (2026-10-05). That version added a
> permutation test for the change-point score, and the evidence updater stopped turning a
> tool's "not significant" verdict into support or refutation by the sign of the shift.
> Before it, pure noise often produced a strength-1.0 shift, and the loop could support,
> or refute, a directional trend on data with no trend. The `fixture` rows are unaffected:
> same pass rates, and the one changed route (`noise_is_not_a_trend` now concludes
> `insufficient_evidence` rather than `refuted`) still passes. The `gpt-5.4-mini` rows
> have not been re-measured since.

> **The hard tier is saturated.** `gpt-5.4-mini` now passes 5 of 5, so this tier no longer has
> headroom: it still separates a rule engine from a model — the baseline scores 0% — but it
> cannot rank two competent agents, which is the job it was built for. That is the same
> condition that retired `suite_agency_v1` as a discriminator, arriving one tier up. New hard
> cases are needed before the next measurement means anything.

[`suite_agency_v2`](agency-evaluation.md) scores whether the investigation loop *reasons* well —
concludes when the evidence supports it, revises when contradicted, tests a claim before
accepting it, declines when the data cannot answer the question, and answers every part of a
question that has more than one. Every check is derived from persisted typed state, so a verdict
is reproducible and never a model judging a model.

Two tiers. **Core** was the frozen `suite_agency_v1`, 13 cases, when this was measured; it has
since gained regression cases the baseline passes, so a fresh core run reports a larger total
than the rows below. `suite_agency_v1` itself is unchanged, which is what keeps this
measurement readable. **Hard** is 5 cases admitted only if they defeat the deterministic
baseline — see [the admission rule](agency-evaluation.md#admission-rule).

## Result

| policy | tier | trials | pass rate | avoids_redundant | calibrated_confidence | challenges_before | path_adapts | preserves_contradicting | right_disposition | respects_budget | revises_under_contradiction | right_termination | mean $ | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixture | core | 5 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0.0000 | 0.00 |
| fixture | hard | 5 | **0%** | 100% | 0% | — | 0% | — | 0% | 0% | 0% | 0% | 0.0000 | 0.01 |
| gpt-5.4-mini | core | 5 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0.1273 | 8.30 |
| gpt-5.4-mini | hard | 5 | **100%** | 100% | 100% | — | 100% | — | 100% | 100% | 100% | 100% | 0.0435 | 12.27 |

`—` marks a property no case in that tier asserts. Total spend: **$0.85**. Zero unstable cases in
any row, and no row truncated.

## What changed, and what did not cause it

Both cases the model failed at `1.0.1` now pass on all five trials, so the hard tier goes
**60% -> 100%**. The tempting story is that the prompt change shipped alongside this measurement
fixed them. It did not, and the prompt files say so.

`1.0.6` changed **one** role. Diffing each 1.0.6 body against its 1.0.5 predecessor, ignoring the
version line, the goal interpreter, hypothesis generator, experiment selector and answer writer
are byte-identical; only the critic differs. The two cases that flipped are
`unanswerable_premise_is_declined`, which is decided by the goal interpreter's `answerable`
field, and `two_part_goal_resolves_both_clauses`, which fails at interpretation before the
critic is ever consulted. Neither is reachable from a critic edit.

What actually moved them shipped earlier and was never measured. The published rows were taken at
`1.0.1` while the code had advanced to `1.0.5` — four versions, of which the goal interpreter
gained 69 changed lines and the hypothesis generator 13. **The scoreboard had been
under-reporting the system for some time**, and re-measuring surfaced work that was already done
rather than work done here.

An exact attribution would need a control run at `1.0.5`, which has not been made. The honest
claim is bounded: the improvement lies somewhere in `1.0.2`-`1.0.5`, and the critic change is
ruled out by inspection rather than by measurement.

## What the critic change did cost

Mean spend per core trial roughly doubled, `$0.0524 -> $0.1273`, and hard-tier p95 latency rose
from 8.71s to 12.27s. Some of that is the wider critic gate doing exactly what it was built to
do — a run that previously made zero critic calls now makes up to one per iteration. Some of it
is four versions of prompt growth. As above, the two are not separated by this run.

The trade is deliberate and worth stating plainly: the loop now examines itself in cases where it
used to conclude unchallenged, and that costs roughly twice as much per question.

## The case that could not exist before

`two_part_goal_resolves_both_clauses` was designed during phase 32 and **dropped as unwinnable**.
The planner parameterised every tool from a single `interpretation.metric_hint`, so a second
hypothesis about a second metric stayed `proposed` forever — for *any* policy, however well it
reasoned. A case built on it would have failed a perfect agent.

Phase 33 removed that limitation: experiments are parameterised from the claim they test, and
sufficiency waits for every claim rather than firing on the first one supported. The case is now
winnable — verified against a two-claim policy — and discriminating.

With it, the hard tier defeated **two** of the four model-backed decisions rather than one — at
`1.0.1`. At `1.0.6` it defeats none, which is what saturation means in practice.
`select_experiment` remains uncovered for the unchanged reason that
`expected_information_gain` is a function of the planner's tool ordering rather than of anything
the run has measured.

## Cost and latency

| | mean $ / trial | p95 per investigation |
|---|---|---|
| fixture | $0.0000 | ~0.01 s |
| gpt-5.4-mini core (13 cases) | $0.1273 | 8.30 s |
| gpt-5.4-mini hard (5 cases) | $0.0435 | 12.27 s |

Per-trial cost tracks case count, so the tiers are not directly comparable. Cost is a slight
over-estimate: `ModelPrice` has no cached-input tier, so cached reads are billed at the full
input rate.

## Reproducing

```bash
python -m backend.dev.agency_bench \
  --policy fixture --policy model --model gpt-5.4-mini \
  --trials 5 --tier all --max-cost-usd 3.00 --format both --out scoreboard
```

Budget roughly **$0.85** at current prompt sizes, up from $0.38 at `1.0.1`.

| | |
|---|---|
| Suite | `suite_agency_v2` — 13 core + 5 hard, 9 properties |
| Prompts | All five `edgar.agentic.*` roles @ `1.0.6`, which is what `AGENTIC_PROMPT_VERSION` resolves to — so the command above reproduces these rows as written. |
| Model | `gpt-5.4-mini` → resolved `gpt-5.4-mini-2026-03-17` |
| Pricing | $0.75 / $4.50 per 1M in/out ([source](https://developers.openai.com/api/docs/models/gpt-5.4-mini)) |
| Raw result | `data/evaluation/agency/scoreboard-2026-08-26-v4.json` |

The free offline half — the fixture policy against committed per-property floors, plus a
headroom ceiling on the hard tier — runs on every pull request. The model rows are on-demand only.

---

## Previous measurements

Retained because the core tier is frozen and these results stay reproducible.

**2026-08-08 · `suite_agency_v2` @ prompts `1.0.1`** — fixture core 100% / hard 0%;
`gpt-5.4-mini` core 100% / hard **60%** (3/5), $0.38, zero unstable cases. The two failures were
`unanswerable_premise_is_declined` (concluded `supported` at 0.95 over a dataset holding no
measure of what was asked) and `two_part_goal_resolves_both_clauses` (read a two-clause question
as a ranking problem and ran only `rank_entities`). Both pass at `1.0.6`. This row stood as the
published result while the shipped prompts advanced to `1.0.5` without being re-measured, which
is why the 2026-08-26 run reads as a jump rather than a series. Raw:
`data/evaluation/agency/scoreboard-2026-08-08-v3.json`.

**2026-08-07 · `suite_agency_v2`, 4 hard cases** — fixture core 100% / hard 0%; `gpt-5.4-mini`
core 100% / hard 75%. Zero unstable cases, $0.35. The run that first showed the suite could
discriminate. Raw: `data/evaluation/agency/scoreboard-2026-08-07-v2.json`.

**2026-08-07 · `suite_agency_v1`, 13 cases** — both policies 100% on all nine properties, $0.35.
That run's finding was **saturation**: the suite could not tell a keyword-matching rule engine
from a frontier model. It also caught a real defect — on prompt `1.0.0` the model scored 62%,
because the critic prompt told it to decline "with nulls" while only two of
`CritiqueProposal`'s five fields are nullable; a null `message` tripped the loop's fail-safe and
lost every converging case. Prompt `1.0.1` fixed the contract description and the same suite
scored 13/13. Raw: `data/evaluation/agency/scoreboard-2026-08-07.json`.
