"""
The agency case set, in two tiers.

Cases are paired on purpose. For every fixture that punishes overclaiming there is one that
punishes hedging, because the two failure modes are opposite and a suite that only tests one
is trivially gamed: an agent that always concludes "insufficient evidence" passes every
negative case and is useless.

**Tiers.** `suite_agency_v1` — the 13 :data:`SUITE_V1_CASES` — is frozen. A published
measurement reports against it (``docs/agent/agency-scoreboard.md``), and growing it in place
would invalidate that result and destroy comparability, so it stays runnable on its own
forever. That measurement also found it *saturated*: a keyword-matching rule engine and a
frontier model both score 100%, so it separates broken agents from working ones but cannot
rank competent ones.

:data:`CaseTier.hard` is the headroom. A hard case must **fail**
:class:`~agentic.agent.fixture_policy.FixtureAgentPolicy`, on a named reasoning property —
that is the operational definition of "discriminating", it is free and deterministic to check,
and ``tests/agentic/test_agency_tiers.py`` enforces it. A case the rule engine passes does not
belong in the tier.

The rule engine is the yardstick of convenience, not the target: each hard case must also read
as a fair test on its own terms. If the only argument for a case is that it breaks
`FixtureAgentPolicy`, it is a trick rather than a measurement.

:data:`CaseTier.core` may still grow, and that is not in tension with the freeze: a core case
is one the deterministic baseline *passes*, so it guards a behaviour rather than measuring
headroom, and `SUITE_V1_CASE_IDS` keeps the published subset intact underneath it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from agentic.domain.common import DomainModel
from agentic.evaluation.agency import AgencyExpectations


class CaseTier(str, Enum):
    """Which bar a case sets."""

    core = "core"
    """The frozen `suite_agency_v1` set. The deterministic baseline passes all of these."""

    hard = "hard"
    """Requires judgement a keyword-matching rule engine cannot supply — it must fail one."""

    generated = "generated"
    """Parameterised cases from :mod:`agentic.evaluation.generated`. Never in
    :data:`AGENCY_CASES`, so ``suite_agency_v2`` numbers keep their meaning. Not subject to the
    hard tier's admission rule: these measure calibration across signal strength, not headroom
    over the rule engine."""


#: The frozen, published suite. Never add to this.
SUITE_V1_ID = "suite_agency_v1"

#: The full set, core + hard. What `run_agency_suite()` reports by default.
SUITE_ID = "suite_agency_v2"

# Termination / disposition vocabulary, kept as literals so a case reads declaratively.
SUFFICIENT = "sufficient_evidence"
INSUFFICIENT = "insufficient_evidence"
NO_VALID_EXPERIMENT = "no_valid_experiment"
BUDGET_EXHAUSTED = "budget_exhausted"
MAX_ITERATIONS = "max_iterations"

SUPPORTED = "supported"
REFUTED = "refuted"
INCONCLUSIVE = "inconclusive"
MIXED = "mixed"


class AgencyCase(DomainModel):
    """One adversarial scenario: a goal over a fixture, plus what good behavior looks like."""

    case_id: str
    description: str
    goal: str
    fixture_id: str
    expectations: AgencyExpectations
    #: Structural hints the adapter needs; never domain vocabulary. Cases over non-financial
    #: fixtures override these, which is how the suite proves nothing depends on column names.
    time_field: str | None = "period"
    entity_id_fields: list[str] = Field(default_factory=lambda: ["entity"])
    metric_field: str = "value"
    max_experiments: int | None = None
    #: `core` unless the case is designed to defeat the deterministic baseline. Defaulted so
    #: the frozen v1 cases below need no annotation and stay byte-for-byte what was published.
    tier: CaseTier = CaseTier.core


AGENCY_CASES: tuple[AgencyCase, ...] = (
    # -- positive controls: an agent that always hedges must fail these ------
    AgencyCase(
        case_id="clear_rising_is_concluded",
        description="An unambiguous upward trend must actually be concluded, not hedged.",
        goal="value is increasing over time",
        fixture_id="clear_rising",
        expectations=AgencyExpectations(
            termination_reason_in=[SUFFICIENT],
            disposition_in=[SUPPORTED],
            hypothesis_status_any=[SUPPORTED],
            min_confidence=0.5,
        ),
    ),
    AgencyCase(
        case_id="clear_falling_is_concluded",
        description="The mirror control: a clear decline must be concluded too.",
        goal="value is decreasing over time",
        fixture_id="clear_falling",
        expectations=AgencyExpectations(
            termination_reason_in=[SUFFICIENT],
            disposition_in=[SUPPORTED],
            hypothesis_status_any=[SUPPORTED],
            min_confidence=0.5,
        ),
    ),
    # -- overclaiming: manufacturing a signal that is not there --------------
    AgencyCase(
        case_id="flat_data_is_not_a_trend",
        description="Flat data cannot support a directional claim.",
        goal="value is increasing over time",
        fixture_id="flat",
        expectations=AgencyExpectations(
            disposition_in=[INSUFFICIENT, INCONCLUSIVE],
            hypothesis_status_not_in=[SUPPORTED],
            max_confidence=0.5,
        ),
    ),
    AgencyCase(
        case_id="noise_is_not_a_trend",
        description="Variation without direction must not be read as a trend.",
        goal="value is increasing over time",
        fixture_id="noisy_no_trend",
        expectations=AgencyExpectations(
            hypothesis_status_not_in=[SUPPORTED],
            max_confidence=0.6,
        ),
    ),
    AgencyCase(
        case_id="two_points_are_not_a_trend",
        description="A two-point series is too short to claim a trend from.",
        goal="value is increasing over time",
        fixture_id="too_short",
        expectations=AgencyExpectations(
            hypothesis_status_not_in=[SUPPORTED],
            max_confidence=0.6,
        ),
    ),
    # -- revision: the evidence contradicts the question ---------------------
    AgencyCase(
        case_id="contradicted_claim_is_revised",
        description="A hypothesis the data opposes must not end up supported.",
        goal="value is increasing over time",
        fixture_id="clear_falling",
        expectations=AgencyExpectations(
            hypothesis_status_not_in=[SUPPORTED],
            disposition_in=[REFUTED, INSUFFICIENT, INCONCLUSIVE],
        ),
    ),
    AgencyCase(
        case_id="opposing_entities_are_not_cherry_picked",
        description="With entities moving in opposite directions, the contradiction is kept.",
        goal="value is increasing over time",
        fixture_id="opposing_entities",
        expectations=AgencyExpectations(
            hypothesis_status_not_in=[SUPPORTED],
            require_contradicting_evidence=True,
        ),
    ),
    # -- adaptivity: the path must reflect the question -----------------------
    AgencyCase(
        case_id="comparison_goal_uses_comparison_tools",
        description="A between-group question must not be answered with a trend experiment.",
        goal="compare value between groups",
        fixture_id="separated_groups",
        expectations=AgencyExpectations(
            expect_any_tool=["compare_groups", "rank_entities"],
            forbid_tools=["analyze_time_series_trend", "detect_change_points"],
        ),
    ),
    AgencyCase(
        case_id="trend_goal_uses_trend_tools",
        description="A temporal question must reach for temporal experiments.",
        goal="what is the trend in value over time?",
        fixture_id="clear_rising",
        expectations=AgencyExpectations(
            expect_any_tool=["analyze_time_series_trend", "detect_change_points"],
            forbid_tools=["compare_groups"],
        ),
    ),
    AgencyCase(
        case_id="weakest_entity_goal_ranks_from_the_bottom",
        description=(
            "Asking which entity is *weakest* must be answered with the weakest one. Ranking "
            "descending here returns 'south', a correct number and the opposite answer, so "
            "asserting that rank_entities ran would not catch it."
        ),
        goal="which region has the weakest revenue?",
        fixture_id="regional_revenue_spread",
        time_field="quarter",
        entity_id_fields=["region"],
        metric_field="revenue",
        expectations=AgencyExpectations(
            # north is the only region whose revenue declines, and it holds the lowest mean by
            # a wide margin; south holds the highest. The two are unambiguous and far apart, so
            # this asserts direction rather than a tie-break.
            #
            # Deliberately no confidence bound. The run that exposed this declined honestly —
            # neutral evidence, `unresolved`, `insufficient_evidence` at 0.2 — and still does,
            # but capping confidence would make that the *required* answer, and these regions
            # genuinely do differ. A policy that concluded so on evidence would be right to.
            expect_ranked_first="north",
        ),
    ),
    # -- input-agnosticism: a different domain, different column names --------
    #
    # These matter because the platform's whole generalization claim rests on the loop
    # reasoning over *roles* declared by an adapter, never over column names. If anything
    # in the loop quietly special-cased financial vocabulary, these are what would fail.
    AgencyCase(
        case_id="non_financial_trend_is_concluded",
        description="Rainfall at a weather station — a rising signal must conclude the same way.",
        goal="rainfall_mm is increasing over time",
        fixture_id="rainfall_rising",
        time_field="month",
        entity_id_fields=["station"],
        metric_field="rainfall_mm",
        expectations=AgencyExpectations(
            termination_reason_in=[SUFFICIENT],
            disposition_in=[SUPPORTED],
            hypothesis_status_any=[SUPPORTED],
            min_confidence=0.5,
        ),
    ),
    AgencyCase(
        case_id="non_financial_flat_is_not_a_trend",
        description="Service latency that is not moving must not be read as a trend either.",
        goal="latency_ms is increasing over time",
        fixture_id="response_latency_flat",
        time_field="day",
        entity_id_fields=["service"],
        metric_field="latency_ms",
        expectations=AgencyExpectations(
            hypothesis_status_not_in=[SUPPORTED],
            max_confidence=0.6,
        ),
    ),
    # -- adversariality: a claim must survive a test before it is accepted -----
    AgencyCase(
        case_id="supported_claim_is_challenged_before_concluding",
        description=(
            "An unambiguous trend must be tested by an independent method before it is "
            "concluded — and testing it must not become an excuse to run everything."
        ),
        goal="value is increasing over time",
        fixture_id="clear_rising",
        # Deliberately no `max_experiments` on the case: capping the *loop* at 2 would make
        # the expectation below unfalsifiable, since the run could never exceed it. The loop
        # keeps its default budget and the expectation does the asserting.
        expectations=AgencyExpectations(
            termination_reason_in=[SUFFICIENT],
            disposition_in=[SUPPORTED],
            require_challenge=True,
            # The two failures are opposite and this bound catches the second. A critic that
            # never fires reaches the same disposition at the same confidence by exhausting
            # the tool list, so it costs an extra experiment; a critic that fires
            # indiscriminately costs several. Two is what a single, targeted challenge takes.
            max_experiments=2,
        ),
    ),
    # -- discipline -----------------------------------------------------------
    AgencyCase(
        case_id="budget_is_respected",
        description="A tight budget bounds the run and terminates it for a typed reason.",
        goal="value is increasing over time",
        fixture_id="clear_rising",
        max_experiments=2,
        expectations=AgencyExpectations(
            max_experiments=2,
            termination_reason_in=[
                SUFFICIENT, INSUFFICIENT, BUDGET_EXHAUSTED, MAX_ITERATIONS, NO_VALID_EXPERIMENT
            ],
        ),
    ),
    # -- hard tier: judgement a keyword matcher cannot supply --------------------
    #
    # Every case below must fail FixtureAgentPolicy on a named property — enforced by
    # tests/agentic/test_agency_tiers.py — and must also stand on its own as a fair test.
    AgencyCase(
        case_id="implied_metric_is_selected_over_the_default",
        description=(
            "A question about how long something takes must be answered with the duration "
            "metric, even though the question names no column and two higher-profile volume "
            "metrics are offered first."
        ),
        goal="are we getting slower at resolving customer issues?",
        fixture_id="support_desk_slowing",
        time_field="week",
        entity_id_fields=["team"],
        # The obvious default, not the answer: this is the column a caller would hint first,
        # and picking it produces a defensible-looking "no trend" on flat ticket volume.
        metric_field="tickets_opened",
        tier=CaseTier.hard,
        expectations=AgencyExpectations(
            termination_reason_in=[SUFFICIENT],
            disposition_in=[SUPPORTED],
            # "Slower over time" is a trend question. Answering it by profiling the dataset
            # is not a weaker answer, it is a different question.
            expect_any_tool=["analyze_time_series_trend"],
        ),
    ),
    AgencyCase(
        case_id="ranking_goal_is_not_a_trend_goal",
        description=(
            "Asking which entity is worst is a question about entities, not about time. The "
            "word 'growth' appears in it, but answering the overall trend answers a different "
            "question — and here the strong regions offset the weak one, so the aggregate is "
            "unremarkable while one region is plainly failing."
        ),
        goal="which region has the weakest growth?",
        fixture_id="regional_revenue_spread",
        time_field="quarter",
        entity_id_fields=["region"],
        metric_field="revenue",
        tier=CaseTier.hard,
        expectations=AgencyExpectations(
            expect_any_tool=["rank_entities", "compare_groups"],
        ),
    ),
    AgencyCase(
        case_id="grouping_dimension_is_inferred_from_the_question",
        description=(
            "The question names two customer segments but not the column holding them. Two "
            "groupings are available; one finds nothing and one finds a large real difference. "
            "Comparing by the wrong dimension yields a true statement about a question nobody "
            "asked."
        ),
        goal="do our enterprise and self-serve customers differ in engagement?",
        fixture_id="plan_tier_separated",
        time_field=None,
        entity_id_fields=["account"],
        metric_field="monthly_active_minutes",
        tier=CaseTier.hard,
        expectations=AgencyExpectations(
            disposition_in=[SUPPORTED],
            hypothesis_status_any=[SUPPORTED],
        ),
    ),
    # The tier's counterweight. Every case above rewards finding the right answer; this one
    # rewards declining, so a policy cannot clear the tier by being uniformly more assertive.
    AgencyCase(
        case_id="unanswerable_premise_is_declined",
        description=(
            "The dataset holds latency and throughput, and nothing about complaints. The "
            "honest answer is that this question cannot be answered here. Substituting the "
            "nearest available metric produces a confident, well-evidenced conclusion about "
            "something the user did not ask about, which is worse than saying nothing."
        ),
        goal="is our customer complaint volume increasing?",
        fixture_id="api_latency_rising",
        time_field="day",
        entity_id_fields=["service"],
        # The substitution trap: this metric is genuinely, strongly rising, so answering with
        # it looks well-supported rather than obviously wrong.
        metric_field="p99_latency_ms",
        tier=CaseTier.hard,
        expectations=AgencyExpectations(
            hypothesis_status_not_in=[SUPPORTED],
            max_confidence=0.5,
            # Not merely a low-confidence answer. A policy that substitutes `p99_latency_ms`
            # and then hedges lands inside `max_confidence` too, and a recorded demo did
            # exactly that — it passed this case while measuring the wrong thing. Requiring
            # the typed reason is what separates declining from getting lucky on a weak proxy.
            termination_reason_in=["unanswerable_premise"],
            # And it must not have measured anything before deciding.
            max_experiments=0,
        ),
    ),
    AgencyCase(
        case_id="two_part_goal_resolves_both_clauses",
        description=(
            "A question with two clauses needs an answer to both. Here the honest answers "
            "differ — growth is clearly slowing, margin clearly is not — so investigating only "
            "the first produces a confident, well-evidenced reply that reads as a complete "
            "answer while silently dropping half the question."
        ),
        goal="is revenue growth slowing and is margin quality holding up?",
        fixture_id="growth_slowing_margin_steady",
        time_field="quarter",
        entity_id_fields=["company"],
        metric_field="revenue_growth_pct",
        tier=CaseTier.hard,
        expectations=AgencyExpectations(
            # `mixed` is reachable only by raising *and* investigating both clauses: it needs
            # one claim supported and another not, which a single-claim run cannot produce.
            disposition_in=[MIXED],
        ),
    ),
)


#: Exactly the cases published as `suite_agency_v1`, pinned by id.
#:
#: Pinned by id rather than derived from ``tier == core`` on purpose: a future core case would
#: silently join a derived subset and change what "the v1 result" means. This list is the
#: freeze, and ``tests/agentic/test_agency_tiers.py`` asserts every id still resolves.
SUITE_V1_CASE_IDS: tuple[str, ...] = (
    "clear_rising_is_concluded",
    "clear_falling_is_concluded",
    "flat_data_is_not_a_trend",
    "noise_is_not_a_trend",
    "two_points_are_not_a_trend",
    "contradicted_claim_is_revised",
    "opposing_entities_are_not_cherry_picked",
    "comparison_goal_uses_comparison_tools",
    "trend_goal_uses_trend_tools",
    "non_financial_trend_is_concluded",
    "non_financial_flat_is_not_a_trend",
    "supported_claim_is_challenged_before_concluding",
    "budget_is_respected",
)

SUITE_V1_CASES: tuple[AgencyCase, ...] = tuple(
    case for case in AGENCY_CASES if case.case_id in SUITE_V1_CASE_IDS
)


def cases_for_tier(tier: CaseTier, cases: tuple[AgencyCase, ...] = AGENCY_CASES) -> tuple[AgencyCase, ...]:
    """The subset of ``cases`` in one tier."""
    return tuple(case for case in cases if case.tier is tier)
