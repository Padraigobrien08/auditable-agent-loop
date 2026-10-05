"""
The model-backed agency bench harness.

Every test here runs offline against the deterministic fixture policy or an injected stub —
the harness must be exercisable without a provider, or it could only be tested by spending
money. The behaviours worth pinning are the ones that would corrupt a published scoreboard:
silently reporting a fixture result under a model's name, and overrunning the cost ceiling.
"""

from __future__ import annotations

import json

import pytest

from agentic.agent.fixture_policy import FixtureAgentPolicy
from agentic.agent.policy import (
    CritiqueProposal,
    ExperimentChoice,
    GoalInterpretation,
    HypothesisProposals,
)
from agentic.evaluation.cases import AGENCY_CASES, SUITE_ID
from agentic.evaluation.runner import run_agency_suite, run_case
from backend.config.settings import Settings
from backend.dev.agency_bench import main, run_policy_rows


def _settings() -> Settings:
    return Settings(agent_completion_model="test-model")


def _fixture_factory(kind: str, settings: Settings):  # noqa: ANN001,ARG001 - test double
    return FixtureAgentPolicy()


class _CostlyPolicy:
    """Fixture behaviour plus a fixed per-decision cost, to drive the ceiling."""

    def __init__(self, cost_per_call: float = 1.0) -> None:
        self._inner = FixtureAgentPolicy()
        self._cost_per_call = cost_per_call
        self._pending = 0.0

    def interpret_goal(self, goal_text: str, *, capability_summary: dict) -> GoalInterpretation:
        self._pending += self._cost_per_call
        return self._inner.interpret_goal(goal_text, capability_summary=capability_summary)

    def generate_hypotheses(self, interpretation, *, metric_names, dimension_names, goal_text="") -> HypothesisProposals:  # noqa: ANN001
        self._pending += self._cost_per_call
        return self._inner.generate_hypotheses(
            interpretation, metric_names=metric_names, dimension_names=dimension_names
        )

    def select_experiment(self, *, goal_summary: dict, candidates: list[dict]) -> ExperimentChoice:
        self._pending += self._cost_per_call
        return self._inner.select_experiment(goal_summary=goal_summary, candidates=candidates)

    def critique(self, *, strongest_claim, available_tools, **_) -> CritiqueProposal:  # noqa: ANN001
        self._pending += self._cost_per_call
        return self._inner.critique(
            strongest_claim=strongest_claim, available_tools=available_tools
        )

    def drain_cost_usd(self) -> float:
        pending, self._pending = self._pending, 0.0
        return pending


# -- rows --------------------------------------------------------------------


def test_fixture_row_runs_offline_and_reports_its_trials() -> None:
    rows = run_policy_rows(
        ["fixture"], trials=2, settings=_settings(), policy_factory=_fixture_factory
    )

    # One row per (policy, tier): core and hard are reported separately, never averaged.
    assert [(r.label, r.tier) for r in rows] == [("fixture", "core"), ("fixture", "hard")]
    assert all(r.trials == 2 for r in rows)
    assert not any(r.truncated for r in rows)


def test_the_deterministic_baseline_is_stable_across_trials() -> None:
    """If the fixture policy ever flaps, the suite's determinism claim is broken."""
    rows = run_policy_rows(
        ["fixture"], trials=3, settings=_settings(), policy_factory=_fixture_factory
    )

    assert rows[0].unstable_cases == []
    assert rows[0].fully_stable


def test_a_model_row_without_a_provider_is_refused_not_faked(monkeypatch) -> None:
    """
    `build_agent_policy` degrades to the fixture policy when no provider is configured.
    Publishing that as a model result would silently invalidate the scoreboard's whole claim,
    so the harness must refuse the row instead.
    """
    with pytest.raises(SystemExit, match="no LLM provider is configured"):
        run_policy_rows(
            ["model"],
            model="some-model",
            trials=1,
            settings=_settings(),
            policy_factory=_fixture_factory,
        )


def test_an_unpriced_model_row_is_refused_before_spending() -> None:
    """
    Unpriced models cost 0.0 by design, which would make the scoreboard's cost column
    meaningless and leave --max-cost-usd unable to ever fire. Fail before the first call.
    """
    with pytest.raises(SystemExit, match="no price configured"):
        run_policy_rows(
            ["model"],
            model="unpriced-model",
            trials=1,
            settings=_settings(),
            policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=0.0),
        )


def test_allow_unpriced_opts_into_a_quality_only_run() -> None:
    rows = run_policy_rows(
        ["model"],
        model="unpriced-model",
        trials=1,
        allow_unpriced=True,
        settings=_settings(),
        policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=0.0),
    )

    assert rows[0].trials == 1
    assert rows[0].total_cost_usd == 0.0


def test_a_priced_model_row_proceeds() -> None:
    priced = Settings(
        agent_completion_model="test-model",
        llm_model_prices={"priced-model": {"input_per_1m": 0.15, "output_per_1m": 0.60}},
    )

    rows = run_policy_rows(
        ["model"],
        model="priced-model",
        trials=1,
        settings=priced,
        policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=1.0),
    )

    assert rows[0].label == "priced-model"
    assert rows[0].total_cost_usd > 0


def test_a_resolved_snapshot_mismatch_is_caught_after_one_trial() -> None:
    """
    The static price check keys off the configured id, but the API bills a dated snapshot and
    the lookup is exact. Observed spend of $0.00 across real model calls is the only signal
    that the price key missed, so the run must stop after one trial rather than complete.
    """
    priced = Settings(
        agent_completion_model="test-model",
        llm_model_prices={"priced-model": {"input_per_1m": 0.15, "output_per_1m": 0.60}},
    )

    with pytest.raises(SystemExit, match="does not match the id the API billed against"):
        run_policy_rows(
            ["model"],
            model="priced-model",
            trials=5,
            settings=priced,
            # Configured id is priced, but this policy accrues nothing — the shape of a
            # snapshot-id mismatch.
            policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=0.0),
        )


def test_all_tiers_produce_one_row_each() -> None:
    rows = run_policy_rows(
        ["fixture"], trials=1, settings=_settings(), policy_factory=_fixture_factory
    )

    assert [(r.label, r.tier) for r in rows] == [("fixture", "core"), ("fixture", "hard")]


def test_tier_core_measures_only_the_frozen_v1_cases() -> None:
    from agentic.evaluation.cases import SUITE_V1_CASES, CaseTier

    rows = run_policy_rows(
        ["fixture"],
        trials=1,
        tiers=(CaseTier.core,),
        settings=_settings(),
        policy_factory=_fixture_factory,
    )

    assert len(rows) == 1
    assert rows[0].tier == "core"
    # The core tier is the frozen suite today; a divergence means v1 stopped being reproducible.
    assert rows[0].mean_pass_rate == 1.0
    assert len(SUITE_V1_CASES) == 13


def test_the_hard_tier_row_shows_the_baseline_failing() -> None:
    """The headroom, visible in the artifact that gets published."""
    from agentic.evaluation.cases import CaseTier

    rows = run_policy_rows(
        ["fixture"],
        trials=1,
        tiers=(CaseTier.hard,),
        settings=_settings(),
        policy_factory=_fixture_factory,
    )

    assert rows[0].mean_pass_rate == 0.0


def test_the_cost_ceiling_is_shared_across_a_policys_tiers() -> None:
    """
    One ceiling per policy, not one per tier — otherwise a two-tier run silently costs twice
    what the ceiling says.
    """
    from agentic.evaluation.cases import CaseTier

    rows = run_policy_rows(
        ["fixture"],
        trials=4,
        max_cost_usd=0.5,
        tiers=(CaseTier.core, CaseTier.hard),
        settings=_settings(),
        policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=1.0),
    )

    assert rows[0].truncated, "the first tier should have exhausted the shared ceiling"
    # The second tier inherits the exhausted budget rather than starting fresh.
    assert rows[1].trials <= rows[0].trials


def test_multiple_rows_are_measured_in_order() -> None:
    rows = run_policy_rows(
        ["fixture", "fixture"], trials=1, settings=_settings(), policy_factory=_fixture_factory
    )

    assert [(r.label, r.tier) for r in rows] == [
        ("fixture", "core"),
        ("fixture", "hard"),
        ("fixture", "core"),
        ("fixture", "hard"),
    ]


# -- cost ceiling ------------------------------------------------------------


def test_the_cost_ceiling_truncates_remaining_trials() -> None:
    rows = run_policy_rows(
        ["fixture"],
        trials=5,
        max_cost_usd=0.5,
        settings=_settings(),
        policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=1.0),
    )

    assert rows[0].truncated
    assert rows[0].trials < 5
    assert rows[0].total_cost_usd > 0


def test_no_ceiling_runs_every_requested_trial() -> None:
    rows = run_policy_rows(
        ["fixture"],
        trials=3,
        settings=_settings(),
        policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=1.0),
    )

    assert rows[0].trials == 3
    assert not rows[0].truncated


# -- budget passthrough ------------------------------------------------------


def test_a_supplied_budget_does_not_disarm_a_cases_own_experiment_cap() -> None:
    """
    A case's `max_experiments` is part of what it asserts, so it must win over a
    caller-supplied budget. Passing a wide budget used to replace the case's cap wholesale,
    which would let the budget cases pass vacuously whenever the bench set a cost ceiling.

    The shipped `budget_is_respected` case cannot detect this — `clear_rising` converges at
    two experiments unaided, so capped and uncapped runs look identical. A cap of 1 is below
    the natural stopping point and therefore observable.
    """
    from agentic.agent.budget import LoopBudget

    base = next(c for c in AGENCY_CASES if c.case_id == "budget_is_respected")
    tight = base.model_copy(update={"case_id": "probe", "max_experiments": 1})

    result = run_case(tight, budget=LoopBudget(max_cost_usd=5.0, max_experiments=99))

    assert len(result.observed_tools) == 1, (
        "the case's own cap of 1 was overridden by the supplied budget's 99"
    )


def test_a_supplied_budgets_other_fields_survive_the_merge() -> None:
    """The case pins only `max_experiments`; the caller's cost bound must still apply."""
    from agentic.agent.budget import LoopBudget

    base = next(c for c in AGENCY_CASES if c.case_id == "budget_is_respected")
    merged = LoopBudget(max_cost_usd=5.0, max_experiments=99).model_copy(
        update={"max_experiments": base.max_experiments}
    )

    assert merged.max_experiments == base.max_experiments
    assert merged.max_cost_usd == 5.0


def test_suite_level_budget_reaches_every_case() -> None:
    from agentic.agent.budget import LoopBudget

    report = run_agency_suite(budget=LoopBudget(max_experiments=2))

    assert report.total == len(AGENCY_CASES)


# -- CLI ---------------------------------------------------------------------


def test_cli_writes_both_artifacts(tmp_path, capsys) -> None:
    out = tmp_path / "scoreboard"

    code = main(["--policy", "fixture", "--trials", "2", "--out", str(out), "--format", "both"])

    assert code == 0
    assert out.with_suffix(".md").is_file()
    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["suite_id"] == SUITE_ID
    assert payload["requested_trials"] == 2
    assert payload["rows"][0]["label"] == "fixture"
    assert payload["prompt_version"]


def test_cli_defaults_to_the_fixture_row(capsys) -> None:
    assert main(["--trials", "1"]) == 0
    assert "fixture" in capsys.readouterr().out


def test_cli_rejects_a_zero_trial_run() -> None:
    with pytest.raises(SystemExit, match="--trials must be at least 1"):
        main(["--policy", "fixture", "--trials", "0"])


# -- local models --------------------------------------------------------------


def test_a_model_priced_at_zero_is_not_mistaken_for_a_snapshot_mismatch() -> None:
    """
    A locally served model costs nothing, and saying so with an explicit zero price is a
    statement, not a missing entry. ``$0.00`` observed spend is then the expected reading;
    the snapshot guard must not abort the run over it.
    """
    free = Settings(
        agent_completion_model="test-model",
        llm_model_prices={"qwen3:4b": {"input_per_1m": 0.0, "output_per_1m": 0.0}},
    )

    rows = run_policy_rows(
        ["model"],
        model="qwen3:4b",
        trials=2,
        settings=free,
        policy_factory=lambda kind, s: _CostlyPolicy(cost_per_call=0.0),
    )

    assert [r.trials for r in rows] == [2, 2]
    assert all(r.total_cost_usd == 0.0 for r in rows)


def test_max_elapsed_seconds_reaches_every_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slow local hardware must not be scored as the model exhausting its budget."""
    import backend.dev.agency_bench as bench

    seen: list[tuple[object, object]] = []
    real = bench.run_agency_suite

    def spy(**kwargs):  # noqa: ANN003 - pass-through test double
        seen.append((kwargs["budget"], kwargs["safety"]))
        return real(**kwargs)

    monkeypatch.setattr(bench, "run_agency_suite", spy)
    run_policy_rows(
        ["fixture"], trials=1, max_elapsed_seconds=1800.0,
        settings=_settings(), policy_factory=_fixture_factory,
    )

    assert seen
    for budget, safety in seen:
        assert budget.max_elapsed_seconds == 1800.0
        # The hard cap sits above the budget; left at its default it would stop the run
        # first, under a different reason.
        assert safety.absolute_max_elapsed_seconds == 1800.0


def test_safety_cap_is_raised_never_lowered() -> None:
    from agentic.agent.budget import SafetyLimits
    from backend.dev.agency_bench import _safety

    assert _safety(None) is None
    assert _safety(SafetyLimits().absolute_max_elapsed_seconds / 2) is None


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "api.openai.com"),
        ("http://localhost:11434/v1", "localhost:11434"),
        # Credentials and tokens in a proxy URL never reach a published scoreboard.
        ("https://user:secret@proxy.example:8443/v1?token=abc", "proxy.example:8443"),
    ],
)
def test_endpoint_records_host_only(base_url: str | None, expected: str) -> None:
    from backend.dev.agency_bench import _endpoint

    assert _endpoint(Settings(openai_base_url=base_url)) == expected


class _UnreachablePolicy(FixtureAgentPolicy):
    def interpret_goal(self, goal_text: str, *, capability_summary: dict) -> GoalInterpretation:
        from agentic.agent.policy import PolicyTransportError

        raise PolicyTransportError("provider error: 401 invalid api key")


def test_a_provider_that_never_answers_is_reported_as_such() -> None:
    """
    No completion means no spend, which the price guard would read as a price-key mismatch
    and send the reader to the wrong config. A bad key must be named as what it is.
    """
    priced = Settings(
        agent_completion_model="test-model",
        llm_model_prices={"priced-model": {"input_per_1m": 0.15, "output_per_1m": 0.60}},
    )

    with pytest.raises(SystemExit, match="provider never answered.*401 invalid api key"):
        run_policy_rows(
            ["model"], model="priced-model", trials=3, settings=priced,
            policy_factory=lambda kind, s: _UnreachablePolicy(),
        )


def test_an_ablated_row_names_what_was_removed(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    """An ablated row must never be readable as the full loop, in the table or the JSON."""
    out = tmp_path / "board"
    main(["--policy", "fixture", "--trials", "1", "--tier", "core",
          "--ablate", "critic", "--ablate", "typed_termination",
          "--format", "both", "--out", str(out)])

    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["ablated"] == ["critic", "typed_termination"]
    assert [r["label"] for r in payload["rows"]] == ["fixture -critic -typed_termination"]
    assert "| fixture -critic -typed_termination |" in out.with_suffix(".md").read_text()


def test_an_unablated_run_records_no_ablation(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    out = tmp_path / "board"
    main(["--policy", "fixture", "--trials", "1", "--tier", "core", "--format", "json", "--out", str(out)])

    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["ablated"] == []
    assert [r["label"] for r in payload["rows"]] == ["fixture"]
