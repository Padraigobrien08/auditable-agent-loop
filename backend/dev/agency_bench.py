"""
Run ``suite_agency_v1`` against real models and produce a publishable scoreboard.

The agency suite scores whether the investigation loop *reasons* well — concludes when the
evidence supports it, revises when contradicted, declines when it cannot. It was written to
accept any :class:`~agentic.agent.policy.AgentPolicy` precisely so a model could be held to the
same bar as the deterministic baseline. This is the harness that does that.

It lives in ``backend/dev`` rather than in ``agentic/evaluation`` on purpose: assembling a
model-backed policy requires settings, a provider, and the prompt registry, and ``agentic/``
must not import any of them. Keeping the coupling here is what lets
``python -m agentic.evaluation`` stay offline, free, and deterministic.

Usage::

    # deterministic baseline, no provider needed
    python -m backend.dev.agency_bench --policy fixture --trials 3

    # baseline plus a model, under a suite-level cost ceiling
    python -m backend.dev.agency_bench \\
        --policy fixture --policy model --model gpt-5.4-mini \\
        --trials 5 --max-cost-usd 2.00 --out scoreboard --format both

Model rows need a configured provider (``EDGAR_BACKEND_OPENAI_API_KEY`` or
``OPENAI_API_KEY``). Without one, ``build_agent_policy`` degrades to the fixture policy — the
harness detects that and says so rather than reporting a fixture result under a model's name.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import structlog

from agentic.agent.budget import LoopBudget, SafetyLimits
from agentic.agent.policy import AgentPolicy
from agentic.evaluation.agency import AgencyReport
from agentic.evaluation.cases import SUITE_ID, CaseTier
from agentic.evaluation.runner import run_agency_suite
from agentic.evaluation.scoreboard import (
    MetricsObserver,
    PolicyScorecard,
    RunMetrics,
    Scoreboard,
    aggregate_trials,
)
from backend.agents.agentic_model_policy import AGENTIC_PROMPT_VERSION, build_agent_policy
from backend.config.settings import Settings, get_settings
from backend.llm.pricing import parse_model_prices

log = structlog.get_logger(__name__)

FIXTURE = "fixture"
MODEL = "model"

#: Returns the policy for one row. Injectable so tests never construct a provider.
PolicyFactory = Callable[[str, Settings], AgentPolicy]


def _default_policy_factory(kind: str, settings: Settings) -> AgentPolicy:
    if kind == FIXTURE:
        from agentic.agent.fixture_policy import FixtureAgentPolicy

        return FixtureAgentPolicy()
    return build_agent_policy(settings)


def _label(kind: str, model: str | None) -> str:
    return FIXTURE if kind == FIXTURE else (model or "model")


def _assert_priced(settings: Settings, model: str | None, label: str) -> None:
    """
    Refuse a model row whose model has no configured price.

    Unpriced models cost ``0.0`` by design (see :mod:`backend.llm.pricing`), which is the
    right call for a deployment but corrosive here: it makes the scoreboard's cost column
    read ``$0.0000`` for every model, and it disarms ``--max-cost-usd``, since the ceiling
    sums a quantity that is always zero. A paid benchmark with a ceiling that cannot fire is
    worse than no ceiling, because it looks protected.
    """
    prices = parse_model_prices(settings.llm_model_prices)
    if prices and (model is None or model in prices):
        return
    raise SystemExit(
        f"no price configured for {label!r}, so cost tracking would report $0.00 and "
        "--max-cost-usd could never fire.\n"
        "Set EDGAR_BACKEND_LLM_MODEL_PRICES, e.g.\n"
        f'  EDGAR_BACKEND_LLM_MODEL_PRICES=\'{{"{model or "<model-id>"}": '
        '{"input_per_1m": 0.15, "output_per_1m": 0.60}}\'\n'
        "Pass --allow-unpriced to measure quality only, accepting a meaningless cost column."
    )


def _is_known_free(settings: Settings, model: str | None) -> bool:
    """
    True when the model is *priced* at zero, as a locally served open-weight model is.

    Distinct from unpriced: an explicit ``0.0`` entry is a statement that the model costs
    nothing, so ``$0.00`` observed spend is the expected reading rather than evidence that
    the price key missed the id the provider billed against.
    """
    price = parse_model_prices(settings.llm_model_prices).get(model or "")
    return price is not None and price.input_per_1m == 0.0 and price.output_per_1m == 0.0


def _endpoint(settings: Settings) -> str:
    """Host the model rows were served from, recorded so a local run is never read as a hosted one.

    Host and port only: a proxy URL can carry credentials or tokens in its userinfo or path.
    """
    if not settings.openai_base_url:
        return "api.openai.com"
    parts = urlsplit(settings.openai_base_url)
    return parts.hostname + (f":{parts.port}" if parts.port else "") if parts.hostname else "custom"


def _budget(budget_cost_usd: float | None, max_elapsed_seconds: float | None) -> LoopBudget | None:
    overrides: dict[str, float] = {}
    if budget_cost_usd:
        overrides["max_cost_usd"] = budget_cost_usd
    if max_elapsed_seconds:
        overrides["max_elapsed_seconds"] = max_elapsed_seconds
    return LoopBudget.model_validate(overrides) if overrides else None


def _safety(max_elapsed_seconds: float | None) -> SafetyLimits | None:
    # The safety cap sits above the budget and would otherwise stop a long local run first,
    # under a different reason. Raised only as far as the budget, never lowered.
    if not max_elapsed_seconds:
        return None
    default = SafetyLimits()
    if max_elapsed_seconds <= default.absolute_max_elapsed_seconds:
        return None
    return SafetyLimits(absolute_max_elapsed_seconds=max_elapsed_seconds)


def run_policy_rows(
    kinds: list[str],
    *,
    model: str | None = None,
    trials: int = 3,
    max_cost_usd: float | None = None,
    budget_cost_usd: float | None = None,
    max_elapsed_seconds: float | None = None,
    allow_unpriced: bool = False,
    tiers: tuple[CaseTier | None, ...] = (CaseTier.core, CaseTier.hard),
    settings: Settings | None = None,
    policy_factory: PolicyFactory = _default_policy_factory,
) -> list[PolicyScorecard]:
    """
    Run the suite ``trials`` times per requested policy and aggregate one scorecard each.

    Produces one row per (policy, tier). Core and hard are never merged into a single number:
    the core tier is saturated by design history and the hard tier is where the headroom is,
    so an average would let either hide inside the other.

    Trials stop early — and the row is marked ``truncated`` — once accumulated spend crosses
    ``max_cost_usd``. That ceiling sits on top of the per-run ``LoopBudget.max_cost_usd``:
    the budget bounds one investigation, this bounds the whole benchmark. Spend accumulates
    across a policy's tiers, so one ceiling covers the whole policy.

    ``max_elapsed_seconds`` overrides the per-investigation wall-clock budget. Hosted models
    never approach the default, but a model served on a laptop can, and the run would then be
    scored as ``budget_exhausted`` — a measurement of the hardware filed as one of the model.
    """
    base = settings if settings is not None else get_settings()
    rows: list[PolicyScorecard] = []

    for kind in kinds:
        row_settings = base.model_copy(update={"agent_completion_model": model}) if (
            kind == MODEL and model
        ) else base
        policy = policy_factory(kind, row_settings)
        label = _label(kind, model)

        if kind == MODEL and type(policy).__name__ == "FixtureAgentPolicy":
            # Reporting a fixture result under a model's name would silently corrupt the
            # scoreboard's central claim, so refuse the row instead.
            log.error("agency_bench.no_provider", label=label)
            raise SystemExit(
                f"--policy model requested for {label!r} but no LLM provider is configured; "
                "set EDGAR_BACKEND_OPENAI_API_KEY (or OPENAI_API_KEY) or drop the model row."
            )

        if kind == MODEL and not allow_unpriced:
            _assert_priced(row_settings, model, label)

        budget = _budget(budget_cost_usd, max_elapsed_seconds)
        safety = _safety(max_elapsed_seconds)
        free = kind == MODEL and _is_known_free(row_settings, model)
        # Spend accumulates across every tier this policy is measured on, so a two-tier run
        # is bounded by one ceiling rather than one per tier.
        policy_metrics: list[RunMetrics] = []

        for tier in tiers:
            reports: list[AgencyReport] = []
            metrics: list[RunMetrics] = []
            truncated = False

            for trial in range(trials):
                observer = MetricsObserver()
                reports.append(
                    run_agency_suite(
                        policy=policy, observer=observer, budget=budget, tier=tier, safety=safety)
                )
                fresh = observer.drain()
                metrics.extend(fresh)
                policy_metrics.extend(fresh)
                spent = sum(m.cost_usd for m in policy_metrics)
                log.info(
                    "agency_bench.trial",
                    label=label,
                    tier=tier.value if tier else "all",
                    trial=trial + 1,
                    of=trials,
                    pass_rate=reports[-1].pass_rate,
                    spent_usd=round(spent, 4),
                )

                # The static check above prices the *configured* id, but the API bills against
                # a resolved snapshot ("gpt-5.4-mini" -> "gpt-5.4-mini-2026-03-17") and the
                # price lookup is an exact match. Only observed spend can catch that mismatch,
                # so fail after one trial rather than completing a whole paid benchmark at $0.
                calls = sum(m.model_calls for m in policy_metrics)
                first_trial = tier is tiers[0] and trial == 0
                if (kind == MODEL and not allow_unpriced and not free and first_trial
                        and calls > 0 and spent == 0.0):
                    log.error("agency_bench.unpriced_resolved_model", label=label, model_calls=calls)
                    raise SystemExit(
                        f"{label!r} made {calls} model calls that cost $0.00, so its price key "
                        "does not match the id the API billed against.\n"
                        "The provider resolves an alias to a dated snapshot; price that snapshot "
                        "id (check the 'llm_model_unpriced' log line for the exact value), or "
                        "pass --allow-unpriced to measure quality only."
                    )

                if max_cost_usd is not None and spent >= max_cost_usd and trial + 1 < trials:
                    log.warning(
                        "agency_bench.cost_ceiling",
                        label=label,
                        tier=tier.value if tier else "all",
                        spent_usd=round(spent, 4),
                        ceiling_usd=max_cost_usd,
                        completed_trials=trial + 1,
                        requested_trials=trials,
                    )
                    truncated = True
                    break

            rows.append(
                aggregate_trials(
                    label, reports, metrics,
                    truncated=truncated, tier=tier.value if tier else "",
                )
            )

    return rows


def _render_json(
    board: Scoreboard, *, trials: int, model: str | None, endpoint: str | None = None,
    max_elapsed_seconds: float | None = None,
) -> str:
    payload = {
        "suite_id": board.suite_id,
        "prompt_version": AGENTIC_PROMPT_VERSION,
        "requested_trials": trials,
        "model": model,
        "endpoint": endpoint,
        "max_elapsed_seconds": max_elapsed_seconds,
        "rows": [row.model_dump(mode="json") for row in board.rows],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m backend.dev.agency_bench",
        description="Score policies on suite_agency_v1 over repeated trials.",
    )
    p.add_argument(
        "--policy",
        action="append",
        choices=[FIXTURE, MODEL],
        help="Policy row to measure; repeat to compare (default: fixture).",
    )
    p.add_argument("--model", default=None, help="Model id for the 'model' row.")
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Suite runs per policy. More than one is required for a model row to mean anything.",
    )
    p.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Suite-level spend ceiling; remaining trials are skipped once crossed.",
    )
    p.add_argument(
        "--budget-cost-usd",
        type=float,
        default=None,
        help="Per-investigation LoopBudget.max_cost_usd.",
    )
    p.add_argument(
        "--max-elapsed-seconds",
        type=float,
        default=None,
        help=(
            "Per-investigation wall-clock budget (default: LoopBudget's). Raise it for a model "
            "served locally, so slow hardware is not scored as budget exhaustion."
        ),
    )
    p.add_argument(
        "--tier",
        choices=[t.value for t in CaseTier] + ["all"],
        default="all",
        help=(
            "Which tier(s) to measure. 'all' produces one row per (policy, tier); core and "
            "hard are never averaged together."
        ),
    )
    p.add_argument(
        "--allow-unpriced",
        action="store_true",
        help=(
            "Measure a model row with no configured price. The cost column will read $0.00 "
            "and --max-cost-usd will never fire; quality scores remain valid."
        ),
    )
    p.add_argument("--out", default=None, help="Path stem for written output (no extension).")
    p.add_argument("--format", choices=["json", "md", "both"], default="md")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kinds = args.policy or [FIXTURE]
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    rows = run_policy_rows(
        kinds,
        model=args.model,
        trials=args.trials,
        max_cost_usd=args.max_cost_usd,
        budget_cost_usd=args.budget_cost_usd,
        max_elapsed_seconds=args.max_elapsed_seconds,
        allow_unpriced=args.allow_unpriced,
        tiers=(
            (CaseTier.core, CaseTier.hard) if args.tier == "all" else (CaseTier(args.tier),)
        ),
    )
    board = Scoreboard(suite_id=SUITE_ID, rows=rows)

    markdown = board.to_markdown()
    payload = _render_json(
        board, trials=args.trials, model=args.model,
        endpoint=_endpoint(get_settings()) if MODEL in kinds else None,
        max_elapsed_seconds=args.max_elapsed_seconds,
    )

    if args.format in ("md", "both"):
        print(markdown)
    if args.format == "json":
        print(payload)

    if args.out:
        stem = Path(args.out)
        if args.format in ("md", "both"):
            stem.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
        if args.format in ("json", "both"):
            stem.with_suffix(".json").write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
