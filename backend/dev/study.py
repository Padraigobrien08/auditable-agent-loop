"""
Run the scaffold-vs-model study and report it.

S5 of ``docs/decisions/2026-10-05-scaffold-vs-model.md``. The design is models × conditions ×
tiers × trials, about 1,350 runs in full, which is overnight work on a laptop. So this runner
persists every model × condition cell to its own file the moment it finishes, and skips cells
already on disk: an interrupted run resumes where it stopped, and adding a model later re-runs
nothing that exists.

Each cell goes through :func:`backend.dev.agency_bench.run_policy_rows`, so every guard the
bench applies (an unpriced model, a provider that never answers, the cost ceiling, a model row
that silently became the fixture policy) applies here too. The report is computed afterwards by
the pure :mod:`agentic.evaluation.study`, from the persisted reports alone.

Usage::

    # the pilot: two sizes, full loop and bare, every tier
    python -m backend.dev.study run --out data/evaluation/agency/study \\
        --model qwen3:1.7b=1.7 --model qwen3:8b=8 --condition A --condition C \\
        --tier core --tier hard --tier generated --trials 1 --max-elapsed-seconds 1800

    # the rule-based baseline (size 0: the scaffold with no model), full and fully ablated
    python -m backend.dev.study run --out ... --model fixture --condition A --condition B

    # the signal-strength curve for the same models and conditions (36 runs each, unscored)
    python -m backend.dev.study run --out ... --model qwen3:8b=8 --condition A --condition C \
        --tier core --trials 1 --sweep

    python -m backend.dev.study report data/evaluation/agency/study
    python -m backend.dev.study report data/evaluation/agency/study --png   # + PNGs for Substack

Conditions: ``A`` is the full loop; ``B`` the loop with critic, typed termination and the
mutual-exclusivity check all off; ``B-critic`` (etc.) one component off; ``C`` the bare model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from agentic.agent.ablations import LoopAblations
from agentic.evaluation.agency import AgencyReport
from agentic.evaluation.cases import CaseTier
from agentic.evaluation.runner import SweepPoint, run_signal_sweep
from agentic.evaluation.scoreboard import RunMetrics
from agentic.evaluation.study import Observation, SweepCurve, analyse, observations, sweep_curve
from backend.agents.agentic_model_policy import AGENTIC_PROMPT_VERSION
from backend.config.settings import get_settings
from backend.dev.agency_bench import (
    BARE,
    FIXTURE,
    LOOP,
    MODEL,
    BareFactory,
    PolicyFactory,
    _assert_priced,
    _budget,
    _default_bare_factory,
    _default_policy_factory,
    _safety,
    run_policy_rows,
)
from backend.dev.bare_model import BARE_PROMPT_VERSION, run_bare_sweep
from backend.dev.study_figures import headline_svg, sweep_svg

CONDITIONS: tuple[str, ...] = (
    "A", "B", *(f"B-{name}" for name in LoopAblations.model_fields), "C",
)


def condition_settings(condition: str) -> tuple[str, LoopAblations | None]:
    """The bench's (condition, ablations) for one study condition."""
    if condition == "A":
        return LOOP, None
    if condition == "B":
        return LOOP, LoopAblations.without(*LoopAblations.model_fields)
    if condition.startswith("B-"):
        return LOOP, LoopAblations.without(condition.removeprefix("B-"))
    if condition == "C":
        return BARE, None
    raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")


def parse_model(spec: str) -> tuple[str, float | None]:
    """``name`` or ``name=size_in_billions``. The rule-based policy is size 0 by definition."""
    name, _, size = spec.partition("=")
    if size:
        return name, float(size)
    return name, 0.0 if name == FIXTURE else None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def cell_path(out: Path, model: str, condition: str) -> Path:
    return out / "cells" / f"{_slug(model)}__{_slug(condition)}.json"


def run_cell(
    out: Path, *, model: str, size_b: float | None, condition: str, tiers: tuple[CaseTier, ...],
    trials: int, max_elapsed_seconds: float | None = None, max_cost_usd: float | None = None,
    allow_unpriced: bool = False, policy_factory: PolicyFactory = _default_policy_factory,
    bare_factory: BareFactory = _default_bare_factory, settings: Any = None,
) -> Path | None:
    """Run one model × condition over every tier and persist it. ``None`` if it already exists."""
    path = cell_path(out, model, condition)
    if path.exists():
        return None
    bench_condition, ablations = condition_settings(condition)
    if bench_condition == BARE and model == FIXTURE:
        raise SystemExit("condition C is a model with no loop; the rule-based policy has no bare form")

    captured: dict[str, Any] = {}

    def keep(label: str, tier: CaseTier | None, reports: list[AgencyReport],
             metrics: list[RunMetrics], truncated: bool) -> None:
        captured[tier.value if tier else "all"] = {
            "label": label,
            "reports": [r.model_dump(mode="json") for r in reports],
            "metrics": [m.model_dump(mode="json") for m in metrics],
            "truncated": truncated,
        }

    run_policy_rows(
        [FIXTURE if model == FIXTURE else MODEL], model=None if model == FIXTURE else model,
        trials=trials, tiers=tiers, condition=bench_condition, ablations=ablations,
        max_elapsed_seconds=max_elapsed_seconds, max_cost_usd=max_cost_usd,
        allow_unpriced=allow_unpriced, policy_factory=policy_factory, bare_factory=bare_factory,
        settings=settings, on_tier=keep,
    )
    payload = {
        "model": model, "size_b": size_b, "condition": condition, "trials": trials,
        "prompt_version": AGENTIC_PROMPT_VERSION if bench_condition == LOOP else None,
        "bare_prompt_version": BARE_PROMPT_VERSION if bench_condition == BARE else None,
        "tiers": captured,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written whole and renamed, so an interrupted run never leaves a half cell that a resume
    # would mistake for a finished one.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def sweep_path(out: Path, model: str, condition: str) -> Path:
    return out / "sweeps" / f"{_slug(model)}__{_slug(condition)}.json"


def run_sweep(
    out: Path, *, model: str, size_b: float | None, condition: str,
    max_elapsed_seconds: float | None = None, allow_unpriced: bool = False,
    policy_factory: PolicyFactory = _default_policy_factory,
    bare_factory: BareFactory = _default_bare_factory, settings: Any = None,
) -> Path | None:
    """
    Run the unscored signal sweep for one model × condition and persist it. ``None`` if it exists.

    The same refusals as a cell: a model row with no provider, an unpriced model, and a sweep in
    which the provider never answered are stopped rather than written, so a curve on disk is
    always a curve of the model it is named after.
    """
    path = sweep_path(out, model, condition)
    if path.exists():
        return None
    bench_condition, ablations = condition_settings(condition)
    if bench_condition == BARE and model == FIXTURE:
        raise SystemExit("condition C is a model with no loop; the rule-based policy has no bare form")
    base = settings if settings is not None else get_settings()
    kind = FIXTURE if model == FIXTURE else MODEL
    row_settings = base if kind == FIXTURE else base.model_copy(update={"agent_completion_model": model})
    if kind == MODEL and not allow_unpriced:
        _assert_priced(row_settings, model, model)

    points: list[SweepPoint]
    if bench_condition == BARE:
        responder, _drain = bare_factory(row_settings)
        points = run_bare_sweep(responder)
    else:
        policy = policy_factory(kind, row_settings)
        if kind == MODEL and type(policy).__name__ == "FixtureAgentPolicy":
            raise SystemExit(f"sweep for {model!r} requested but no LLM provider is configured")
        points = run_signal_sweep(policy=policy, ablations=ablations,
                                  budget=_budget(None, max_elapsed_seconds), safety=_safety(max_elapsed_seconds))

    if kind == MODEL and points and all((p.policy_failure or "").endswith(":transport") for p in points):
        raise SystemExit(f"every sweep point for {model!r} failed because the provider never answered")

    payload = {"model": model, "size_b": size_b, "condition": condition,
               "points": [p.model_dump(mode="json") for p in points]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_sweeps(out: Path) -> list[SweepCurve]:
    curves = []
    for path in sorted((out / "sweeps").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        points = [SweepPoint.model_validate(p) for p in data["points"]]
        curves.append(sweep_curve(points, model=data["model"], size_b=data["size_b"], condition=data["condition"]))
    return curves


def load_observations(out: Path) -> list[Observation]:
    obs: list[Observation] = []
    for path in sorted((out / "cells").glob("*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        for tier, data in cell["tiers"].items():
            reports = [AgencyReport.model_validate(r) for r in data["reports"]]
            obs += observations(reports, model=cell["model"], size_b=cell["size_b"],
                                condition=cell["condition"], tier=tier)
    return obs


def write_report(out: Path, *, png: bool = False) -> str:
    report = analyse(load_observations(out), load_sweeps(out))
    markdown = report.to_markdown()
    (out / "study.md").write_text(markdown + "\n", encoding="utf-8")
    (out / "study.json").write_text(report.model_dump_json(indent=1) + "\n", encoding="utf-8")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    if report.cells:
        (figures / "headline.svg").write_text(headline_svg(report), encoding="utf-8")
    if report.sweeps:
        (figures / "sweep.svg").write_text(sweep_svg(report.sweeps), encoding="utf-8")
    if png:
        from backend.dev.study_png import export_all

        export_all(figures)
    return markdown


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m backend.dev.study", description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run every missing model × condition cell.")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--model", action="append", required=True,
                     help="name or name=size_in_billions (repeatable). 'fixture' is the rule-based policy.")
    run.add_argument("--condition", action="append", choices=CONDITIONS, required=True)
    run.add_argument("--tier", action="append", choices=[t.value for t in CaseTier], required=True)
    run.add_argument("--trials", type=int, default=3)
    run.add_argument("--max-elapsed-seconds", type=float, default=None)
    run.add_argument("--max-cost-usd", type=float, default=None)
    run.add_argument("--allow-unpriced", action="store_true")
    run.add_argument("--sweep", action="store_true",
                     help="Also run the unscored signal sweep (36 runs) per model × condition.")

    rep = sub.add_parser("report", help="Analyse the persisted cells.")
    rep.add_argument("out", type=Path)
    rep.add_argument("--png", action="store_true",
                     help="Also write light and dark PNGs of each figure (needs Chrome's headless shell).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "report":
        print(write_report(args.out, png=args.png))
        return 0

    tiers = tuple(CaseTier(t) for t in args.tier)
    for spec in args.model:
        model, size_b = parse_model(spec)
        for condition in args.condition:
            if model == FIXTURE and condition == "C":
                continue
            written = run_cell(
                args.out, model=model, size_b=size_b, condition=condition, tiers=tiers,
                trials=args.trials, max_elapsed_seconds=args.max_elapsed_seconds,
                max_cost_usd=args.max_cost_usd, allow_unpriced=args.allow_unpriced,
            )
            print(f"{model} {condition}: {'written ' + str(written) if written else 'already done, skipped'}")
            if args.sweep:
                swept = run_sweep(
                    args.out, model=model, size_b=size_b, condition=condition,
                    max_elapsed_seconds=args.max_elapsed_seconds, allow_unpriced=args.allow_unpriced,
                )
                print(f"{model} {condition} sweep: {'written ' + str(swept) if swept else 'already done, skipped'}")
    print(write_report(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
