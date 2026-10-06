"""
The study's figures render from the report alone, stay readable, and never repaint a condition.

The SVG is checked structurally: it must parse, carry a title and description, select dark mode
with its own palette, and keep each condition on its fixed colour slot whatever else is present.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from agentic.evaluation.study import Cell, StudyReport, SweepCurve, SweepLevel
from backend.dev.study_figures import headline_svg, sweep_svg

_NS = "{http://www.w3.org/2000/svg}"


def _cell(model: str, size: float | None, condition: str, tier: str = "core", rate: float = 0.5) -> Cell:
    return Cell(model=model, size_b=size, condition=condition, tier=tier, rate=rate,
                ci_low=max(0.0, rate - 0.1), ci_high=min(1.0, rate + 0.1), raw_rate=rate, n_cases=10, n_runs=10)


_RATES = {"A": 0.8, "B": 0.6, "C": 0.4}


def _report() -> StudyReport:
    cells = [_cell(f"m{s}", s, c, tier, rate=_RATES[c])
             for tier in ("core", "hard") for c in ("A", "B", "C") for s in (1.7, 8.0)]
    cells += [_cell("fixture", 0.0, "A"), _cell("gpt-5.4-mini", None, "A", rate=0.9)]
    return StudyReport(cells=cells)


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_headline_parses_and_describes_itself() -> None:
    root = _parse(headline_svg(_report()))

    assert root.find(f"{_NS}title").text == "Honest pass rate by model size"
    assert "study.md" in root.find(f"{_NS}desc").text, "the description must point at the table view"


def test_dark_mode_is_selected_with_its_own_palette() -> None:
    svg = headline_svg(_report())

    assert "@media (prefers-color-scheme: dark)" in svg
    assert "#1a1a19" in svg and "#3987e5" in svg


def test_each_condition_keeps_its_slot_when_others_are_missing() -> None:
    """Colour follows the condition, never its rank: dropping B must not repaint C."""
    only_c = StudyReport(cells=[_cell("m", 1.7, "C"), _cell("m", 8.0, "C")])

    assert 'stroke="var(--s3)"' in headline_svg(only_c)
    assert "var(--s1)" not in headline_svg(only_c).split("</style>")[1]


def test_every_line_is_direct_labelled_and_the_no_model_point_is_placed() -> None:
    svg = headline_svg(_report())

    for condition, rate in _RATES.items():
        assert f">{condition} {rate:.0%}<" in svg
    assert "no model" in svg
    assert "gpt-5.4-mini (A) 90%" in svg


def test_sweep_parses_and_labels_t50() -> None:
    def levels(rates: list[float]) -> list[SweepLevel]:
        return [SweepLevel(target_t=t, mean_realised_t=t, claim_rate=r, n=3) for t, r in zip((0, 2, 4), rates)]

    curves = [SweepCurve(model="m", size_b=1.7, condition="A", levels=levels([0.0, 0.4, 1.0]), t50=2.4),
              SweepCurve(model="m", size_b=1.7, condition="C", levels=levels([0.7, 1.0, 1.0]),
                         t50_note="claims at every level")]

    root = _parse(sweep_svg(curves))
    text = ET.tostring(root, encoding="unicode")
    assert "A t50 2.4" in text
    assert "C always claims" in text


def test_figures_render_with_nothing_to_plot() -> None:
    _parse(headline_svg(StudyReport()))
    _parse(sweep_svg([]))


def test_the_report_writes_both_figures(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    from agentic.evaluation.cases import CaseTier
    from backend.dev.study import run_cell, run_sweep, write_report

    run_cell(tmp_path, model="fixture", size_b=0.0, condition="A", tiers=(CaseTier.core,), trials=1)
    run_sweep(tmp_path, model="fixture", size_b=0.0, condition="A")
    write_report(tmp_path)

    for name in ("headline.svg", "sweep.svg"):
        _parse((tmp_path / "figures" / name).read_text())


def test_identical_series_share_one_label_and_the_reference_is_drawn_last() -> None:
    """Identical lines draw exactly on top of each other; the figure must say they are equal."""
    same = StudyReport(cells=[_cell("m", s, c) for c in ("A", "B") for s in (1.7, 8.0)]
                       + [_cell("fixture", 0.0, c, rate=1.0) for c in ("A", "B")])
    svg = headline_svg(same)

    assert ">A = B 50%<" in svg
    assert "A = B 100%" in svg, "the no-model points coincide too, and must say so"
    body = svg.split("</style>")[1]
    assert body.rfind('stroke="var(--s1)"') > body.rfind('stroke="var(--s2)"'), "A must be drawn on top"


def test_identical_sweep_curves_share_one_label() -> None:
    levels = [SweepLevel(target_t=t, mean_realised_t=t, claim_rate=r, n=3) for t, r in [(0, 0.0), (4, 1.0)]]
    curves = [SweepCurve(model="m", size_b=1.7, condition=c, levels=levels, t50=2.0) for c in ("A", "B")]

    assert "A = B t50 2.0" in sweep_svg(curves)


def test_a_merge_never_hides_a_differing_value() -> None:
    """Same points but different labels (as a hand-built input could give) stay two labels."""
    levels = [SweepLevel(target_t=t, mean_realised_t=t, claim_rate=r, n=3) for t, r in [(0, 0.0), (4, 1.0)]]
    curves = [SweepCurve(model="m", size_b=1.7, condition="A", levels=levels, t50=2.4),
              SweepCurve(model="m", size_b=1.7, condition="C", levels=levels, t50=2.5)]
    svg = sweep_svg(curves)

    assert "A t50 2.4" in svg and "C t50 2.5" in svg
