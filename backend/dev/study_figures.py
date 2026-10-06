"""
The study's two figures, rendered from ``study.json`` as self-contained SVG.

S6 of ``docs/decisions/2026-10-05-scaffold-vs-model.md``:

- **headline.svg**: honest pass rate against model size, one panel per tier, one line per
  condition with its 95% interval as a faint band. The rule-based policy (the scaffold with no
  model) sits at its own "no model" slot left of the size axis, because size 0 has no place on
  a log scale. A reference model of unknown size (the frontier row) is a labelled horizontal
  hairline.
- **sweep.svg**: the signal-strength curve, claim rate against realised slope t, one panel per
  model, one line per condition.

Pure Python with no plotting dependency, so the figures are reproducible from the persisted
study alone. Colours are the dataviz reference palette's first three categorical slots, which
validate all-pairs in both modes; light/dark is selected with ``prefers-color-scheme`` inside
the SVG. Every line is direct-labelled and the report's tables carry every value, because the
third slot (aqua) is below 3:1 on the light surface.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from html import escape

from agentic.evaluation.study import StudyReport, SweepCurve

#: Condition -> categorical slot. Fixed per condition, never by rank, so a figure missing a
#: condition does not repaint the others.
_SLOT = {"A": 1, "B": 2, "C": 3}
_EXTRA_SLOTS = (4, 5, 6)  # B-<component> variants, in order of first appearance
_LIGHT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4", 6: "#008300"}
_DARK = {1: "#3987e5", 2: "#d95926", 3: "#199e70", 4: "#c98500", 5: "#d55181", 6: "#008300"}

#: The report's full wording, shortened to fit a direct label.
_SHORT_NOTES = {"claims at every level": "always claims", "never claims": "never claims"}

_NAMES = {"A": "A · full loop", "B": "B · loop, scaffold off", "C": "C · model alone"}

_W, _H = 360, 250            # one panel
#: The right margin holds the direct labels at each line's end.
_M = {"l": 44, "r": 96, "t": 34, "b": 40}
_TITLE_H = 64                # figure title + legend row


def _style(slots: dict[str, int]) -> str:
    def block(palette: dict[int, str], surface: str, ink: str, ink2: str, muted: str, grid: str, axis: str) -> str:
        series = " ".join(f"--s{slot}: {palette[slot]};" for slot in sorted(set(slots.values())))
        return (f"--surface: {surface}; --ink: {ink}; --ink2: {ink2}; --muted: {muted}; "
                f"--grid: {grid}; --axis: {axis}; {series}")

    light = block(_LIGHT, "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
    dark = block(_DARK, "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835")
    return (
        "<style>"
        f"svg {{ {light} font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; }}"
        f"@media (prefers-color-scheme: dark) {{ svg {{ {dark} }} }}"
        ".bg { fill: var(--surface); } .grid { stroke: var(--grid); stroke-width: 1; }"
        ".axis { stroke: var(--axis); stroke-width: 1; } .t1 { fill: var(--ink); }"
        ".t2 { fill: var(--ink2); } .tm { fill: var(--muted); font-variant-numeric: tabular-nums; }"
        "</style>"
    )


def _slots(conditions: Iterable[str]) -> dict[str, int]:
    slots: dict[str, int] = {}
    extra = iter(_EXTRA_SLOTS)
    for c in conditions:
        if c not in slots:
            slots[c] = _SLOT.get(c) or next(extra, 6)
    return slots


def _name(condition: str) -> str:
    return _NAMES.get(condition, f"{condition} · one component off" if condition.startswith("B-") else condition)


def _legend(slots: dict[str, int], x: float, y: float) -> str:
    out, cursor = [], x
    for condition, slot in slots.items():
        label = _name(condition)
        out.append(f'<line x1="{cursor}" y1="{y - 4}" x2="{cursor + 16}" y2="{y - 4}" '
                   f'stroke="var(--s{slot})" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<text x="{cursor + 22}" y="{y}" class="t2" font-size="12">{escape(label)}</text>')
        cursor += 30 + 6.6 * len(label)
    return "".join(out)


def _draw_order(slots: dict[str, int]) -> list[tuple[str, int]]:
    """Reverse slot order, so the reference condition (A) is drawn last and never hidden."""
    return list(reversed(list(slots.items())))


def _merge_identical(entries: list[tuple[float, str, str, object]]) -> list[tuple[float, str, int]]:
    """
    One label per distinct series: ``(y, condition, value, signature)`` entries that share a
    signature become "A = B value". Identical series draw exactly on top of each other, so
    without this one of them looks absent rather than equal.
    """
    groups: dict[object, list[tuple[float, str, str]]] = {}
    for y, condition, value, signature in entries:
        # The shown value is part of the key: two series merge only when what the label would
        # say is the same too, so a merge can never hide a value.
        groups.setdefault((signature, value), []).append((y, condition, value))
    merged = []
    for members in groups.values():
        names = " = ".join(sorted(c for _, c, _ in members))
        merged.append((members[0][0], f"{names} {members[0][2]}", 0))
    return merged


def _spread(labels: list[tuple[float, str, int]], gap: float = 13.0) -> list[tuple[float, str, int]]:
    """Nudge end labels apart vertically so none overlap, keeping their order."""
    placed: list[tuple[float, str, int]] = []
    for y, text, slot in sorted(labels):
        if placed and y - placed[-1][0] < gap:
            y = placed[-1][0] + gap
        placed.append((y, text, slot))
    return placed


def _y_axis(ox: float, oy: float, ph: float, pw: float) -> str:
    out = []
    for v in (0, 25, 50, 75, 100):
        y = oy + ph - ph * v / 100
        cls = "axis" if v == 0 else "grid"
        out.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox + pw}" y2="{y:.1f}" class="{cls}"/>')
        out.append(f'<text x="{ox - 6}" y="{y + 4:.1f}" class="tm" font-size="11" text-anchor="end">{v}%</text>')
    return "".join(out)


def _marker(x: float, y: float, slot: int) -> str:
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--s{slot})" '
            f'stroke="var(--surface)" stroke-width="2"/>')


def _svg(width: float, height: float, title: str, desc: str, body: str, slots: dict[str, int]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img">'
        f"<title>{escape(title)}</title><desc>{escape(desc)}</desc>{_style(slots)}"
        f'<rect class="bg" width="100%" height="100%"/>{body}</svg>\n'
    )


# -- headline ----------------------------------------------------------------------------------


def headline_svg(report: StudyReport) -> str:
    tiers = sorted({c.tier for c in report.cells})
    slots = _slots(sorted({c.condition for c in report.cells}, key=lambda c: (_SLOT.get(c, 9), c)))
    sized = sorted({c.size_b for c in report.cells if c.size_b})
    lo, hi = (math.log(sized[0]), math.log(sized[-1])) if sized else (0.0, 1.0)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5

    pw, ph = _W - _M["l"] - _M["r"], _H - _M["t"] - _M["b"]
    zero_slot, scale_left = 22.0, 70.0  # the "no model" slot, then a gap, then the log axis

    def x_of(size: float | None, ox: float) -> float:
        if not size:
            return ox + zero_slot
        span = pw - scale_left - 8
        return ox + scale_left + span * (math.log(size) - lo) / (hi - lo)

    body = [
        '<text x="16" y="24" class="t1" font-size="15" font-weight="600">'
        "Honest pass rate by model size</text>",
        _legend(slots, 16, 48),
    ]
    for i, tier in enumerate(tiers):
        ox, oy = i * _W + _M["l"], _TITLE_H + _M["t"]
        body.append(f'<text x="{ox}" y="{oy - 12}" class="t1" font-size="13" font-weight="600">'
                    f"{escape(tier)}</text>")
        body.append(_y_axis(ox, oy, ph, pw))
        if any(c.size_b == 0 for c in report.cells if c.tier == tier):
            body.append(f'<text x="{ox + zero_slot}" y="{oy + ph + 16}" class="tm" font-size="10" '
                        'text-anchor="middle">no model</text>')
        for s in sized:
            body.append(f'<text x="{x_of(s, ox):.1f}" y="{oy + ph + 16}" class="tm" font-size="11" '
                        f'text-anchor="middle">{s:g}B</text>')
        body.append(f'<text x="{ox + scale_left + (pw - scale_left) / 2:.1f}" y="{oy + ph + 32}" '
                    'class="t2" font-size="11" text-anchor="middle">parameters (log scale)</text>')

        def y_of(v: float) -> float:
            return oy + ph - ph * v  # noqa: B023 - bound per panel, used within the iteration

        labels: list[tuple[float, str, str, object]] = []
        no_model: list[tuple[float, str, str, object]] = []
        for condition, slot in _draw_order(slots):
            cells = [c for c in report.cells if c.tier == tier and c.condition == condition]
            line = sorted((c for c in cells if c.size_b), key=lambda c: c.size_b or 0)
            if line:
                upper = " ".join(f"{x_of(c.size_b, ox):.1f},{y_of(c.ci_high):.1f}" for c in line)
                lower = " ".join(f"{x_of(c.size_b, ox):.1f},{y_of(c.ci_low):.1f}" for c in reversed(line))
                body.append(f'<polygon points="{upper} {lower}" fill="var(--s{slot})" fill-opacity="0.1"/>')
                path = " ".join(f"{x_of(c.size_b, ox):.1f},{y_of(c.rate):.1f}" for c in line)
                body.append(f'<polyline points="{path}" fill="none" stroke="var(--s{slot})" stroke-width="2" '
                            'stroke-linejoin="round" stroke-linecap="round"/>')
                end = line[-1]
                signature = tuple((c.size_b, c.rate, c.ci_low, c.ci_high) for c in line)
                labels.append((y_of(end.rate), condition, f"{end.rate:.0%}", signature))
            for c in cells:
                if c.size_b == 0:
                    no_model.append((0.0, condition, f"{c.rate:.0%}", (c.rate, c.ci_low, c.ci_high)))
                    x = x_of(0, ox)
                    body.append(f'<line x1="{x:.1f}" y1="{y_of(c.ci_low):.1f}" x2="{x:.1f}" '
                                f'y2="{y_of(c.ci_high):.1f}" stroke="var(--s{slot})" stroke-width="1"/>')
                    body.append(_marker(x, y_of(c.rate), slot))
            for c in line:
                body.append(_marker(x_of(c.size_b, ox), y_of(c.rate), slot))
            for ref in (c for c in cells if c.size_b is None):
                y = y_of(ref.rate)
                body.append(f'<line x1="{ox + scale_left}" y1="{y:.1f}" x2="{ox + pw}" y2="{y:.1f}" '
                            f'stroke="var(--s{slot})" stroke-width="1"/>')
                body.append(f'<text x="{ox + scale_left + 2}" y="{y + 12:.1f}" class="t2" font-size="10">'
                            f"{escape(ref.model)} ({condition}) {ref.rate:.0%}</text>")
        for y, text, _slot in _spread(_merge_identical(labels)):
            body.append(f'<text x="{ox + pw + 4}" y="{y + 4:.1f}" class="t2" font-size="11">{escape(text)}</text>')
        if no_model:
            # A second line under the "no model" tick, not beside the points: the slot is narrow,
            # the first model's line starts just to its right, and the header holds the title.
            summary = " · ".join(text for _, text, _ in sorted(_merge_identical(no_model), key=lambda m: m[1]))
            body.append(f'<text x="{ox + zero_slot}" y="{oy + ph + 29}" class="t2" font-size="10" '
                        f'text-anchor="middle">{escape(summary)}</text>')

    width = max(1, len(tiers)) * _W
    desc = ("Honest pass rate (answer properties, structural failures counted as failed) against model "
            "size, one panel per tier, one line per condition with its 95% case-bootstrap interval. "
            "Every value is in study.md.")
    return _svg(width, _TITLE_H + _H, "Honest pass rate by model size", desc, "".join(body), slots)


# -- sweep -------------------------------------------------------------------------------------


def sweep_svg(curves: list[SweepCurve]) -> str:
    models = sorted({(c.size_b is None, c.size_b or 0, c.model) for c in curves})
    slots = _slots(sorted({c.condition for c in curves}, key=lambda c: (_SLOT.get(c, 9), c)))
    t_max = max((lv.mean_realised_t for c in curves for lv in c.levels), default=1.0)
    t_max = max(1.0, math.ceil(t_max))
    pw, ph = _W - _M["l"] - _M["r"], _H - _M["t"] - _M["b"]

    body = [
        '<text x="16" y="24" class="t1" font-size="15" font-weight="600">'
        "When does it start claiming a trend?</text>",
        _legend(slots, 16, 48),
    ]
    for i, (_unknown, _size, model) in enumerate(models):
        ox, oy = i * _W + _M["l"], _TITLE_H + _M["t"]
        body.append(f'<text x="{ox}" y="{oy - 12}" class="t1" font-size="13" font-weight="600">'
                    f"{escape(model)}</text>")
        body.append(_y_axis(ox, oy, ph, pw))
        step = 2 if t_max > 6 else 1
        for t in range(0, int(t_max) + 1, step):
            x = ox + pw * t / t_max
            body.append(f'<text x="{x:.1f}" y="{oy + ph + 16}" class="tm" font-size="11" '
                        f'text-anchor="middle">{t}</text>')
        body.append(f'<text x="{ox + pw / 2:.1f}" y="{oy + ph + 32}" class="t2" font-size="11" '
                    'text-anchor="middle">realised slope t (signal strength)</text>')
        labels: list[tuple[float, str, str, object]] = []
        for condition, slot in _draw_order(slots):
            curve = next((c for c in curves if c.model == model and c.condition == condition), None)
            if curve is None or not curve.levels:
                continue
            pts = [(ox + pw * min(lv.mean_realised_t, t_max) / t_max, oy + ph - ph * lv.claim_rate)
                   for lv in curve.levels]
            body.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                        f'" fill="none" stroke="var(--s{slot})" stroke-width="2" '
                        'stroke-linejoin="round" stroke-linecap="round"/>')
            # Line only, with an end dot: twelve markers per line crowd the steep region.
            body.append(_marker(*pts[-1], slot))
            t50 = f"t50 {curve.t50:.1f}" if curve.t50 is not None else _SHORT_NOTES.get(curve.t50_note, curve.t50_note)
            signature = tuple((lv.mean_realised_t, lv.claim_rate) for lv in curve.levels)
            labels.append((pts[-1][1], condition, t50, signature))
        for y, text, _slot in _spread(_merge_identical(labels)):
            body.append(f'<text x="{ox + pw + 4}" y="{y + 4:.1f}" class="t2" font-size="11">{escape(text)}</text>')

    width = max(1, len(models)) * _W
    desc = ("Share of runs claiming a trend against the realised slope t-statistic, one panel per model, "
            "one line per condition. t50 is where claims reach 50%. Every value is in study.md.")
    return _svg(width, _TITLE_H + _H, "When does it start claiming a trend?", desc, "".join(body), slots)
