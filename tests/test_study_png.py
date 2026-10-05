"""
PNG export of the study figures: the right browser, the right size, or an error.

Full Chrome's headless mode clips the figure while still writing a PNG of the requested size,
so the exporter only accepts the headless shell and checks every image's dimensions. These
tests pin both, and render for real when a headless shell is installed.
"""

from __future__ import annotations

import stat
import struct
import zlib
from pathlib import Path

import pytest

from agentic.evaluation.study import Cell, StudyReport
from backend.dev.study_figures import headline_svg
from backend.dev.study_png import SCALE, export_all, find_headless_shell, png_size, render, svg_size


def _figure(tmp_path: Path) -> Path:
    cells = [Cell(model="m", size_b=s, condition="A", tier="core", rate=0.5, ci_low=0.4, ci_high=0.6,
                  raw_rate=0.5, n_cases=10, n_runs=10) for s in (1.7, 8.0)]
    svg = tmp_path / "headline.svg"
    svg.write_text(headline_svg(StudyReport(cells=cells)), encoding="utf-8")
    return svg


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _fake_browser(tmp_path: Path, width: int, height: int) -> str:
    """A 'browser' that writes a PNG of a fixed size wherever --screenshot points."""
    image = tmp_path / "fixed.png"
    image.write_bytes(_png(width, height))
    script = tmp_path / "fake-shell"
    script.write_text(
        "#!/bin/sh\nfor a in \"$@\"; do case \"$a\" in --screenshot=*) "
        f"cp {image} \"${{a#--screenshot=}}\";; esac; done\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_png_size_reads_the_header(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    path.write_bytes(_png(7, 3))
    assert png_size(path) == (7, 3)


def test_a_correctly_sized_render_is_accepted(tmp_path: Path) -> None:
    svg = _figure(tmp_path)
    width, height = svg_size(svg)
    browser = _fake_browser(tmp_path, width * SCALE, height * SCALE)

    out = render(svg, tmp_path / "out.png", browser=browser)
    assert png_size(out) == (width * SCALE, height * SCALE)


def test_a_wrongly_sized_render_is_refused_and_removed(tmp_path: Path) -> None:
    """The failure full Chrome produces looks like this from outside: a PNG, just not the figure."""
    svg = _figure(tmp_path)
    browser = _fake_browser(tmp_path, 10, 10)

    with pytest.raises(RuntimeError, match="refusing a possibly clipped figure"):
        render(svg, tmp_path / "out.png", browser=browser)
    assert not (tmp_path / "out.png").exists()


def test_no_headless_shell_is_an_error_with_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDY_HEADLESS_SHELL", str(tmp_path / "missing"))

    assert find_headless_shell() is None
    with pytest.raises(SystemExit, match="headless shell"):
        render(_figure(tmp_path), tmp_path / "out.png")


@pytest.mark.skipif(find_headless_shell() is None, reason="no Chrome headless shell installed")
def test_real_render_writes_light_and_dark_at_twice_the_size(tmp_path: Path) -> None:
    svg = _figure(tmp_path)
    width, height = svg_size(svg)

    written = export_all(tmp_path)

    assert {p.name for p in written} == {"headline.png", "headline-dark.png"}
    for png in written:
        assert png_size(png) == (width * SCALE, height * SCALE)
    assert (tmp_path / "headline.png").read_bytes() != (tmp_path / "headline-dark.png").read_bytes()
