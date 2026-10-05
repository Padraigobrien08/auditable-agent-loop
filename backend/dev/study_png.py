"""
PNG copies of the study's SVG figures, for places that will not take SVG (Substack, slides).

The SVGs are the source of truth; this rasterises them with Chrome's *headless shell* at 2x
for a crisp image, once in light mode and once in dark (the SVG selects its palette with
``prefers-color-scheme``, so the browser's setting picks which one is drawn).

Only the headless shell is accepted, not full Chrome. Full Chrome's headless mode renders into a
viewport shorter than the window it is given, so the bottom of the figure is cut off while the
PNG still has the requested dimensions. Nothing in the file would show it. The headless shell
renders exactly the requested viewport, and every PNG's size is checked against the SVG's before
it is accepted, so a clipped image is an error rather than a figure.

Get one with ``npx playwright install chromium-headless-shell`` or
``npx @puppeteer/browsers install chrome-headless-shell@stable``, or point
``STUDY_HEADLESS_SHELL`` at the binary.
"""

from __future__ import annotations

import glob
import os
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SCALE = 2

_INSTALL_HINT = (
    "PNG export needs Chrome's headless shell (full Chrome clips the figure). Install one with\n"
    "  npx playwright install chromium-headless-shell\n"
    "or\n"
    "  npx @puppeteer/browsers install chrome-headless-shell@stable\n"
    "and, if it is not found automatically, set STUDY_HEADLESS_SHELL to the binary."
)


def find_headless_shell() -> str | None:
    """The headless shell binary: ``STUDY_HEADLESS_SHELL``, then ``PATH``, then Playwright's caches."""
    explicit = os.environ.get("STUDY_HEADLESS_SHELL")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    for name in ("chrome-headless-shell", "headless_shell"):
        found = shutil.which(name)
        if found:
            return found
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), str(Path.home() / ".cache" / "ms-playwright"),
             str(Path.home() / "Library" / "Caches" / "ms-playwright")]
    for root in filter(None, roots):
        matches = sorted(glob.glob(os.path.join(root, "chromium_headless_shell-*", "*", "headless_shell")))
        matches += sorted(glob.glob(os.path.join(root, "chromium_headless_shell-*", "*",
                                                 "chrome-headless-shell*", "chrome-headless-shell")))
        if matches:
            return matches[-1]  # newest revision
    return None


def svg_size(svg: Path) -> tuple[int, int]:
    root = ET.fromstring(svg.read_text(encoding="utf-8"))
    return int(float(root.attrib["width"])), int(float(root.attrib["height"]))


def png_size(png: Path) -> tuple[int, int]:
    header = png.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{png} is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def render(svg: Path, png: Path, *, dark: bool = False, browser: str | None = None, scale: int = SCALE) -> Path:
    """Rasterise one SVG. Raises rather than write an image of the wrong size."""
    binary = browser or find_headless_shell()
    if binary is None:
        raise SystemExit(_INSTALL_HINT)
    width, height = svg_size(svg)
    with tempfile.TemporaryDirectory() as tmp:
        # A page holding the figure at its own size, so the viewport is exactly the figure.
        shutil.copy(svg, Path(tmp) / "figure.svg")
        page = Path(tmp) / "page.html"
        page.write_text(
            '<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0}'
            f'img{{display:block}}</style></head><body><img src="figure.svg" width="{width}" '
            f'height="{height}"></body></html>',
            encoding="utf-8",
        )
        command = [
            binary,
            # The page is a local file this process just wrote; the sandbox also cannot start as
            # root in a container, which is where these figures are usually built.
            "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
            f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
            f"--blink-settings=preferredColorScheme={0 if dark else 1}",
            f"--screenshot={png.resolve()}", page.resolve().as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if not png.exists():
        raise RuntimeError(f"{binary} wrote no PNG for {svg.name}: {result.stderr[-500:]}")
    expected, actual = (width * scale, height * scale), png_size(png)
    if actual != expected:
        png.unlink()
        raise RuntimeError(f"{png.name} rendered at {actual}, expected {expected}; "
                           "refusing a possibly clipped figure")
    return png


def export_all(figures: Path, *, browser: str | None = None) -> list[Path]:
    """``name.png`` (light) and ``name-dark.png`` for every SVG in ``figures``."""
    written = []
    for svg in sorted(figures.glob("*.svg")):
        written.append(render(svg, svg.with_suffix(".png"), browser=browser))
        written.append(render(svg, svg.with_name(f"{svg.stem}-dark.png"), dark=True, browser=browser))
    return written
