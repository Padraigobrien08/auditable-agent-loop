"""
The manual agency-bench workflow's environment is enough to start the bench.

``tests/conftest.py`` supplies a JWT secret, an ops token and a registration mode to every
test, which is exactly why the suite never noticed the workflow supplied none of them: the
workflow failed settings validation before running a single case. So this test runs the
bench in a fresh interpreter whose environment is *only* what the workflow step sets,
with secrets replaced by placeholders.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "agency-bench.yml"

#: Placeholders for the secrets the step reads. The fixture-only run below never calls a
#: provider, so a fake key is never sent anywhere.
_SECRETS = {
    "EDGAR_BACKEND_OPENAI_API_KEY": "sk-placeholder",
    "EDGAR_BACKEND_LLM_MODEL_PRICES": '{"gpt-5.4-mini": {"input_per_1m": 0.15, "output_per_1m": 0.6}}',
}


def _bench_step_env() -> dict[str, str]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["bench"]["steps"]
    step = next(s for s in steps if "agency_bench" in str(s.get("run", "")))

    def resolve(value: object) -> str:
        text = str(value)
        text = text.replace("${{ github.workspace }}", str(ROOT))
        return re.sub(r"\$\{\{\s*secrets\.(\w+)\s*\}\}", lambda m: _SECRETS[m.group(1)], text)

    return {key: resolve(value) for key, value in step["env"].items()}


def _run(code: str) -> subprocess.CompletedProcess[str]:
    # Only PATH survives from this process; everything else the bench sees is the workflow's.
    env = {"PATH": os.environ.get("PATH", ""), **_bench_step_env()}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300
    )


def test_workflow_env_passes_settings_validation_and_enables_the_provider() -> None:
    result = _run(
        "from backend.config.settings import Settings\n"
        "s = Settings()\n"
        "assert s.llm_provider == 'openai', s.llm_provider\n"
    )

    assert result.returncode == 0, result.stderr[-2000:]


def test_workflow_env_runs_the_bench() -> None:
    result = _run(
        "from backend.dev.agency_bench import main\n"
        "raise SystemExit(main(['--policy', 'fixture', '--trials', '1', '--tier', 'hard']))\n"
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "| fixture | hard |" in result.stdout
