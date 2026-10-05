# Auditable agent loop

Auditable agentic analysis over tabular data, with SEC EDGAR as the flagship dataset.
Python/FastAPI/SQLAlchemy backend + worker, two MCP servers, Next.js frontend, Postgres.

**The governing invariant: the LLM plans and interprets; deterministic code computes.
No number in a trace may originate from a language model.** Every claim links to evidence;
every run is reproducible from persisted structured state.

`insufficient_evidence` and `rejected` are first-class outcomes. Never turn uncertainty into
an error, a silent fallback, or a fabricated number.

Active plan: `docs/decisions/2026-10-05-scaffold-vs-model.md` (hosting decisions still from `2026-08-11-showcase-direction.md`). `.planning/` is local-only and gitignored; never cite it from tracked files.

## Commands

```bash
python -m pytest tests/ -q              # full suite; pytest.ini already sets pythonpath
python -m ruff check .                  # BLOCKING in CI — run before every push
python -m mypy backend                  # report-only; ~50 known findings, not a gate
npm --prefix frontend run lint          # frontend gate
python -m alembic upgrade head          # needs the env vars below
./scripts/stack up | down | smoke       # docker compose wrapper
python3 scripts/check-lockfile-drift.py # requirements*.txt vs requirements*.lock
```

Anything that loads `backend.config.settings` in a fresh process needs these, or it exits on validation before doing any work:

```bash
EDGAR_BACKEND_JWT_SECRET=<32+ chars> EDGAR_BACKEND_ALLOW_OPEN_REGISTRATION=true \
EDGAR_BACKEND_OPS_API_TOKEN=<any> EDGAR_BACKEND_DATABASE_URL=<url>
```

Postgres-only tests skip silently unless `EDGAR_TEST_POSTGRES_URL` is set.

## Gotchas that have actually cost time

- **CI builds the PR _merge_ commit, not your branch tip.** A branch that is green locally fails once `main` moves. Merge `main` in before trusting a run.
- **Two migrations whose `down_revision` is the same revision = two alembic heads**, and `alembic upgrade head` exits 255 with a message CI may not surface. This is the most likely way a migration PR breaks after `main` advances.
- **Migrations must apply on SQLite as well as Postgres.** SQLite has no `ALTER TABLE ... ADD CONSTRAINT`: use `op.batch_alter_table` for any constraint or column-type change. Drop indexes _before_ a batch block that removes their columns.
- **Tests build schema with `Base.metadata.create_all`, not migrations**, so a broken or drifting migration passes the suite. `tests/test_migration_metadata_parity.py` is the guard — a new migration and its model must agree, checked with `compare_type` and `compare_server_default`. When you add a column, update both sides.
- **`ruff` is blocking and `mypy` is not.** A clean local `pytest` is not evidence CI will pass.
- **Your local package versions probably differ from CI**, which installs `requirements*.lock`. Verify version-sensitive work against those pins, not whatever is on your machine.
- **The suite cannot reach a live model provider**; `tests/conftest.py` blocks it. Opt in with `EDGAR_TESTS_ALLOW_LIVE_LLM=1` only for `-m integration`.
- Revision ids exceed alembic's default 32-char `version_num`; `alembic/env.py` pre-creates a wider table. Don't "simplify" that away.

## Boundaries that are rules, not layout

These are **enforced**, not just asserted: `tests/agentic/test_domain_boundary.py` and
`tests/test_backend_boundaries.py` parse the imports with `ast` and fail on a violation.
Add a rule here only with a test that checks it — the one rule that had no test is the one
that was broken.

- `src/` — deterministic EDGAR computation. Reached only through `edgar_project/mcp/`. Never import `src/*` from `backend/`; add a re-export to `edgar_project/mcp/adapters.py` instead.
- `agentic/` — the adaptive investigation loop. Standalone and offline-safe: **never** imports `backend/`. It may reach `edgar_project/`+`src/` from the two EDGAR bridge modules only (`agentic/adapters/edgar.py`, `agentic/experiments/tools/edgar_tools.py`), function-locally, so the generic path never pays for them — the allowlist in `test_domain_boundary.py` is the definition. `agentic/domain` has no SQLAlchemy. `agentic/` emits no logs, metrics or traces directly — instrumentation goes through the `AgentObserver` seam. Single wiring point: `backend/services/agentic_investigation_execution_service.py`.
- `backend/mcp/` — exposes the platform by calling `/v1` over HTTP. It is a client, not a second implementation; auth and owner scoping are inherited, never reimplemented. It may not import `backend/repositories|models|db|services`.
- `edgar_project/orchestration/` — planner is pure (no `src/*`, no network); only the executor touches tools.
- `backend/api/` holds HTTP, auth and ownership; it delegates to `backend/services/`, which use `backend/repositories/` and `backend/models/`.
- Frontend talks to FastAPI **server-side only** (`frontend/src/lib/api/`, `actions/`, `app/api/**/route.ts`). Browser components consume derived view models; JWTs never reach browser JavaScript.

## Conventions worth stating

- Services raise domain exceptions (`backend/services/exceptions.py`); routes translate them to `HTTPException` after `db.rollback()`. Don't raise HTTP errors from services.
- Ownership checks return **404, not 403**, for both missing and unauthorized — deliberate.
- MCP tools never let exceptions cross the boundary: they return `ToolResponseEnvelope` with `status`/`errors[]`. A 429 is no exception to this.
- Alembic migrations are additive and reversible. Write a real `downgrade()`.
- Keep suppressions narrow and justified (`# noqa: BLE001` at a defensive boundary, not blanket).
- Explain _why_ in comments, not what. Module docstrings state the boundary the file owns.

## Repo etiquette

- Never commit to `main`; branch, PR, and let CI run.
- Commits never end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and do not co author commits/prs
- `docs/api/openapi.json` is checked in and CI fails if stale: `python3 scripts/export-openapi.py`.
- Don't add TODO/FIXME to product code without a linked follow-up.
