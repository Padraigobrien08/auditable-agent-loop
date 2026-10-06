# Direction: hosted showcase demo

Date: 2026-08-11 · Status: **accepted**

Supersedes the sequencing in `.planning/ROADMAP.md` and `.planning/AGENT-PLATFORM-ROADMAP.md`
(local-only since 2026-10-05; they remain in git history). Its own sequencing is in turn
superseded by [`2026-10-05-scaffold-vs-model.md`](./2026-10-05-scaffold-vs-model.md); the
decisions below still hold.

---

## 1. Decisions

### D1 — The audience is a technical reviewer, not an end user

This is a **portfolio / capability showcase**. Success is a stranger forming an accurate,
favourable judgement of the engineering in under five minutes. It is not signups, retention,
or analytical throughput.

Consequence: work that a reviewer cannot see is not worth doing. Work that is invisible but
*enables* the visible demo (spend guards, deploy) is worth exactly as much as the demo it
protects, and no more.

### D2 — The demo is hosted and clickable

A public URL, not a README the reviewer has to believe, and not a `docker compose up` they
have to attempt. This is the decision that makes deployment real work rather than theater.

### D3 — Three access tiers

| Tier | Who | Engine | Model spend |
|---|---|---|---|
| **Replay** | everyone, no account | pre-recorded investigations served from persisted state | zero |
| **Guest** | one-click guest session | deterministic EDGAR chain + narrative phases | ~$0.01/run (mini model) |
| **Adaptive** | accounts holding the invite code | full agentic investigation loop | ~$0.10–1.00/run |

The replay tier carries most of the demo weight. It is real persisted state — real hypotheses,
real evidence, real critic falsification, real trace — rendered by the same UI as a live run, so
it is not a mockup. The machinery already exists ([`agentic/agent/replay.py`](../../agentic/agent/replay.py),
[`backend/services/investigation_replay_service.py`](../../backend/services/investigation_replay_service.py)).

### D4 — The agentic loop becomes the default engine for entitled accounts

`EDGAR_BACKEND_AGENTIC_ENGINE_ENABLED` flips on, and the chat surface routes to the
investigation loop for the Adaptive tier. Guests keep the deterministic chain.

This resolves the diagnosis in the agent-platform roadmap — "the flagship loop and the
flagship dataset never meet." It is scoped to entitled accounts by cost, not by preference.

### D5 — Generality is demonstrated, not rebuilt

The product claim is auditable agentic analysis over **any tabular dataset**, with EDGAR as the
flagship. In this push, generality is proved by a recorded non-EDGAR CSV investigation through
the existing [`/investigations/new`](../../frontend/src/app/projects/[projectId]/investigations/new/page.tsx)
surface — not by the unified-entry rewrite.

The unified entry (EDGAR as one adapter alongside upload) remains the correct target
architecture and is deferred, not abandoned.

### D6 — Frontend on Vercel, backend on one small box, CORS stays closed

Next.js on Vercel; `db` / `migrate` / `api` / `worker` in Compose on a single VPS behind Caddy
for TLS. Postgres runs in Compose on that box — managed Postgres alone would consume the budget.

**Next.js keeps proxying server-side.** The browser never calls FastAPI directly, the JWT stays
in the HttpOnly cookie on the Vercel origin, and `cors_allow_origins` stays empty. The cost is
one cross-internet hop per render; mitigate by co-locating the VPS with the Vercel function
region. This preserves the auth posture the codebase already documents and defends.

### D7 — Budget ceiling is $25/month all-in

Hosting (~$6–12 VPS, Vercel free tier) plus model spend. Every cap in the plan derives from
this number.

### D8 — O2 (backup / restore / DR) is closed as won't-do

There is no user data worth recovering, and a reviewer cannot see a restore procedure. Record
it as a deliberate scope decision so it is not mistaken for an oversight. If the product ever
takes real users, it reopens first.

---

## 2. Conflicts surfaced, and how they resolve

### Conflict 1 — open registration vs. near-zero model spend

**Open self-serve registration into the Adaptive tier is not affordable at $25/month.** One
motivated visitor could exhaust a month's budget in an afternoon; the per-investigation ceiling
(`agent_max_cost_usd`, default $1.00) bounds a single run, not a user.

**Resolution (approved 2026-08-11):** registration stays open, but a new account lands in the
**Guest** tier economics — the deterministic chain with narrative phases, roughly a cent a run.
The **Adaptive** tier is unlocked by an **invite code**, published in the README and on the CV.
A reviewer who wants the live loop self-serves with the code; a drive-by cannot.

The code is a shared secret, not a per-person token — cheap to build, and the per-account spend
cap in [S0](#s0--spend-guard--blocks-everything-public) is what actually bounds the bill. If
the code leaks, rotating it is a config change and the caps hold in the meantime. That ordering
matters: **the cap is the control, the code is the filter.** Do not let the code's existence
justify skipping the cap.

### Conflict 2 — four tracks vs. a weeks-long timeline

All four open tracks is roughly a quarter of work. Against a job-search deadline the plan below
carries **only the demo-critical path**. Everything else is listed as deferred with a reason, so
the cut is legible rather than silent.

### Conflict 3 — `agentic_engine_enabled` gates the whole engine, not a tier

The flag is currently global. D4 needs per-user routing, so the flag becomes "the engine is
available in this deployment" and entitlement becomes a user attribute. That is an additive
change, but it is a change to the flag's meaning and should be documented where it is defined.

---

## 3. Sequence

Ordered by dependency, then by demo value per hour. Each ships independently.

### S0 — Spend guard · **landed 2026-08-11**

Nothing goes on the public internet with a live API key and no per-account ceiling.

Implemented in [`backend/services/spend_guard.py`](../../backend/services/spend_guard.py), with
`users.access_tier` (migration `017`), the invite code on `POST /v1/auth/register`, and
admission checks on every path that starts execution. 17 tests in
[`tests/test_spend_guard.py`](../../tests/test_spend_guard.py); full suite 1135 passed.

**Three findings worth keeping:**

1. **Two pricing configs existed, with different key names.** `llm_model_prices`
   (`input_per_1m`/`output_per_1m`) drives the loop's own budget; `agent_llm_pricing_json`
   (`prompt_usd_per_1m`/`completion_usd_per_1m`) drives the usage API. A cap built on the wrong
   one — or on an unconfigured one — reads $0.00 and never fires. The guard uses the former, so
   the per-account cap and the per-investigation budget agree, and it exposes
   `cost_priced: bool` so "unpriced" can never be misread as "spent nothing". **The run-count
   ceiling is the real backstop**; USD ceilings are secondary because they depend on operator
   config. `log_spend_guard_posture` warns at startup when the USD half is inert.
2. **The engine gate is enforced inside `select_run_engine`**, not at the call sites, so the
   API route and the worker are covered by one check and a future caller cannot forget it. It
   fails closed: an unresolvable user yields the deterministic engine.
3. **A pre-existing bug: guest mode was broken end to end.** Guest emails used `@demo.local`,
   which `EmailStr` rejects as a special-use domain when `UserRead` validates the *response* —
   so `GET /v1/auth/me` returned 500 for every guest, and the frontend layout calls it on each
   render. Now `@guest.example.com` (RFC 2606), with a regression test. The guest tier in D3
   would not have worked without this.

- Per-account spend: aggregate `ModelCall.{prompt,completion}_tokens` over a user's runs and
  price via [`backend/llm/pricing.py`](../../backend/llm/pricing.py). The aggregation helper
  already exists (`aggregate_llm_usage_for_calls`, used by `GET /v1/runs/{id}/llm-usage`) —
  this is a query and a pre-flight check, **not a new subsystem or table**.
- Entitlement: additive nullable column on `User` (e.g. `agentic_tier`), reversible migration.
  Absent/false → deterministic chain. Refuse the loop with a typed service error, not a 500.
- Invite code: an optional field at registration, compared against a settings secret
  (`EDGAR_BACKEND_ADAPTIVE_INVITE_CODE`); a match sets the entitlement. Unset in settings → no
  self-serve path to the adaptive tier, which is the correct default for any deployment that is
  not this demo. Compare in constant time and rate-limit it on the existing auth limiter, or it
  becomes a guessable oracle on an endpoint that is already public.
- Global kill switch: a monthly ceiling in settings. Once crossed, live runs stop and the
  surface falls back to the replay tier with an honest message. A demo that degrades visibly
  beats a demo that silently drains a card.
- Guest sessions pinned to the deterministic chain regardless of the engine flag.

Risk: touches the shared execution path. Mitigate the way the codebase already does — service
layer raises, route translates, tests assert the refusal.

### S1 — The recorded runs · **machinery landed, recordings blocked**

**Landed 2026-08-11:** the replay tier itself — `investigations.demo_slug` (migration `018`),
publication-as-authorization in
[`backend/services/demo_publication_service.py`](../../backend/services/demo_publication_service.py),
the unauthenticated `/v1/demos` routes, the
[`publish_demo`](../../backend/maintenance/publish_demo.py) CLI, a deterministic non-financial
dataset ([`scripts/build_demo_dataset.py`](../../scripts/build_demo_dataset.py)), and the
[recording harness](../../scripts/record_demo.py). 25 tests.

**Found and fixed en route:** the agentic loop made real model calls but persisted **no**
`ModelCall` rows — it tracked cost only in memory. That silently broke two things already
claimed to work: `GET /v1/runs/{id}/llm-usage` was empty for the flagship engine, and S0's
per-account USD ceiling summed zero rows, so it read $0.00 forever while *appearing* enforced.
`CostTrackingResponder` now routes through `RecordedChatCompletionService`.

**Blocked: the recordings are not good enough to publish.** Two live runs on the tabular
dataset produced 1 hypothesis, 1–3 experiments out of 8–12 allowed, **zero critiques**, and
`insufficient_evidence` at iteration 1. The persisted decisions show why, and it is not
plumbing:

1. **Goal interpretation misframes the question.** "Is service quality degrading, or is rising
   volume the explanation?" was read as a *comparison* goal, yielding the hypothesis
   "avg_delivery_days differs meaningfully from the other available service metrics" — a
   category error. The planner then had one candidate experiment (`rank_entities`), and the
   selector's own rationale says so: *"This is the only available comparison experiment."*
2. **The termination policy has no patience.** `provenance.source: agent_llm`,
   `at_iteration: 1` — the model-backed policy stops after a single inconclusive experiment
   rather than trying an alternative. The critic never runs, so the falsification beat the
   demo is built around never happens.

Publishing a one-experiment trace would actively undermine the pitch, and cherry-picking a
flattering run is ruled out by [`demo-script.md`](../demo-script.md).

**Update 2026-08-11 — goal interpretation fixed; termination was not the defect.**

Prompt `agentic_goal_interpreter` **1.0.2** teaches three distinctions the 1.0.1 prompt left
open: `comparison` is between *groups*, never metric-vs-metric; a question about change over
time stays `trend` even when it offers a competing explanation; "does X explain Y?" is
`correlation` on the outcome metric. One constant versions all four agentic prompts, so the
other three files are copies of 1.0.1.

Measured on the same goal and dataset:

| | before (1.0.1) | after (1.0.2) |
|---|---|---|
| intent | `comparison` | `trend` |
| hypothesis | *"avg_delivery_days differs from the other available metrics"* | *"on_time_rate decreases over time"* |
| experiments | 1 | 3 |
| evidence | 1 | 6 (support 3 / refute 2) |

**The termination policy turned out not to be broken.** The premature stop was entirely
downstream of the misclassification: `comparison` offers two tools, one failed validation, so
the run had a single candidate and had to conclude. With the right intent the loop works
through all three trend tools and stops when they are genuinely exhausted — which is correct.

A planner "broadening" fallback was written and then **reverted**. It let a starved run reach
for off-intent tools, and the agency suite failed it against
`comparison_goal_uses_comparison_tools` — a case asserting that a between-group question must
not be answered by trend analysis. The suite's principle is the better one, and it is this
product's thesis: a run that stays on-question and honestly reports `insufficient_evidence`
beats one that pads with experiments that answer a different question.

**Update 2026-08-11 (later) — competing hypotheses and the critic gate. First recording published.**

Two further fixes, and the first one was structural rather than a prompt problem:

1. **The goal text never reached the hypothesis generator.** `generate_hypotheses` received
   only the `GoalInterpretation` — an intent, one metric hint, a direction. An alternative
   explanation offered by the goal ("…or is rising volume the cause?") cannot survive that
   classification, so the generator could not have proposed the competing claim no matter how
   the prompt was worded. `goal_text` is now threaded through
   ([`policy.py`](../../agentic/agent/policy.py), defaulted so other implementations keep
   working), and the prompt tells the generator to treat a named alternative as a second claim
   rather than padding.
2. **The critic was gated on `supported`.** It now also challenges a claim carrying **both**
   supporting and refuting evidence. Mixed evidence is exactly where a competing explanation
   is informative; the old gate meant no run ending inconclusive was ever challenged. Claims
   with no evidence, one-sided evidence, or a settled status are still left alone, and
   `supported` keeps priority — so the agency suite's `require_challenge` routes are unchanged
   and the whole suite stays green. Covered by
   [`test_critic_mixed_evidence.py`](../../tests/agentic/test_critic_mixed_evidence.py).

Cumulative effect on the same goal and dataset:

| | 1.0.1 | + intent fix | + both |
|---|---|---|---|
| hypotheses | 1 (category error) | 1 | **2, competing** |
| experiments | 1 | 3 | **6** |
| evidence | 1 | 6 | **12** |
| critiques | 0 | 0 | **1, acted on** |
| spend | $0.004 | $0.004 | $0.012 |

The critique proposed `detect_change_points` as its falsification tool and the loop **ran** it —
an acted-on challenge by the agency suite's own standard, not merely a note. Both hypotheses end
`weakened`: neither explanation is cleanly supported, which is the honest reading of a confound
built to be genuinely ambiguous.

**Update 2026-08-11 (later still) — both recordings published. S1 complete.**

The EDGAR recording ran against live SEC data (AAPL, MSFT, NVDA) and is the stronger of the two:

| | EDGAR | tabular CSV |
|---|---|---|
| slug | `edgar-margin-vs-growth` | `csv-delivery-delays` |
| status | **converged** | exhausted |
| termination | **sufficient_evidence** | insufficient_evidence |
| hypotheses | 2 — one **rejected** (0.05), one **supported** (0.95) | 2, both weakened |
| experiments | 7 | 6 |
| evidence / artifacts | 7 / 10 | 12 / 5 |
| critiques | 1, acted on | 1, acted on |
| model calls | 11 | 14 |
| spend | $0.0101 | $0.0121 |

`edgar_trend_break_analysis` **led the run** — the EDGAR domain tools are reachable and chosen
first, which is what `B1` on the agent-platform roadmap set out to achieve. The loop then
rejected the premise the question assumed ("margin has deteriorated") and supported the
alternative it offered ("revenue growth is the explanation"). The critic challenged the
*supported* claim with `summarize_distribution` — checking whether the support was an artefact
of skew or outliers — and the loop ran it.

Together the two cover the demo script's range: one run that converges and discriminates
between competing claims, one that honestly refuses to.

**Fixed while verifying:** `InvestigationState.confidence` was never written — it sat at its
`0.0` default even on a converged run whose conclusion said `0.5`. That field is what
`InvestigationSummary` surfaces, so the public listing advertised every demo at zero
confidence. `set_conclusion` now carries the conclusion's confidence onto the state.

S1 is complete. Next is **S2 — hosted deploy**.

Original scope, still the target:

1. **EDGAR, live SEC data.** Goal → hypotheses → experiments → *contradicting* evidence →
   critic falsification → typed termination reason → evidence-linked conclusion. This is
   `B2` from the agent-platform roadmap and the single highest-value artifact in the plan.
2. **Non-EDGAR CSV.** The same loop over an unrelated tabular dataset, via the existing
   investigations form. Evidence for D5.

A run that terminates `insufficient_evidence` is a *good* recorded run if the evidence genuinely
was insufficient — it demonstrates the honesty property the whole design argues for. Do not
re-roll until the answer is flattering.

One-time cost: a few dollars. This is the best money in the plan.

**Update 2026-08-14 — ranking goals were answered from the wrong end.**

Found on the same delivery dataset: *"Which region has the weakest on-time delivery rate
overall?"* was interpreted `ranking`, ran `rank_entities`, and recorded *"Top entity by
mean(on_time_rate) is 'east'"* — the **best** region. `InvestigationPlanner._params_for` never
passed `ascending`, and the tool defaults to descending, so every weakest/worst/lowest goal was
answered with the strongest entity.

`GoalInterpretation.direction` already existed to carry this; prompt **1.0.3** extends it from
`trend` to `ranking` (weakest → `down`), the fixture policy reads superlatives via
`parse_extreme`, and the planner turns `down` into `ascending=True`. `rank_entities` now says
"Lowest entity" when it ranked ascending — a claim that named the top of a bottom-up ranking
would be false about a correctly computed number, which is the one thing the deterministic
layer must never produce.

**What the loop got right, and what had to survive the fix:** it did *not* assert 'east' as the
weakest region. The evidence was recorded `neutral`, the hypothesis left `unresolved`, and the
run terminated `insufficient_evidence` at 0.2 — the honesty property this plan argues for,
holding on a run whose parameters were wrong. That behaviour is unchanged: the equivalent run
still declines at 0.2, now over the right entity. The new core case
`weakest_entity_goal_ranks_from_the_bottom` pins the entity the ranking leads with; it
deliberately does *not* cap confidence, because entities in that fixture genuinely do differ
and a future policy that concluded so with evidence would be right to.

### S2 — Hosted deploy · **assets landed; provisioning is yours**

Everything that can be built without your accounts is in place. Runbook:
[`docs/deploy.md`](../deploy.md).

- [`docker-compose.prod.yml`](../../docker-compose.prod.yml) — standalone (not an overlay), so
  the production topology is readable in one file and nothing is inherited invisibly. No `web`
  service; nothing publishes a host port but Caddy; every secret is `:?` so a half-filled
  `.env` fails at `config` time instead of booting a dev posture on the public internet
  (verified).
- [`ops/caddy/Caddyfile`](../../ops/caddy/Caddyfile) — TLS, and 404s `/metrics` and
  `/v1/worker/health` so ops surfaces are unreachable from the internet rather than merely
  token-gated. Validated against real Caddy.
- [`.env.production.example`](../../.env.production.example) — full posture including both
  spend-guard control families. Needed a `.gitignore` negation: `.env.*` is ignored by design,
  so the template was invisible to git.
- [`.github/workflows/deploy.yml`](../../.github/workflows/deploy.yml) — manual
  `workflow_dispatch` (a portfolio demo has no reason to redeploy while someone is looking at
  it). Runs suite + ruff + API-contract check, then SSH `git reset --hard` and
  `up -d --build`, then asserts `/v1/health` is 200 **and** `/v1/demos` is non-empty — a deploy
  that leaves the replay tier empty is a broken demo even when health is green. Builds on the
  server: the heavy build is on Vercel, so skipping a registry removes credential handling from
  both CI and the host.

**No exporter was built.** Moving a demo means moving an investigation, its whole child graph,
its analysis run, its artifacts *and* their blobs, across SQLite→Postgres. Recording directly
against the deployed stack costs ~$0.02, takes under a minute, and doubles as an end-to-end
smoke test. It must run **inside** the container (`compose exec api`) so blobs land on the
server's artifact volume — a row pointing at a `local:` key that exists only on a laptop is a
broken demo.

**Found while writing the runbook:** the image did not copy `scripts/`, so the documented
`compose exec` seeding commands would have failed on a fresh host. `Dockerfile` now copies it;
verified by building the image and running `build_demo_dataset.py` inside it.

Remaining, and it needs your accounts: provision the VPS, point DNS, fill `.env`, create the
Vercel project with `API_URL`, then run the two recordings from step 4 of the runbook.

### S2 — Hosted deploy (original scope)

- Vercel project for `frontend/`, `API_URL` pointing at the backend origin.
- VPS: Compose `db` / `migrate` / `api` / `worker`, Caddy TLS, firewall closed except 80/443.
- Production posture: `ALLOW_SQLITE=false`, real `JWT_SECRET`, `OPS_API_TOKEN` set,
  `ALLOW_OPEN_REGISTRATION=true`, `ALLOW_GUEST_DEMO=true`, `AGENTIC_ENGINE_ENABLED=true`.
- E5-lite: GitHub Actions builds and pushes the image, deploys over SSH. Enough for a visible
  green pipeline; not a full CD story.
- The observability stack (`docker-compose.observability.yml`) stays **local-only** — Grafana
  and Prometheus on the demo box cost RAM the box does not have. Dashboards ship as screenshots.

### S3 — README and demo narrative · **landed 2026-08-11**

Rewritten around the five-minute path in [`demo-script.md`](../demo-script.md): 367 lines → 228,
and the recorded EDGAR investigation opens the file instead of a feature list.

Every figure is from the published run and was checked against the database rather than
recalled — 7 experiments, 7 evidence, 10 artifacts, 11 model calls, $0.0101, the two hypothesis
statements and their confidences verbatim. All 36 local links resolve.

The lead is that **the loop rejected the premise the question assumed**, and the second
recording is presented as ending `insufficient_evidence` on purpose — a run that stops at "I
cannot separate these" is a correct outcome, and saying so is more persuasive to this audience
than a clean answer would be. Known limits stay in, sharpened: the model is named as the weak
link, not the loop.

Two things it deliberately does **not** claim: the hosted demo is marked *not yet live*
(provisioning pending), and the Grafana stack is labelled local-only. Overstating either is
exactly what the reviewer is checking for.

Remaining: swap the placeholder for the real URL once the host is up.

---

## 4. Deferred, with reasons

| Item | Why not now |
|---|---|
| ~~Unified BYO-data entry (EDGAR as one adapter)~~ | **Done 2026-08-12.** Not only a frontend change: `/v1/investigations` hardcoded `adapter: in_memory`, so EDGAR was unreachable through it even though the execution layer already supported the adapter. `dataset.source` (`tabular` \| `edgar`) now selects it, defaulted so every pre-existing caller is unaffected. The form gains a source picker, a ticker field prefilled from project scope, and CSV file upload. EDGAR always enqueues — its panel is built from live SEC fetches before the loop starts, which is not a request to hold open. |
| ~~v1.5 Durable Chat History (phases 27–30)~~ | **Done 2026-08-12** (PR #68). The bug was real but misplaced: persistence and listing were already correct, and the fault was `ChatShell` reconciling instead of remounting across a thread change, so `useState(props)` kept the previous thread's messages, draft, and run-progress poll. Fixed with `key={conversationId}`. Reproducing before diagnosing is what moved the search from the data layer to the shell. |
| ~~O5 load / performance testing~~ | **Done 2026-08-12** — [`docs/performance.md`](../performance.md), measured with [`scripts/loadtest.py`](../../scripts/loadtest.py). Headline: a full investigation with the fixture policy runs in **42 ms**, so >99% of a live run's wall clock is model latency and optimising the loop would be pointless. Read paths are 3–8 ms unloaded and saturate at 150–460 rps with **zero errors** across ~40k requests. The engines are deliberately excluded from load — hammering them would measure an SEC rate limit and a bill. |
| ~~Replay HTTP route~~ | **Done 2026-08-12.** `POST /v1/investigations/{id}/replay`, returning the conclusion-first verdict (`identical` / `same_conclusion` / `diverged`). **Admin-only**, because replay deliberately never persists its candidate — so it produces no `AnalysisRun` and no `ModelCall` rows and the spend guard cannot see or count what it spends. An endpoint that spends money invisibly to the ceilings cannot be open to ordinary accounts. |
| ~~MCP tool-call rate limiting~~; unauthenticated handshake | **Rate limiting done 2026-08-12** — per-caller sliding window in [`backend/mcp/rate_limit.py`](../../backend/mcp/rate_limit.py), enforced in the server's `_guarded` wrapper so every tool is bounded by construction. Keyed by a hash of the bearer token, since stdio has no IP and hosted callers share a proxy. The **unauthenticated handshake remains open** — it exposes schema only, which is normal for MCP; bind loopback behind a reverse proxy to close it. |
| Log aggregation, SLOs | The remaining half of `O4`. No reviewer-visible payoff at this scale. |
| **O2 backup / restore / DR** | **Won't-do.** See D8. |

---

## 5. What this plan is betting on

That a reviewer trusts *demonstrated* engineering over *claimed* engineering. The repository's
existing strength is real — deterministic core with frozen regression tests, lease-based worker
verified against Postgres, ownership-404 semantics, additive reversible migrations, bounded-
cardinality metrics. None of it is visible until someone can click something.

The plan therefore spends almost nothing on new capability and almost everything on making
existing capability legible. The one exception is S0, which exists solely so S2 cannot become
an unbounded bill.
