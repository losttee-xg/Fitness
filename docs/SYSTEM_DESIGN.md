# Rally Health System Blueprint

## Purpose and status

This repository is a modular starter codebase for the requested eight workstreams. It is not a complete or clinically validated health product. The architecture is deliberately designed so that health values, calculations, content review and action permissions are explicit and testable.

## Runtime architecture

```text
Workbook/device/manual/report inputs
  → source-specific adapters and validation
  → immutable observations + provenance
  → reviewed metric engine + baseline/quality status
  → insight/coach/predictive domain services
  → AI orchestration over approved facts and tools
  → safety, authorization and response validation
  → UI display or explicit action proposal
  → app confirmation / bounded grant
  → schedule or memory write with audit
```

### Packages

- `app/rally_health/core`: settings, development authentication boundary, request context/logging.
- `app/rally_health/domain`: typed observations, statuses, confidence, assistant response and schedule contracts.
- `app/rally_health/repositories`: SQLite starter schema and transaction helpers.
- `app/rally_health/importer.py`: workbook ingestion and provenance.
- `app/rally_health/services`: observation validation, metrics, insights, workouts, coach, safety, schedule, memory and predictive interfaces.
- `app/rally_health/integrations`: optional Anthropic provider adapter; disabled by default.
- `app/rally_health/api/routes.py`: versioned API surface.

## Safety/data rules encoded in the scaffold

1. Missing values remain `null` with an explicit status; they are not replaced with zero.
2. Rally calculation prose is documentation, never executable code. Only explicitly reviewed formulas should be registered in `MetricEngine`.
3. Observation units and value ranges are validated before storage; provenance is stored alongside each observation.
4. Workout catalog membership is not individual medical clearance. Unreviewed exercises are blocked from automated plan drafts.
5. Schedule writes use a server-side proposal followed by a separate confirm endpoint. A client-supplied boolean alone is not accepted as proof of confirmation.
6. Memory writes use proposal/confirm flow. Current deletion only covers SQLite; until other stores exist, deletion propagation must not be claimed as complete.
7. Planned roadmap content and DrKegel competitor research are excluded from factual retrieval/recommendation.
8. Forecast/anomaly routes are placeholders and do not issue user-facing alerts.

## Main API groups

- Health: `GET /healthz`, `GET /readyz`.
- Catalog: `GET /v1/catalog/metrics`, `GET /v1/metrics/{code}/definition`, `GET /v1/catalog/workouts`.
- User-scoped data: `POST /v1/observations`, metric result/baseline/trend endpoints.
- AI foundations: `POST /v1/assistant/turns`, daily insight endpoint.
- Coach: `POST /v1/coach/drafts`.
- Schedule: proposal, separate confirmation, read; direct write is disabled.
- Memory: list, proposal, confirmation, delete.
- Predictive: placeholder candidate/forecast endpoints only.
- Sleep summary is exposed as a non-reviewed interval-union prototype over a local-noon-to-noon window; it must not be treated as a validated Rally sleep formula.
- Video: metadata registration only, no generation.

## Workbook import results

The importer reports: 87 Rally metric rows (filename says 86), 31 supplemental rules, 212 female exercises, 63 male exercises, 32 editorial topics. Known gaps include two Rally metric calculation descriptions, 94 missing female session tags, two missing female cycle-phase values, 18 missing male video URLs and 26 missing timestamps. All roadmap content remains ineligible for retrieval; DrKegel remains competitor research only.

## Security boundary

`X-API-Key` plus `X-User-ID` is a local development scaffold, not production authentication. Production user-scoped routes are disabled until a trusted identity provider is configured. Do not expose this app publicly or send real health data until authorization, consent, encryption/key management, rate limiting, audit policy, retention, backups, deletion propagation and privacy review are implemented.

## Provider boundary

Claude integration uses the official Anthropic Python SDK and is disabled by default. It only receives explicitly supplied facts/citations and must not calculate, diagnose or execute write actions. LLM output validation, approval UX, prompt-injection evaluation, streaming, latency/cost measurements and a full tool registry remain required before product use.

## Release gates

- Approve metric formulas and per-metric data sufficiency/quality rules.
- Review exercise constraints with qualified domain reviewers.
- Add a representative, consented, independent evaluation dataset.
- Validate per-user auth/privacy and deletion before ingesting personal data.
- Validate action permissions and calendar conflicts before enabling writes.
- Define ground truth and error targets before enabling predictive functions.
- Human-review generated exercise media before release.
