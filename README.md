# Rally Health — modular backend foundation

The codebase is kept in this directory. It is an extensible development scaffold for the eight requested workstreams, not a finished or clinically validated health product.

## Run locally (PowerShell, from `D:\Fitness`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set a private random `RALLY_API_KEY` in `.env`. Keep `RALLY_ENVIRONMENT=development`. Import the source workbooks:

```powershell
python -I app\scripts\import_workbooks.py --source . --output data\normalized
```

Start the API:

```powershell
$env:PYTHONPATH = 'app'
uvicorn rally_health.main:app --app-dir app --host 127.0.0.1 --port 8000
```

Run tests:

```powershell
$env:PYTHONPATH = 'app'
.\.venv\Scripts\python.exe -m pytest -q
```

All `/v1/*` requests require `X-API-Key` and a development-only `X-User-ID`. That header-based identity is not production authentication. Production mode intentionally disables user-scoped routes until a trusted identity provider is integrated. Do not expose this API publicly or put real health data in the local SQLite database until consent, access control, encryption/key management, retention, backups and deletion are implemented and reviewed.

## Included modules

- `app/rally_health/core`: configuration, development auth boundary, request IDs/logging.
- `app/rally_health/domain`: typed observation, provenance, confidence/status, assistant and schedule contracts.
- `app/rally_health/repositories`: SQLite schemas and transaction helpers.
- `app/rally_health/importer.py`: workbook ingestion and quality report.
- `app/rally_health/services`: provenance-aware batch observations, validation/normalization, prototype metric aggregation/baseline, insight/trend, workouts, safety, coach, schedule, memory and prediction placeholders.
- `app/rally_health/integrations`: Apple Health export XML parser; GPT-4o vision extraction adapter for health reports (off by default); optional Claude adapter using the official Anthropic Python SDK, off by default.
- `app/rally_health/api/ingestion_routes.py`: batch/manual ingest, Apple Health XML export ingest, profile/questionnaire storage and report intake/review routes.
- `app/rally_health/api/routes.py`: other versioned API surfaces.

## API groups

- `GET /healthz`, `GET /readyz`
- `GET /v1/catalog/metrics`, `GET /v1/metrics/{code}/definition`
- `POST /v1/observations`, `POST /v1/ingest/batches`, `POST /v1/ingest/manual`, `POST /v1/ingest/apple-health-export`
- `POST /v1/reports/upload-intake` and `POST /v1/reports/extraction-jobs` store reports and, when OCR is enabled, return GPT-4o candidates awaiting explicit user confirmation; `GET /v1/reports/extraction-jobs/{job_id}` retrieves review status and `/file` retrieves that user's original report.
- `GET /v1/metrics/{code}/result`, `GET /v1/metrics/{code}/baseline`, `GET /v1/metrics/{code}/trend`, `GET /v1/metrics/SL-02/sleep-summary?night=YYYY-MM-DD&timezone=IANA`
- `GET/PUT /v1/profiles/me`, `POST/GET /v1/questionnaires/submissions`
- `POST /v1/schedule/events/proposals`, `POST /v1/schedule/events/proposals/{id}/confirm`, `GET /v1/schedule/events`
- `GET /v1/memories`, `POST /v1/memories/proposals`, `POST /v1/memories/proposals/{id}/confirm`, `DELETE /v1/memories/{id}`
- `GET /v1/predictive/anomalies`, `GET /v1/predictive/forecast/{code}`
- `POST /v1/reports/extraction-jobs`, `POST /v1/assistant/turns`, `POST /v1/video/assets`
- `POST /v1/admin/catalog/reimport` (development scaffold; must be protected with a real admin role before production)

## Important limitations

- Report uploads are limited by extension, content type, size, and PDF/PNG/JPEG file signature; local per-user storage folders use a non-reversible user ID digest. This is not encryption or a substitute for protected object storage/access controls.
- `RALLY_OCR_ENABLED=false` disables report OCR by default. When explicitly enabled with a server-side `RALLY_OCR_API_KEY`, report images and PDFs (rendered to page images, up to 10 pages) are sent to the OpenAI-compatible endpoint using the fixed `gpt-4o` model and `temperature=0`. The adapter requests verbatim transcription plus structured candidates; candidates must be supported by a quote contained in that transcription and are still unverified until user confirmation. These checks do not prove the transcription matches the image.
- `app/scripts/evaluate_clinocr.py --root data/research/ocr/clinocr-bench-v1.0 --output data/research/ocr/clinocr-bench-inventory.json` creates/refreshes a no-model benchmark inventory. Add `--live-gpt4o --allow-billable-tests --max-images N` only when a new `RALLY_OCR_API_KEY` is configured and billable synthetic-image requests are explicitly permitted; full benchmark scoring is not run by default.
- The OCR provider endpoint may receive sensitive health documents. Do not enable it with real patient data until the provider, endpoint operator, retention, region, contractual and regulatory requirements have been reviewed and approved.
- Questionnaire responses and profile are stored separately from sensor observations; questionnaire scoring and metric fallback rules are not implemented.
- Batch ingest stores per-user idempotency results and accepted/rejected observation events in one SQLite transaction. A crashed legacy `PROCESSING` batch is recovered and safely retried; replaying a completed batch returns its saved response.
- Sleep interval summary uses a local-noon-to-noon window, unions overlapping asleep intervals and is explicitly a non-reviewed prototype; time in bed, a reviewed Rally sleep score and calibrated confidence remain unavailable.
- Profile timezone is stored and used by metric results/baselines for local-day grouping; `tzdata` supplies IANA zones on Windows. Confidence remains heuristic for raw metrics and unavailable for baselines.
- Only a limited set of raw instantaneous metric medians is implemented. This is not the full 87-metric formula engine or calibrated confidence system.
- Rally formula prose is never executed as code. Unreviewed formulas return not-computable or insufficient-data states.
- Female workbook exercise IDs are generated and provisional. Workout tags are not individual medical clearance; automated coach drafts reject catalog items not marked approved.
- Current catalog has known missing workout tags/video metadata. Roadmap topics are Planned and excluded from evidence retrieval. DrKegel is isolated as competitor research.
- Schedule uses a local proposal/confirm flow and conflict/idempotency checks, but app-issued confirmation credentials, user authentication, update/delete and Auto grants still need production implementation.
- Apple Health export XML is imported, but native HealthKit/iOS and other direct wearable adapters, per-user production identity, consent scopes and authorization are not implemented.
- OCR candidate fields now require a known catalog analyte/alias plus a source quote containing matching value, unit and date. This check is not proof that the quote exists on the image; user review stays mandatory. A user-scoped report-file endpoint supports side-by-side review by a future UI.
- Video route registers metadata only; there is no generation pipeline. `pushup_remake.mp4` was not present in the inspected directory.

See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) and [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) before extending or deploying.
