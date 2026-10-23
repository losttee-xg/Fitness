# Task 5 expanded dataset: schema and enrichment rules

## Status

The source files for Zenodo record 8018238 remain under `D:\Fitness\dataset\fatigue_zenodo_8018238`; LifeSnaps source files remain under `D:\Fitness\dataset\rais_anonymized`; harmonized tables and the initial fatigue MVP remain where previously generated. The planned expanded master derivative tables have **not** been generated. The derivation job was blocked by the workspace's sensitive-source provenance control for the combination of the Zenodo fatigue source and `D:\Fitness\dataset\harmonized\master`.

## Planned outputs

- `observations_long.csv`: one row per observed participant/day/metric, preserving the original metric name and value. Rows retain cohort/source version, original unit or an explicit unverified-unit marker, source file and row, and a `synthetic` flag.
- `labels_long.csv`: separate row per survey response, keeping raw answer text, original question, survey date, target period, survey domain, source row, and label semantics. Fatigue self-report, tiredness EMA, sleep/pain/mood responses, STAI, PANAS, BREQ, TTM, SEMA, weekly questionnaires and baseline/screener fields must remain distinguishable.
- `fatigue_survey_feature_windows.csv`: fatigue-cohort responses joined only to prior wearable days under a documented lookback window. The original ordinal/text response remains the target; numerical encoding is a separate proxy column, not a newly observed score.
- `lifesnaps_survey_feature_windows.csv`: LifeSnaps survey features only where participant ID and temporal semantics can be verified. Cohort IDs remain namespaced and must never be joined across datasets.
- `master_manifest.json`: source record IDs, license, citations, hashes, counts, date spans, field mapping provenance, missingness and validation notes.
- Optional `synthetic_engineering_only/`: deterministic engineering fixtures for UI/rule smoke tests only. These must be marked synthetic, physically kept separate from observed tables, and never passed to training or evaluation.

## Source-specific limits

### Zenodo 8018238 fatigue cohort

The source daily question `tired_yesterday` is worded “Did you feel tired yesterday?” Its five source categories are preserved verbatim. An ordinal convenience mapping can be used for an exploratory model, but it is not a clinical scale or a validated interval measure. Submission timestamp is the next day; when constructing prior wearable features, define the label day as survey date minus one calendar day and make the feature window end strictly before the label day. The dataset's activity dictionary describes Fitbit/Evidation daily aggregates, but some units and metric semantics remain unspecified; do not invent them.

Weekly surveys contain fatigue/exhaustion, energy, usual activities and other symptom fields. Retain each raw answer and its full question text, preserve week number and survey dates, and use the stated prior-week target period. Do not collapse all answers into a single fatigue score unless the source's validated scoring instructions explicitly define it.

### LifeSnaps/RAIS

Keep RMSSD, NREM heart rate, RHR, consumer/vendor stress/sleep scores, sleep duration, SEMA momentary flags, and scored questionnaires separate. Dataset-specific participant IDs are not globally unique. `TIRED`/`RESTED` EMA flags are not interchangeable with STAI/PANAS or the separate fatigue cohort's question. Preserve original values and units. Where definitions are uncertain, the table should say unverified and downstream modeling should exclude the field.

### Other local sources

BIDSleep labels require parsing and epoch-level timestamp alignment before being usable as labels; partial motion files stay excluded. PAMAP2 relative elapsed time cannot be turned into calendar dates. PPG-DaLiA sensor payloads require a reviewed safe conversion path; do not deserialize untrusted pickle. Datasheet1 aggregate statistics are never participant-level training rows.

## Enrichment policy

1. **Observed**: source measurements or source survey responses with provenance.
2. **Derived**: deterministic features such as rolling mean, within-person deviation, measurement coverage and missingness. Each requires formula/version and a past-only temporal window.
3. **Proxy label**: an explicitly named source question such as self-reported tiredness. Never rename this as Recovery/readiness.
4. **Synthetic**: generated scenarios or composites for software tests only, stored separately and excluded from all training/evaluation.
5. **LLM proposals**: may suggest metadata mapping with source citation, but require human review; never generate labels or measurements.

## Completion/quality gates after explicit data-derivation authorization

- Verify source and destination; preserve the downloaded sources untouched.
- Check Zenodo checksums where supplied and calculate local SHA-256 hashes.
- Validate row counts, unique participant/date keys, duplicate survey submissions, missingness, timestamp parsing and cross-table key joins.
- Assert every joined feature date is earlier than the label's target period; assert survey and wearable participant IDs intersect only within the same dataset.
- Keep all source answers and categorical labels unchanged; keep numeric encodings in separate derived fields.
- Assert synthetic rows never appear in observed labels, training matrices or evaluation reports.
- Re-run participant-grouped CV and compare with a training-fold baseline; do not ship a target when the model fails the baseline gate.
- Record dataset citation and license. Zenodo 8018238 is CC BY 4.0; preserve attribution and review any additional source/content rights for intended use.

## Existing MVP artifacts

The initial fatigue cohort examples, grouped-CV report, and non-deployable model already exist under `D:\Fitness\dataset\harmonized\fatigue_mvp`. That prior training run used a 7-day prior window and did not beat the training-fold median baseline on MAE. It is research-only and not a Recovery/Fatigue product model.
