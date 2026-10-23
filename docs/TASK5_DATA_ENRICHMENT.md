# Task 5 data enrichment

## Goal and scope

Create auditable analysis features from observed LifeSnaps/RAIS data and align them with independently recorded STAI/PANAS questionnaire scores. This is exploratory research, not a clinical model. The pipeline does not use an LLM to fabricate physiological values, infer unobserved fatigue/recovery labels, or fill missing days.

## Planned sequence

1. Keep the harmonized source tables immutable and record their versions/hashes.
2. Produce a metric catalog from the harmonizer's source mapping, with units, definitions, and unresolved semantics. Descriptions are reviewable metadata, not labels.
3. For each LifeSnaps participant and STAI/PANAS questionnaire date, compute features only from the preceding seven calendar days. The target day and future days are excluded.
4. Add transparent features per metric: observed-day count and coverage, mean, population standard deviation, last observed value, and first-to-last delta. Preserve missingness; do not impute.
5. Store the questionnaire target and metadata separately from feature columns, retaining target source, date, and row provenance. Exclude SEMA binary flags from questionnaire targets in this first version.
6. Mark rows with fewer than three total observed metric-days in the lookback window ineligible for modeling; retain them for coverage analysis.
7. Validate the temporal cutoff and synthetic exclusion with tests. Run feature generation and inspect counts before any training.
8. Only after source/codebook review, evaluate one target at a time with participant-held-out splits. Keep an untouched test group; compare against simple baselines and report coverage, MAE/RMSE or classification metrics as appropriate. No notification or health recommendation is produced.
9. If LLM assistance is later used, restrict it to proposing metadata mappings with citations and human approval. Store proposal, source citation, reviewer decision, model/version, and prompt hash; never accept a generated measurement or target label as data.

## Generated files

- `lifesnaps_survey_features.csv`: one row per supported questionnaire score, with a strictly prior feature window.
- `feature_catalog.json`: definitions, formula descriptions, and cautions.
- `build_report.json`: row/participant/target counts and eligibility summary.

## Run

From `D:\Fitness`:

```powershell
python -I app/scripts/enrich_task5_features.py `
  --observations "D:\Fitness\dataset\harmonized\observations.csv" `
  --labels "D:\Fitness\dataset\harmonized\labels.csv" `
  --output-dir "D:\Fitness\dataset\harmonized\enriched" `
  --lookback-days 7 `
  --min-observed-days 3
```

The default outputs stay in a subfolder and do not overwrite the harmonized source tables.

## MVP benchmark outputs (2026-10-23)

- Survey-score prediction: grouped five-fold evaluation by participant found that the model did not beat the training-fold median baseline for STAI stress, PANAS positive affect or PANAS negative affect. Full metrics are in `survey_model_evaluation.json`.
- One-day metric forecast: rolling-median baseline outperformed last-value baseline in scored contiguous windows for RHR (893 target-days, 65 participants; MAE 1.689 vs 2.521 bpm), sleep minutes (574 days, 48 participants; 63.936 vs 87.429 min), and source RMSSD (381 days, 32 participants; 8.192 vs 11.255 ms mapping). This does not validate an ML model or clinical recovery prediction.
- Habit association: exact-date exploratory Pearson comparisons between steps and health metrics are in `habit_associations.json`; participant-demeaned correlations are also reported to reduce simple between-person confounding. They are not causal effects.
- Anomaly candidate counts and rate-limit policy simulation are in the corresponding JSON files. They are shadow software tests only; no notifications were sent.
- Only metrics with non-unverified catalog units are included in the regenerated questionnaire feature table. The source codebook still does not authoritatively define every Fitbit field. Value ranges are saved in `lifesnaps_metric_audit.json`, explicitly as range checks, not unit validation.

## Interpretation and limits

- STAI/PANAS scores are self-reports, not clinical diagnoses or direct fatigue/readiness truth.
- A preceding-week association is not causal evidence.
- Wearable metric definitions and several vendor units/scores still require source documentation review.
- The feature table is a research-ready modeling input only after leakage-safe participant/time splitting and target-specific analysis; it does not establish product performance.
