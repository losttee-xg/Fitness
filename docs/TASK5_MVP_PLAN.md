# Task 5 MVP: dataset, model baselines, and limits

## What this MVP does

The MVP has three independent research tracks instead of one unsupported “health score” model:

1. **Short-horizon metric forecasts**: daily RHR, RMSSD, sleep minutes, and eligible activity metrics; predict the next observed date from recent history and compare with last-value/rolling baselines. Current implementation is benchmark-only and requires contiguous daily windows; sparse LifeSnaps can lead to few or no scored participants.
2. **Questionnaire association prototype**: features from the seven calendar days strictly before each STAI/PANAS score, evaluated with participant-grouped cross-validation against a training-fold median baseline. These are self-reported psychological questionnaire targets, not fatigue or recovery ground truth.
3. **Anomaly candidates in shadow mode**: per-person rolling robust-MAD deviations for selected observed metrics. Candidates are not alerts, and no precision/recall claim is possible without independently adjudicated event labels.

Habit-health association may be explored using participant-level historical steps/activity and the same participant’s later survey score, but it is observational and non-causal. No recommendations or notifications are sent by this MVP.

## Data dictionary and unit gate

The LifeSnaps Zenodo record describes 71 Fitbit Sense participants over more than four months and provides daily/hourly Fitbit/EMA exports, scored surveys and MongoDB collections under CC BY 4.0. It does not provide field-by-field definitions for all Fitbit columns in the record. The current local folder also has no authoritative field-level codebook. Therefore:

- Preserve source values and raw fields. Do not rescale RMSSD, resting HR, or sleep duration based on guessed units.
- Use RMSSD only as RMSSD; never rename it SDNN.
- Keep NREM HR, RHR, vendor sleep/stress scores, and sleep duration distinct.
- Exclude metrics with unverified source units and vendor scores from the current survey model features. The feature script accepts only catalogued metrics whose unit isn't marked unverified; verify this catalog against original source documentation before expanding it.
- Labels remain separate from wearable observations. SEMA flags are not clinical fatigue labels.
- Before product use, maintain source/license attribution and have privacy/legal review the dataset terms and intended commercial use.

## Dataset choices

- **LifeSnaps/RAIS**: primary longitudinal cohort already in the workspace, useful for wearable trends/anomaly exploration and exploratory association with EMA/surveys. It is not a validated Recovery/Fatigue cohort.
- **MMASH**: possible independent exploratory cohort for HRV, sleep, activity and questionnaire associations; only 22 participants and roughly 24 hours, so not useful for short-horizon longitudinal forecasting. Review its ODbL terms before commercial use.
- **BIDSleep**: sleep-stage classification validation after correctly parsing its 30-second reference labels and aligning timestamps; not a daily fatigue/recovery dataset.
- **No dataset identified here supplies a validated Recovery/Fatigue target suitable for supervised product claims.** Do not synthesize one with LLMs. Synthetic sequences are permitted only in a separate engineering test fixture, marked synthetic and excluded from model evaluation.

## Current exploratory model evaluation

### Observable metric forecasts

One-day-ahead per-person chronological benchmarks on consecutive daily windows. The report evaluates the test tail of each individual series. Weighted pooled absolute error across scored target-days:

| Metric | Participants with scored series | Test target-days | Last-value MAE | Rolling-median MAE |
|---|---:|---:|---:|---:|
| RHR | 65 | 893 | 2.521 bpm | 1.689 bpm |
| Sleep minutes asleep | 48 | 574 | 87.429 min | 63.936 min |
| RMSSD | 32 | 381 | 11.255 ms (source mapping) | 8.192 ms (source mapping) |

The rolling median is a benchmark, not an approved product model. Since it outperformed last-value in these cohorts, do not claim a learned model advantage. These results are available only for people with sufficient consecutive observed days and are not external validation.

### Questionnaire target models

Created with grouped five-fold cross-validation, with participants held out as groups and a training-fold median baseline:

| Target | Usable rows | Participants | Model MAE | Baseline MAE | Beats baseline? |
|---|---:|---:|---:|---:|---|
| STAI stress score | 278 | 53 | 4.64 | 4.36 | No |
| PANAS positive affect | 267 | 51 | 6.77 | 5.91 | No |
| PANAS negative affect | 267 | 51 | 6.66 | 6.37 | No |

These results do not support shipping predictive questionnaire insights. Model performance was worse than the simple baseline on all three targets. Treat them as a useful negative result and dataset/feature diagnostic, not as evidence of no biological relationship.

### Habit association exploration

Pairwise descriptive comparisons require exact dates and no imputation:

| Habit | Outcome | Pairs | Participants | Raw pooled r | Participant-demeaned r |
|---|---|---:|---:|---:|---:|
| Steps | Next-day sleep minutes | 3,499 | 62 | -0.078 | -0.007 |
| Steps | Next-day RHR | 4,270 | 69 | -0.186 | 0.067 |
| Steps | Next-day RMSSD | 2,445 | 41 | 0.076 | -0.048 |
| Steps | Same-day RHR | 4,411 | 69 | -0.190 | 0.034 |

These are descriptive, non-causal summaries. The change after participant demeaning shows that pooled correlations can be dominated by between-person differences. Do not turn these results into user-facing advice.

### Shadow anomaly and notification policy

- RHR robust-MAD produced 157 candidate points across 71 participants; RMSSD produced 86 across 43 participants. The detector also abstains or emits no-candidate outcomes. These counts do not estimate precision, false-positive rate, or health events.
- A software-only rate-limit simulation received 243 candidate points and selected 108 after the placeholder policy of at most one notification per rolling seven days plus a three-day cooldown. No notification was sent. This only checks policy code; it is not evidence the frequency is safe or useful.

## Reproduction

From PowerShell at `D:\Fitness`:

```powershell
.\.venv\Scripts\python.exe -I app/scripts/harmonize_local_datasets.py `
  --dataset-root "D:\Fitness\dataset" `
  --output "D:\Fitness\dataset\harmonized"

.\.venv\Scripts\python.exe -I app/scripts/enrich_task5_features.py `
  --observations "D:\Fitness\dataset\harmonized\observations.csv" `
  --labels "D:\Fitness\dataset\harmonized\labels.csv" `
  --output-dir "D:\Fitness\dataset\harmonized\enriched" `
  --lookback-days 7 `
  --min-observed-days 3

.\.venv\Scripts\python.exe -I app/scripts/evaluate_task5_survey_models.py `
  --features "D:\Fitness\dataset\harmonized\enriched\lifesnaps_survey_features.csv" `
  --output "D:\Fitness\dataset\harmonized\enriched\survey_model_evaluation.json" `
  --folds 5
```

The optional dependencies are in `requirements-task5.txt`. Tests: `python -I -m pytest tests/test_task5_enrichment.py tests/test_task5_survey_models.py tests/test_task5_research_pipeline.py -q`.

## MVP product behavior (safe prototype)

- Show metric trends and data-coverage status, not a combined “recovery score.”
- Show anomaly candidates only in an internal review view; no push notification until false-positive burden and user-benefit policy have been validated.
- Habit insights must say “associated with” rather than “caused by”; require sufficient within-person history and disclose missing data.
- If there is insufficient data or uncertainty, abstain and explain why.
- Any real user-facing proactive recommendations require prospective evaluation, consent, safety review and a rate-limited notification policy.

## Next gates

1. Obtain/verify authoritative field-level Fitbit data dictionary and establish exact definitions for each included feature.
2. Narrow feature set to verified units; rerun group-held-out benchmarks and check performance across folds, not only aggregate MAE.
3. Obtain a longitudinal cohort with validated/stated daily recovery/fatigue labels if supervised prediction is a product requirement; otherwise scope MVP to descriptive trends, forecast observable metrics, and low-risk statistical coaching hypotheses.
4. Parse BIDSleep only for its proper task (sleep-stage recognition), and assess PPG-DaLiA raw payload with a safe, reviewed conversion process before signal experiments.
5. Define product success criteria and alert budgets before enabling notifications; build offline safety/timing evaluation, then a prospective pilot.
