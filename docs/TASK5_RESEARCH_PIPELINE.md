# Task 5 public-data research kit

This kit downloads an allowlisted set of publicly accessible research datasets, records source/license/provenance metadata, builds a daily panel from already-normalized observations, and runs **offline baseline benchmarks**. It is a research scaffold, not a clinical model or a production notification system.

## Important limits

- The included downloader supports only `sleepaccel`, `wearable_device`, and `pamap2`. It does not download gated datasets (GLOBEM, TILES), sources with unclear commercial terms, or WESAD (scientific non-commercial use only).
- Do not assume a repository's code license covers its dataset. Check each dataset's own license and terms before using it for commercial training or distributing model weights.
- SleepAccel and structured stress/exercise sessions are not longitudinal consumer-health datasets. They are useful for parsing, signal/label exploration and limited activity/sleep experiments, but they do not alone validate next-day Recovery/Fatigue forecasts.
- The daily panel builder accepts a normalized long-format CSV. It does not infer dates, units, metric variants, sleep labels, or missing observations. Relative lab time must not be converted to fake calendar dates.
- Forecast training requires repeated consecutive daily observations per participant. With a single sleep-lab night, the correct result is “insufficient longitudinal data,” not a trained model.
- Anomaly output is `SHADOW_CANDIDATES_ONLY`; no notification is sent. MAD thresholds are exploratory, not clinically or statistically calibrated false-positive thresholds.
- Training/evaluation with these public research cohorts does not validate performance on real product users.

## 1. Install optional training dependencies

From `D:\Fitness` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-task5.txt
```

The downloader and panel/anomaly scripts use the Python standard library. The supervised forecast trainer additionally uses scikit-learn and joblib.

## 2. Download allowlisted data

The downloader checks the redirect host, verifies ZIP CRCs, refuses overwrite, uses a temporary partial file, safely extracts path traversal-resistant members, and writes a manifest with SHA-256 hashes and citations.

```powershell
$env:PYTHONPATH = 'app'
.\.venv\Scripts\python.exe -m scripts.download_task5_datasets `
  --dataset sleepaccel `
  --dataset wearable_device `
  --root data\research\task5
```

Optional PAMAP2 activity data:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_task5_datasets `
  --dataset pamap2 `
  --root data\research\task5
```

The downloader deliberately refuses to overwrite an existing archive/extraction. Review `data\research\task5\download_manifest.json` and the source license before using the files. Large downloads can take time. No dataset from a gated source is fetched automatically.

## 3. Normalize downloaded or already-present sources

Use the source adapters for lab/session data. They preserve relative-time semantics instead of inventing calendar dates:

```powershell
$env:PYTHONPATH = 'app'
.\.venv\Scripts\python.exe -m scripts.normalize_task5_sources `
  --sleepaccel-root data\research\task5\extracted\sleepaccel `
  --wearable-root data\research\task5\extracted\wearable_device `
  --output data\research\task5\normalized
```

You may point SleepAccel at an already downloaded folder, for example `data\research\raw\sleepaccel` if it contains participant subfolders. The script writes separate JSONL + manifests per dataset. Wearable E4 parsing handles known single-channel CSV structures only; inspect its manifest and source data constraints. It deliberately leaves session-relative timestamps without UTC dates and does not derive HRV from IBI.

The daily panel input is a CSV with these required fields:

```text
dataset_id,participant_id,date,metric_code,value,unit
```

Optional fields include `metric_variant`, `source`, `quality_flags`, `synthetic`, `label`, and `label_source`. Each row must represent a normalized observation with a real date and a documented semantic meaning. For sleep-night data, choose and document a consistent night date (for example, local wake date), retaining original timestamps and timezone in the source data.

Do not map the following to one another without a reviewed adapter:

- HRV SDNN and HRV RMSSD;
- HR and IBI;
- PSG/EEG sleep-stage reference labels and consumer-watch stage estimates;
- protocol stress labels and general everyday stress;
- self-reported fatigue/readiness and wearable-derived scores.

Create a daily median panel, preserving dataset and participant identities:

```powershell
.\.venv\Scripts\python.exe -m scripts.task5_research_pipeline build-panel `
  --input data\research\normalized\your_observations.csv `
  --output data\research\task5\normalized\daily_panel.csv
```

The panel builder does not merge across datasets. It groups by dataset, participant, date, metric and unit; it records observation count and source row numbers. If multiple observations occur within a day, the daily median is only a neutral aggregation default. Select a domain-specific aggregation when scientifically justified.

## 4. Run an offline forecast benchmark

Example for one metric in one dataset:

```powershell
.\.venv\Scripts\python.exe -m scripts.task5_research_pipeline forecast `
  --panel data\research\task5\normalized\daily_panel.csv `
  --metric TST `
  --horizon 1 `
  --min-history 7 `
  --output data\research\task5\evaluation\tst_h1.json
```

This compares last-value and rolling-median baselines using chronological holdout **within each participant**. It reports per-participant results; it does not pool incompatible devices or labels. It is not a model-training run and is useful as the first benchmark.

To train a small global regression model, and hold out whole participants:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_task5_forecast `
  --panel data\research\task5\normalized\daily_panel.csv `
  --dataset-id my-longitudinal-dataset `
  --metric TST `
  --history-days 7 `
  --horizon-days 1 `
  --output-dir data\research\task5\models
```

The trainer refuses fewer than six participants, rejects duplicate daily targets and mixed units per participant, splits by participant, compares against last-value prediction, and writes a joblib model plus JSON report. **The trainer is a starter benchmark, not a production training recipe.** Its participant holdout does not replace a temporal external test; it does not create prediction intervals, tune hyperparameters, handle censoring, validate clinical outcomes, or approve alert thresholds.

## 5. Generate anomaly candidates in shadow mode

```powershell
.\.venv\Scripts\python.exe -m scripts.task5_research_pipeline anomaly `
  --panel data\research\task5\normalized\daily_panel.csv `
  --metric RHR `
  --window 28 `
  --min-baseline 14 `
  --threshold 3.0 `
  --output data\research\task5\evaluation\rhr_candidates.json
```

The script excludes rows marked `synthetic=true`, uses prior observations only, and abstains when MAD is zero. It reports candidate crossings only; it never sends an alert. A threshold crossing is not proof of a health event. To estimate precision/recall, create independently adjudicated event labels and a locked participant/time test set.

## 6. Dataset-specific adapters still required

The allowlisted downloader is not a universal parser. For each downloaded archive:

1. Read its data dictionary, citations, version and license.
2. Inspect headers and timestamp semantics before writing an adapter.
3. Preserve the raw archive and hashes.
4. Convert raw samples into source-aware observations; retain original units and timestamps.
5. Parse ground-truth labels from their authoritative source, and keep labels in a separate table/field.
6. Write tests for units, timestamp zones, label alignment, missingness, duplicates and participant isolation.
7. Generate the normalized panel only after an independent review of the mapping.

Never deserialize unknown pickle files or execute code shipped inside a dataset. For the already-present WESAD archive, usage is restricted to scientific non-commercial purposes and the project has intentionally not deserialized participant pickle signals. Do not use WESAD in commercial training unless the owner grants additional permission.

## 7. Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task5_research_pipeline.py -q
```

Tests are deterministic unit tests for panel aggregation, calendar gaps, anomaly abstention and synthetic-row exclusion. They do not score model quality or establish product readiness.
