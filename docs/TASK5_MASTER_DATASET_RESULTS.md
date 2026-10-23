# Expanded Task 5 data bundle completed

## Outcome

Built an expanded source-aware dataset bundle under `D:\Fitness\dataset\harmonized\master` from public fatigue and LifeSnaps data, preserving source identity and survey answers. The master dataset contains 802,292 non-missing measurement rows and 106,518 non-empty survey-response rows. Observations and labels remain in separate tables; cohort IDs are never cross-joined.

## Files

| File | Description |
|---|---|
| `observations_long.csv` | 621,544 Zenodo fatigue-cohort wearable features + 180,748 LifeSnaps values |
| `labels_long.csv` | Daily/weekly/baseline/screener and scored LifeSnaps survey responses |
| `fatigue_daily_training_examples.csv` | 5,221 daily “tired yesterday” targets with strictly prior seven-day features, 203 participants |
| `lifesnaps_survey_feature_windows.csv` | 815 STAI/PANAS targets with strictly prior seven-day features, 53 participants |
| `master_manifest.json` | File hashes, source/record citations, licensing notes, counts and limitations |
| `synthetic_engineering_only/` | Isolated relative-wellness proxy rows for software/UI tests only; never valid targets or training data |

## Data/target counts

- Daily fatigue self-report: 5,221 target rows.
- Four weekly questionnaires: 780 rows each for physically exhausted, mentally exhausted, fatigued, tired, difficulty starting/finishing due to tiredness, and related tiredness/energy items. Original survey item and answer are preserved individually rather than collapsed into a fabricated common score.
- LifeSnaps scored surveys: STAI stress 279; PANAS positive 268; PANAS negative 268. BREQ, personality and TTM item responses are retained separately in `labels_long.csv`.
- An optional synthetic-derived wellness proxy has 2,465 fatigue-cohort and 1,822 LifeSnaps rows. Every row is marked synthetic and kept in its own directory. It is unsuitable as Recovery ground truth and excluded from model datasets.

## Validation

- `observations_long.csv`: 802,292 rows; no synthetic rows.
- `labels_long.csv`: 106,518 rows; no synthetic rows.
- Fatigue daily training examples: 5,221 `tired_yesterday` targets with strictly prior seven-day features; no leakage violations and no synthetic rows.
- Weekly fatigue questionnaire feature windows are separately represented in `labels_long.csv` with target periods; these retrospective answers are not in the daily training file to prevent target-period overlap.
- LifeSnaps feature windows: 815 rows; no temporal window ends after the survey date.
- Validation suite: 14 tests passed; the master builder compiles successfully.

## Limits

- Zenodo fatigue labels are self-reported tiredness/exhaustion, not a diagnosis or validated Recovery score.
- The prior tiredness model did not beat the baseline MAE.
- Wearable feature units are not verified for all columns; original names and values are retained.
- Pooled observations across datasets are not one population; models and evaluation must filter to a single cohort and target definition.
- Synthetic relative score is an engineering test fixture, not a patient outcome. Do not train on it or present it as a health score.

Sources: [Zenodo fatigue cohort 8018238](https://zenodo.org/records/8018238), [LifeSnaps/RAIS 7229547](https://zenodo.org/records/7229547).