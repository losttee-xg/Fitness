# health_synth.csv — Data dictionary

SYNTHETIC data (seed 42, 300 users, 36,553 user-days, ~60–180 days/user). Row = one user-day indexed by MORNING date.
Sleep/physiology = the night just ended. `*_prev` behaviour columns = the previous day/evening.

## Identity / split
| Column | Meaning |
|---|---|
| user_id, date, day_index (0-based), day_of_week (0=Mon) | keys; `day_index` supports the "enough history" gate |
| split | train/val/test, **split by user** (70/15/15), no user in >1 split |
| age, sex | static attributes |

## Habits (previous day)
alcohol_units_prev, caffeine_mg_prev, caffeine_after_2pm_mg_prev, screen_time_late_min_prev, stress_self_prev (1–10),
workout_min_prev, workout_intensity_prev (0–5), steps_prev, strain_prev (0–10, derived from workout+steps)

## Sleep & physiology (observed; NaN = wear gap)
bedtime_hour (0–24 clock, e.g. 0.5 = 00:30), sleep_latency_min, sleep_duration_h, sleep_efficiency (%), deep_sleep_min, rem_sleep_min,
hrv_rmssd_ms, rhr_bpm, resp_rate_bpm, skin_temp_dev_c

## Scores (computed from OBSERVED data)
- sleep_score (0–100) = 0.40·duration/need + 0.25·efficiency + 0.20·deep+REM share + 0.15·latency (each scaled 0–100)
- recovery_score (1–99) = 52 + 14·z(log HRV) − 8·z(RHR) + 8·z(sleep_score); z = robust z vs the user's trailing 28-day median/MAD (shifted 1 day, min 10 days → NaN early). **A PROXY, not ground truth.**
- fatigue_self_report (1–10): simulated self-report, 30% missing

## Anomaly labels (ground truth)
- is_anomaly = 1 when episode severity ≥ 0.25; anomaly_type ∈ illness, overtraining, sleep_restriction, high_stress, jetlag; anomaly_severity 0–1
- anomaly_episode_id: whole episode incl. weak tail days (is_anomaly=0 there) → use `anomaly_episodes.csv` for event-level evaluation
- is_sensor_artifact = 1: HRV spiked ×1.7–2.2 (not an anomaly; tests robustness)

## Forecast targets (next day, same user)
y_next_sleep_score, y_next_recovery_score, y_next_fatigue, y_next_is_anomaly (NaN on each user's last day / missing values)

## Built-in difficulty
~10% anomaly days · 4% wear gaps · 2% HRV-only dropouts · 0.5% artifacts · heavy-drinking nights lower HRV but are NOT anomalies (hard negatives) · AR(1) physiology, weekday effects, per-user baselines.

## Assumed causal structure (made up, simple, directionally plausible)
alcohol↓HRV/deep/REM, ↑RHR, ↑latency · late caffeine/screen↑latency · high strain↓HRV · sleep debt↓HRV, ↑fatigue · illness: HRV −, RHR +, temp +, resp + · overtraining: strain +, HRV −, RHR + · sleep restriction: duration −2h · high stress: latency +, REM −, HRV − · jetlag: bedtime shift.
