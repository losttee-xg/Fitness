
## Gap-aware chronological evaluation (7-day embargo)

To avoid evaluating adjacent survey rows whose prior-7-day wearable windows overlap training labels, a separate rolling-origin test was run with a 7-day date gap. It is a diagnostic, single-cohort split; it must not be reused repeatedly for model selection.

| Block | Train rows | Test rows with prior history | Wearable-only MAE | Wearable + prior self-report MAE | Last-report MAE | Rolling-3 self-report MAE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2,710 | 876 | 0.848 | 0.635 | 0.658 | 0.642 |
| 2 | 3,760 | 320 | 0.838 | 0.631 | 0.613 | 0.650 |
| 3 | 4,717 | 85 | 0.847 | 0.682 | 0.824 | 0.788 |

Interpretation: across these three blocks, adding prior self-report categories improves over wearable-only regression. It is only better than the rolling-three self-report baseline in the first block; the baseline is better in blocks 2 and 3. The third block has only 85 test rows. Do not claim a robust model advantage. This supports the hypothesis that previous self-reported tiredness drives much of the apparent signal; an app can present a self-reported trend, but this does not validate a wearable-derived fatigue forecast.

The evaluation deliberately uses only earlier survey labels at each known-user prediction point and inserts a full seven-day embargo before each test block. It is not prospective external validation.
