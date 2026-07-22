# Feature-Masking Re-score Summary

Status: run

Source feature matrix: `artifacts/baselines/lgbm_numeric_v1_subsample/X_test.parquet`

| Mask rate | Rows | Mean base score | Mean masked score | Mean abs score delta | Action change rate | Step-up/review rate | HIGH risk rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1000 | 0.0405 | 0.0405 | 0.0000 | 0.0000 | 0.0220 | 0.0080 |
| 0.3 | 1000 | 0.0405 | 0.0267 | 0.0269 | 0.0200 | 0.0070 | 0.0000 |
| 0.6 | 1000 | 0.0405 | 0.0332 | 0.0280 | 0.0210 | 0.0110 | 0.0000 |

Interpretation: This inference-only masking re-score applies 0%, 30% and 60% masking to a persisted model-ready feature matrix and re-scores the trained LightGBM model. It strengthens thin-file robustness evidence if the persisted feature matrix is the same matrix used for the reported baseline evaluation. It remains a controlled stress test rather than a production outage simulation.
