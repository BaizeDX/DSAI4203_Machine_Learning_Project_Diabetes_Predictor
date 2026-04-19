# Tail Cutoff Comparison

## Ordinary 5-Fold CV Benchmark
| benchmark | mean_auc | std_auc | oof_auc |
| --- | --- | --- | --- |
| ordinary_stratified_cv | 0.701444 | 0.006585 | 0.701259 |

## Tail Holdout Results
| cutoff | cutoff_percent | train_size | validation_size | train_positive_rate | validation_positive_rate | auc | auc_minus_cv_mean | preferred_cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pct:0.03 | 3.000000 | 29100 | 900 | 0.619347 | 0.644444 | 0.705894 | 0.004450 | False |
| pct:0.04 | 4.000000 | 28800 | 1200 | 0.619549 | 0.633333 | 0.716734 | 0.015290 | False |
| pct:0.05 | 5.000000 | 28500 | 1500 | 0.619333 | 0.634667 | 0.710243 | 0.008799 | False |
| pct:0.10 | 10.000000 | 27000 | 3000 | 0.618852 | 0.631333 | 0.698694 | -0.002750 | True |
| pct:0.15 | 15.000000 | 25500 | 4500 | 0.618902 | 0.626889 | 0.702098 | 0.000654 | False |
| pct:0.20 | 20.000000 | 24000 | 6000 | 0.619042 | 0.624333 | 0.703240 | 0.001796 | False |

## Recommendation
- Main development validation: Tail 10%
- Stress-test validation: Tail 3% or Tail 5%
- Ordinary 5-fold CV remains useful as a stability reference, but it should not be the only decision metric.

## Interpretation
- Smaller cutoffs are more tail-focused but noisier.
- Larger cutoffs are more stable but less tail-focused.
- Based on our current experiments, 10% appears to be the most practical and balanced cutoff.
