# Fold Count Comparison

## Comparison Results
| fold_count | mean_auc | std_auc | oof_auc | runtime_seconds | preferred_fold |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.725611 | 0.000835 | 0.725604 | 413.659437 | False |
| 7 | 0.725852 | 0.001294 | 0.725845 | 504.849922 | True |
| 10 | 0.725908 | 0.001466 | 0.725905 | 741.510129 | False |

## Recommendation
- Preferred validation setting: 7-fold
- Primary reason: 7-fold is selected because it has lower fold-to-fold variance.
- Mean AUC and runtime remain supporting diagnostics rather than the main decision rule.