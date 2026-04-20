# Fold Count Comparison

## Comparison Results
| fold_count | mean_auc | std_auc | oof_auc | runtime_seconds | preferred_fold |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.725611 | 0.000835 | 0.725604 | 409.458535 | False |
| 7 | 0.725852 | 0.001294 | 0.725845 | 513.805212 | True |
| 10 | 0.725908 | 0.001466 | 0.725905 | 676.504081 | False |

## Recommendation
- Preferred validation setting: 7-fold
- Primary reason: 7-fold is selected as the best balance between validation performance, stability, and runtime.
- 5-fold is more stable and faster, while 10-fold is slightly stronger on mean AUC but more expensive and less stable.