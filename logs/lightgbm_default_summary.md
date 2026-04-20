# Default LightGBM Summary

## Metrics
- Baseline Decision Tree AUC: 0.6825
- Default XGBoost AUC: 0.7192
- Default LightGBM AUC: 0.7224
- Difference vs default XGBoost: +0.0032

## Split
- Train shape: (560000, 42)
- Validation shape: (140000, 42)

## Notes
- This notebook mirrors 04_xgboost_default as closely as possible.
- The only intentional model change is replacing XGBoost with default LightGBM.
