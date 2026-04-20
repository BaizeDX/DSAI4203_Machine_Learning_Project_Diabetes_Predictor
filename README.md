# Diabetes Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Diabetes%20Prediction-20beff.svg)](https://www.kaggle.com/)

A comprehensive machine learning project for diabetes prediction using XGBoost with proper cross-validation and feature engineering. The retained final workflow is a **7-fold** XGBoost pipeline with **mean CV AUC 0.725852**, **std 0.001294**, and **OOF AUC 0.725845**.

## 📊 Project Overview

This project develops a systematic machine learning pipeline for predicting diabetes using clinical and demographic data. The dataset contains 700,000 training samples and 300,000 test samples with 26 columns in the training file and 25 in the test file. The retained final workflow uses **7-fold stratified cross-validation**, **fold-wise categorical encoding**, engineered features, and **out-of-fold evaluation** to produce a more reliable estimate of generalization performance.

### Key Results

| Metric | Value |
|--------|-------|
| **7-Fold CV AUC** | 0.725852 ± 0.001294 |
| **OOF AUC** | 0.725845 |
| **Kaggle Public Score** | 0.69626 |
| **Kaggle Private Score** | 0.69300 |

## 📁 Project Structure

```
machinelearning_project/
├── README.md                      # Project overview and usage notes
├── requirements.txt               # Python dependencies
│
├── data/                          # Kaggle input files
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/                           # Final reproducible code pipeline
│   ├── config.py                  # Paths, constants, experiment config
│   ├── features.py                # Final engineered features
│   ├── preprocessing.py           # Shared preprocessing helpers
│   ├── models.py                  # CV training, baselines, model logic
│   ├── train.py                   # Additional training entry point
│   ├── train_xgb_cv.py            # Main training entry point
│   ├── report_figures.py          # Final report figure generator
│   ├── evaluate.py                # Optional evaluation helpers
│   ├── utils.py                   # Utility functions
│   └── __init__.py
│
├── logs/                          # Structured training outputs and supporting analysis artifacts
│   ├── model_comparison.csv       # Baseline vs final model comparison
│   ├── oof_predictions.csv        # OOF predictions for analysis/ROC
│   ├── feature_importance.csv     # Final feature importance table
│   ├── summary.json               # Fold AUCs, mean/std AUC, OOF AUC
│   ├── tail_cutoff_comparison.*   # Tail-holdout comparison artifacts
│   ├── fold_count_comparison.*    # 5/7/10-fold comparison artifacts
│   ├── lightgbm_default_*         # Supporting LightGBM baseline artifacts
│   └── 04-11_* / exp*_summary.*   # Historical experiment logs and summaries
│
├── report_figures/                # Final report-ready figures from src/report_figures.py
│   ├── figure_1_7fold_cv_aucs.(png/pdf)
│   ├── figure_2_feature_importance.(png/pdf)
│   ├── figure_3_feature_importance_pie.(png/pdf)
│   ├── figure_4_roc_curve.(png/pdf)
│   ├── figure_5_prediction_distribution.(png/pdf)
│   ├── figure_6_auc_boxplot.(png/pdf)
│   └── figure_7_kaggle_score.png
│
├── submissions/                   # Kaggle submission outputs
│   ├── final_submission.csv
│   ├── lightgbm_default.csv
│   ├── xgboost_default.csv
│   └── 04-11_* / 11_xgboost_cv_*  # Historical experiment submissions
│
├── notebooks/                     # EDA, model exploration, summaries, and report notebooks
│   ├── 01_EDA.ipynb
│   ├── 02.5_data_explore_report.md
│   ├── 02_baseline.ipynb
│   ├── 03_knn_mlp_hybrid.ipynb
│   ├── 04_xgboost_default.ipynb
│   ├── 04.5_lightgbm_default.ipynb
│   ├── 05-10_*.ipynb
│   ├── 10.5_conclusion_of_1-10.md
│   ├── 11.5_summary_of_11.md
│   ├── 11_xgboost_cv_final.ipynb
│   ├── 12_tail_cutoff_comparison.ipynb
│   ├── 13_fold_count_comparison.ipynb
│   ├── 13.5_overall_experiment_summary.ipynb
│   ├── 14_visualization.ipynb
│   └── 14.5_modularize_report.md
│
├── models/                        # Final and archived intermediate models
│   ├── xgb_cv_final/              # Saved XGBoost fold-model directory used by the CV pipeline
│   ├── baseline_dt.pkl
│   ├── xgboost_default.pkl
│   ├── lightgbm_default.pkl
│   └── 04-08_*                    # Historical experiment model files
│
├── figures/                       # Notebook-generated figures for analysis and report drafting
│   ├── figure_1_7fold_cv_aucs.png
│   ├── figure_2_feature_importance.png
│   ├── figure_3_feature_importance_pie.png
│   ├── figure_4_roc_curve.png
│   ├── figure_5_prediction_distribution.png
│   ├── figure_6_auc_boxplot.png
│   └── legacy and notebook-specific PNG/PDF exports
│
└── reports/                       # Final report assets
    ├── reports.tex
    └── reports.pdf
```

## 📊 Data Exploration

### Dataset Statistics

| Property | Value |
|----------|-------|
| Training samples | 700,000 |
| Test samples | 300,000 |
| Columns in `train.csv` | 26 (24 predictors + `id` + target) |
| Target distribution | 62.33% diabetic, 37.67% non-diabetic |

### Feature Correlations

| Feature | Correlation with Target |
|---------|------------------------|
| family_history_diabetes | +0.211 |
| age | +0.161 |
| systolic_bp | +0.107 |
| bmi | +0.106 |
| physical_activity_minutes_per_week | -0.170 |

### Key Findings

- **No missing values** - simplifies preprocessing
- **Class imbalance** - requires careful evaluation
- **Strong predictive signals** from family history and age
- **Outliers present** in physical activity and sleep hours

## 🧠 Methodology

### Baseline Models

1. **Dummy Classifier** (AUC = 0.500)
   - Majority class predictor
   - Establishes lower bound

2. **Decision Tree** (AUC = 0.6825)
   - Max depth = 5
   - Interpretable baseline

### Early Alternative Exploration: KNN + MLP Hybrid

Before fully committing to boosted tree models, the project also explored a small KNN / MLP branch in [03_knn_mlp_hybrid.ipynb](/Users/junhaohuang/Documents/Playground/DSAI4203_Machine_Learning_Project_Diabetes_Predictor-main/notebooks/03_knn_mlp_hybrid.ipynb). This notebook compared three simple alternatives under the same early-stage experimental setting:

| Model | Validation AUC | Note |
|-------|----------------|------|
| KNN | 0.6447 | Distance-based baseline, sensitive to high-dimensional mixed-type features |
| MLP | 0.6890 | Best result within this branch |
| KNN + MLP Hybrid | 0.6763 | Simple probability-average blend |

This comparison was useful because it tested a different model family before the project moved further into boosting. The outcome suggested that although the MLP was stronger than KNN, the hybrid branch still underperformed the later boosted-tree baselines. As a result, the main development path shifted toward XGBoost and LightGBM rather than neural-network or distance-based methods.

### Feature Engineering

Based on clinical domain knowledge, five engineered features were added:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `age_family_history` | age × family_history_diabetes | Age effect amplified with family history |
| `age_bmi` | age × bmi | Obesity risk increases with age |
| `cardio_risk_score` | weighted sum of cardiovascular factors | Composite risk score |
| `cholesterol_ratio` | total_cholesterol / (hdl + 1) | Clinical risk indicator (>4 = high risk) |
| `non_hdl_cholesterol` | total_cholesterol - hdl | Recognized cardiovascular marker |

### Critical Methodological Improvements

#### From Fixed Holdout to Cross-Validation
- **Before**: Single 80/20 split with fixed random seed
- **After**: 7-fold stratified cross-validation in the retained final workflow
- **Benefit**: Each sample validated exactly once, reliable estimates

#### Fold-Wise Categorical Encoding
```python
# Encoder fitted ONLY on fold-train to prevent leakage
def _encode_categorical_fold(X_train_raw, X_valid_raw, categorical_cols):
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train_raw[col])      # Only on fold-train
        X_train_encoded[col] = le.transform(X_train_raw[col])
        X_valid_encoded[col] = le.transform(X_valid_raw[col])
```

#### Out-of-Fold Evaluation
- Collects validation predictions from all folds
- Provides unbiased estimate of generalization performance

#### Early Stopping
- Stops training when validation AUC doesn't improve for 100 rounds
- Automatically determines optimal tree count

### Final Model Parameters

```python
{
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'n_estimators': 2000,           # Early stopping active
    'learning_rate': 0.03,
    'max_depth': 4,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'early_stopping_rounds': 100
}
```

## 📈 Results

### Cross-Validation Performance

| Fold | AUC | Best Iterations |
|------|-----|-----------------|
| 1 | 0.725554 | 1999 |
| 2 | 0.725768 | 1998 |
| 3 | 0.725870 | 1987 |
| 4 | 0.723834 | 1999 |
| 5 | 0.725908 | 1999 |
| 6 | 0.728571 | 1994 |
| 7 | 0.725461 | 1999 |
| **Mean** | **0.725852** | **1996** |
| **Std** | **0.001294** | - |
| **OOF AUC** | **0.725845** | - |

### Model Comparison

| Model | Mean CV AUC | Std | OOF AUC |
|-------|-------------|-----|---------|
| Dummy Classifier | 0.500000 | 0.000000 | 0.500000 |
| Decision Tree | 0.685722 | 0.001715 | 0.685839 |
| **XGBoost (Final)** | **0.725852** | **0.001294** | **0.725845** |

### Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | family_history_diabetes | 0.4251 |
| 2 | age_family_history | 0.2975 |
| 3 | age_bmi | 0.0677 |
| 4 | cardio_risk_score | 0.0498 |
| 5 | physical_activity_minutes_per_week | 0.0345 |

### Visualizations

| Plot | Description |
|------|-------------|
| Fold AUCs | Broken-axis view of 7-fold CV stability across folds |
| Feature Importance | Top 15 features with importance scores |
| Feature Importance Share | Top 5 feature-importance share |
| ROC Curve | OOF predictions, AUC = 0.725845 |
| Prediction Distribution | OOF prediction distribution by true class |
| AUC Boxplot | Fold-wise spread around the 7-fold mean AUC |
| Kaggle Result | Final public/private leaderboard score summary |

## 🔧 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
lightgbm>=3.3.0
```

## 📝 Experiment Log

| Experiment | Description | AUC |
|------------|-------------|-----|
| 01 | Exploratory Data Analysis | - |
| 02 | Decision Tree Baseline | 0.6825 |
| 03 | KNN / MLP / Hybrid exploration | 0.6890 (best MLP) |
| 04 | Default XGBoost baseline | 0.7192 |
| 04.5 | Default LightGBM baseline | 0.7248 |
| 05 | XGBoost + 300 trees + class weight | 0.7255 |
| 06-08 | XGBoost refinement and feature engineering | 0.7264 (best holdout) |
| 09 | Ensemble (XGB + LGB) | 0.7256 |
| 10 | Holdout-stage experiment comparison | Summary notebook |
| **11** | **XGBoost + 7-fold CV + OOF** | **0.725845** |
| 12 | Tail cutoff comparison | 0.6987 (preferred pct:0.10) |
| 13 | 5-fold vs 7-fold vs 10-fold comparison | 0.725852 (preferred 7-fold) |
| 14 | Final report visualizations | Figure generation notebook |

## 💡 Key Lessons Learned

### What Worked
- ✅ **Stratified cross-validation** - reliable performance estimates
- ✅ **Fold-wise categorical encoding** - prevents data leakage
- ✅ **Early stopping** - automatic model complexity control
- ✅ **Domain-informed features** - validates clinical knowledge

### What Could Be Improved
- 🔄 More extensive hyperparameter search
- 🔄 Ensemble methods (XGBoost + LightGBM + CatBoost)
- 🔄 SHAP analysis for interpretability
- 🔄 Neural network architectures

## 🙏 Acknowledgments

- Dataset provided by Kaggle's Tabular Playground Series
- Inspired by the Diabetes Health Indicators Dataset
- XGBoost library by Tianqi Chen and contributors
