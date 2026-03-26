# Diabetes Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Diabetes%20Prediction-20beff.svg)](https://www.kaggle.com/)

A comprehensive machine learning project for diabetes prediction using XGBoost with proper cross-validation and feature engineering. Achieves **0.7256 CV AUC** with stable performance across 5 folds.

## 📊 Project Overview

This project develops a systematic machine learning pipeline for predicting diabetes using clinical and demographic data. The dataset contains 700,000 training samples and 300,000 test samples with 26 features. The final model uses **5-fold stratified cross-validation**, **fold-wise categorical encoding**, and **out-of-fold evaluation** to ensure reliable generalization estimates.

### Key Results

| Metric | Value |
|--------|-------|
| **5-Fold CV AUC** | 0.7256 ± 0.0008 |
| **OOF AUC** | 0.7256 |
| **Kaggle Public Score** | 0.6956 |
| **Kaggle Private Score** | 0.6931 |

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
│   ├── train_xgb_cv.py            # Main training entry point
│   ├── report_figures.py          # Final report figure generator
│   ├── evaluate.py                # Optional evaluation helpers
│   ├── utils.py                   # Utility functions
│   └── __init__.py
│
├── logs/                          # Final structured outputs from src/
│   ├── model_comparison.csv       # Baseline vs final model comparison
│   ├── oof_predictions.csv        # OOF predictions for analysis/ROC
│   ├── feature_importance.csv     # Final feature importance table
│   └── summary.json               # Fold AUCs, mean/std AUC, OOF AUC
│
├── report_figures/                # Final report-ready figures from src/report_figures.py
│   ├── 01_cv_fold_aucs.png
│   ├── 02_feature_importance.png
│   ├── 03_feature_importance_pie.png
│   ├── 04_roc_curve.png
│   ├── 05_prediction_distribution.png
│   └── 06_auc_boxplot.png
│
├── submissions/                   # Kaggle submission outputs
│   └── final_submission.csv
│
├── notebooks/                     # EDA and historical experiments
│   ├── 01_EDA.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_xgboost_default.ipynb
│   ├── 04-10_*.ipynb
│   ├── 11_xgboost_cv_final.ipynb
│   └── 12_visualization.ipynb
│
├── models/                        # Archived intermediate models / old experiments
├── figures/                       # Legacy notebook-generated figures
└── reports/                       # Final report assets
```

## 📊 Data Exploration

### Dataset Statistics

| Property | Value |
|----------|-------|
| Training samples | 700,000 |
| Test samples | 300,000 |
| Features | 26 |
| Target distribution | 62.3% diabetic, 37.7% non-diabetic |

### Feature Correlations

| Feature | Correlation with Target |
|---------|------------------------|
| family_history_diabetes | +0.211 |
| age | +0.161 |
| systolic_bp | +0.107 |
| bmi | +0.106 |
| physical_activity | -0.170 |

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
- **After**: 5-fold stratified cross-validation
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
| 1 | 0.72596 | 1993 |
| 2 | 0.72428 | 1997 |
| 3 | 0.72515 | 1999 |
| 4 | 0.72675 | 1998 |
| 5 | 0.72591 | 1999 |
| **Mean** | **0.72561** | **1997** |
| **Std** | **0.00084** | - |
| **OOF AUC** | **0.72560** | - |

### Model Comparison

| Model | Mean CV AUC | Std | OOF AUC |
|-------|-------------|-----|---------|
| Dummy Classifier | 0.5000 | 0.0000 | 0.5000 |
| Decision Tree | 0.6825 | 0.0012 | 0.6824 |
| **XGBoost (Final)** | **0.7256** | **0.0008** | **0.7256** |

### Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | family_history_diabetes | 42.7% |
| 2 | age_family_history | 29.4% |
| 3 | age_bmi | 6.9% |
| 4 | cardio_risk_score | 4.6% |
| 5 | physical_activity_minutes_per_week | 3.5% |

### Visualizations

| Plot | Description |
|------|-------------|
| ROC Curve | OOF predictions, AUC = 0.7256 |
| Calibration Curve | Model calibration analysis |
| Prediction Distribution | Class separation visualization |
| Fold AUCs | CV stability across folds |
| Feature Importance | Top 15 features with scores |

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
| 03 | XGBoost Default | 0.7192 |
| 04 | XGBoost + 300 trees + class weight | 0.7255 |
| 05 | XGBoost + 500 trees + depth tuning | 0.7264 |
| 06 | XGBoost + Feature Engineering | 0.7251 |
| 07 | XGBoost + Feature Selection | 0.7251 |
| 08 | LightGBM | 0.7248 |
| 09 | Ensemble (XGB + LGB) | 0.7256 |
| **11** | **XGBoost + 5-fold CV + OOF** | **0.7256** |

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
