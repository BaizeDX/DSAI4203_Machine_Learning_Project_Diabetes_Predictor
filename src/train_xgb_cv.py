# src/train_xgb_cv.py
"""Main training script for XGBoost with proper cross-validation."""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import json  # 添加 json 导入

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from src
from src.config import (
    DATA_DIR, SUBMISSION_DIR, LOG_DIR, FIGURE_DIR,
    RANDOM_STATE, N_SPLITS, ExperimentConfig
)
from src.features import add_engineered_features
from src.models import (
    run_xgb_cv, run_all_baselines,
    create_comparison_table
)
from src.preprocessing import encode_categorical

warnings.filterwarnings('ignore')


def load_data():
    """Load train, test, and sample submission data."""
    print("\n1. Loading data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    
    return train_df, test_df, sample_sub


def prepare_features(train_df, test_df, config=None):
    """Prepare features for modeling."""
    if config is None:
        config = ExperimentConfig()
    
    print("\n2. Preparing features...")
    
    # Separate features and target
    X = train_df.drop(columns=['id', 'diagnosed_diabetes'])
    y = train_df['diagnosed_diabetes'].astype(int)
    X_test = test_df.drop(columns=['id'])
    
    print(f"   Original features: {X.shape[1]}")
    
    # Apply feature engineering (deterministic, no leakage)
    X = add_engineered_features(X, config)
    X_test = add_engineered_features(X_test, config)
    print(f"   After feature engineering: {X.shape[1]}")
    
    # Encode categorical features
    categorical_cols = ['gender', 'ethnicity', 'education_level',
                        'income_level', 'smoking_status', 'employment_status']
    
    X = encode_categorical(X, categorical_cols)
    X_test = encode_categorical(X_test, categorical_cols)
    
    print(f"   After encoding: {X.shape[1]}")
    
    return X, y, X_test, categorical_cols


def validate_submission_predictions(predictions, sample_sub, pred_col):
    """Run safety checks on predictions."""
    errors = []
    
    if len(predictions) != len(sample_sub):
        errors.append(f"Length mismatch: {len(predictions)} vs {len(sample_sub)}")
    
    if np.isnan(predictions).any():
        errors.append(f"NaN values: {np.isnan(predictions).sum()}")
    
    if (predictions < 0).any() or (predictions > 1).any():
        errors.append("Values outside [0,1]")
    
    if errors:
        print("❌ Submission validation failed:")
        for e in errors:
            print(f"   - {e}")
        return False
    
    print("✓ Submission validation passed")
    return True


def save_final_results(artifacts, submission_df, pred_col):
    """Save final results to submissions/ and logs/."""
    from src.config import SUBMISSION_DIR, LOG_DIR
    
    # 1. Save Kaggle submission (final_submission.csv)
    submission = submission_df.copy()
    submission[pred_col] = artifacts.test_pred
    submission_path = SUBMISSION_DIR / "final_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✓ Kaggle submission saved: {submission_path}")
    
    # 2. Save feature importance
    artifacts.feature_importance.to_csv(LOG_DIR / "feature_importance.csv", index=False)
    print(f"✓ Feature importance saved: {LOG_DIR / 'feature_importance.csv'}")
    
    # 3. Save summary JSON for report
    summary = {
        'fold_aucs': artifacts.fold_scores,
        'mean_fold_auc': artifacts.mean_auc,
        'std_fold_auc': artifacts.std_auc,
        'oof_auc': artifacts.oof_auc,
        'best_iterations': artifacts.best_iterations,
        'top_features': artifacts.feature_importance.head(10).to_dict('records')
    }
    
    with open(LOG_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {LOG_DIR / 'summary.json'}")
    
    # 4. Save OOF predictions
    # Note: artifacts.oof_pred doesn't have id mapping, we need y_true
    # This is handled in main() separately


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("XGBoost Training with Proper Cross-Validation")
    print("=" * 60)
    
    # 1. Load data
    train_df, test_df, sample_sub = load_data()
    
    # 2. Prepare features
    X, y, X_test, categorical_cols = prepare_features(train_df, test_df)
    
    # 3. Run baseline models
    print("\n3. Running baseline models...")
    baseline_results = run_all_baselines(
        X, y, categorical_cols, 
        n_splits=N_SPLITS, 
        random_state=RANDOM_STATE
    )
    
    # 4. Run XGBoost with CV
    print("\n4. Running XGBoost with 5-fold CV...")
    artifacts = run_xgb_cv(
        X, y, X_test, categorical_cols,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    # 5. Create comparison table
    print("\n5. Creating comparison table...")
    comparison_df = create_comparison_table(baseline_results, artifacts)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv(LOG_DIR / 'model_comparison.csv', index=False)
    print(f"\n✓ Comparison table saved: {LOG_DIR / 'model_comparison.csv'}")
    
    # 6. Validate and save predictions
    print("\n6. Validating and saving predictions...")
    pred_col = sample_sub.columns[1]
    
    if validate_submission_predictions(artifacts.test_pred, sample_sub, pred_col):
        save_final_results(artifacts, sample_sub, pred_col)
    
    # 7. Save OOF predictions (for analysis)
    oof_df = pd.DataFrame({
        'id': train_df['id'],
        'y_true': y,
        'y_pred': artifacts.oof_pred
    })
    oof_df.to_csv(LOG_DIR / 'oof_predictions.csv', index=False)
    print(f"✓ OOF predictions saved: {LOG_DIR / 'oof_predictions.csv'}")
    
    # 8. Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"""
Model Performance:
- Dummy Classifier (baseline):  {baseline_results['dummy']['oof_auc']:.6f}
- Decision Tree (baseline):     {baseline_results['decision_tree']['oof_auc']:.6f}
- XGBoost (final):               {artifacts.oof_auc:.6f}

Cross-Validation:
- Mean CV AUC:  {artifacts.mean_auc:.6f}
- Std CV AUC:   {artifacts.std_auc:.6f}
- Fold AUCs:    {[round(x, 6) for x in artifacts.fold_scores]}

Top 5 Features:
""")
    for i, row in artifacts.feature_importance.head(5).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n✓ Submission file: {SUBMISSION_DIR / 'final_submission.csv'}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
