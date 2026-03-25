# src/models.py
"""Model training and evaluation with proper cross-validation."""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 直接从 config 导入需要的常量
from src.config import RANDOM_STATE, N_SPLITS, ExperimentConfig
from src.features import add_engineered_features

warnings.filterwarnings("ignore")


@dataclass
class ModelArtifacts:
    """Container for model training outputs."""
    oof_pred: np.ndarray
    fold_scores: List[float]
    best_iterations: List[Optional[int]]
    feature_importance: pd.DataFrame
    test_pred: np.ndarray
    mean_auc: float
    std_auc: float
    oof_auc: float
    config: Dict[str, Any] = field(default_factory=dict)


def build_xgb_model(seed: int = RANDOM_STATE) -> XGBClassifier:
    """Build XGBoost model with optimized parameters."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=100,
        verbosity=0,
    )


def build_dummy_model() -> DummyClassifier:
    """Build dummy classifier for baseline."""
    return DummyClassifier(strategy="most_frequent")


def build_decision_tree_model(seed: int = RANDOM_STATE) -> DecisionTreeClassifier:
    """Build decision tree for baseline."""
    return DecisionTreeClassifier(max_depth=5, random_state=seed)


# src/models.py - _encode_categorical_fold 函数（已存在，确保正确）

def _encode_categorical_fold(
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features within a single fold.
    Fitted ONLY on fold-train to avoid data leakage.
    """
    X_train_encoded = X_train_raw.copy()
    X_valid_encoded = X_valid_raw.copy()

    for col in categorical_cols:
        if col not in X_train_encoded.columns:
            continue

        le = LabelEncoder()
        # 🔴 关键：只 fit fold-train
        X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))

        # Transform fold-valid
        try:
            X_valid_encoded[col] = le.transform(X_valid_encoded[col].astype(str))
        except ValueError:
            # Unseen category in validation - assign -1
            X_valid_encoded[col] = -1

    # Align columns
    missing_valid = set(X_train_encoded.columns) - set(X_valid_encoded.columns)
    for col in missing_valid:
        X_valid_encoded[col] = 0
    X_valid_encoded = X_valid_encoded[X_train_encoded.columns]

    return X_train_encoded, X_valid_encoded


def run_baseline_cv(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    model_fn: Callable,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run stratified k-fold CV for baseline models.
    Uses SAME fold-wise encoding as XGBoost for fair comparison.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred = np.zeros(len(X))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        X_train_raw = X.iloc[train_idx]
        X_valid_raw = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # Fold-wise categorical encoding
        X_train_encoded, X_valid_encoded = _encode_categorical_fold(
            X_train_raw, X_valid_raw, categorical_cols
        )

        # Train model
        model = model_fn()
        model.fit(X_train_encoded, y_train)

        # Predict
        valid_pred = model.predict_proba(X_valid_encoded)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        oof_pred[valid_idx] = valid_pred
        fold_scores.append(fold_auc)

        if verbose:
            print(f"  Fold {fold}: AUC = {fold_auc:.6f}")

    return {
        "fold_aucs": fold_scores,
        "mean_auc": np.mean(fold_scores),
        "std_auc": np.std(fold_scores),
        "oof_auc": roc_auc_score(y, oof_pred),
        "oof_pred": oof_pred,
    }


def run_xgb_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_cols: List[str],
    config: ExperimentConfig = None,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> ModelArtifacts:
    """
    Run stratified k-fold CV for XGBoost with proper fold-wise encoding.
    """
    if config is None:
        config = ExperimentConfig()

    # X = add_engineered_features(X, config)
    # X_test = add_engineered_features(X_test, config)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize storage
    oof_pred = np.zeros(len(X))
    fold_scores = []
    best_iterations = []
    feature_importance_sum = np.zeros(X.shape[1])
    test_preds = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        if verbose:
            print(f"\n{'='*40}")
            print(f"Fold {fold}/{n_splits}")
            print(f"{'='*40}")

        X_train_raw = X.iloc[train_idx]
        X_valid_raw = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        if verbose:
            print(f"  Training: {len(X_train_raw)} samples")
            print(f"  Validation: {len(X_valid_raw)} samples")

        # Fold-wise categorical encoding
        X_train_encoded, X_valid_encoded = _encode_categorical_fold(
            X_train_raw, X_valid_raw, categorical_cols
        )

        # Encode test data using fold-train encoder
        X_test_encoded = X_test.copy()
        for col in categorical_cols:
            if col in X_test_encoded.columns and col in X_train_raw.columns:
                le = LabelEncoder()
                le.fit(X_train_raw[col].astype(str))
                try:
                    X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
                except ValueError:
                    X_test_encoded[col] = -1

        # Align columns
        missing_test = set(X_train_encoded.columns) - set(X_test_encoded.columns)
        for col in missing_test:
            X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[X_train_encoded.columns]

        # Train model
        model = build_xgb_model(seed=random_state + fold)
        model.fit(
            X_train_encoded,
            y_train,
            eval_set=[(X_valid_encoded, y_valid)],
            verbose=False,
        )

        # Validation predictions
        valid_pred = model.predict_proba(X_valid_encoded)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        oof_pred[valid_idx] = valid_pred
        fold_scores.append(fold_auc)

        # Track best iteration
        best_iter = getattr(model, "best_iteration", None)
        best_iterations.append(best_iter)

        # Accumulate feature importance
        feature_importance_sum += model.feature_importances_

        # Test predictions
        test_preds.append(model.predict_proba(X_test_encoded)[:, 1])

        if verbose:
            print(f"  Validation AUC: {fold_auc:.6f}")
            if best_iter:
                print(f"  Best iteration: {best_iter}")

    # Average test predictions
    test_pred = np.mean(test_preds, axis=0)

    # Compute overall metrics
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    oof_auc = roc_auc_score(y, oof_pred)

    # Build feature importance dataframe
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": feature_importance_sum / n_splits}
    ).sort_values("importance", ascending=False)

    if verbose:
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")
        print(f"  Fold AUCs: {[round(s, 6) for s in fold_scores]}")
        print(f"  Mean AUC: {mean_auc:.6f}")
        print(f"  Std AUC: {std_auc:.6f}")
        print(f"  OOF AUC: {oof_auc:.6f}")
        print(f"\n  Top 5 Features:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")

    return ModelArtifacts(
        oof_pred=oof_pred,
        fold_scores=fold_scores,
        best_iterations=best_iterations,
        feature_importance=feature_importance,
        test_pred=test_pred,
        mean_auc=mean_auc,
        std_auc=std_auc,
        oof_auc=oof_auc,
        config={
            "n_splits": n_splits,
            "random_state": random_state,
            "feature_flags": {
                "use_engineered_features": config.use_engineered_features,
                "use_family_history_features": config.use_family_history_features,
                "use_cholesterol_features": config.use_cholesterol_features,
            },
        },
    )


def run_all_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """Run all baseline models with consistent CV protocol."""
    results = {}

    print("\n" + "=" * 60)
    print("Running Baseline Models")
    print("=" * 60)

    print("\n[1/2] Dummy Classifier (most frequent)...")
    results["dummy"] = run_baseline_cv(
        X, y, categorical_cols, build_dummy_model, n_splits, random_state
    )
    print(f"  → OOF AUC: {results['dummy']['oof_auc']:.6f}")

    print("\n[2/2] Decision Tree (max_depth=5)...")
    results["decision_tree"] = run_baseline_cv(
        X, y, categorical_cols, build_decision_tree_model, n_splits, random_state
    )
    print(f"  → OOF AUC: {results['decision_tree']['oof_auc']:.6f}")

    return results


def create_comparison_table(
    baseline_results: Dict[str, Dict[str, Any]],
    xgb_results: ModelArtifacts,
) -> pd.DataFrame:
    """Create unified comparison table for report."""
    rows = []

    # Baseline models
    rows.append({
        "Model": "Dummy Classifier",
        "CV AUC (mean)": baseline_results["dummy"]["mean_auc"],
        "CV AUC (std)": baseline_results["dummy"]["std_auc"],
        "OOF AUC": baseline_results["dummy"]["oof_auc"],
        "Note": "Majority class predictor (lower bound)",
    })

    rows.append({
        "Model": "Decision Tree",
        "CV AUC (mean)": baseline_results["decision_tree"]["mean_auc"],
        "CV AUC (std)": baseline_results["decision_tree"]["std_auc"],
        "OOF AUC": baseline_results["decision_tree"]["oof_auc"],
        "Note": "Max depth=5, interpretable baseline",
    })

    # XGBoost final model
    rows.append({
        "Model": "XGBoost (Final)",
        "CV AUC (mean)": xgb_results.mean_auc,
        "CV AUC (std)": xgb_results.std_auc,
        "OOF AUC": xgb_results.oof_auc,
        "Note": "With engineered features, early stopping, 5-fold CV",
    })

    return pd.DataFrame(rows)


def save_xgb_results(
    artifacts: ModelArtifacts,
    submission_df: pd.DataFrame,
    pred_col: str,
    output_dir: Path,
) -> None:
    """Save all XGBoost results with validation checks."""
    # Validate submission
    errors = []

    if len(submission_df) != len(artifacts.test_pred):
        errors.append(f"Length mismatch: submission={len(submission_df)}, predictions={len(artifacts.test_pred)}")

    preds = artifacts.test_pred
    if np.isnan(preds).any():
        errors.append(f"NaN values detected: {np.isnan(preds).sum()} NaNs")

    if (preds < 0).any() or (preds > 1).any():
        errors.append(f"Predictions outside [0,1]")

    if errors:
        print("❌ Submission validation failed:")
        for e in errors:
            print(f"  - {e}")
        raise ValueError("Submission validation failed")

    print("✓ Submission validation passed")

    # Save submission
    submission = submission_df.copy()
    submission[pred_col] = artifacts.test_pred
    submission.to_csv(output_dir / "11_xgboost_cv_submission.csv", index=False)
    print(f"✓ Submission saved: {output_dir / '11_xgboost_cv_submission.csv'}")

    # Save feature importance
    artifacts.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"✓ Feature importance saved: {output_dir / 'feature_importance.csv'}")

    # Save summary JSON
    summary = {
        "fold_aucs": artifacts.fold_scores,
        "mean_auc": artifacts.mean_auc,
        "std_auc": artifacts.std_auc,
        "oof_auc": artifacts.oof_auc,
        "best_iterations": artifacts.best_iterations,
        "top_features": artifacts.feature_importance.head(10).to_dict("records"),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {output_dir / 'summary.json'}")
