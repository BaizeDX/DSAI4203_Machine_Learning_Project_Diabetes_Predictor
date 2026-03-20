# src/train.py
"""Training pipeline with cross-validation."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import N_SPLITS, RANDOM_STATE
from .models import get_xgb_model


def run_cv(X, y, model_fn=None, n_splits=N_SPLITS):
    """Run stratified k-fold cross-validation.
    
    Returns:
        oof_pred: Out-of-fold predictions
        fold_scores: AUC scores for each fold
        best_iters: Best iterations (if early stopping enabled)
        oof_auc: Overall OOF AUC
    """
    if model_fn is None:
        model_fn = get_xgb_model
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    oof_pred = np.zeros(len(X), dtype=float)
    fold_scores = []
    best_iters = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = model_fn()
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        
        valid_pred = model.predict_proba(X_valid)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)
        oof_pred[valid_idx] = valid_pred
        fold_scores.append(fold_auc)
        
        if hasattr(model, 'best_iteration'):
            best_iters.append(model.best_iteration)
        
        model.save_model(f'models/xgb_fold_{fold}.json')
    
    oof_auc = roc_auc_score(y, oof_pred)
    return oof_pred, fold_scores, best_iters, oof_auc


def train_full(X, y, model_fn=None):
    """Train on full dataset."""
    if model_fn is None:
        model_fn = get_xgb_model
    
    model = model_fn()
    model.fit(X, y)
    return model


def predict(model, X_test):
    """Generate predictions."""
    return model.predict_proba(X_test)[:, 1]
