# src/evaluate.py
"""Evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def get_classification_metrics(y_true, y_pred):
    """Return key classification metrics."""
    auc = roc_auc_score(y_true, y_pred)
    return {'auc': auc}


def plot_roc_curve(y_true, y_pred, ax=None, title=None):
    """Plot ROC curve."""
    if ax is None:
        fig, ax = plt.subplots()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], '--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title or 'ROC Curve')
    ax.legend()
    return ax


def plot_calibration_curve(y_true, y_pred, ax=None, title=None):
    """Plot calibration curve."""
    if ax is None:
        fig, ax = plt.subplots()
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    ax.plot(prob_pred, prob_true, marker='o', label='XGBoost')
    ax.plot([0, 1], [0, 1], '--', label='Perfect')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title or 'Calibration Curve')
    ax.legend()
    return ax


def plot_prediction_distribution(y_true, y_pred, ax=None, title=None):
    """Plot prediction distribution by class."""
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Non-diabetic')
    ax.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Diabetic')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title(title or 'Prediction Distribution')
    ax.legend()
    return ax


def create_comparison_table(results_dict):
    """Create model comparison table."""
    df = pd.DataFrame([
        {'Model': name, 'CV AUC': scores[0], 'OOF AUC': scores[1]}
        for name, scores in results_dict.items()
    ])
    return df
