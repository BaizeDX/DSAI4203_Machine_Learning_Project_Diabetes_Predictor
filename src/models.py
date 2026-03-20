# src/models.py
"""Model definitions."""

from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

from .config import XGB_PARAMS, RANDOM_STATE


def get_xgb_model():
    """Return configured XGBoost model."""
    return XGBClassifier(**XGB_PARAMS)


def get_dummy_baseline():
    """Return dummy classifier for lower-bound reference."""
    return DummyClassifier(strategy='most_frequent')
