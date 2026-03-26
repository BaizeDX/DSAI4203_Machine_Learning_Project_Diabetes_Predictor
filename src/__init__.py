"""Machine Learning Project - Diabetes Prediction.

This package contains all modules for the diabetes prediction project.
Use the main training script via: python -m src.train_xgb_cv
"""

from src.config import (
    DATA_DIR, LOG_DIR, MODEL_DIR, SUBMISSION_DIR, FIGURE_DIR,
    RANDOM_STATE, N_SPLITS, ExperimentConfig, XGB_PARAMS
)
from src.features import add_engineered_features
from src.preprocessing import encode_categorical

# Do not import models or train_xgb_cv here to avoid circular imports

__all__ = [
    'DATA_DIR', 'LOG_DIR', 'MODEL_DIR', 'SUBMISSION_DIR', 'FIGURE_DIR',
    'RANDOM_STATE', 'N_SPLITS', 'ExperimentConfig', 'XGB_PARAMS',
    'add_engineered_features', 'encode_categorical'
]
