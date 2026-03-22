# src/__init__.py
"""Machine Learning Project - Diabetes Prediction.

This package contains all modules for the diabetes prediction project.
Use the main training script via: python -m src.train_xgb_cv
"""

# 只导入配置和基础函数，避免循环引用
from src.config import (
    DATA_DIR, LOG_DIR, MODEL_DIR, SUBMISSION_DIR, FIGURE_DIR,
    RANDOM_STATE, N_SPLITS, ExperimentConfig, XGB_PARAMS
)
from src.features import add_engineered_features
from src.preprocessing import encode_categorical

# 不在这里导入 models 和 train_xgb_cv，避免循环引用
# 这些应该在需要时直接导入

__all__ = [
    'DATA_DIR', 'LOG_DIR', 'MODEL_DIR', 'SUBMISSION_DIR', 'FIGURE_DIR',
    'RANDOM_STATE', 'N_SPLITS', 'ExperimentConfig', 'XGB_PARAMS',
    'add_engineered_features', 'encode_categorical'
]
