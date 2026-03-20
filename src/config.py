# src/config.py
"""Configuration parameters for the project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'
LOG_DIR = PROJECT_ROOT / 'logs'

# Model parameters
TARGET = 'diagnosed_diabetes'
RANDOM_STATE = 42
N_SPLITS = 5

# XGBoost parameters
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'max_depth': 4,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'early_stopping_rounds': 100,
    'verbosity': 0
}
