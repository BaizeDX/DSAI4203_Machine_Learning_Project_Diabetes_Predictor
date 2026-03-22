# src/config.py
"""Configuration parameters for the project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models/xgb_cv_final'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'
LOG_DIR = PROJECT_ROOT / 'logs'
FIGURE_DIR = PROJECT_ROOT / 'figures'

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TARGET = 'diagnosed_diabetes'
RANDOM_STATE = 42
N_SPLITS = 5

# XGBoost parameters (for compatibility with models.py)
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


class ExperimentConfig:
    """Configuration for experiments with ablation support."""
    
    def __init__(self):
        # Feature flags
        self.use_engineered_features = True
        self.use_family_history_features = True
        self.use_cholesterol_features = True
        
        # Model params
        self.xgb_params = XGB_PARAMS.copy()
    
    def get_description(self) -> str:
        """Return config description for logging."""
        flags = [
            f"eng_feat={self.use_engineered_features}",
            f"family_hist={self.use_family_history_features}",
            f"cholesterol={self.use_cholesterol_features}"
        ]
        return "XGBoost_" + "_".join(flags)
