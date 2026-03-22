# src/features.py
"""Feature engineering functions."""

import pandas as pd
from src.config import ExperimentConfig


def add_engineered_features(
    df: pd.DataFrame,
    config: ExperimentConfig = None
) -> pd.DataFrame:
    """Add engineered features with ablation support."""
    if config is None:
        config = ExperimentConfig()
    
    df = df.copy()
    
    # Age-family history interaction
    if config.use_family_history_features:
        df['age_family_history'] = df['age'] * df['family_history_diabetes']
        df['age_bmi'] = df['age'] * df['bmi']
        df['cardio_risk_score'] = (
            df['family_history_diabetes'] * 3 +
            df['hypertension_history'] * 2 +
            df['cardiovascular_history'] * 2
        )
    
    # Cholesterol metrics
    if config.use_cholesterol_features:
        df['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
        df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    return df


def get_feature_rationale() -> dict:
    """Return explanation for each engineered feature."""
    return {
        'age_family_history': 'Age effect is amplified when family history exists',
        'age_bmi': 'Obesity risk increases with age',
        'cardio_risk_score': 'Composite risk score from cardiovascular factors',
        'cholesterol_ratio': 'Cholesterol ratio >4 is a clinical risk indicator',
        'non_hdl_cholesterol': 'Non-HDL cholesterol is a recognized risk marker'
    }
