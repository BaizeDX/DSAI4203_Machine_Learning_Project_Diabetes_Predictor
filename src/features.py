# src/features.py
"""Feature engineering functions."""

import pandas as pd


def add_engineered_features(df):
    """Add domain-inspired features while retaining original ones.
    
    Features added:
    - age_family_history: interaction between age and family history
    - age_bmi: interaction between age and BMI
    - cardio_risk_score: composite cardiovascular risk score
    - cholesterol_ratio: total cholesterol / HDL
    - non_hdl_cholesterol: total - HDL
    """
    df = df.copy()
    
    # Age interactions
    df['age_family_history'] = df['age'] * df['family_history_diabetes']
    df['age_bmi'] = df['age'] * df['bmi']
    
    # Cardiovascular risk score (higher = more risk factors)
    df['cardio_risk_score'] = (
        df['family_history_diabetes'] * 3 +
        df['hypertension_history'] * 2 +
        df['cardiovascular_history'] * 2
    )
    
    # Cholesterol metrics (clinical indicators)
    df['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    return df


def get_feature_names():
    """Return list of all features after engineering."""
    return [
        # Original features
        'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
        'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
        'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
        'triglycerides', 'family_history_diabetes', 'hypertension_history',
        'cardiovascular_history',
        # Engineered features
        'age_family_history', 'age_bmi', 'cardio_risk_score',
        'cholesterol_ratio', 'non_hdl_cholesterol'
    ]


def get_categorical_columns():
    """Return list of categorical columns."""
    return ['gender', 'ethnicity', 'education_level',
            'income_level', 'smoking_status', 'employment_status']


def get_feature_rationale():
    """Return explanation for each engineered feature."""
    return {
        'age_family_history': 'Age effect is amplified when family history exists',
        'age_bmi': 'Obesity risk increases with age',
        'cardio_risk_score': 'Composite risk score from multiple cardiovascular factors',
        'cholesterol_ratio': 'Cholesterol ratio >4 is a clinical risk indicator',
        'non_hdl_cholesterol': 'Non-HDL cholesterol is a recognized cardiovascular risk marker'
    }
