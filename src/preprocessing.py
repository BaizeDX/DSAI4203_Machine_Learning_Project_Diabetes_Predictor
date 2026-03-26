"""Data preprocessing functions."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Label encode categorical columns."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('Unknown')
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure test set has same columns as training set."""
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0
    test_df = test_df[train_df.columns]
    return test_df
