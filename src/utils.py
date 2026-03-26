"""Utility functions."""

import json
import pandas as pd


def save_results(results, filepath):
    """Save results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    """Load results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_submission(test_ids, predictions, filepath):
    """Save Kaggle submission."""
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': predictions
    })
    submission.to_csv(filepath, index=False)
