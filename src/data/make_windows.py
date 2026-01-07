import numpy as np
import pandas as pd


def make_fixed_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    window_size: int = 20,
    step: int = 5,
):
    """
    Create fixed-length sliding windows from tabular network flow data.

    Returns:
        X: np.ndarray of shape [N, T, F]
        y: np.ndarray of shape [N]
    """
    X, y = [], []

    values = df[feature_cols].to_numpy()
    labels = df[label_col].to_numpy()

    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size

        X.append(values[start:end])

        window_labels = labels[start:end]

        # Majority label for the window (works for strings OR ints)
        uniq, counts = np.unique(window_labels, return_counts=True)
        y.append(uniq[counts.argmax()])

    return np.array(X), np.array(y)
