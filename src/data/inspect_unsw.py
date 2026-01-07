from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_unsw import load_unsw_csv


def guess_label_col(cols: list[str]) -> str | None:
    # common variants seen in UNSW / cleaned versions
    candidates = ["label", "attack_cat", "attack", "target", "y"]
    lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand in lower:
            return cols[lower.index(cand)]
    return None


def guess_time_col(cols: list[str]) -> str | None:
    candidates = ["timestamp", "time", "ts", "stime", "ltime", "starttime", "start_time", "date"]
    lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand in lower:
            return cols[lower.index(cand)]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to UNSW CSV (local file, not committed)")
    parser.add_argument("--nrows", type=int, default=20000, help="Rows to sample for inspection")
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Read a sample to stay fast
    df = pd.read_csv(path, nrows=args.nrows)
    df.columns = [c.lower() for c in df.columns]

    print("\n=== BASIC INFO ===")
    print("shape:", df.shape)

    print("\n=== FIRST 30 COLUMNS ===")
    print(df.columns.tolist()[:30])

    label_col = guess_label_col(df.columns.tolist())
    time_col = guess_time_col(df.columns.tolist())

    print("\n=== GUESSES ===")
    print("label_col:", label_col)
    print("time_col :", time_col)

    if label_col:
        print("\n=== LABEL DISTRIBUTION (TOP 15) ===")
        print(df[label_col].value_counts().head(15))

    # Candidate numeric features (sequence models need numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col and label_col in numeric_cols:
        numeric_cols.remove(label_col)

    print("\n=== NUMERIC FEATURE CANDIDATES ===")
    print(f"count: {len(numeric_cols)}")
    print("first 30:", numeric_cols[:30])

    # Quick missingness signal
    if numeric_cols:
        miss = df[numeric_cols].isna().mean().sort_values(ascending=False).head(10)
        print("\n=== TOP 10 NUMERIC COLS BY MISSINGNESS ===")
        print(miss)

    print("\nDone. Next: choose feature_cols + label_col, then build windows on a sample.")


if __name__ == "__main__":
    main()
