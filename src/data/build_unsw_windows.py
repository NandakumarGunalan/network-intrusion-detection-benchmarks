from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.load_unsw import load_unsw_csv
from src.data.make_windows import make_fixed_windows


# ---------------- CONFIG ----------------
DATA_PATH = Path(
    "/Users/nandakumargunalan/Documents/Documents - Nanda’s MacBook Air/"
    "05_Projects_Personal/Ml_Portfolio/data/unsw400k_full400k.csv"
)

LABEL_COL = "label"
TIME_COL = "ts"

FEATURE_COLS = [
    "duration_conn",
    "orig_bytes_conn",
    "resp_bytes_conn",
    "missed_bytes_conn",
    "orig_pkts_conn",
    "orig_ip_bytes_conn",
    "resp_pkts_conn",
    "resp_ip_bytes_conn",
    "qclass_dns",
    "qtype_dns",
    "rcode_dns",
    "request_body_len_http",
    "response_body_len_http",
    "status_code_http",
]

WINDOW_SIZE = 20
STEP = 5
N_NORMAL = 20000
N_ATTACK = 20000
# ----------------------------------------


def main():
    print("Loading UNSW data...")
    df = load_unsw_csv(DATA_PATH)

    print("Original shape:", df.shape)

    # Binary label: normal=0, attack=1
    df["binary_label"] = (df[LABEL_COL] != "normal").astype(int)

    # Separate normal and attack
    normal_df = df[df["binary_label"] == 0].head(N_NORMAL)
    attack_df = df[df["binary_label"] == 1].head(N_ATTACK)

    df = pd.concat([normal_df, attack_df], axis=0)

    # Sort by time to form meaningful sequences
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    print("Sampled shape:", df.shape)
    print("Binary label distribution:")
    print(df["binary_label"].value_counts())

    # -------- Feature scaling --------
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    print("Building sequence windows...")
    X, y = make_fixed_windows(
        df,
        feature_cols=FEATURE_COLS,
        label_col="binary_label",
        window_size=WINDOW_SIZE,
        step=STEP,
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y distribution:", np.bincount(y))


if __name__ == "__main__":
    main()
