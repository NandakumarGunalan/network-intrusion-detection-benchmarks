from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

from src.data.build_unsw_windows import (
    DATA_PATH,
    FEATURE_COLS,
    LABEL_COL,
    TIME_COL,
    WINDOW_SIZE,
    STEP,
    N_NORMAL,
    N_ATTACK,
)
from src.data.load_unsw import load_unsw_csv
from src.data.make_windows import make_fixed_windows
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_data():
    df = load_unsw_csv(DATA_PATH)
    df["binary_label"] = (df[LABEL_COL] != "normal").astype(int)

    normal_df = df[df["binary_label"] == 0].head(N_NORMAL)
    attack_df = df[df["binary_label"] == 1].head(N_ATTACK)
    df = pd.concat([normal_df, attack_df]).sort_values(TIME_COL)

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    X, y = make_fixed_windows(
        df,
        feature_cols=FEATURE_COLS,
        label_col="binary_label",
        window_size=WINDOW_SIZE,
        step=STEP,
    )
    return X, y


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(X_train.shape[1:])

    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=64,
        verbose=2,
    )
    model.save("artifacts/gru_baseline.keras")

    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
