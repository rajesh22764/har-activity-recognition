"""Shared paths and data loaders for the HAR project."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "UCI HAR Dataset"
FIG = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

for _p in (FIG, REPORTS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)


def activity_labels() -> dict:
    df = pd.read_csv(RAW / "activity_labels.txt", sep=r"\s+", header=None)
    return dict(zip(df[0], df[1]))


def feature_names() -> list:
    """561 feature names, made unique (the raw file has 84 duplicates)."""
    raw = pd.read_csv(RAW / "features.txt", sep=r"\s+", header=None)[1].tolist()
    seen, out = {}, []
    for n in raw:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}__{seen[n]}")
    return out


def load_features(split: str):
    """Return X (DataFrame), y (activity name Series), subject (Series)."""
    X = pd.read_csv(RAW / split / f"X_{split}.txt", sep=r"\s+", header=None)
    X.columns = feature_names()
    y = pd.read_csv(RAW / split / f"y_{split}.txt", header=None)[0].map(activity_labels())
    s = pd.read_csv(RAW / split / f"subject_{split}.txt", header=None)[0]
    return X, y.rename("activity"), s.rename("subject")


def load_raw_signals(split: str) -> np.ndarray:
    """Raw inertial windows, shape (n_windows, 128, 9)."""
    cols = []
    for sig in SIGNALS:
        f = RAW / split / "Inertial Signals" / f"{sig}_{split}.txt"
        cols.append(pd.read_csv(f, sep=r"\s+", header=None).values)
    return np.stack(cols, axis=-1).astype("float32")


def savefig(fig, name: str):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    print(f"saved {FIG / name}")
