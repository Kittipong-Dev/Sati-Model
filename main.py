"""
Sati Clip MVP Training Pipeline
================================

Two-model pipeline:
1. Posture / movement risk classifier
   - Input: windowed physics + IMU features
   - Output: posture class, e.g. safe_lift, stoop_risk, twist_risk, non_risk_bend, normal_activity, transit_ignore

2. Gaussian fatigue proxy model
   - Input: vibration / instability features from IMU windows
   - Train: fresh_baseline only
   - Output: fatigue_proxy_score / anomaly flag

Expected CSV schema, minimum:
trial_id,subject_id,timestamp_ms,label,risk_label,session_phase,rpe,load_level,label_quality,
acc_up_x,acc_up_y,acc_up_z,
gyro_up_x,gyro_up_y,gyro_up_z,
acc_low_x,acc_low_y,acc_low_z,
gyro_low_x,gyro_low_y,gyro_low_z,
mag_low_x,mag_low_y,mag_low_z,
pitch_up,roll_up,yaw_up,
pitch_low,roll_low,yaw_low

If pitch/roll/yaw are not available, this script can still run using raw IMU features,
but orientation features related to spine_flexion/spine_roll_delta will be skipped unless orientation is computed later.
"""

from __future__ import annotations

import os
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance


# =========================
# Config
# =========================

@dataclass
class PipelineConfig:
    data_dir: str = "data/raw"
    output_dir: str = "artifacts"
    sampling_hz: int = 50

    # Posture model window: 1.5 seconds is reasonable for lift/bend/twist motion.
    posture_window_sec: float = 1.5
    posture_overlap: float = 0.5

    # Fatigue model window: longer window helps FFT stability.
    fatigue_window_sec: float = 4.0
    fatigue_overlap: float = 0.5

    # Data augmentation target per train class.
    augment_target_per_class: int = 300
    random_state: int = 42

    # Gaussian anomaly threshold.
    # Mahalanobis score has no universal threshold here; tune using validation set.
    fatigue_anomaly_threshold: float = 12.0


CONFIG = PipelineConfig()


RAW_IMU_COLUMNS = [
    "acc_up_x", "acc_up_y", "acc_up_z",
    "gyro_up_x", "gyro_up_y", "gyro_up_z",
    "acc_low_x", "acc_low_y", "acc_low_z",
    "gyro_low_x", "gyro_low_y", "gyro_low_z",
]

MAG_COLUMNS = [
    "mag_low_x", "mag_low_y", "mag_low_z",
]

ORIENTATION_COLUMNS = [
    "pitch_up", "roll_up", "yaw_up",
    "pitch_low", "roll_low", "yaw_low",
]

META_COLUMNS = [
    "trial_id", "subject_id", "timestamp_ms", "label", "risk_label",
    "session_phase", "rpe", "load_level", "label_quality",
]


# Map fine labels from collection into first MVP train classes.
TRAIN_CLASS_MAP = {
    "safe_lift_motion": "safe_lift",
    "unsafe_flexion_motion": "stoop_risk",
    "unsafe_twist_motion": "twist_risk",
    "bend_no_load": "non_risk_bend",
    "sit_bend": "non_risk_bend",
    "walk": "normal_activity",
    "sit": "normal_activity",
    "transit_noise": "transit_ignore",
}


# =========================
# Utility
# =========================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert selected columns to numeric, coercing invalid values to NaN."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def gyro_mag(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return np.sqrt(
        df[f"gyro_{prefix}_x"].to_numpy() ** 2
        + df[f"gyro_{prefix}_y"].to_numpy() ** 2
        + df[f"gyro_{prefix}_z"].to_numpy() ** 2
    )


def acc_mag(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return np.sqrt(
        df[f"acc_{prefix}_x"].to_numpy() ** 2
        + df[f"acc_{prefix}_y"].to_numpy() ** 2
        + df[f"acc_{prefix}_z"].to_numpy() ** 2
    )


def jerk(signal: np.ndarray, sampling_hz: int) -> np.ndarray:
    """Approximate jerk as derivative of acceleration magnitude."""
    if len(signal) < 2:
        return np.array([0.0])
    return np.diff(signal) * sampling_hz


def band_energy(signal: np.ndarray, sampling_hz: int, low_hz: float, high_hz: float) -> float:
    """FFT band energy in [low_hz, high_hz]."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)
    if len(signal) < 4:
        return 0.0

    yf = np.abs(rfft(signal)) ** 2
    xf = rfftfreq(len(signal), d=1.0 / sampling_hz)
    band = (xf >= low_hz) & (xf <= high_hz)
    return float(np.sum(yf[band]))


def spectral_entropy(signal: np.ndarray, sampling_hz: int) -> float:
    """Simple normalized spectral entropy."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)
    if len(signal) < 4:
        return 0.0

    power = np.abs(rfft(signal)) ** 2
    power_sum = np.sum(power)
    if power_sum <= 1e-12:
        return 0.0

    p = power / power_sum
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)) / np.log(len(power)))


def pearson_corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# =========================
# Load + EDA
# =========================

def load_csv_folder(data_dir: str) -> pd.DataFrame:
    """Load all CSV files recursively from data_dir."""
    files = list(Path(data_dir).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        "timestamp_ms", "rpe",
        *RAW_IMU_COLUMNS,
        *MAG_COLUMNS,
        *ORIENTATION_COLUMNS,
    ]
    data = safe_numeric(data, numeric_cols)

    # Normalize missing metadata columns.
    for col in META_COLUMNS:
        if col not in data.columns:
            data[col] = "unknown"

    if "label_quality" in data.columns:
        data["label_quality"] = data["label_quality"].fillna("unknown")

    return data


def run_eda(data: pd.DataFrame, output_dir: str) -> None:
    """Save basic EDA summaries to CSV/JSON."""
    ensure_dir(output_dir)

    summary = {
        "num_rows": int(len(data)),
        "num_trials": int(data["trial_id"].nunique()) if "trial_id" in data.columns else None,
        "num_subjects": int(data["subject_id"].nunique()) if "subject_id" in data.columns else None,
        "labels": data["label"].value_counts(dropna=False).to_dict(),
        "risk_labels": data["risk_label"].value_counts(dropna=False).to_dict(),
        "session_phase": data["session_phase"].value_counts(dropna=False).to_dict(),
        "label_quality": data["label_quality"].value_counts(dropna=False).to_dict(),
        "missing_rate": data.isna().mean().sort_values(ascending=False).head(30).to_dict(),
    }

    with open(Path(output_dir) / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Per-trial row counts and approximate Hz.
    rows = []
    for trial_id, g in data.groupby("trial_id"):
        if "timestamp_ms" in g.columns and g["timestamp_ms"].notna().sum() >= 2:
            t_min = g["timestamp_ms"].min()
            t_max = g["timestamp_ms"].max()
            duration_s = max((t_max - t_min) / 1000.0, 1e-9)
            approx_hz = len(g) / duration_s
        else:
            duration_s = np.nan
            approx_hz = np.nan

        rows.append({
            "trial_id": trial_id,
            "subject_id": g["subject_id"].iloc[0] if "subject_id" in g.columns else "unknown",
            "label": g["label"].iloc[0] if "label" in g.columns else "unknown",
            "rows": len(g),
            "duration_s": duration_s,
            "approx_hz": approx_hz,
            "source_file": g["source_file"].iloc[0] if "source_file" in g.columns else "unknown",
        })

    pd.DataFrame(rows).to_csv(Path(output_dir) / "eda_trials.csv", index=False)

    print("EDA saved:")
    print(f"- {Path(output_dir) / 'eda_summary.json'}")
    print(f"- {Path(output_dir) / 'eda_trials.csv'}")


# =========================
# Windowing + Feature Extraction
# =========================

def make_windows(
    data: pd.DataFrame,
    window_sec: float,
    overlap: float,
    sampling_hz: int,
) -> List[pd.DataFrame]:
    """Create sliding windows inside each trial_id."""
    window_size = int(window_sec * sampling_hz)
    stride = max(1, int(window_size * (1.0 - overlap)))

    windows: List[pd.DataFrame] = []
    for trial_id, g in data.groupby("trial_id"):
        g = g.sort_values("timestamp_ms").reset_index(drop=True)
        if len(g) < window_size:
            continue
        for start in range(0, len(g) - window_size + 1, stride):
            w = g.iloc[start:start + window_size].copy()
            w["window_start_idx"] = start
            windows.append(w)

    return windows


def extract_posture_features(window_df: pd.DataFrame, sampling_hz: int) -> Dict[str, float]:
    """Physics-informed + IMU summary features for posture classifier."""
    f: Dict[str, float] = {}

    # Raw IMU statistical features.
    for col in RAW_IMU_COLUMNS:
        if col in window_df.columns:
            x = window_df[col].to_numpy(dtype=float)
            f[f"{col}_mean"] = float(np.nanmean(x))
            f[f"{col}_std"] = float(np.nanstd(x))
            f[f"{col}_min"] = float(np.nanmin(x))
            f[f"{col}_max"] = float(np.nanmax(x))

    # Magnitude and jerk features.
    for prefix in ["up", "low"]:
        if all(c in window_df.columns for c in [f"acc_{prefix}_x", f"acc_{prefix}_y", f"acc_{prefix}_z"]):
            a_mag = acc_mag(window_df, prefix)
            j = jerk(a_mag, sampling_hz)
            f[f"acc_{prefix}_mag_mean"] = float(np.nanmean(a_mag))
            f[f"acc_{prefix}_mag_std"] = float(np.nanstd(a_mag))
            f[f"jerk_{prefix}_max"] = float(np.nanmax(np.abs(j)))
            f[f"jerk_{prefix}_mean"] = float(np.nanmean(np.abs(j)))

        if all(c in window_df.columns for c in [f"gyro_{prefix}_x", f"gyro_{prefix}_y", f"gyro_{prefix}_z"]):
            g_mag = gyro_mag(window_df, prefix)
            f[f"gyro_{prefix}_mag_mean"] = float(np.nanmean(g_mag))
            f[f"gyro_{prefix}_mag_std"] = float(np.nanstd(g_mag))
            f[f"gyro_{prefix}_energy"] = float(np.nanmean(g_mag ** 2))

    twist_rate_proxy = np.abs(
        window_df["gyro_up_z"].to_numpy(dtype=float) - window_df["gyro_low_z"].to_numpy(dtype=float)
    )
    f["twist_rate_proxy_mean"] = float(np.nanmean(twist_rate_proxy))
    f["twist_rate_proxy_max"] = float(np.nanmax(twist_rate_proxy))
    f["twist_rate_proxy_energy"] = float(np.nanmean(twist_rate_proxy ** 2))

    # Orientation / biomechanics features if available.
    has_orientation = all(c in window_df.columns for c in ORIENTATION_COLUMNS)
    if has_orientation:
        pitch_up = window_df["pitch_up"].to_numpy(dtype=float)
        pitch_low = window_df["pitch_low"].to_numpy(dtype=float)
        roll_up = window_df["roll_up"].to_numpy(dtype=float)
        roll_low = window_df["roll_low"].to_numpy(dtype=float)
        yaw_up = window_df["yaw_up"].to_numpy(dtype=float)
        yaw_low = window_df["yaw_low"].to_numpy(dtype=float)

        spine_flexion = np.abs(pitch_up - pitch_low)
        spine_roll_delta = np.abs(roll_up - roll_low)
        # spine_yaw_delta = np.abs(yaw_up - yaw_low)

        f["pitch_up_mean"] = float(np.nanmean(pitch_up))
        f["pitch_low_mean"] = float(np.nanmean(pitch_low))
        f["spine_flexion_mean"] = float(np.nanmean(spine_flexion))
        f["spine_flexion_max"] = float(np.nanmax(spine_flexion))
        f["spine_roll_delta_mean"] = float(np.nanmean(spine_roll_delta))
        f["spine_roll_delta_max"] = float(np.nanmax(spine_roll_delta))
        # f["spine_yaw_delta_mean"] = float(np.nanmean(spine_yaw_delta))
        # f["spine_yaw_delta_max"] = float(np.nanmax(spine_yaw_delta))

    # Common-mode vibration: high correlation can indicate whole-body/environment vibration.
    try:
        up_g = gyro_mag(window_df, "up")
        low_g = gyro_mag(window_df, "low")
        f["upper_lower_gyro_corr"] = pearson_corr_safe(up_g, low_g)
    except Exception:
        f["upper_lower_gyro_corr"] = 0.0

    return f


def extract_fatigue_features(window_df: pd.DataFrame, sampling_hz: int) -> Dict[str, float]:
    """Vibration/FFT features for Gaussian fatigue proxy."""
    f: Dict[str, float] = {}

    up_g = gyro_mag(window_df, "up")
    low_g = gyro_mag(window_df, "low") if all(c in window_df.columns for c in ["gyro_low_x", "gyro_low_y", "gyro_low_z"]) else np.zeros_like(up_g)
    up_a = acc_mag(window_df, "up")

    total_energy = band_energy(up_g, sampling_hz, 0.5, sampling_hz / 2 - 1)
    energy_4_8 = band_energy(up_g, sampling_hz, 4.0, 8.0)
    energy_8_12 = band_energy(up_g, sampling_hz, 8.0, 12.0)

    f["gyro_up_var"] = float(np.nanvar(up_g))
    f["gyro_up_rms"] = float(np.sqrt(np.nanmean(up_g ** 2)))
    f["acc_up_var"] = float(np.nanvar(up_a))
    f["jerk_up_mean_abs"] = float(np.nanmean(np.abs(jerk(up_a, sampling_hz))))
    f["fft_energy_4_8"] = float(energy_4_8)
    f["fft_energy_8_12"] = float(energy_8_12)
    f["fft_band_ratio_8_12"] = float(energy_8_12 / (total_energy + 1e-9))
    f["spectral_entropy"] = spectral_entropy(up_g, sampling_hz)
    f["upper_lower_gyro_corr"] = pearson_corr_safe(up_g, low_g)

    return f


def build_feature_table(
    windows: List[pd.DataFrame],
    feature_fn,
    sampling_hz: int,
) -> pd.DataFrame:
    rows = []
    for w in windows:
        meta = {
            "trial_id": w["trial_id"].iloc[0],
            "subject_id": w["subject_id"].iloc[0],
            "label": w["label"].iloc[0],
            "risk_label": w["risk_label"].iloc[0],
            "session_phase": w["session_phase"].iloc[0],
            "rpe": w["rpe"].iloc[0],
            "load_level": w["load_level"].iloc[0],
            "label_quality": w["label_quality"].iloc[0],
            "window_start_ms": w["timestamp_ms"].iloc[0],
            "window_end_ms": w["timestamp_ms"].iloc[-1],
        }
        features = feature_fn(w, sampling_hz)
        rows.append({**meta, **features})
    return pd.DataFrame(rows)


# =========================
# Augmentation
# =========================

def augment_feature_table(
    df: pd.DataFrame,
    target_col: str,
    target_per_class: int,
    random_state: int = 42,
    noise_scale: float = 0.03,
) -> pd.DataFrame:
    """
    Simple feature-level augmentation.
    Use only on train set, never validation/test.
    Adds small Gaussian noise to numeric feature columns.
    """
    rng = np.random.default_rng(random_state)
    meta_cols = {
        "trial_id", "subject_id", "label", "risk_label", "session_phase", "rpe",
        "load_level", "label_quality", "window_start_ms", "window_end_ms", target_col,
    }
    feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    augmented = [df]
    for cls, count in df[target_col].value_counts().items():
        if count >= target_per_class:
            continue
        cls_df = df[df[target_col] == cls]
        if cls_df.empty:
            continue

        needed = target_per_class - count
        samples = cls_df.sample(n=needed, replace=True, random_state=random_state).copy()

        for col in feature_cols:
            col_std = df[col].std()
            if not np.isfinite(col_std) or col_std == 0:
                col_std = 1.0
            samples[col] = samples[col] + rng.normal(0.0, noise_scale * col_std, size=len(samples))

        samples["trial_id"] = samples["trial_id"].astype(str) + "__aug"
        augmented.append(samples)

    return pd.concat(augmented, ignore_index=True)


# =========================
# Model 1: Posture Classifier
# =========================

def prepare_posture_dataset(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df = df[df["label_quality"] == "clean"].copy()
    df["train_class"] = df["label"].map(TRAIN_CLASS_MAP)
    df = df[df["train_class"].notna()].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df


def split_by_subject(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Person-based split.
    If fewer than 3 subjects, fallback to trial/window random split.
    """
    subjects = sorted(df["subject_id"].dropna().unique())
    if len(subjects) >= 3:
        train_subjects = subjects[:-2]
        val_subjects = [subjects[-2]]
        test_subjects = [subjects[-1]]
        train_df = df[df["subject_id"].isin(train_subjects)].copy()
        val_df = df[df["subject_id"].isin(val_subjects)].copy()
        test_df = df[df["subject_id"].isin(test_subjects)].copy()
        return train_df, val_df, test_df

    warnings.warn("Fewer than 3 subjects found. Falling back to random split; this may overestimate performance.")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=CONFIG.random_state, stratify=df["train_class"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=CONFIG.random_state, stratify=temp_df["train_class"])
    return train_df, val_df, test_df


def train_posture_classifier(features_df: pd.DataFrame, output_dir: str) -> None:
    ensure_dir(output_dir)
    df = prepare_posture_dataset(features_df)
    train_df, val_df, test_df = split_by_subject(df)

    meta_cols = {
        "trial_id", "subject_id", "label", "risk_label", "session_phase", "rpe",
        "load_level", "label_quality", "window_start_ms", "window_end_ms", "train_class",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    train_aug = augment_feature_table(
        train_df,
        target_col="train_class",
        target_per_class=CONFIG.augment_target_per_class,
        random_state=CONFIG.random_state,
    )

    X_train = train_aug[feature_cols].to_numpy(dtype=float)
    y_train = train_aug["train_class"].to_numpy()
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["train_class"].to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["train_class"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=CONFIG.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    for split_name, X, y in [
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        pred = model.predict(X)
        print(f"\n=== Posture classifier: {split_name} ===")
        print(classification_report(y, pred, zero_division=0))
        print(confusion_matrix(y, pred, labels=model.classes_))

        report = classification_report(y, pred, zero_division=0, output_dict=True)
        with open(Path(output_dir) / f"posture_{split_name}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    artifacts = {
        "model": model,
        "feature_cols": feature_cols,
        "classes": list(model.classes_),
        "config": CONFIG.__dict__,
    }
    joblib.dump(artifacts, Path(output_dir) / "posture_classifier.joblib")
    print(f"Saved posture classifier to {Path(output_dir) / 'posture_classifier.joblib'}")


# =========================
# Model 2: Gaussian Fatigue Proxy
# =========================

def prepare_fatigue_dataset(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df = df[df["label_quality"] == "clean"].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df


def train_gaussian_fatigue_model(features_df: pd.DataFrame, output_dir: str) -> None:
    ensure_dir(output_dir)
    df = prepare_fatigue_dataset(features_df)

    # Train only on fresh baseline.
    baseline_df = df[df["session_phase"] == "fresh_baseline"].copy()
    eval_df = df[df["session_phase"].isin(["fresh_baseline", "repeated_task", "fatigue_like", "recovery"])].copy()

    if len(baseline_df) < 10:
        raise ValueError("Not enough fresh_baseline windows for Gaussian fatigue model. Need at least ~10 windows.")

    meta_cols = {
        "trial_id", "subject_id", "label", "risk_label", "session_phase", "rpe",
        "load_level", "label_quality", "window_start_ms", "window_end_ms",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    scaler = StandardScaler()
    X_base = scaler.fit_transform(baseline_df[feature_cols].to_numpy(dtype=float))

    gaussian = EmpiricalCovariance().fit(X_base)

    def score(x_raw: np.ndarray) -> np.ndarray:
        x_scaled = scaler.transform(x_raw)
        # mahalanobis returns squared Mahalanobis distance.
        return gaussian.mahalanobis(x_scaled)

    X_eval = eval_df[feature_cols].to_numpy(dtype=float)
    scores = score(X_eval)
    eval_out = eval_df[["trial_id", "subject_id", "session_phase", "rpe", "label"]].copy()
    eval_out["fatigue_proxy_score"] = scores
    eval_out["fatigue_anomaly"] = scores > CONFIG.fatigue_anomaly_threshold
    eval_out.to_csv(Path(output_dir) / "fatigue_eval_scores.csv", index=False)

    print("\n=== Gaussian fatigue proxy score by session_phase ===")
    print(eval_out.groupby("session_phase")["fatigue_proxy_score"].describe())

    artifacts = {
        "scaler": scaler,
        "gaussian": gaussian,
        "feature_cols": feature_cols,
        "threshold": CONFIG.fatigue_anomaly_threshold,
        "config": CONFIG.__dict__,
    }
    joblib.dump(artifacts, Path(output_dir) / "gaussian_fatigue_proxy.joblib")
    print(f"Saved Gaussian fatigue proxy to {Path(output_dir) / 'gaussian_fatigue_proxy.joblib'}")


# =========================
# Inference examples
# =========================

def predict_posture_from_window(window_df: pd.DataFrame, model_path: str) -> Dict[str, object]:
    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]

    features = extract_posture_features(window_df, CONFIG.sampling_hz)
    x = pd.DataFrame([features]).reindex(columns=feature_cols).fillna(0.0)

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    return {
        "posture_class": pred,
        "confidence": float(np.max(proba)),
        "class_probabilities": dict(zip(model.classes_, map(float, proba))),
    }


def score_fatigue_from_window(window_df: pd.DataFrame, model_path: str) -> Dict[str, object]:
    artifacts = joblib.load(model_path)
    scaler = artifacts["scaler"]
    gaussian = artifacts["gaussian"]
    feature_cols = artifacts["feature_cols"]
    threshold = artifacts["threshold"]

    features = extract_fatigue_features(window_df, CONFIG.sampling_hz)
    x = pd.DataFrame([features]).reindex(columns=feature_cols).fillna(0.0)
    x_scaled = scaler.transform(x)
    score = float(gaussian.mahalanobis(x_scaled)[0])

    return {
        "fatigue_proxy_score": score,
        "fatigue_anomaly": score > threshold,
        "threshold": threshold,
    }


# =========================
# Main
# =========================

def main() -> None:
    ensure_dir(CONFIG.output_dir)

    print("Loading CSV dataset...")
    data = load_csv_folder(CONFIG.data_dir)

    # Optional: keep only clean/uncertain during EDA, but models use clean only.
    run_eda(data, CONFIG.output_dir)

    print("Creating posture windows...")
    posture_windows = make_windows(
        data,
        window_sec=CONFIG.posture_window_sec,
        overlap=CONFIG.posture_overlap,
        sampling_hz=CONFIG.sampling_hz,
    )
    posture_features = build_feature_table(posture_windows, extract_posture_features, CONFIG.sampling_hz)
    posture_features.to_csv(Path(CONFIG.output_dir) / "posture_features.csv", index=False)

    print("Creating fatigue windows...")
    fatigue_windows = make_windows(
        data,
        window_sec=CONFIG.fatigue_window_sec,
        overlap=CONFIG.fatigue_overlap,
        sampling_hz=CONFIG.sampling_hz,
    )
    fatigue_features = build_feature_table(fatigue_windows, extract_fatigue_features, CONFIG.sampling_hz)
    fatigue_features.to_csv(Path(CONFIG.output_dir) / "fatigue_features.csv", index=False)

    print("Training posture classifier...")
    train_posture_classifier(posture_features, CONFIG.output_dir)

    print("Training Gaussian fatigue proxy...")
    train_gaussian_fatigue_model(fatigue_features, CONFIG.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
