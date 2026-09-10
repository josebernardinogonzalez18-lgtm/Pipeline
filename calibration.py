"""Calibracion de probabilidades: ECE (20 bins) y curva de confiabilidad."""
from __future__ import annotations
import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 20) -> float:
    """ECE con bins uniformes en [0,1]. ECE = sum( (|B_m|/N) * |acc_m - conf_m| )."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # bin derecho cerrado para capturar p = 1.0
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_curve_data(y_true: np.ndarray, y_prob: np.ndarray,
                           n_bins: int = 20) -> list[dict]:
    """Datos para el calibration plot: confianza media vs accuracy por bin."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        rows.append({
            "bin": b,
            "bin_lower": round(float(bins[b]), 4),
            "bin_upper": round(float(bins[b + 1]), 4),
            "n": int(mask.sum()),
            "mean_confidence": round(float(y_prob[mask].mean()), 4) if mask.any() else None,
            "empirical_accuracy": round(float(y_true[mask].mean()), 4) if mask.any() else None,
        })
    return rows


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray,
                            eps: float = 1e-15) -> float:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
