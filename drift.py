"""Monitoreo de drift: PSI (poblacional) y Kolmogorov-Smirnov por feature."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index usando buckets por cuantiles de la referencia."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    qs = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    eps = 1e-6
    exp_counts, _ = np.histogram(expected, bins=qs)
    act_counts, _ = np.histogram(actual, bins=qs)
    exp_pct = np.clip(exp_counts / max(len(expected), 1), eps, None)
    act_pct = np.clip(act_counts / max(len(actual), 1), eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                         psi_warning: float, psi_critical: float,
                         ks_pvalue_warning: float) -> dict:
    """Reporte de drift por feature + banderas globales."""
    rows = []
    for col in reference.columns:
        if not pd.api.types.is_numeric_dtype(reference[col]):
            continue
        ref = reference[col].dropna().values
        cur = current[col].dropna().values
        if len(ref) < 20 or len(cur) < 5:
            continue
        p = psi(ref, cur)
        ks_stat, ks_p = ks_2samp(ref, cur)
        rows.append({
            "feature": col,
            "psi": round(p, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 6),
            "alert": "CRITICAL" if p >= psi_critical else
                     "WARNING" if p >= psi_warning or ks_p < ks_pvalue_warning
                     else "OK",
        })
    if not rows:
        return {"features": [], "status": "OK", "n_alerts": 0}
    worst = max(rows, key=lambda r: r["psi"])
    status = "CRITICAL" if any(r["alert"] == "CRITICAL" for r in rows) else              "WARNING" if any(r["alert"] == "WARNING" for r in rows) else "OK"
    return {"features": rows, "status": status, "n_alerts": sum(r["alert"] != "OK" for r in rows),
            "worst_feature": worst["feature"], "worst_psi": worst["psi"]}
