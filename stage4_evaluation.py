"""STAGE 4 - Evaluacion, auditoria de calibracion (ECE 20 bins) y drift.

- ECE con EXACTAMENTE 20 bins uniformes + curva de confiabilidad (PNG + CSV).
- SHAP para verificar influencia de los sabermetrics de primer nivel.
- Drift: PSI + KS por feature (data drift) y caida de log-loss reciente
  (concept drift) -> banderas WARNING / CRITICAL que disparan re-entrenamiento.
"""
from __future__ import annotations
import datetime as dt
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config_loader import load_config, ensure_parent
from src.calibration import expected_calibration_error, reliability_curve_data, negative_log_likelihood
from src.drift import feature_drift_report


def _plot_reliability(rows: list[dict], ece: float, path: str) -> None:
    xs = [r["mean_confidence"] for r in rows if r["n"] > 0]
    ys = [r["empirical_accuracy"] for r in rows if r["n"] > 0]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(xs, ys, "o-", label=f"Modelo (ECE={ece:.4f})")
    plt.xlabel("Confianza media (20 bins)"); plt.ylabel("Accuracy empirica")
    plt.title("Curva de calibracion - validation out-of-time")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def _shap_summary(bundle: dict, X: pd.DataFrame, path_txt: str) -> str:
    """Verificacion SHAP: ranking de importancia. Devuelve top-5 como texto."""
    try:
        import shap
        model = bundle["model"]
        if bundle["model_name"].startswith("xgboost"):
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(lambda d: model.predict_proba(d)[:, 1], X)
            vals = explainer(X).values
        mean_abs = np.abs(vals).mean(axis=0)
        ranking = sorted(zip(X.columns, mean_abs), key=lambda t: -t[1])
        lines = [f"{i+1}. {c}: {v:.4f}" for i, (c, v) in enumerate(ranking[:10])]
        with open(path_txt, "w", encoding="utf-8") as fh:
            fh.write("SHAP mean(|value|) - top 10\n" + "\n".join(lines))
        return ", ".join(c for c, _ in ranking[:5])
    except Exception as exc:  # shap puede fallar en CI -> no romper el pipeline
        return f"SHAP no disponible ({type(exc).__name__})"


def run(cfg: dict) -> dict:
    bundle = joblib.load(cfg["paths"]["model"])
    feats = pd.read_parquet(cfg["paths"]["features"])
    df = feats.dropna(subset=["home_win"]).sort_values("date")
    X, y = df[bundle["features"]], df["home_win"].astype(int)
    n_bins = cfg["training"]["n_bins_ece"]

    # Predicciones calibradas out-of-time (slice de calibracion guardado implicito:
    # recalculamos con embargo = ultimo validation_size, sin tocar datos futuros)
    n_val = max(int(len(X) * cfg["training"]["validation_size"]), 50)
    raw = bundle["model"].predict_proba(X.iloc[-n_val:])[:, 1]
    p_cal = bundle["calibrator"].predict(raw)
    y_val = y.iloc[-n_val:].values

    ece = expected_calibration_error(y_val, p_cal, n_bins)
    brier = float(np.mean((p_cal - y_val) ** 2))
    nll = negative_log_likelihood(y_val, p_cal)
    acc = float((p_cal.round() == y_val).mean())

    # Concept drift: log-loss en ventana reciente vs ventana completa de validacion
    w = min(cfg["drift"]["window_recent_games"], len(y_val) // 3)
    ll_recent = negative_log_likelihood(y_val[-w:], p_cal[-w:])
    concept_drift = "CRITICAL" if ll_recent - nll > cfg["drift"]["retrain_accuracy_drop"] * 2 else                     "WARNING" if ll_recent - nll > cfg["drift"]["retrain_accuracy_drop"] else "OK"

    # Data drift vs referencia de entrenamiento
    ref = pd.read_parquet(cfg["paths"]["reference_features"])
    cur = X.iloc[-w:]
    drift_rep = feature_drift_report(ref[bundle["features"]], cur,
                                     cfg["drift"]["psi_warning"], cfg["drift"]["psi_critical"],
                                     cfg["drift"]["ks_pvalue_warning"])
    data_status = drift_rep["status"]
    overall = "CRITICAL" if "CRITICAL" in (concept_drift, data_status) else               "WARNING" if "WARNING" in (concept_drift, data_status) else "OK"
    retrain_triggered = overall == "CRITICAL" or (concept_drift == "WARNING" and data_status == "WARNING")

    # Artefactos
    rows = reliability_curve_data(y_val, p_cal, n_bins)
    rel_csv = cfg["paths"]["calibration_plot"].replace(".png", "_bins.csv")
    ensure_parent(rel_csv)
    pd.DataFrame(rows).to_csv(rel_csv, index=False)
    ensure_parent(cfg["paths"]["calibration_plot"])
    _plot_reliability(rows, ece, cfg["paths"]["calibration_plot"])
    top5 = _shap_summary(bundle, X.iloc[-n_val:], "reports/shap_top10.txt")

    report = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "out_of_time_validation": {"n_games": int(n_val), "split": "ultimo 20% temporal",
                                   "strategy": bundle["split_strategy"]},
        "metrics": {"ece_20bins": round(ece, 4), "brier": round(brier, 4),
                    "log_loss": round(nll, 4), "accuracy": round(acc, 4),
                    "log_loss_recent_window": round(ll_recent, 4)},
        "calibration_plot": cfg["paths"]["calibration_plot"],
        "reliability_bins_csv": rel_csv,
        "shap_top5": top5,
        "drift": {"status": overall, "data_drift": data_status,
                  "concept_drift": concept_drift,
                  "retrain_triggered": bool(retrain_triggered),
                  "features": drift_rep["features"]},
    }
    ensure_parent(cfg["paths"]["drift_report"])
    with open(cfg["paths"]["drift_report"], "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[stage4] ECE({n_bins})={ece:.4f} acc={acc:.3f} | drift={overall} "
          f"(data={data_status}, concept={concept_drift}) retrain={retrain_triggered}")
    return report


if __name__ == "__main__":
    run(load_config())
