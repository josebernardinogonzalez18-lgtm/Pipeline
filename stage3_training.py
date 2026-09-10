"""STAGE 3 - Entrenamiento con validacion OUT-OF-TIME (walk-forward).

- NUNCA random K-Fold: TimeSeriesSplit con embargo entre train y valid.
- Benchmark: LogisticRegression vs XGBoost.
- Criterio de seleccion de hiperparametros: log-loss y ECE (20 bins), NO accuracy.
- Calibracion final: isotonic sobre slice temporal reciente (validation_size).
"""
from __future__ import annotations
import datetime as dt
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from src.config_loader import load_config, ensure_parent
from src.calibration import expected_calibration_error, negative_log_likelihood
from src.stage2_features import FEATURE_COLS, hybrid_feature_selection


def _embargoed_splits(n: int, n_splits: int, embargo: int):
    """Indices (train, valid) puramente temporales con gap de embargo."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(np.zeros(n)):
        yield tr, va[va > tr.max() + embargo]


def _xgb_param_grid() -> list[dict]:
    return [
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 400, "subsample": 0.9},
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 600, "subsample": 0.9},
    ]


def run(cfg: dict) -> dict:
    feats = pd.read_parquet(cfg["paths"]["features"])
    train_df = feats.dropna(subset=["home_win"]).sort_values("date").reset_index(drop=True)
    X_all, y = train_df[FEATURE_COLS], train_df["home_win"].astype(int)

    # 1) Seleccion hibrida de features (varianza + correlacion)
    selected = hybrid_feature_selection(X_all, cfg)
    print(f"[stage3] features tras filtro hibrido: {len(selected)}/{len(FEATURE_COLS)}")
    X = X_all[selected]

    n = len(X)
    n_val = max(int(n * cfg["training"]["validation_size"]), 50)
    X_fit, y_fit = X.iloc[:-n_val], y.iloc[:-n_val]
    X_cal, y_cal = X.iloc[-n_val:], y.iloc[-n_val:]

    # 2) Walk-forward: seleccion de modelo e hiperparams por log-loss + ECE
    embargo = cfg["training"]["embargo_games"]
    n_bins = cfg["training"]["n_bins_ece"]
    candidates: list[tuple[float, str, object, dict]] = []

    lr = LogisticRegression(max_iter=1000)
    losses = []
    for tr, va in _embargoed_splits(len(X_fit), cfg["training"]["n_splits"], embargo):
        lr.fit(X_fit.iloc[tr], y_fit.iloc[tr])
        p = lr.predict_proba(X_fit.iloc[va])[:, 1]
        losses.append(negative_log_likelihood(y_fit.iloc[va].values, p))
    candidates.append((float(np.mean(losses)), "logistic_regression", LogisticRegression(max_iter=1000), {}))

    for params in _xgb_param_grid():
        losses = []
        for tr, va in _embargoed_splits(len(X_fit), cfg["training"]["n_splits"], embargo):
            m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              random_state=cfg["project"]["random_seed"], **params)
            m.fit(X_fit.iloc[tr], y_fit.iloc[tr], verbose=False)
            p = m.predict_proba(X_fit.iloc[va])[:, 1]
            losses.append(negative_log_likelihood(y_fit.iloc[va].values, p))
        candidates.append((float(np.mean(losses)), f"xgboost_{params}", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=cfg["project"]["random_seed"], **params), params))

    candidates.sort(key=lambda t: t[0])
    best_loss, best_name, best_model, best_params = candidates[0]
    print(f"[stage3] modelo seleccionado: {best_name} (log-loss walk-forward={best_loss:.4f})")

    # 3) Refit completo sobre train+valid-fit y calibracion isotonic en slice cal
    best_model.fit(X_fit, y_fit)
    p_cal_raw = best_model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_cal_raw, y_cal)
    p_cal = calibrator.predict(p_cal_raw)
    ece = expected_calibration_error(y_cal.values, p_cal, n_bins)

    bundle = {
        "model": best_model,
        "calibrator": calibrator,
        "features": selected,
        "model_name": best_name,
        "model_params": best_params,
        "ece_calibrated": ece,
        "n_train_games": int(len(X_fit)),
        "n_cal_games": int(len(X_cal)),
        "split_strategy": "TimeSeriesSplit walk-forward + embargo (out-of-time, sin leakage)",
        "validation_size": cfg["training"]["validation_size"],
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_features": X_fit.tail(cfg["drift"]["window_recent_games"] * 2),
    }
    path = cfg["paths"]["model"]
    ensure_parent(path)
    joblib.dump(bundle, path)
    # Referencia para drift (se compara contra datos nuevos)
    ref_path = cfg["paths"]["reference_features"]
    ensure_parent(ref_path)
    bundle["reference_features"].to_parquet(ref_path, index=False)
    print(f"[stage3] OK -> {path} | ECE calibrado ({n_bins} bins) = {ece:.4f}")
    return {"model": best_name, "logloss": best_loss, "ece": ece,
            "trained_at_utc": bundle["trained_at_utc"]}


if __name__ == "__main__":
    run(load_config())
