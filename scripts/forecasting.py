"""
Forecasting Baselines and Holt-Winters Script
Part of the 16-20 Hour Google Cluster Trace MVP (Hour 6-7).

Models Implemented:
1. Baseline B0 (Persistence / Naive): y_hat[t] = y[t-1] across continuous boundary.
2. Baseline B0b (Seasonal Naive): y_hat[t] = y[t-288] (24h period) across continuous boundary.
3. Baseline B2 (Holt-Winters / Exponential Smoothing):
   - Per-machine seasonal period grid search: m in {48, 144, 288} (4h, 12h, 24h).
   - Tuned on Validation MAE.
   - Refit best configuration on (Train + Val) -> Predict on Test.
   - Predictions clipped to [0.0, 1.0].
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress statsmodels convergence warnings during grid search
warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
INPUT_PARQUET = DATA_DIR / "google_features_5min.parquet"

NAIVE_CSV = MODELS_DIR / "naive_predictions.csv"
SEASONAL_NAIVE_CSV = MODELS_DIR / "seasonal_naive_predictions.csv"
HW_CSV = MODELS_DIR / "hw_predictions.csv"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def fit_and_eval_hw(train_series: np.ndarray, val_series: np.ndarray, m: int):
    """
    Fit Holt-Winters on train_series with seasonal period m and evaluate on val_series.
    """
    try:
        if len(train_series) < 2 * m:
            # If train length < 2*m, fallback to smaller period or simple additive
            m_use = max(12, len(train_series) // 3)
        else:
            m_use = m

        model = ExponentialSmoothing(
            train_series,
            seasonal_periods=m_use,
            trend="add",
            seasonal="add",
            initialization_method="estimated"
        ).fit(optimized=True)
        val_pred = model.forecast(len(val_series))
        val_pred = np.clip(val_pred, 0.0, 1.0)
        mae, _ = calculate_metrics(val_series, val_pred)
        return mae, model
    except Exception:
        # Fallback to simple exponential smoothing if additive seasonal fails
        try:
            model = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            val_pred = model.forecast(len(val_series))
            val_pred = np.clip(val_pred, 0.0, 1.0)
            mae, _ = calculate_metrics(val_series, val_pred)
            return mae, model
        except Exception:
            # Extreme fallback to mean
            val_pred = np.full(len(val_series), np.mean(train_series))
            mae, _ = calculate_metrics(val_series, val_pred)
            return mae, None


def run_baselines_and_hw():
    print("=" * 60)
    print(" Google Cluster Trace MVP — Baselines & Holt-Winters (Hour 6–7)")
    print("=" * 60)

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input file {INPUT_PARQUET} not found. Run features.py first!")

    print(f"[*] Loading dataset from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    df.sort_values(by=["machine_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    machines = df["machine_id"].unique()
    print(f"[+] Loaded {len(df):,} rows across {len(machines)} machines.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    naive_rows = []
    seasonal_naive_rows = []
    hw_rows = []

    m_candidates = [48, 144, 288]  # 4h, 12h, 24h seasonality
    hw_best_m_counts = {m: 0 for m in m_candidates}

    print(f"[*] Processing 100 machines (B0 Naive, B0b Seasonal Naive, B2 Holt-Winters)...")

    for idx, m_id in enumerate(machines):
        m_df = df[df["machine_id"] == m_id].copy().reset_index(drop=True)
        
        y = m_df["cpu"].values
        timestamps = m_df["timestamp"].values
        splits = m_df["split"].values

        train_mask = (splits == "train")
        val_mask = (splits == "val")
        test_mask = (splits == "test")

        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        # -------------------------------------------------------------
        # 1. Baseline B0: Naive Persistence (y_hat[t] = y[t-1])
        # Continuous series indexing ensures test[0] uses val[-1]
        # -------------------------------------------------------------
        naive_pred_all = np.zeros_like(y)
        naive_pred_all[0] = y[0]
        naive_pred_all[1:] = y[:-1]

        # -------------------------------------------------------------
        # 2. Baseline B0b: Seasonal Naive (y_hat[t] = y[t-288])
        # 288 * 5 min = 24 hours. Fallback to y[t-1] if t < 288
        # -------------------------------------------------------------
        seasonal_pred_all = np.zeros_like(y)
        for t in range(len(y)):
            if t >= 288:
                seasonal_pred_all[t] = y[t - 288]
            else:
                seasonal_pred_all[t] = y[t - 1] if t > 0 else y[0]

        # -------------------------------------------------------------
        # 3. Baseline B2: Holt-Winters with Validation Grid Search
        # -------------------------------------------------------------
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]

        best_m = m_candidates[0]
        best_val_mae = float("inf")

        for m in m_candidates:
            val_mae, _ = fit_and_eval_hw(y_train, y_val, m)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_m = m

        hw_best_m_counts[best_m] += 1

        # Refit best HW model on (Train + Val)
        y_train_val = np.concatenate([y_train, y_val])
        try:
            m_use = best_m if len(y_train_val) >= 2 * best_m else max(12, len(y_train_val) // 3)
            refit_model = ExponentialSmoothing(
                y_train_val,
                seasonal_periods=m_use,
                trend="add",
                seasonal="add",
                initialization_method="estimated"
            ).fit(optimized=True)
            hw_test_pred = np.clip(refit_model.forecast(len(y_test)), 0.0, 1.0)
        except Exception:
            # Fallback
            try:
                refit_model = ExponentialSmoothing(
                    y_train_val,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated"
                ).fit(optimized=True)
                hw_test_pred = np.clip(refit_model.forecast(len(y_test)), 0.0, 1.0)
            except Exception:
                hw_test_pred = np.full(len(y_test), np.mean(y_train_val))

        # HW predictions for val: generate using train-only model
        _, val_model = fit_and_eval_hw(y_train, y_val, best_m)
        if val_model is not None:
            try:
                hw_val_pred = np.clip(val_model.forecast(len(y_val)), 0.0, 1.0)
            except Exception:
                hw_val_pred = np.full(len(y_val), np.mean(y_train))
        else:
            hw_val_pred = np.full(len(y_val), np.mean(y_train))

        # Store predictions
        for t_idx in range(len(y)):
            sp = splits[t_idx]
            ts = timestamps[t_idx]
            act = y[t_idx]

            # Naive
            naive_rows.append({
                "machine_id": m_id,
                "timestamp": ts,
                "split": sp,
                "actual": act,
                "predicted": naive_pred_all[t_idx]
            })

            # Seasonal Naive
            seasonal_naive_rows.append({
                "machine_id": m_id,
                "timestamp": ts,
                "split": sp,
                "actual": act,
                "predicted": seasonal_pred_all[t_idx]
            })

        # HW Val rows
        for i, val_pos in enumerate(val_indices):
            hw_rows.append({
                "machine_id": m_id,
                "timestamp": timestamps[val_pos],
                "split": "val",
                "actual": y[val_pos],
                "predicted": hw_val_pred[i],
                "best_m": best_m
            })

        # HW Test rows
        for i, test_pos in enumerate(test_indices):
            hw_rows.append({
                "machine_id": m_id,
                "timestamp": timestamps[test_pos],
                "split": "test",
                "actual": y[test_pos],
                "predicted": hw_test_pred[i],
                "best_m": best_m
            })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(machines):
            print(f"    Progress: {idx + 1}/{len(machines)} machines completed...")

    # Convert to DataFrames and save CSVs
    df_naive = pd.DataFrame(naive_rows)
    df_seasonal = pd.DataFrame(seasonal_naive_rows)
    df_hw = pd.DataFrame(hw_rows)

    df_naive.to_csv(NAIVE_CSV, index=False)
    df_seasonal.to_csv(SEASONAL_NAIVE_CSV, index=False)
    df_hw.to_csv(HW_CSV, index=False)

    print(f"\n[+] Saved Naive predictions to {NAIVE_CSV}")
    print(f"[+] Saved Seasonal Naive predictions to {SEASONAL_NAIVE_CSV}")
    print(f"[+] Saved Holt-Winters predictions to {HW_CSV}")
    print(f"[+] Holt-Winters Seasonality Distribution across 100 machines: {hw_best_m_counts}")

    # Evaluate on Test Set
    df_naive_test = df_naive[df_naive["split"] == "test"]
    df_seasonal_test = df_seasonal[df_seasonal["split"] == "test"]
    df_hw_test = df_hw[df_hw["split"] == "test"]

    mae_naive, rmse_naive = calculate_metrics(df_naive_test["actual"].values, df_naive_test["predicted"].values)
    mae_seasonal, rmse_seasonal = calculate_metrics(df_seasonal_test["actual"].values, df_seasonal_test["predicted"].values)
    mae_hw, rmse_hw = calculate_metrics(df_hw_test["actual"].values, df_hw_test["predicted"].values)

    print("\n" + "=" * 60)
    print(" BASELINE & HOLT-WINTERS TEST PERFORMANCE SUMMARY (30,300 test points)")
    print("=" * 60)
    print(f"  B0  Naive (Persistence):     MAE = {mae_naive:.4f} | RMSE = {rmse_naive:.4f}")
    print(f"  B0b Seasonal Naive (24h):   MAE = {mae_seasonal:.4f} | RMSE = {rmse_seasonal:.4f}")
    print(f"  B2  Holt-Winters (Tuned m):  MAE = {mae_hw:.4f} | RMSE = {rmse_hw:.4f}")
    print("=" * 60)

    # Save summary metrics
    summary_metrics = {
        "B0_Naive": {"MAE": mae_naive, "RMSE": rmse_naive},
        "B0b_Seasonal_Naive": {"MAE": mae_seasonal, "RMSE": rmse_seasonal},
        "B2_Holt_Winters": {"MAE": mae_hw, "RMSE": rmse_hw},
        "HW_seasonality_distribution": hw_best_m_counts
    }
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    return summary_metrics


if __name__ == "__main__":
    run_baselines_and_hw()
