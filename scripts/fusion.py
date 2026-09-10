"""
Hybrid Fusion Forecasting Script
Part of the 16-20 Hour Google Cluster Trace MVP (Phase 6 / Hour 9-10).

Specification:
- Formula: y_hat_fused = alpha * y_hat_HW + (1 - alpha) * y_hat_GRU
- Grid search alpha in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} on Validation MAE.
- Select alpha* minimizing Validation MAE.
- Generate and save fused predictions across all splits to models/fusion_predictions.csv.
- Save full forecasting results table to results/forecasting_results.csv.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

HW_CSV = MODELS_DIR / "hw_predictions.csv"
GRU_CSV = MODELS_DIR / "gru_predictions.csv"
NAIVE_CSV = MODELS_DIR / "naive_predictions.csv"
SEASONAL_NAIVE_CSV = MODELS_DIR / "seasonal_naive_predictions.csv"

FUSION_CSV = MODELS_DIR / "fusion_predictions.csv"
RESULTS_CSV = RESULTS_DIR / "forecasting_results.csv"
RESULTS_JSON = RESULTS_DIR / "forecasting_results.json"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def run_hybrid_fusion():
    print("=" * 60)
    print(" Google Cluster Trace MVP — Hybrid Fusion (Phase 6 / Hour 9–10)")
    print("=" * 60)

    # 1. Load predictions
    if not HW_CSV.exists() or not GRU_CSV.exists():
        raise FileNotFoundError(f"Missing {HW_CSV} or {GRU_CSV}.")

    print(f"[*] Loading HW predictions from {HW_CSV}...")
    df_hw = pd.read_csv(HW_CSV)
    # Standardize column name for predicted
    hw_pred_col = "pred" if "pred" in df_hw.columns else "predicted"
    df_hw.rename(columns={hw_pred_col: "hw_pred"}, inplace=True)

    print(f"[*] Loading GRU predictions from {GRU_CSV}...")
    df_gru = pd.read_csv(GRU_CSV)
    gru_pred_col = "pred" if "pred" in df_gru.columns else "predicted"
    df_gru.rename(columns={gru_pred_col: "gru_pred"}, inplace=True)

    # Merge on (machine_id, timestamp, split)
    df_merged = pd.merge(
        df_hw[["machine_id", "timestamp", "split", "actual", "hw_pred"]],
        df_gru[["machine_id", "timestamp", "split", "gru_pred"]],
        on=["machine_id", "timestamp", "split"],
        how="inner"
    )
    df_merged.sort_values(by=["machine_id", "timestamp"], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    print(f"[+] Total aligned records: {len(df_merged):,} across {df_merged['machine_id'].nunique()} machines.")

    # 2. Grid search alpha on Validation Set
    df_val = df_merged[df_merged["split"] == "val"].copy()
    y_val_act = df_val["actual"].values
    y_val_hw = df_val["hw_pred"].values
    y_val_gru = df_val["gru_pred"].values

    alpha_candidates = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]
    grid_results = []

    print("\n[*] Grid Search on Validation Set (alpha in [0.0, 1.0]):")
    best_alpha = 0.0
    best_val_mae = float("inf")
    best_val_rmse = float("inf")

    for alpha in alpha_candidates:
        fused_val = alpha * y_val_hw + (1.0 - alpha) * y_val_gru
        fused_val = np.clip(fused_val, 0.0, 1.0)
        mae, rmse = calculate_metrics(y_val_act, fused_val)
        grid_results.append({"alpha": alpha, "val_mae": mae, "val_rmse": rmse})
        print(f"    alpha = {alpha:.1f} | Val MAE = {mae:.5f} | Val RMSE = {rmse:.5f}")

        if mae < best_val_mae:
            best_val_mae = mae
            best_val_rmse = rmse
            best_alpha = alpha

    print(f"\n[+] Selected Optimal alpha* = {best_alpha:.2f} (Val MAE = {best_val_mae:.5f})")

    # 3. Generate Fused Predictions for all splits using best alpha
    df_merged["fused_pred"] = np.clip(
        best_alpha * df_merged["hw_pred"].values + (1.0 - best_alpha) * df_merged["gru_pred"].values,
        0.0,
        1.0
    )

    # Save models/fusion_predictions.csv
    df_fusion_out = df_merged[["machine_id", "timestamp", "split", "actual", "hw_pred", "gru_pred", "fused_pred"]].copy()
    df_fusion_out.rename(columns={"fused_pred": "pred"}, inplace=True)
    df_fusion_out.to_csv(FUSION_CSV, index=False)
    print(f"[+] Saved fused predictions to {FUSION_CSV}")

    # 4. Evaluate on Test Set
    df_test = df_merged[df_merged["split"] == "test"]
    y_test_act = df_test["actual"].values
    y_test_fused = df_test["fused_pred"].values
    y_test_hw = df_test["hw_pred"].values
    y_test_gru = df_test["gru_pred"].values

    test_mae_fused, test_rmse_fused = calculate_metrics(y_test_act, y_test_fused)
    test_mae_hw, test_rmse_hw = calculate_metrics(y_test_act, y_test_hw)
    test_mae_gru, test_rmse_gru = calculate_metrics(y_test_act, y_test_gru)

    # Load naive baselines for complete summary
    df_naive = pd.read_csv(NAIVE_CSV)
    df_snaive = pd.read_csv(SEASONAL_NAIVE_CSV)
    naive_pred_col = "pred" if "pred" in df_naive.columns else "predicted"
    snaive_pred_col = "pred" if "pred" in df_snaive.columns else "predicted"

    df_naive_test = df_naive[df_naive["split"] == "test"]
    df_snaive_test = df_snaive[df_snaive["split"] == "test"]

    test_mae_naive, test_rmse_naive = calculate_metrics(df_naive_test["actual"].values, df_naive_test[naive_pred_col].values)
    test_mae_snaive, test_rmse_snaive = calculate_metrics(df_snaive_test["actual"].values, df_snaive_test[snaive_pred_col].values)

    df_naive_val = df_naive[df_naive["split"] == "val"]
    df_snaive_val = df_snaive[df_snaive["split"] == "val"]
    val_mae_naive, val_rmse_naive = calculate_metrics(df_naive_val["actual"].values, df_naive_val[naive_pred_col].values)
    val_mae_snaive, val_rmse_snaive = calculate_metrics(df_snaive_val["actual"].values, df_snaive_val[snaive_pred_col].values)

    val_mae_hw, val_rmse_hw = calculate_metrics(y_val_act, y_val_hw)
    val_mae_gru, val_rmse_gru = calculate_metrics(y_val_act, y_val_gru)

    # 5. Build Final Comparison Table
    results_summary = [
        {
            "Config": "B0 (Naive Persistence)",
            "Model": "y[t-1]",
            "Val_MAE": val_mae_naive,
            "Val_RMSE": val_rmse_naive,
            "Test_MAE": test_mae_naive,
            "Test_RMSE": test_rmse_naive
        },
        {
            "Config": "B0b (Seasonal Naive)",
            "Model": "y[t-288] (24h)",
            "Val_MAE": val_mae_snaive,
            "Val_RMSE": val_rmse_snaive,
            "Test_MAE": test_mae_snaive,
            "Test_RMSE": test_rmse_snaive
        },
        {
            "Config": "B2 (Holt-Winters)",
            "Model": "HW (m in {48,144,288})",
            "Val_MAE": val_mae_hw,
            "Val_RMSE": val_rmse_hw,
            "Test_MAE": test_mae_hw,
            "Test_RMSE": test_rmse_hw
        },
        {
            "Config": "B1 (Global GRU)",
            "Model": "GRU(7 feat, hidden=32)",
            "Val_MAE": val_mae_gru,
            "Val_RMSE": val_rmse_gru,
            "Test_MAE": test_mae_gru,
            "Test_RMSE": test_rmse_gru
        },
        {
            "Config": "Fusion (HW + GRU)",
            "Model": f"alpha*={best_alpha:.2f}",
            "Val_MAE": best_val_mae,
            "Val_RMSE": best_val_rmse,
            "Test_MAE": test_mae_fused,
            "Test_RMSE": test_rmse_fused
        }
    ]

    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"[+] Saved forecasting comparison table to {RESULTS_CSV}")

    # JSON export
    full_export = {
        "best_alpha": best_alpha,
        "grid_search_val": grid_results,
        "models": results_summary
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(full_export, f, indent=2)

    print("\n" + "=" * 80)
    print(" COMPLETE FORECASTING EVALUATION (VAL & TEST ACROSS 100 MACHINES)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    return full_export


if __name__ == "__main__":
    run_hybrid_fusion()
