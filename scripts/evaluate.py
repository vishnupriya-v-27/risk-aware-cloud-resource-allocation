"""
Phase 10: Final 9-Configuration Benchmark Evaluation Script
Part of the 16-20 Hour Google Cluster Trace MVP (Hour 14.5-16).

Configurations Evaluated (Test Set: 30,300 points, 100 machines):
1. B0:  Naive Persistence (zero buffer)
2. B0b: Seasonal Naive (zero buffer)
3. B1:  Global GRU (zero buffer)
4. B2:  Holt-Winters (zero buffer)
5. B3:  Hybrid Fusion + Fixed Buffer (beta * mean(U_val))
6. B4a: Hybrid Fusion + U1 (Disagreement) Buffer (beta * U1)
7. B4b: Hybrid Fusion + U2 (Error EWMA) Buffer (beta * U2)
8. B4c: Hybrid Fusion + U3 (Volatility) Buffer (beta * U3)
9. P:   Proposed Adaptive Hybrid Fusion + U* Buffer (N*=75, beta*=1.0)

Outputs:
- results/final_results.csv
- results/final_results.json
- figures/results_table.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UNCERTAINTY_DIR = BASE_DIR / "uncertainty"
SELECTOR_DIR = BASE_DIR / "selector"
ALLOCATION_DIR = BASE_DIR / "allocation"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

NAIVE_CSV = MODELS_DIR / "naive_predictions.csv"
SNAIVE_CSV = MODELS_DIR / "seasonal_naive_predictions.csv"
HW_CSV = MODELS_DIR / "hw_predictions.csv"
GRU_CSV = MODELS_DIR / "gru_predictions.csv"
FUSION_CSV = MODELS_DIR / "fusion_predictions.csv"
UNCERTAINTY_CSV = UNCERTAINTY_DIR / "uncertainty.csv"
SELECTOR_CSV = SELECTOR_DIR / "selector_results.csv"
ALLOCATION_CSV = ALLOCATION_DIR / "allocation.csv"

FINAL_RESULTS_CSV = RESULTS_DIR / "final_results.csv"
FINAL_RESULTS_JSON = RESULTS_DIR / "final_results.json"
RESULTS_TABLE_FIG = FIGURES_DIR / "results_table.png"

DELTA_GRACE = 0.005

def compute_metrics(actual: np.ndarray, pred: np.ndarray, alloc: np.ndarray, delta: float = DELTA_GRACE):
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))

    diff = alloc - actual
    breach_mask = actual > (alloc + delta)
    sla_breach_rate = float(np.mean(breach_mask))
    overshoot_rate = float(np.mean(alloc > actual))
    grace_zone_rate = float(np.mean((alloc < actual) & (actual <= alloc + delta)))
    over_provisioning = float(np.mean(np.maximum(0.0, diff)))
    under_provisioning = float(np.mean(np.maximum(0.0, -diff)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "SLA_Breach_Rate": sla_breach_rate,
        "Overshoot_Rate": overshoot_rate,
        "Grace_Zone_Rate": grace_zone_rate,
        "Over_Provisioning": over_provisioning,
        "Under_Provisioning": under_provisioning
    }


def run_benchmark():
    print("=" * 70, flush=True)
    print(" Google Cluster Trace MVP — Final 9-Configuration Benchmark (Phase 10)", flush=True)
    print("=" * 70, flush=True)

    # Load selected beta from Phase 9 allocation metrics
    alloc_metrics_file = RESULTS_DIR / "allocation_metrics.json"
    if alloc_metrics_file.exists():
        with open(alloc_metrics_file) as f:
            alloc_meta = json.load(f)
            beta_star = float(alloc_meta.get("selected_beta", 3.0))
    else:
        beta_star = 3.0

    print(f"[*] Loaded calibrated beta* = {beta_star:.2f} from Phase 9 allocation metrics.", flush=True)

    # 1. Load test data across all models
    print("[*] Loading datasets and verifying alignment across 100 machines...", flush=True)
    
    df_naive = pd.read_csv(NAIVE_CSV)
    df_snaive = pd.read_csv(SNAIVE_CSV)
    df_hw = pd.read_csv(HW_CSV)
    df_gru = pd.read_csv(GRU_CSV)
    df_fusion = pd.read_csv(FUSION_CSV)
    df_unc = pd.read_csv(UNCERTAINTY_CSV)
    df_sel = pd.read_csv(SELECTOR_CSV)

    # Filter to test split
    t_naive = df_naive[df_naive["split"] == "test"].sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
    t_snaive = df_snaive[df_snaive["split"] == "test"].sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
    t_hw = df_hw[df_hw["split"] == "test"].sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
    t_gru = df_gru[df_gru["split"] == "test"].sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
    t_fusion = df_fusion[df_fusion["split"] == "test"].sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
    t_unc = df_unc[df_unc["split"] == "test"].sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
    t_sel = df_sel[df_sel["split"] == "test"].sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)

    # Verification of identical lengths and machines
    n_test = len(t_naive)
    assert n_test == 30300, f"Expected 30,300 test steps, got {n_test}"
    for name, df_check in [
        ("SNaive", t_snaive), ("HW", t_hw), ("GRU", t_gru),
        ("Fusion", t_fusion), ("Unc", t_unc), ("Sel", t_sel)
    ]:
        assert len(df_check) == 30300, f"{name} length mismatch: {len(df_check)}"
        assert (df_check["machine_id"].values == t_naive["machine_id"].values).all(), f"{name} machine_id alignment mismatch"
        assert (df_check["timestamp"].values == t_naive["timestamp"].values).all(), f"{name} timestamp alignment mismatch"

    print(f"[✓] Test Alignment Verified: Identical 30,300 timesteps across all 100 machines.")

    # Ground truth actual and individual predictions
    y_act = t_naive["actual"].values.astype(np.float64)
    pred_naive = t_naive["predicted" if "predicted" in t_naive.columns else "pred"].values.astype(np.float64)
    pred_snaive = t_snaive["predicted" if "predicted" in t_snaive.columns else "pred"].values.astype(np.float64)
    pred_hw = t_hw["predicted" if "predicted" in t_hw.columns else "pred"].values.astype(np.float64)
    pred_gru = t_gru["predicted" if "predicted" in t_gru.columns else "pred"].values.astype(np.float64)
    pred_fusion = t_fusion["pred" if "pred" in t_fusion.columns else "fused_pred"].values.astype(np.float64)

    # Uncertainty signals (raw and calibrated)
    u1_test = t_unc["u1"].values.astype(np.float64)
    u2_test = t_unc["u2"].values.astype(np.float64)
    u3_test = t_unc["u3"].values.astype(np.float64)
    u1_cal_test = t_unc["u1_cal"].values.astype(np.float64)
    u2_cal_test = t_unc["u2_cal"].values.astype(np.float64)
    u3_cal_test = t_unc["u3_cal"].values.astype(np.float64)

    u_star_raw_test = t_sel["selected_uncertainty"].values.astype(np.float64)
    u_star_cal_test = t_sel["selected_uncertainty_cal"].values.astype(np.float64)

    # B3 fixed buffer value: beta * mean(U_val)
    val_unc = df_unc[df_unc["split"] == "val"]
    val_mean_u_raw = float(val_unc[["u1", "u2", "u3"]].mean().mean())
    val_mean_u_cal = float(val_unc[["u1_cal", "u2_cal", "u3_cal"]].mean().mean())
    
    fixed_buffer_val = beta_star * val_mean_u_cal

    # 2. Build 9 Configurations (Method 1: Calibrated Scale)
    configs = [
        {
            "Config": "B0",
            "Name": "Naive (Persistence)",
            "Forecast_Model": "y[t-1]",
            "Buffer_Type": "Zero Buffer",
            "pred": pred_naive,
            "alloc": np.clip(pred_naive, 0.0, 1.0)
        },
        {
            "Config": "B0b",
            "Name": "Seasonal Naive",
            "Forecast_Model": "y[t-288] (24h)",
            "Buffer_Type": "Zero Buffer",
            "pred": pred_snaive,
            "alloc": np.clip(pred_snaive, 0.0, 1.0)
        },
        {
            "Config": "B1",
            "Name": "Global GRU",
            "Forecast_Model": "GRU (7 feat)",
            "Buffer_Type": "Zero Buffer",
            "pred": pred_gru,
            "alloc": np.clip(pred_gru, 0.0, 1.0)
        },
        {
            "Config": "B2",
            "Name": "Holt-Winters",
            "Forecast_Model": "HW (Tuned m)",
            "Buffer_Type": "Zero Buffer",
            "pred": pred_hw,
            "alloc": np.clip(pred_hw, 0.0, 1.0)
        },
        {
            "Config": "B3",
            "Name": "GRU + Fixed Mean Buffer",
            "Forecast_Model": "GRU-based",
            "Buffer_Type": f"Fixed Scalar ({fixed_buffer_val:.4f})",
            "pred": pred_fusion,
            "alloc": np.clip(pred_fusion + fixed_buffer_val, 0.0, 1.0)
        },
        {
            "Config": "B4a",
            "Name": "GRU + U1 Calibrated (Disagreement)",
            "Forecast_Model": "GRU + HW Sensor",
            "Buffer_Type": f"beta* U1_cal (beta={beta_star:.2f})",
            "pred": pred_fusion,
            "alloc": np.clip(pred_fusion + beta_star * u1_cal_test, 0.0, 1.0)
        },
        {
            "Config": "B4b",
            "Name": "GRU + U2 Calibrated (Error EWMA)",
            "Forecast_Model": "GRU-based",
            "Buffer_Type": f"beta* U2_cal (beta={beta_star:.2f})",
            "pred": pred_fusion,
            "alloc": np.clip(pred_fusion + beta_star * u2_cal_test, 0.0, 1.0)
        },
        {
            "Config": "B4c",
            "Name": "GRU + U3 Calibrated (Volatility)",
            "Forecast_Model": "GRU-based",
            "Buffer_Type": f"beta* U3_cal (beta={beta_star:.2f})",
            "pred": pred_fusion,
            "alloc": np.clip(pred_fusion + beta_star * u3_cal_test, 0.0, 1.0)
        },
        {
            "Config": "P",
            "Name": "Proposed Adaptive Calibrated",
            "Forecast_Model": "GRU-based (Adaptive)",
            "Buffer_Type": f"beta* U*_cal (N*=75, beta={beta_star:.2f})",
            "pred": pred_fusion,
            "alloc": np.clip(pred_fusion + beta_star * u_star_cal_test, 0.0, 1.0)
        }
    ]

    # 3. Compute Metrics for all 9 configurations
    results_rows = []
    print("\n[*] Evaluating All 9 Configurations on Test Set...", flush=True)

    for c in configs:
        m = compute_metrics(y_act, c["pred"], c["alloc"], DELTA_GRACE)
        row = {
            "Config": c["Config"],
            "Name": c["Name"],
            "Forecast_Model": c["Forecast_Model"],
            "Buffer_Type": c["Buffer_Type"],
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "SLA_Breach_Pct": m["SLA_Breach_Rate"] * 100.0,
            "Overshoot_Pct": m["Overshoot_Rate"] * 100.0,
            "Grace_Zone_Pct": m["Grace_Zone_Rate"] * 100.0,
            "Over_Provisioning": m["Over_Provisioning"],
            "Under_Provisioning": m["Under_Provisioning"]
        }
        results_rows.append(row)

    df_final = pd.DataFrame(results_rows)

    # -----------------------------------------------------------------
    # Multi-Beta Pareto Frontier Computation (Calibrated & Uncalibrated)
    # -----------------------------------------------------------------
    beta_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    pareto_data = []

    for b in beta_grid:
        # B3 Calibrated
        b3_alloc = np.clip(pred_fusion + b * val_mean_u_cal, 0.0, 1.0)
        m_b3 = compute_metrics(y_act, pred_fusion, b3_alloc, DELTA_GRACE)
        pareto_data.append({"Strategy": "B3 (Fixed Mean Cal)", "beta": b, "SLA_Breach_Pct": m_b3["SLA_Breach_Rate"] * 100.0, "Waste": m_b3["Over_Provisioning"], "Deficit": m_b3["Under_Provisioning"]})

        # B4a Calibrated (U1)
        b4a_alloc = np.clip(pred_fusion + b * u1_cal_test, 0.0, 1.0)
        m_b4a = compute_metrics(y_act, pred_fusion, b4a_alloc, DELTA_GRACE)
        pareto_data.append({"Strategy": "B4a (U1 Calibrated)", "beta": b, "SLA_Breach_Pct": m_b4a["SLA_Breach_Rate"] * 100.0, "Waste": m_b4a["Over_Provisioning"], "Deficit": m_b4a["Under_Provisioning"]})

        # B4b Calibrated (U2)
        b4b_alloc = np.clip(pred_fusion + b * u2_cal_test, 0.0, 1.0)
        m_b4b = compute_metrics(y_act, pred_fusion, b4b_alloc, DELTA_GRACE)
        pareto_data.append({"Strategy": "B4b (U2 Calibrated)", "beta": b, "SLA_Breach_Pct": m_b4b["SLA_Breach_Rate"] * 100.0, "Waste": m_b4b["Over_Provisioning"], "Deficit": m_b4b["Under_Provisioning"]})

        # B4c Calibrated (U3)
        b4c_alloc = np.clip(pred_fusion + b * u3_cal_test, 0.0, 1.0)
        m_b4c = compute_metrics(y_act, pred_fusion, b4c_alloc, DELTA_GRACE)
        pareto_data.append({"Strategy": "B4c (U3 Calibrated)", "beta": b, "SLA_Breach_Pct": m_b4c["SLA_Breach_Rate"] * 100.0, "Waste": m_b4c["Over_Provisioning"], "Deficit": m_b4c["Under_Provisioning"]})

        # Proposed Calibrated (U*_cal)
        p_alloc = np.clip(pred_fusion + b * u_star_cal_test, 0.0, 1.0)
        m_p = compute_metrics(y_act, pred_fusion, p_alloc, DELTA_GRACE)
        pareto_data.append({"Strategy": "P (Proposed Calibrated U*)", "beta": b, "SLA_Breach_Pct": m_p["SLA_Breach_Rate"] * 100.0, "Waste": m_p["Over_Provisioning"], "Deficit": m_p["Under_Provisioning"]})

        # Proposed Uncalibrated (Reference)
        p_raw_alloc = np.clip(pred_fusion + b * u_star_raw_test, 0.0, 1.0)
        m_p_raw = compute_metrics(y_act, pred_fusion, p_raw_alloc, DELTA_GRACE)
        pareto_data.append({"Strategy": "P_raw (Uncalibrated U*)", "beta": b, "SLA_Breach_Pct": m_p_raw["SLA_Breach_Rate"] * 100.0, "Waste": m_p_raw["Over_Provisioning"], "Deficit": m_p_raw["Under_Provisioning"]})

    df_pareto = pd.DataFrame(pareto_data)
    df_pareto.to_csv(RESULTS_DIR / "pareto_results.csv", index=False)

    # 4. Save Final CSV and JSON
    df_final.to_csv(FINAL_RESULTS_CSV, index=False)
    print(f"\n[+] Saved final benchmark results to {FINAL_RESULTS_CSV}", flush=True)

    json_export = {
        "test_timesteps": n_test,
        "num_machines": 100,
        "delta_grace": DELTA_GRACE,
        "beta_star": beta_star,
        "fixed_buffer_val_b3": fixed_buffer_val,
        "val_mean_u_cal": val_mean_u_cal,
        "val_mean_u_raw": val_mean_u_raw,
        "configurations": results_rows,
        "pareto_summary": df_pareto.to_dict(orient="records")
    }
    with open(FINAL_RESULTS_JSON, "w") as f:
        json.dump(json_export, f, indent=2)
    print(f"[+] Saved final benchmark results JSON to {FINAL_RESULTS_JSON}", flush=True)

    # -----------------------------------------------------------------
    # Render and Save figures/results_table.png and figures/pareto_analysis.png
    # -----------------------------------------------------------------
    generate_results_table_image(df_final)
    generate_pareto_image(df_pareto, beta_star)

    # Print clean formatted text table
    print("\n" + "=" * 115, flush=True)
    print(" FINAL 9-CONFIGURATION TEST BENCHMARK RESULTS (30,300 observations across 100 machines)")
    print("=" * 115, flush=True)
    print(f"{'Config':<7} | {'Name':<28} | {'MAE':<7} | {'RMSE':<7} | {'SLA Breach':<10} | {'Overshoot':<9} | {'Waste (Over)':<12} | {'Deficit (Under)':<15}")
    print("-" * 115, flush=True)
    for r in results_rows:
        print(f"{r['Config']:<7} | {r['Name']:<28} | {r['MAE']:<7.4f} | {r['RMSE']:<7.4f} | {r['SLA_Breach_Pct']:>8.2f}% | {r['Overshoot_Pct']:>7.2f}% | {r['Over_Provisioning']:>10.4f}  | {r['Under_Provisioning']:>13.4f}")
    print("=" * 115, flush=True)

    return df_final


def generate_pareto_image(df_pareto: pd.DataFrame, beta_star: float):
    """
    Renders SLA Breach % vs Over-Provisioning Waste Pareto curve.
    """
    PARETO_FIG = FIGURES_DIR / "pareto_analysis.png"
    plt.figure(figsize=(10, 6.5))

    colors = {
        "B3 (Fixed Mean Cal)": "#64748b",
        "B4a (U1 Calibrated)": "#ef4444",
        "B4b (U2 Calibrated)": "#f59e0b",
        "B4c (U3 Calibrated)": "#10b981",
        "P (Proposed Calibrated U*)": "#4f46e5",
        "P_raw (Uncalibrated U*)": "#94a3b8"
    }
    markers = {
        "B3 (Fixed Mean Cal)": "s",
        "B4a (U1 Calibrated)": "^",
        "B4b (U2 Calibrated)": "D",
        "B4c (U3 Calibrated)": "v",
        "P (Proposed Calibrated U*)": "o",
        "P_raw (Uncalibrated U*)": "x"
    }
    linestyles = {
        "B3 (Fixed Mean Cal)": "-",
        "B4a (U1 Calibrated)": "-",
        "B4b (U2 Calibrated)": "-",
        "B4c (U3 Calibrated)": "-",
        "P (Proposed Calibrated U*)": "-",
        "P_raw (Uncalibrated U*)": "--"
    }

    for strat in df_pareto["Strategy"].unique():
        sdf = df_pareto[df_pareto["Strategy"] == strat].sort_values(by="Waste")
        plt.plot(
            sdf["Waste"],
            sdf["SLA_Breach_Pct"],
            label=strat,
            color=colors.get(strat, "#000000"),
            marker=markers.get(strat, "o"),
            linestyle=linestyles.get(strat, "-"),
            linewidth=2.2 if "Proposed Calibrated" in strat else (1.5 if "P_raw" in strat else 1.6),
            markersize=6 if "Proposed" in strat else 5
        )

    plt.axhline(y=2.0, color="#dc2626", linestyle=":", alpha=0.8, linewidth=1.5, label="2% SLA Target (Constraint)")
    plt.xlabel("Over-Provisioning / Waste (cores per step)", fontsize=11, fontweight="bold")
    plt.ylabel("SLA Breach Rate (%)", fontsize=11, fontweight="bold")
    plt.title("Reliability vs Resource Waste Pareto Trade-off Across beta in [0.5, 3.0] (Calibrated vs Uncalibrated)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(PARETO_FIG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Pareto trade-off curve to {PARETO_FIG}", flush=True)


def generate_results_table_image(df: pd.DataFrame):
    """
    Renders publication-quality table figure saved to figures/results_table.png.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    ax.axis("tight")

    # Format table data
    table_data = []
    headers = [
        "Config", "Configuration Name", "Forecast Model", "Buffer Strategy",
        "MAE", "RMSE", "SLA Breach %", "Overshoot %", "Waste (Over)", "Deficit (Under)"
    ]

    for _, r in df.iterrows():
        table_data.append([
            r["Config"],
            r["Name"],
            r["Forecast_Model"],
            r["Buffer_Type"],
            f"{r['MAE']:.4f}",
            f"{r['RMSE']:.4f}",
            f"{r['SLA_Breach_Pct']:.2f}%",
            f"{r['Overshoot_Pct']:.2f}%",
            f"{r['Over_Provisioning']:.4f}",
            f"{r['Under_Provisioning']:.4f}"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header and rows
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1e293b")
        elif table_data[i - 1][0] == "P":
            # Highlight proposed configuration row
            cell.set_text_props(weight="bold", color="#1e1b4b")
            cell.set_facecolor("#e0e7ff")
        else:
            cell.set_facecolor("#f8fafc" if i % 2 == 0 else "#ffffff")

    plt.title("Google Cluster Trace MVP: Final 9-Configuration Evaluation Summary", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_TABLE_FIG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved results table figure to {RESULTS_TABLE_FIG}", flush=True)


if __name__ == "__main__":
    run_benchmark()
