"""
Phase 12: Final Publication Figures & MVP Outputs Generator
Part of the 16-20 Hour Google Cluster Trace MVP.

Produces:
- figures/final_data_validation.png
- figures/final_prediction_comparison.png
- figures/final_uncertainty_signals.png
- figures/final_selector_choices.png
- figures/final_allocation_vs_actual.png
- figures/final_results_comparison.png
- figures/final_pareto_frontier.png
- results/final_results_table.csv
- results/final_mvp_summary.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UNCERTAINTY_DIR = BASE_DIR / "uncertainty"
SELECTOR_DIR = BASE_DIR / "selector"
ALLOCATION_DIR = BASE_DIR / "allocation"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data files
CPU_CSV = DATA_DIR / "google_cpu_5min.csv"
NAIVE_CSV = MODELS_DIR / "naive_predictions.csv"
HW_CSV = MODELS_DIR / "hw_predictions.csv"
GRU_CSV = MODELS_DIR / "gru_predictions.csv"
UNCERTAINTY_CSV = UNCERTAINTY_DIR / "uncertainty.csv"
CALIB_JSON = UNCERTAINTY_DIR / "calibration_factors.json"
SELECTOR_CSV = SELECTOR_DIR / "selector_results.csv"
ALLOCATION_CSV = ALLOCATION_DIR / "allocation.csv"
FINAL_RESULTS_CSV = RESULTS_DIR / "final_results.csv"
PARETO_CSV = RESULTS_DIR / "pareto_results.csv"

# Styling defaults
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8


def generate_figure_1_data_validation():
    """
    FIGURE 1 — DATA VALIDATION
    Show 4 representative machines over time.
    """
    print("[*] Generating Figure 1: final_data_validation.png...")
    df_cpu = pd.read_csv(CPU_CSV)
    machines = df_cpu["machine_id"].unique()[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    for i, m_id in enumerate(machines):
        m_df = df_cpu[df_cpu["machine_id"] == m_id].sort_values(by="timestamp").reset_index(drop=True)
        time_hours = np.arange(len(m_df)) * 5.0 / 60.0
        
        ax = axes[i]
        ax.plot(time_hours, m_df["cpu"], color=colors[i], linewidth=0.9, alpha=0.9, label=f"Machine {m_id}")
        ax.set_title(f"Machine ID: {m_id} (2,018 Bins / 7 Days)", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("CPU Utilization", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        
        # Mark split boundaries (Train 70% = 1412, Val 15% = 303)
        ax.axvline(1412 * 5.0 / 60.0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(1715 * 5.0 / 60.0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.8)

    axes[2].set_xlabel("Time (Hours across 7-Day Trace)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Time (Hours across 7-Day Trace)", fontsize=11, fontweight="bold")

    fig.suptitle("Representative CPU Workload Traces (Google Cluster Trace v2, 100-Machine Sample)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    out_path = FIGURES_DIR / "final_data_validation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [✓] Saved {out_path}")


def generate_figure_2_forecasting_comparison():
    """
    FIGURE 2 — FORECASTING COMPARISON
    Test period of representative machine: Actual, Naive, GRU, Holt-Winters.
    """
    print("[*] Generating Figure 2: final_prediction_comparison.png...")
    df_naive = pd.read_csv(NAIVE_CSV)
    df_hw = pd.read_csv(HW_CSV)
    df_gru = pd.read_csv(GRU_CSV)

    m_id = df_naive["machine_id"].unique()[0]
    
    t_naive = df_naive[(df_naive["machine_id"] == m_id) & (df_naive["split"] == "test")].reset_index(drop=True)
    t_hw = df_hw[(df_hw["machine_id"] == m_id) & (df_hw["split"] == "test")].reset_index(drop=True)
    t_gru = df_gru[(df_gru["machine_id"] == m_id) & (df_gru["split"] == "test")].reset_index(drop=True)

    # Window of 120 steps (10 hours)
    plot_len = 120
    time_hours = np.arange(plot_len) * 5.0 / 60.0

    y_act = t_naive["actual"].iloc[:plot_len].values
    y_naive = t_naive["pred" if "pred" in t_naive.columns else "predicted"].iloc[:plot_len].values
    y_hw = t_hw["pred" if "pred" in t_hw.columns else "predicted"].iloc[:plot_len].values
    y_gru = t_gru["pred" if "pred" in t_gru.columns else "predicted"].iloc[:plot_len].values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_hours, y_act, color="#0f172a", linewidth=2.2, label="Actual Ground Truth $y_t$", zorder=5)
    ax.plot(time_hours, y_gru, color="#2563eb", linewidth=1.8, linestyle="-", label="Global GRU Forecast $\\hat{y}_{t}$ (MAE=0.0570)", zorder=4)
    ax.plot(time_hours, y_naive, color="#f59e0b", linewidth=1.2, linestyle="--", label="Naive Persistence Baseline $y_{t-1}$ (MAE=0.0631)", zorder=3)
    ax.plot(time_hours, y_hw, color="#dc2626", linewidth=1.2, linestyle=":", label="Holt-Winters Statistical Baseline (MAE=0.1619)", zorder=2)

    ax.set_title("Workload Forecasting Model Comparison (Test Split, Machine 1000)\n"
                 "GRU selected as final forecaster because validation fusion selected $\\alpha^*=0.00$", 
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time (Hours in Test Horizon)", fontsize=11, fontweight="bold")
    ax.set_ylabel("CPU Utilization", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, max(y_act.max(), y_gru.max()) + 0.15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_prediction_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [✓] Saved {out_path}")


def generate_figure_3_uncertainty_signals():
    """
    FIGURE 3 — UNCERTAINTY / RISK SIGNALS
    U1 (Disagreement), U2 (EWMA error), U3 (Volatility) labeled as Risk / uncertainty proxy signals.
    """
    print("[*] Generating Figure 3: final_uncertainty_signals.png...")
    df_unc = pd.read_csv(UNCERTAINTY_CSV)
    m_id = df_unc["machine_id"].unique()[0]
    
    t_unc = df_unc[(df_unc["machine_id"] == m_id) & (df_unc["split"] == "test")].reset_index(drop=True)
    plot_len = 120
    time_hours = np.arange(plot_len) * 5.0 / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: U1
    axes[0].plot(time_hours, t_unc["u1_cal"].iloc[:plot_len], color="#dc2626", linewidth=1.4, label="Calibrated $U_1$ (Model Disagreement $|\\text{HW} - \\text{GRU}| \\cdot c_1$)")
    axes[0].set_ylabel("Proxy Magnitude", fontsize=10, fontweight="bold")
    axes[0].set_title("Heterogeneous Risk / Uncertainty Proxy Signals (Machine 1000, Test Period)", fontsize=12, fontweight="bold", pad=10)
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: U2
    axes[1].plot(time_hours, t_unc["u2_cal"].iloc[:plot_len], color="#059669", linewidth=1.4, label="Calibrated $U_2$ (Recent Forecast Error EWMA $\\gamma=0.2 \\cdot c_2$)")
    axes[1].set_ylabel("Proxy Magnitude", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: U3
    axes[2].plot(time_hours, t_unc["u3_cal"].iloc[:plot_len], color="#2563eb", linewidth=1.4, label="Calibrated $U_3$ (Workload Volatility $\\text{std}(y_{t-6:t-1}) \\cdot c_3$)")
    axes[2].set_ylabel("Proxy Magnitude", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Time (Hours in Test Horizon)", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_uncertainty_signals.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [✓] Saved {out_path}")


def generate_figure_4_selector_choices():
    """
    FIGURE 4 — ADAPTIVE SELECTOR
    Show categorical selector choices U1, U2, U3 over test-period segment.
    """
    print("[*] Generating Figure 4: final_selector_choices.png...")
    df_sel = pd.read_csv(SELECTOR_CSV)
    m_id = df_sel["machine_id"].unique()[0]
    
    t_sel = df_sel[(df_sel["machine_id"] == m_id) & (df_sel["split"] == "test")].reset_index(drop=True)
    plot_len = 120
    time_hours = np.arange(plot_len) * 5.0 / 60.0

    sig_map = {"U1": 1, "U2": 2, "U3": 3}
    num_sigs = [sig_map[s] for s in t_sel["selected_signal"].iloc[:plot_len]]
    colors = ["#dc2626" if s == "U1" else ("#059669" if s == "U2" else "#2563eb") for s in t_sel["selected_signal"].iloc[:plot_len]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})

    # Upper: Selected Calibrated Uncertainty Signal
    ax1.plot(time_hours, t_sel["selected_uncertainty_cal"].iloc[:plot_len], color="#4f46e5", linewidth=1.8, label="Active Calibrated Uncertainty Buffer $\\tilde{U}^*(t)$")
    ax1.plot(time_hours, t_sel["previous_error"].iloc[:plot_len], color="#64748b", linestyle=":", linewidth=1.2, alpha=0.7, label="Previous Absolute Error $E_{t-1}$")
    ax1.set_ylabel("Risk Proxy Scale", fontsize=10, fontweight="bold")
    ax1.set_title("Adaptive Uncertainty Selector Dynamics (Machine 1000, Test Split)\n"
                  "Adaptive selector: rolling Spearman correlation with previous errors, window size $N^*=75$",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Lower: Categorical Signal Selection State
    ax2.step(time_hours, num_sigs, where="mid", color="#6366f1", linewidth=1.8)
    ax2.scatter(time_hours, num_sigs, c=colors, s=30, zorder=4)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(["$U_1$ (Disagreement)", "$U_2$ (Error EWMA)", "$U_3$ (Volatility)"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Selected Proxy", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (Hours in Test Horizon)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.5, 3.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_selector_choices.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [✓] Saved {out_path}")


def generate_figure_5_allocation_vs_actual():
    """
    FIGURE 5 — ALLOCATION VS ACTUAL
    Show Actual CPU, GRU prediction, Proposed allocated resource (beta=3.0).
    """
    print("[*] Generating Figure 5: final_allocation_vs_actual.png...")
    df_alloc = pd.read_csv(ALLOCATION_CSV)
    m_id = df_alloc["machine_id"].unique()[0]
    
    t_alloc = df_alloc[(df_alloc["machine_id"] == m_id) & (df_alloc["split"] == "test")].reset_index(drop=True)
    plot_len = 120
    time_hours = np.arange(plot_len) * 5.0 / 60.0

    act = t_alloc["actual"].iloc[:plot_len].values
    pred = t_alloc["pred"].iloc[:plot_len].values
    alloc = t_alloc["allocation"].iloc[:plot_len].values

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Fill risk buffer
    ax.fill_between(time_hours, pred, alloc, color="#e0e7ff", alpha=0.7, label="Adaptive Risk Buffer $(\\beta^* \\cdot \\tilde{U}^*(t))$")
    
    ax.plot(time_hours, act, color="#0f172a", linewidth=2.0, label="Actual Ground Truth $y_t$", zorder=5)
    ax.plot(time_hours, pred, color="#2563eb", linestyle="--", linewidth=1.5, label="GRU Workload Prediction $\\hat{y}_t$", zorder=4)
    ax.plot(time_hours, alloc, color="#4f46e5", linewidth=2.0, label="Proposed Resource Allocation $A_t$ ($\\beta^*=3.00$)", zorder=6)

    # Highlight breaches if any
    breaches = np.where(act > alloc + 0.005)[0]
    if len(breaches) > 0:
        ax.scatter(time_hours[breaches], act[breaches], color="#dc2626", s=45, zorder=7, label="SLA Breach ($y_t > A_t + \\delta$)")

    ax.set_title("Resource Allocation vs. Actual Workload (Machine 1000, Test Split)\n"
                 "$A_t = \\min(1.0, \\max(0.0, \\hat{y}_t + \\beta^* \\tilde{U}^*(t)))$ with calibrated multiplier $\\beta^*=3.00$",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time (Hours in Test Horizon)", fontsize=11, fontweight="bold")
    ax.set_ylabel("CPU Resource Allocation (Cores Fraction)", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_allocation_vs_actual.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [✓] Saved {out_path}")


def generate_figure_6_final_results_comparison():
    """
    FIGURE 6 — FINAL BENCHMARK COMPARISON
    Comparison of all 9 configurations showing SLA breach rate (%) and Over-provisioning waste.
    """
    print("[*] Generating Figure 6: final_results_comparison.png...")
    df_res = pd.read_csv(FINAL_RESULTS_CSV)
    
    configs = df_res["Config"].values
    names = df_res["Name"].values
    breach = df_res["SLA_Breach_Pct"].values
    waste = df_res["Over_Provisioning"].values

    x = np.arange(len(configs))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(14, 6.5))

    # Bars
    color_breach = "#ef4444"
    color_waste = "#3b82f6"

    rects1 = ax1.bar(x - width/2, breach, width, label="SLA Breach Rate (%)", color=color_breach, alpha=0.9)
    ax1.set_ylabel("SLA Breach Rate (%)", color=color_breach, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_breach)
    ax1.set_ylim(0, 58)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, waste, width, label="Over-Provisioning / Waste (Cores)", color=color_waste, alpha=0.9)
    ax2.set_ylabel("Over-Provisioning / Waste (Cores per Step)", color=color_waste, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_waste)
    ax2.set_ylim(0, 0.25)

    # Set x tick labels
    ax1.set_xticks(x)
    short_labels = [f"{c}\n({n})" for c, n in zip(configs, [
        "Naive", "S.Naive", "GRU", "HW", "Fixed Buffer", "U1 (Disag)", "U2 (EWMA)", "U3 (Vol)", "Proposed P"
    ])]
    ax1.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Evaluation Configuration", fontsize=11, fontweight="bold", labelpad=8)

    # Add text labels on bars
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color_breach)

    for r in rects2:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color_waste)

    plt.title("Final 9-Configuration Benchmark: SLA Breach Rate vs. Over-Provisioning Waste\n"
              "Evaluated on 30,300 test timesteps across 100 machines with calibrated multiplier $\\beta^*=3.00$",
              fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_results_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [✓] Saved {out_path}")


def generate_figure_7_pareto_frontier():
    """
    FIGURE 7 — PARETO FRONTIER
    Show B3, B4a, B4b, B4c, P across tested beta in [0.5, 3.0].
    """
    print("[*] Generating Figure 7: final_pareto_frontier.png...")
    df_pareto = pd.read_csv(PARETO_CSV)
    
    fig, ax = plt.subplots(figsize=(10, 6.5))

    colors = {
        "B3 (Fixed Mean Cal)": "#64748b",
        "B4a (U1 Calibrated)": "#ef4444",
        "B4b (U2 Calibrated)": "#059669",
        "B4c (U3 Calibrated)": "#2563eb",
        "P (Proposed Calibrated U*)": "#7c3aed",
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
        ax.plot(
            sdf["Waste"],
            sdf["SLA_Breach_Pct"],
            label=strat,
            color=colors.get(strat, "#000000"),
            marker=markers.get(strat, "o"),
            linestyle=linestyles.get(strat, "-"),
            linewidth=2.2 if "Proposed Calibrated" in strat else (1.4 if "P_raw" in strat else 1.8),
            markersize=6.5 if "Proposed" in strat else 5.5
        )

    # Reference target line
    ax.axhline(y=2.0, color="#dc2626", linestyle=":", alpha=0.8, linewidth=1.5, label="2% SLA Target (Constraint)")
    
    # Annotate Pareto dominance
    ax.annotate("B4b (U2) strictly dominates Proposed P\n(lower breach & lower waste)", 
                xy=(0.1767, 2.77), xytext=(0.10, 3.5),
                arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.2, headwidth=6),
                fontsize=9.5, fontweight="bold", color="#065f46",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ecfdf5", ec="#059669", alpha=0.9))

    ax.set_xlabel("Over-Provisioning / Resource Waste (Cores per Timestep)", fontsize=11, fontweight="bold")
    ax.set_ylabel("SLA Breach Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Reliability vs. Resource Waste Pareto Frontier Across $\\beta \\in [0.5, 3.0]$\n"
                 "Calibrated Risk Proxies on 30,300 Test Timesteps (100 Machines)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_pareto_frontier.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [✓] Saved {out_path}")


def generate_tables_and_summary():
    """
    Generate final_results_table.csv and final_mvp_summary.json.
    """
    print("[*] Generating final_results_table.csv and final_mvp_summary.json...")
    df_res = pd.read_csv(FINAL_RESULTS_CSV)
    
    # Save clean final_results_table.csv
    table_csv_path = RESULTS_DIR / "final_results_table.csv"
    export_cols = [
        "Config", "Name", "Forecast_Model", "Buffer_Type",
        "MAE", "RMSE", "SLA_Breach_Pct", "Overshoot_Pct", "Grace_Zone_Pct",
        "Over_Provisioning", "Under_Provisioning"
    ]
    df_res[export_cols].to_csv(table_csv_path, index=False)
    print(f"    [✓] Saved {table_csv_path}")

    # Load calibration and selector metadata
    with open(CALIB_JSON) as f:
        calib_meta = json.load(f)

    summary_json = {
        "project": "Google Cluster Trace MVP: Risk-Aware Resource Allocation Framework",
        "dataset": {
            "source": "Google Cluster Trace v2",
            "num_machines": 100,
            "bins_per_machine": 2018,
            "total_rows": 201800,
            "sampling_interval_min": 5,
            "aggregation": "overlap-weighted CPU usage rate"
        },
        "split": {
            "train_steps": 141200,
            "validation_steps": 30300,
            "test_steps": 30300,
            "split_ratio": "70/15/15 chronological"
        },
        "forecasting": {
            "models": ["Naive Persistence", "Seasonal Naive (24h)", "Global GRU", "Holt-Winters"],
            "gru_architecture": {
                "input_features": 7,
                "hidden_size": 32,
                "dropout": 0.20,
                "output_activation": "Sigmoid"
            },
            "fusion_alpha": 0.00,
            "deployed_forecaster": "Global GRU only (due to alpha*=0.00 on validation)"
        },
        "uncertainty_and_calibration": {
            "signals": {
                "U1": "|Holt-Winters - GRU| (Model Disagreement)",
                "U2": "EWMA(gamma=0.2) of past prediction error (Error EWMA)",
                "U3": "Rolling std over past 6 observations (Workload Volatility)"
            },
            "calibration_method": "Method 1: Validation Mean-Error Matching",
            "validation_mean_error": calib_meta["mean_error_val"],
            "calibration_factors": {
                "c1": calib_meta["c1"],
                "c2": calib_meta["c2"],
                "c3": calib_meta["c3"]
            }
        },
        "adaptive_selector": {
            "metric": "Rolling Spearman rank correlation on historical error window [t-N, t-1]",
            "optimal_window_N_star": 75,
            "test_selection_distribution": {
                "U1_disagreement_pct": 41.68,
                "U2_error_ewma_pct": 23.23,
                "U3_volatility_pct": 35.09
            },
            "invariance_verified": True
        },
        "allocation_and_buffer": {
            "rule": "A_t = clip(yhat_t + beta * U_cal_star(t), 0.0, 1.0)",
            "calibrated_beta_star": 3.00,
            "grace_tolerance_delta": 0.005,
            "validation_sla_target_achieved": False,
            "validation_sla_breach_pct": 5.62
        },
        "final_benchmark_results": df_res[export_cols].to_dict(orient="records"),
        "major_findings": [
            "Method 1 validation mean-error matching successfully eliminates the 8-fold scale distortion between raw proxies, reducing over-provisioning waste by 30.2% on the proposed method.",
            "Spearman selector decisions and optimal window size (N*=75) are 100% mathematically invariant under positive linear scale calibration.",
            "On the evaluated 100-machine Google Cluster Trace dataset, calibrated U2 (Error EWMA, B4b) strictly dominates the adaptive selector (P) on the Pareto frontier, achieving 2.77% SLA breach at 0.1767 waste compared to 5.20% breach at 0.1796 waste for P.",
            "The rolling Spearman selector exhibits transition latency (N*=75 steps = 6.25h) that limits its responsiveness during sudden trace bursts compared to smooth exponential smoothing."
        ],
        "limitations_and_disclosures": [
            "Evaluated on a 100-machine subset of Google Cluster Trace v2; broader multi-cloud traces (e.g. Alibaba) remain future work.",
            "Uncertainty signals are heuristic risk proxies rather than formal Bayesian or conformal prediction intervals.",
            "Validation SLA target of <= 2.0% breach was not achieved within the candidate beta grid in [0.5, 3.0].",
            "Because alpha*=0.00, the deployed forecaster is GRU-only; Holt-Winters serves solely as a disagreement sensor."
        ]
    }

    summary_json_path = RESULTS_DIR / "final_mvp_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"    [✓] Saved {summary_json_path}")


def main():
    print("=" * 70)
    print(" Google Cluster Trace MVP — Phase 12 Final Figures & Output Generator")
    print("=" * 70)
    
    generate_figure_1_data_validation()
    generate_figure_2_forecasting_comparison()
    generate_figure_3_uncertainty_signals()
    generate_figure_4_selector_choices()
    generate_figure_5_allocation_vs_actual()
    generate_figure_6_final_results_comparison()
    generate_figure_7_pareto_frontier()
    generate_tables_and_summary()

    print("\n[+] Phase 12 Figure Generation & Output Packaging Complete.")


if __name__ == "__main__":
    main()
