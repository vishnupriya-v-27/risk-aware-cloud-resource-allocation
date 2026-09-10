"""
Adaptive Uncertainty Selector Script (Optimized Vectorized)
Part of the 16-20 Hour Google Cluster Trace MVP (Phase 8 / Hour 11.5-13).

Mechanism:
1. At timestep t, for each uncertainty signal U_i in {U1, U2, U3}, compute rolling Spearman correlation
   with previous prediction error E[t-N : t-1] where E[tau] = |pred[tau] - actual[tau]|.
2. Window candidates: N in {20, 30, 50, 75, 100}.
3. At each step t, select U*(t) = U_k(t) where k = argmax_i Spearman(U_i[t-N:t-1], E[t-N:t-1]).
4. Tie-breaking rule: Deterministic priority U2 > U1 > U3.
5. Window N* selected strictly using Validation set evaluation.
6. Generate test set outputs with fixed N*.

Outputs:
- selector/selector_results.csv
- uncertainty/selector_log.csv
- results/selector_metrics.json
- figures/selector_choices.png
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UNCERTAINTY_DIR = BASE_DIR / "uncertainty"
SELECTOR_DIR = BASE_DIR / "selector"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

UNCERTAINTY_CSV = UNCERTAINTY_DIR / "uncertainty.csv"
SELECTOR_CSV = SELECTOR_DIR / "selector_results.csv"
SELECTOR_LOG_CSV = UNCERTAINTY_DIR / "selector_log.csv"
METRICS_JSON = RESULTS_DIR / "selector_metrics.json"
SELECTOR_FIG = FIGURES_DIR / "selector_choices.png"


def fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute fast Spearman correlation using ranked Pearson.
    """
    n = len(x)
    if n < 4:
        return 0.0
    rx = rankdata(x)
    ry = rankdata(y)
    std_x = np.std(rx)
    std_y = np.std(ry)
    if std_x < 1e-7 or std_y < 1e-7:
        return 0.0
    cov = np.mean((rx - np.mean(rx)) * (ry - np.mean(ry)))
    r = cov / (std_x * std_y)
    return float(np.clip(r, -1.0, 1.0))


def compute_machine_rolling_corrs(u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, prev_errors: np.ndarray, N: int):
    """
    Compute rolling Spearman correlation for one machine across timesteps.
    """
    L = len(u1)
    rho1 = np.zeros(L, dtype=np.float32)
    rho2 = np.zeros(L, dtype=np.float32)
    rho3 = np.zeros(L, dtype=np.float32)
    selected_sig = []
    selected_u = np.zeros(L, dtype=np.float32)

    for t in range(L):
        if t < 4:
            # Cold start: default to U2
            sig = "U2"
            val = u2[t]
            r1, r2, r3 = 0.0, 0.0, 0.0
        else:
            w_start = max(0, t - N)
            h_err = prev_errors[w_start:t]
            h_u1 = u1[w_start:t]
            h_u2 = u2[w_start:t]
            h_u3 = u3[w_start:t]

            r1 = fast_spearman(h_u1, h_err)
            r2 = fast_spearman(h_u2, h_err)
            r3 = fast_spearman(h_u3, h_err)

            # Tie-breaking priority: U2 > U1 > U3
            max_r = max(r1, r2, r3)
            if r2 == max_r:
                sig = "U2"
                val = u2[t]
            elif r1 == max_r:
                sig = "U1"
                val = u1[t]
            else:
                sig = "U3"
                val = u3[t]

        rho1[t] = r1
        rho2[t] = r2
        rho3[t] = r3
        selected_sig.append(sig)
        selected_u[t] = val

    return rho1, rho2, rho3, selected_sig, selected_u


def run_adaptive_selector():
    print("=" * 60, flush=True)
    print(" Google Cluster Trace MVP — Adaptive Selector (Phase 8)", flush=True)
    print("=" * 60, flush=True)

    if not UNCERTAINTY_CSV.exists():
        raise FileNotFoundError(f"Missing {UNCERTAINTY_CSV}. Run uncertainty.py first!")

    print(f"[*] Loading uncertainty dataset from {UNCERTAINTY_CSV}...", flush=True)
    df_unc = pd.read_csv(UNCERTAINTY_CSV)
    df_unc.sort_values(by=["machine_id", "timestamp"], inplace=True)
    df_unc.reset_index(drop=True, inplace=True)

    SELECTOR_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    machines = df_unc["machine_id"].unique()
    window_candidates = [20, 30, 50, 75, 100]
    val_results = {}
    selector_dfs = {}

    print(f"[*] Pre-extracting machine arrays for {len(machines)} machines...", flush=True)
    machine_data = {}
    for m_id in machines:
        m_df = df_unc[df_unc["machine_id"] == m_id].copy().reset_index(drop=True)
        y_act = m_df["actual"].values.astype(np.float32)
        y_pred = m_df["pred"].values.astype(np.float32)
        u1 = m_df["u1"].values.astype(np.float32)
        u2 = m_df["u2"].values.astype(np.float32)
        u3 = m_df["u3"].values.astype(np.float32)
        u1_cal = m_df["u1_cal"].values.astype(np.float32)
        u2_cal = m_df["u2_cal"].values.astype(np.float32)
        u3_cal = m_df["u3_cal"].values.astype(np.float32)
        splits = m_df["split"].values
        timestamps = m_df["timestamp"].values

        # Previous prediction error series: E[tau] = |y_pred[tau] - y_act[tau]|
        prev_errors = np.abs(y_pred - y_act)

        machine_data[m_id] = {
            "y_act": y_act,
            "y_pred": y_pred,
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "u1_cal": u1_cal,
            "u2_cal": u2_cal,
            "u3_cal": u3_cal,
            "splits": splits,
            "timestamps": timestamps,
            "prev_errors": prev_errors,
            "n_steps": len(m_df)
        }

    print("\n[*] Evaluating Candidate Window Sizes N in {20, 30, 50, 75, 100} on Validation Set...", flush=True)

    for N in window_candidates:
        all_rows = []
        val_machine_corrs = []

        for m_id in machines:
            m = machine_data[m_id]
            rho1, rho2, rho3, sel_sig, sel_u = compute_machine_rolling_corrs(
                m["u1"], m["u2"], m["u3"], m["prev_errors"], N
            )

            # Build rows
            for t in range(m["n_steps"]):
                p_err = m["prev_errors"][t - 1] if t > 0 else 0.0
                sig = sel_sig[t]
                if sig == "U1":
                    val_cal = m["u1_cal"][t]
                elif sig == "U2":
                    val_cal = m["u2_cal"][t]
                else:
                    val_cal = m["u3_cal"][t]

                all_rows.append({
                    "machine_id": m_id,
                    "timestamp": m["timestamps"][t],
                    "split": m["splits"][t],
                    "actual": float(m["y_act"][t]),
                    "pred": float(m["y_pred"][t]),
                    "previous_error": float(p_err),
                    "u1": float(m["u1"][t]),
                    "u2": float(m["u2"][t]),
                    "u3": float(m["u3"][t]),
                    "u1_cal": float(m["u1_cal"][t]),
                    "u2_cal": float(m["u2_cal"][t]),
                    "u3_cal": float(m["u3_cal"][t]),
                    "rho_u1": float(rho1[t]),
                    "rho_u2": float(rho2[t]),
                    "rho_u3": float(rho3[t]),
                    "selected_signal": sig,
                    "selected_uncertainty": float(sel_u[t]),
                    "selected_uncertainty_cal": float(val_cal),
                    "window_size": N
                })

            # Validation correlation for this machine
            val_mask = (m["splits"] == "val")
            m_val_err = np.abs(m["y_pred"][val_mask] - m["y_act"][val_mask])
            m_val_u = sel_u[val_mask]
            val_machine_corrs.append(fast_spearman(m_val_u, m_val_err))

        df_res = pd.DataFrame(all_rows)
        selector_dfs[N] = df_res

        df_val = df_res[df_res["split"] == "val"]
        val_global_corr = fast_spearman(
            df_val["selected_uncertainty"].values,
            np.abs(df_val["pred"].values - df_val["actual"].values)
        )
        val_mean_machine_corr = float(np.mean(val_machine_corrs))
        val_counts = df_val["selected_signal"].value_counts(normalize=True).to_dict()

        val_results[N] = {
            "window_size": N,
            "val_mean_machine_spearman": val_mean_machine_corr,
            "val_global_spearman": val_global_corr,
            "u1_val_pct": float(val_counts.get("U1", 0.0) * 100.0),
            "u2_val_pct": float(val_counts.get("U2", 0.0) * 100.0),
            "u3_val_pct": float(val_counts.get("U3", 0.0) * 100.0)
        }

        print(f"    N = {N:3d} | Val Mean Spearman = {val_mean_machine_corr:.4f} | Global Spearman = {val_global_corr:.4f} | Selection: U1={val_counts.get('U1',0.0)*100:4.1f}%, U2={val_counts.get('U2',0.0)*100:4.1f}%, U3={val_counts.get('U3',0.0)*100:4.1f}%", flush=True)

    # Select best N* using validation mean machine Spearman correlation
    best_N = max(window_candidates, key=lambda n: val_results[n]["val_mean_machine_spearman"])
    print(f"\n[+] Optimal Selector Window Size N* = {best_N} (Selected strictly on Validation Set)", flush=True)

    # Retrieve full dataset with optimal N*
    df_final = selector_dfs[best_N]

    # -----------------------------------------------------------------
    # Verification Checks
    # -----------------------------------------------------------------
    print("\n[*] Performing Phase 8 Verification Checks:", flush=True)
    
    # 1. Row counts & splits
    assert len(df_final) == 201800, f"Expected 201,800 rows, got {len(df_final)}"
    assert df_final["machine_id"].nunique() == 100, f"Expected 100 machines, got {df_final['machine_id'].nunique()}"
    
    split_counts = df_final["split"].value_counts().to_dict()
    assert split_counts.get("train", 0) == 141200, f"Train count mismatch: {split_counts.get('train', 0)}"
    assert split_counts.get("val", 0) == 30300, f"Val count mismatch: {split_counts.get('val', 0)}"
    assert split_counts.get("test", 0) == 30300, f"Test count mismatch: {split_counts.get('test', 0)}"
    print(f"    [✓] Total Rows: 201,800 across 100 machines (141,200 Train / 30,300 Val / 30,300 Test).", flush=True)

    # 2. NaN / Inf
    check_cols = ["u1", "u2", "u3", "u1_cal", "u2_cal", "u3_cal", "previous_error", "selected_uncertainty", "selected_uncertainty_cal"]
    assert not df_final[check_cols].isna().any().any(), "NaN values found in selector results!"
    assert not np.isinf(df_final[check_cols].values).any(), "Inf values found in selector results!"
    print("    [✓] No NaN or Inf values detected.", flush=True)

    # 3. Valid signal names & window
    assert set(df_final["selected_signal"].unique()).issubset({"U1", "U2", "U3"}), "Invalid signal names found!"
    assert df_final["window_size"].iloc[0] == best_N, "Window size mismatch!"
    print(f"    [✓] All selected signals are valid ('U1', 'U2', 'U3') and N*={best_N} is from {{20,30,50,75,100}}.", flush=True)

    # 4. Causality & Leakage Verification
    test_first_indices = df_final.groupby("machine_id").apply(lambda g: g[g["split"] == "test"].index[0], include_groups=False).values
    val_last_indices = df_final.groupby("machine_id").apply(lambda g: g[g["split"] == "val"].index[-1], include_groups=False).values

    for t_idx, v_idx in zip(test_first_indices[:10], val_last_indices[:10]):
        exp_prev_err = np.abs(df_final.loc[v_idx, "pred"] - df_final.loc[v_idx, "actual"])
        actual_prev_err = df_final.loc[t_idx, "previous_error"]
        assert np.isclose(exp_prev_err, actual_prev_err, atol=1e-6), "Leakage/boundary check failed at Val->Test transition!"
    print("    [✓] Boundary Causality Verified: Test[0] previous error equals last Val error (no test actual leakage).", flush=True)

    # Save outputs
    output_cols = [
        "machine_id", "timestamp", "split", "u1", "u2", "u3",
        "u1_cal", "u2_cal", "u3_cal", "previous_error",
        "selected_uncertainty", "selected_uncertainty_cal", "selected_signal", "window_size"
    ]
    df_save = df_final[output_cols]
    df_save.to_csv(SELECTOR_CSV, index=False)
    df_save.to_csv(SELECTOR_LOG_CSV, index=False)
    print(f"\n[+] Saved selector output to {SELECTOR_CSV}", flush=True)
    print(f"[+] Saved selector log to {SELECTOR_LOG_CSV}", flush=True)

    # -----------------------------------------------------------------
    # Signal Selection Distributions
    # -----------------------------------------------------------------
    overall_dist = df_final["selected_signal"].value_counts(normalize=True).to_dict()
    test_df = df_final[df_final["split"] == "test"]
    test_dist = test_df["selected_signal"].value_counts(normalize=True).to_dict()
    val_dist = df_final[df_final["split"] == "val"]["selected_signal"].value_counts(normalize=True).to_dict()

    print("\n" + "=" * 65, flush=True)
    print(f" SIGNAL SELECTION DISTRIBUTION (N* = {best_N})", flush=True)
    print("=" * 65, flush=True)
    print(f"  Overall (201,800 steps): U1 = {overall_dist.get('U1',0.0)*100:5.2f}% | U2 = {overall_dist.get('U2',0.0)*100:5.2f}% | U3 = {overall_dist.get('U3',0.0)*100:5.2f}%", flush=True)
    print(f"  Validation (30,300 steps): U1 = {val_dist.get('U1',0.0)*100:5.2f}% | U2 = {val_dist.get('U2',0.0)*100:5.2f}% | U3 = {val_dist.get('U3',0.0)*100:5.2f}%", flush=True)
    print(f"  Test Set   (30,300 steps): U1 = {test_dist.get('U1',0.0)*100:5.2f}% | U2 = {test_dist.get('U2',0.0)*100:5.2f}% | U3 = {test_dist.get('U3',0.0)*100:5.2f}%", flush=True)
    print("=" * 65, flush=True)

    # Save metrics JSON
    metrics_export = {
        "best_window_size": best_N,
        "validation_tuning_results": val_results,
        "selection_distribution": {
            "overall_pct": {k: float(v * 100.0) for k, v in overall_dist.items()},
            "val_pct": {k: float(v * 100.0) for k, v in val_dist.items()},
            "test_pct": {k: float(v * 100.0) for k, v in test_dist.items()}
        },
        "tie_breaking_rule": "Deterministic priority U2 > U1 > U3"
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_export, f, indent=2)

    # -----------------------------------------------------------------
    # Generate figures/selector_choices.png
    # -----------------------------------------------------------------
    generate_selector_plot(df_final, machines, best_N)

    return metrics_export


def generate_selector_plot(df: pd.DataFrame, machines: list, best_N: int):
    """
    Creates figures/selector_choices.png showing dynamic signal selection timeline.
    """
    sample_m_id = machines[0]
    sample_df = df[df["machine_id"] == sample_m_id].copy().reset_index(drop=True)
    
    test_start_idx = np.where(sample_df["split"] == "test")[0][0]
    # Zoom in on 150 test timesteps (12.5 hours)
    plot_df = sample_df.iloc[test_start_idx : test_start_idx + 150].reset_index(drop=True)

    time_hrs = np.arange(len(plot_df)) * 5.0 / 60.0

    signal_mapping = {"U1": 1, "U2": 2, "U3": 3}
    num_signals = [signal_mapping[s] for s in plot_df["selected_signal"]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.2]})
    fig.suptitle(f"Google Cluster Trace MVP: Adaptive Uncertainty Selector (Machine {sample_m_id}, N*={best_N})", fontsize=14, fontweight="bold", y=0.98)

    # Subplot 1: Uncertainty Signals vs Error
    abs_err = np.abs(plot_df["pred"] - plot_df["actual"])
    axes[0].plot(time_hrs, abs_err, color="#dc2626", linewidth=1.5, label="Forecast Error |y - ŷ|")
    axes[0].plot(time_hrs, plot_df["selected_uncertainty"], color="#1e1b4b", linewidth=1.8, linestyle="-.", label="Adaptive Selected U*(t)")
    axes[0].set_ylabel("Magnitude", fontsize=10)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Individual Candidates U1, U2, U3
    axes[1].plot(time_hrs, plot_df["u1"], color="#2563eb", alpha=0.7, label="U1 (Disagreement)")
    axes[1].plot(time_hrs, plot_df["u2"], color="#059669", alpha=0.7, label="U2 (Error EWMA)")
    axes[1].plot(time_hrs, plot_df["u3"], color="#d97706", alpha=0.7, label="U3 (Volatility)")
    axes[1].set_ylabel("Candidate Signals", fontsize=10)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Discrete Selection State Over Time
    axes[2].step(time_hrs, num_signals, where="mid", color="#7c3aed", linewidth=2.0)
    axes[2].scatter(time_hrs, num_signals, c=["#2563eb" if s=="U1" else ("#059669" if s=="U2" else "#d97706") for s in plot_df["selected_signal"]], s=35, zorder=3)
    axes[2].set_yticks([1, 2, 3])
    axes[2].set_yticklabels(["U1 (Disagreement)", "U2 (EWMA)", "U3 (Volatility)"], fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Time (Hours in Test Horizon)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Selected Signal", fontsize=10)
    axes[2].set_ylim(0.5, 3.5)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(SELECTOR_FIG, dpi=300)
    plt.close()
    print(f"[+] Saved selector choices figure to {SELECTOR_FIG}", flush=True)


if __name__ == "__main__":
    run_adaptive_selector()
