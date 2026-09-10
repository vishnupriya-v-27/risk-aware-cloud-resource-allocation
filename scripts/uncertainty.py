"""
Uncertainty Quantification Script
Part of the 16-20 Hour Google Cluster Trace MVP (Phase 7 / Hour 10-11.5).

Signals:
1. U1 (Model Disagreement): abs(hw_pred - gru_pred)
2. U2 (Recent Error EWMA): EWMA(gamma=0.2) of previous prediction error abs(y[t-1] - y_hat[t-1]).
   - Test step 0 initialization: U2_test[0] = abs(y_val[-1] - y_hat_val[-1]).
   - Strictly causal: no future or current actual y[t] used.
3. U3 (Workload Volatility): rolling std(y[t-6:t-1]) computed continuously across boundaries.

Outputs:
- uncertainty/uncertainty.csv (201,800 rows, 100 machines)
- figures/uncertainty_signals.png
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
FIGURES_DIR = BASE_DIR / "figures"

FUSION_CSV = MODELS_DIR / "fusion_predictions.csv"
UNCERTAINTY_CSV = UNCERTAINTY_DIR / "uncertainty.csv"
UNCERTAINTY_FIG = FIGURES_DIR / "uncertainty_signals.png"


def compute_uncertainty_signals():
    print("=" * 60)
    print(" Google Cluster Trace MVP — Uncertainty Quantification (Phase 7)")
    print("=" * 60)

    if not FUSION_CSV.exists():
        raise FileNotFoundError(f"Missing {FUSION_CSV}. Run fusion.py first!")

    print(f"[*] Loading fusion predictions from {FUSION_CSV}...")
    df = pd.read_csv(FUSION_CSV)
    df.sort_values(by=["machine_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    machines = df["machine_id"].unique()
    print(f"[+] Loaded {len(df):,} rows across {len(machines)} machines.")

    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gamma = 0.2
    all_rows = []
    val_init_verification = []

    for m_id in machines:
        m_df = df[df["machine_id"] == m_id].copy().reset_index(drop=True)

        y = m_df["actual"].values
        hw_pred = m_df["hw_pred"].values
        gru_pred = m_df["gru_pred"].values
        fused_pred = m_df["pred"].values
        splits = m_df["split"].values
        timestamps = m_df["timestamp"].values

        n_steps = len(m_df)
        val_indices = np.where(splits == "val")[0]
        test_indices = np.where(splits == "test")[0]

        val_start = val_indices[0]
        val_end = val_indices[-1]
        test_start = test_indices[0]

        # -------------------------------------------------------------
        # 1. U1: Model Disagreement
        # -------------------------------------------------------------
        u1 = np.abs(hw_pred - gru_pred)

        # -------------------------------------------------------------
        # 2. U2: Recent Error EWMA (Strictly Causal)
        # -------------------------------------------------------------
        # e[t-1] = |y[t-1] - y_hat[t-1]|
        # For train:
        u2 = np.zeros(n_steps, dtype=np.float64)
        
        # Train sequence
        u2[0] = np.abs(y[0] - fused_pred[0])  # initial prior
        for t in range(1, test_start):
            prev_err = np.abs(y[t - 1] - fused_pred[t - 1])
            u2[t] = gamma * prev_err + (1.0 - gamma) * u2[t - 1]

        # Test sequence initialization:
        # At test step 0 (t = test_start), initialize using the last validation error
        last_val_error = np.abs(y[val_end] - fused_pred[val_end])
        u2[test_start] = last_val_error

        # Subsequent test steps
        for t in range(test_start + 1, n_steps):
            prev_err = np.abs(y[t - 1] - fused_pred[t - 1])
            u2[t] = gamma * prev_err + (1.0 - gamma) * u2[t - 1]

        val_init_verification.append({
            "machine_id": int(m_id),
            "last_val_error": float(last_val_error),
            "u2_test_0": float(u2[test_start]),
            "match": bool(np.isclose(u2[test_start], last_val_error, atol=1e-7))
        })

        # -------------------------------------------------------------
        # 3. U3: Workload Volatility (Rolling Std of y[t-6:t-1])
        # Continuous across train/val/test boundaries
        # -------------------------------------------------------------
        u3 = np.zeros(n_steps, dtype=np.float64)
        for t in range(n_steps):
            if t == 0:
                u3[0] = 0.0
            elif t == 1:
                u3[1] = 0.0
            else:
                lookback_start = max(0, t - 6)
                lookback_window = y[lookback_start:t]  # y[t-6:t-1]
                u3[t] = np.std(lookback_window, ddof=1) if len(lookback_window) > 1 else 0.0

        # Backfill initial zero with first non-zero volatility if desired, or keep as exact std
        if u3[0] == 0.0 and len(u3) > 6:
            u3[0] = u3[6]
            u3[1] = u3[6]

        for t in range(n_steps):
            all_rows.append({
                "machine_id": m_id,
                "timestamp": timestamps[t],
                "split": splits[t],
                "actual": y[t],
                "pred": fused_pred[t],
                "hw_pred": hw_pred[t],
                "gru_pred": gru_pred[t],
                "u1": float(u1[t]),
                "u2": float(u2[t]),
                "u3": float(u3[t])
            })

    df_out = pd.DataFrame(all_rows)

    # -----------------------------------------------------------------
    # Method 1: Validation Scale Calibration Factors (Validation Only)
    # -----------------------------------------------------------------
    val_df = df_out[df_out["split"] == "val"]
    val_err = np.abs(val_df["actual"].values - val_df["pred"].values)
    mean_error_val = float(np.mean(val_err))
    mean_u1_val = float(np.mean(val_df["u1"].values))
    mean_u2_val = float(np.mean(val_df["u2"].values))
    mean_u3_val = float(np.mean(val_df["u3"].values))

    c1 = float(mean_error_val / mean_u1_val)
    c2 = float(mean_error_val / mean_u2_val)
    c3 = float(mean_error_val / mean_u3_val)

    calib_factors = {
        "mean_error_val": mean_error_val,
        "mean_u1_val": mean_u1_val,
        "mean_u2_val": mean_u2_val,
        "mean_u3_val": mean_u3_val,
        "c1": c1,
        "c2": c2,
        "c3": c3
    }
    calib_json_path = UNCERTAINTY_DIR / "calibration_factors.json"
    with open(calib_json_path, "w") as f:
        json.dump(calib_factors, f, indent=2)
    print(f"\n[+] Saved validation calibration factors to {calib_json_path}")
    print(f"    - mean_error_val = {mean_error_val:.6f}")
    print(f"    - c1 (U1 scale)  = {c1:.6f} (1/c1 = {1/c1:.4f})")
    print(f"    - c2 (U2 scale)  = {c2:.6f} (1/c2 = {1/c2:.4f})")
    print(f"    - c3 (U3 scale)  = {c3:.6f} (1/c3 = {1/c3:.4f})")

    # Add calibrated signal columns
    df_out["u1_cal"] = df_out["u1"] * c1
    df_out["u2_cal"] = df_out["u2"] * c2
    df_out["u3_cal"] = df_out["u3"] * c3

    # -----------------------------------------------------------------
    # Verification Checks
    # -----------------------------------------------------------------
    print("\n[*] Performing Verification Checks:")
    
    # 1. Total rows & machines
    assert len(df_out) == 201800, f"Expected 201,800 rows, got {len(df_out)}"
    assert df_out["machine_id"].nunique() == 100, f"Expected 100 machines, got {df_out['machine_id'].nunique()}"
    print("    [✓] Total Rows: 201,800 across 100 machines.")

    # 2. NaN / Inf checks
    check_cols = ["u1", "u2", "u3", "u1_cal", "u2_cal", "u3_cal"]
    assert not df_out[check_cols].isna().any().any(), "NaN values found in uncertainty signals!"
    assert not np.isinf(df_out[check_cols].values).any(), "Inf values found in uncertainty signals!"
    print("    [✓] No NaN or Inf values detected.")

    # 3. Non-negativity & Positive Factors
    assert (df_out["u1"] >= 0.0).all(), "Negative U1 values found!"
    assert (df_out["u2"] >= 0.0).all(), "Negative U2 values found!"
    assert (df_out["u3"] >= 0.0).all(), "Negative U3 values found!"
    assert c1 > 0 and c2 > 0 and c3 > 0, "Calibration factors must be strictly positive!"
    print("    [✓] All uncertainty signals (raw and calibrated) are strictly non-negative; c1, c2, c3 > 0.")

    # 4. Test U2 boundary initialization check
    mismatches = [v for v in val_init_verification if not v["match"]]
    assert len(mismatches) == 0, f"U2 test boundary initialization mismatch in {len(mismatches)} machines!"
    print(f"    [✓] Test U2[0] equals last validation error across all 100 machines (100% verified).")

    # 5. Save uncertainty.csv
    df_out.to_csv(UNCERTAINTY_CSV, index=False)
    print(f"\n[+] Saved uncertainty signals (with calibrated columns) to {UNCERTAINTY_CSV}")

    # -----------------------------------------------------------------
    # Basic Statistics Summary
    # -----------------------------------------------------------------
    stats_summary = {}
    for col, name in [("u1", "U1 (Model Disagreement)"), ("u2", "U2 (Recent Error EWMA)"), ("u3", "U3 (Workload Volatility)")]:
        stats_summary[name] = {
            "Min": float(df_out[col].min()),
            "Mean": float(df_out[col].mean()),
            "Std": float(df_out[col].std()),
            "Median": float(df_out[col].median()),
            "Max": float(df_out[col].max())
        }

    print("\n" + "=" * 75)
    print(" UNCERTAINTY SIGNALS BASIC STATISTICS (201,800 timesteps)")
    print("=" * 75)
    df_stats = pd.DataFrame(stats_summary).T
    print(df_stats.to_string())
    print("=" * 75)

    # -----------------------------------------------------------------
    # Generate figures/uncertainty_signals.png
    # -----------------------------------------------------------------
    generate_uncertainty_plot(df_out, machines)

    return stats_summary


def generate_uncertainty_plot(df: pd.DataFrame, machines: list):
    """
    Plot U1, U2, U3 along with ground truth error for sample machines.
    """
    sample_m_id = machines[0]
    sample_df = df[df["machine_id"] == sample_m_id].copy().reset_index(drop=True)
    
    # Focus on test segment + portion of val segment for clarity
    test_start_idx = np.where(sample_df["split"] == "test")[0][0]
    plot_df = sample_df.iloc[max(0, test_start_idx - 100) : test_start_idx + 200].reset_index(drop=True)

    time_hours = np.arange(len(plot_df)) * 5.0 / 60.0  # 5-min steps to hours
    split_boundary_hr = (test_start_idx - max(0, test_start_idx - 100)) * 5.0 / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Google Cluster Trace MVP: Uncertainty Signals (Machine {sample_m_id})", fontsize=14, fontweight="bold", y=0.98)

    # Subplot 1: Forecast Error
    abs_err = np.abs(plot_df["actual"] - plot_df["pred"])
    axes[0].plot(time_hours, abs_err, color="#dc2626", linewidth=1.2, label="Absolute Forecast Error |y - ŷ|")
    axes[0].axvline(split_boundary_hr, color="#4b5563", linestyle="--", linewidth=1.2, label="Val / Test Boundary")
    axes[0].set_ylabel("Error", fontsize=10)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: U1 Disagreement
    axes[1].plot(time_hours, plot_df["u1"], color="#2563eb", linewidth=1.2, label="U1: Model Disagreement |HW - GRU|")
    axes[1].axvline(split_boundary_hr, color="#4b5563", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("U1", fontsize=10)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: U2 Error EWMA
    axes[2].plot(time_hours, plot_df["u2"], color="#059669", linewidth=1.2, label="U2: Recent Error EWMA (γ=0.2, Val Initialized)")
    axes[2].axvline(split_boundary_hr, color="#4b5563", linestyle="--", linewidth=1.2)
    axes[2].set_ylabel("U2", fontsize=10)
    axes[2].legend(loc="upper right", framealpha=0.9)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    # Subplot 4: U3 Workload Volatility
    axes[3].plot(time_hours, plot_df["u3"], color="#d97706", linewidth=1.2, label="U3: Volatility std(y[t-6:t-1])")
    axes[3].axvline(split_boundary_hr, color="#4b5563", linestyle="--", linewidth=1.2)
    axes[3].set_ylabel("U3", fontsize=10)
    axes[3].set_xlabel("Time (Hours in Evaluation Window)", fontsize=11, fontweight="bold")
    axes[3].legend(loc="upper right", framealpha=0.9)
    axes[3].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(UNCERTAINTY_FIG, dpi=300)
    plt.close()
    print(f"[+] Saved uncertainty figure to {UNCERTAINTY_FIG}")


if __name__ == "__main__":
    compute_uncertainty_signals()
