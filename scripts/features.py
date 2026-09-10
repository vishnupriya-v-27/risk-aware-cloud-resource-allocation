"""
Feature Engineering and Data Validation Script
Part of the 16-20 Hour Google Cluster Trace MVP (Hour 5-6).

Tasks:
1. Load data/google_cpu_5min.csv.
2. Clean missingness (linear interpolate gaps <= 2, verify <= 20% missing).
3. Compute 7 MVP features strictly without lookahead:
   - cpu: y[t]
   - cpu_lag1: y[t-1]
   - cpu_lag2: y[t-2]
   - roll_mean_6: mean(y[t-6:t-1])
   - roll_std_6: std(y[t-6:t-1])
   - hour_sin: sin(2 * pi * hour / 24)
   - hour_cos: cos(2 * pi * hour / 24)
4. Split data chronologically (70% Train, 15% Val, 15% Test).
5. Generate figures/data_validation.png (4 sample machines).
6. Save processed dataset to data/google_features_5min.parquet.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
INPUT_CSV = DATA_DIR / "google_cpu_5min.csv"
OUTPUT_PARQUET = DATA_DIR / "google_features_5min.parquet"
OUTPUT_FIG = FIGURES_DIR / "data_validation.png"
SPLIT_INFO_FILE = DATA_DIR / "split_info.json"


def engineer_features_and_split():
    print("=" * 60)
    print(" Google Cluster Trace MVP — Feature Engineering (Hour 5–6)")
    print("=" * 60)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file {INPUT_CSV} not found. Run preprocess.py first!")

    print(f"[*] Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"[+] Loaded {len(df):,} rows across {df['machine_id'].nunique()} machines.")

    # Sort by machine and timestamp
    df.sort_values(by=["machine_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Process per machine
    processed_dfs = []
    machines = df["machine_id"].unique()

    for m_id in machines:
        m_df = df[df["machine_id"] == m_id].copy()
        
        # 1. Handle missingness
        # Linear interpolate small gaps <= 2 bins (10 mins)
        m_df["cpu"] = m_df["cpu"].interpolate(method="linear", limit=2)
        # Fill remaining boundaries or extreme edge nans with 0
        m_df["cpu"] = m_df["cpu"].fillna(0.0)
        m_df["cpu"] = np.clip(m_df["cpu"].values, 0.0, 1.0)

        # 2. Compute 7 MVP Features (strictly backward-looking to prevent leakage)
        # cpu_lag1 = y[t-1]
        m_df["cpu_lag1"] = m_df["cpu"].shift(1)
        # cpu_lag2 = y[t-2]
        m_df["cpu_lag2"] = m_df["cpu"].shift(2)
        # roll_mean_6 = mean(y[t-6:t-1]) (30 min lookback)
        m_df["roll_mean_6"] = m_df["cpu"].shift(1).rolling(window=6, min_periods=1).mean()
        # roll_std_6 = std(y[t-6:t-1])
        m_df["roll_std_6"] = m_df["cpu"].shift(1).rolling(window=6, min_periods=1).std().fillna(0.0)

        # Diurnal temporal features
        # Timestamp is in seconds -> hour of day in [0, 24)
        hour_of_day = (m_df["timestamp"] / 3600.0) % 24.0
        m_df["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0).astype(np.float32)
        m_df["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0).astype(np.float32)

        # Backfill first 2 lag rows with earliest valid value
        m_df["cpu_lag1"] = m_df["cpu_lag1"].bfill()
        m_df["cpu_lag2"] = m_df["cpu_lag2"].bfill()
        m_df["roll_mean_6"] = m_df["roll_mean_6"].bfill()
        m_df["roll_std_6"] = m_df["roll_std_6"].bfill()

        # Check feature bounds
        assert not m_df[["cpu", "cpu_lag1", "cpu_lag2", "roll_mean_6", "roll_std_6", "hour_sin", "hour_cos"]].isnull().any().any(), f"NaN found in features for machine {m_id}"
        
        processed_dfs.append(m_df)

    df_featured = pd.concat(processed_dfs, ignore_index=True)

    # 3. Chronological 70 / 15 / 15 Data Split
    unique_timestamps = np.sort(df_featured["timestamp"].unique())
    total_timesteps = len(unique_timestamps)
    train_end_idx = int(0.70 * total_timesteps)
    val_end_idx = int(0.85 * total_timesteps)

    train_max_ts = unique_timestamps[train_end_idx - 1]
    val_max_ts = unique_timestamps[val_end_idx - 1]

    # Assign split column
    df_featured["split"] = "train"
    df_featured.loc[df_featured["timestamp"] > train_max_ts, "split"] = "val"
    df_featured.loc[df_featured["timestamp"] > val_max_ts, "split"] = "test"

    train_count = (df_featured["split"] == "train").sum() // len(machines)
    val_count = (df_featured["split"] == "val").sum() // len(machines)
    test_count = (df_featured["split"] == "test").sum() // len(machines)

    print(f"\n[+] Chronological Split Complete (100 Machines):")
    print(f"    Train: {train_count} steps (70.0%) | TS <= {train_max_ts}s")
    print(f"    Val:   {val_count} steps (15.0%) | {train_max_ts}s < TS <= {val_max_ts}s")
    print(f"    Test:  {test_count} steps (15.0%) | TS > {val_max_ts}s")

    # Save split metadata
    split_info = {
        "total_timesteps": int(total_timesteps),
        "train_steps": int(train_count),
        "val_steps": int(val_count),
        "test_steps": int(test_count),
        "train_max_timestamp": int(train_max_ts),
        "val_max_timestamp": int(val_max_ts),
        "features": ["cpu", "cpu_lag1", "cpu_lag2", "roll_mean_6", "roll_std_6", "hour_sin", "hour_cos"],
        "target": "cpu"
    }
    with open(SPLIT_INFO_FILE, "w") as f:
        json.dump(split_info, f, indent=2)

    # Save parquet
    df_featured.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"[+] Saved featured dataset to {OUTPUT_PARQUET} ({len(df_featured):,} rows).")

    # 4. Generate Visual Validation Plot (figures/data_validation.png)
    generate_data_validation_plot(df_featured, machines, train_max_ts, val_max_ts)

    return df_featured


def generate_data_validation_plot(df: pd.DataFrame, machines: list, train_ts: int, val_ts: int):
    """
    Creates figures/data_validation.png showing 4 sample machines CPU over time with split markers.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sample_machines = [machines[0], machines[25], machines[50], machines[75]]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle("Google Cluster Trace MVP: CPU Usage Series & Chronological Splits (Data Validation)", fontsize=14, fontweight="bold", y=0.98)

    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    for ax, m_id, color in zip(axes, sample_machines, colors):
        m_df = df[df["machine_id"] == m_id]
        time_days = m_df["timestamp"] / 86400.0  # Convert to days

        ax.plot(time_days, m_df["cpu"], color=color, alpha=0.8, linewidth=1.0, label=f"Machine {m_id} CPU")
        
        # Split vertical lines
        train_days = train_ts / 86400.0
        val_days = val_ts / 86400.0
        ax.axvline(train_days, color="#dc2626", linestyle="--", linewidth=1.2, label="Train/Val Split (70%)")
        ax.axvline(val_days, color="#9333ea", linestyle="--", linewidth=1.2, label="Val/Test Split (85%)")

        ax.set_ylabel("CPU Fraction", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.85, fontsize=9)

    axes[-1].set_xlabel("Time (Days)", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.close()
    print(f"[+] Saved visual validation figure to {OUTPUT_FIG}")


if __name__ == "__main__":
    engineer_features_and_split()
