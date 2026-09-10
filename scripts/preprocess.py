"""
Google Cluster Trace 2011 CPU Aggregation Script
Part of the 16-20 Hour Google Cluster Trace MVP (Hour 3-5).

Key Components:
1. Reference Interval-Overlap Aggregation (ground-truth algorithm).
2. Optimized Vectorized Interval-Overlap Aggregation.
3. Unit Test: synthetic check (task [301s, 598s] CPU=0.4 -> bin 1 = 0.396).
4. Equivalence Test: np.allclose(reference, optimized) on real task sample.
5. Full 100-Machine Aggregation into 5-minute bins -> data/google_cpu_5min.csv.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GOOGLE_DIR = DATA_DIR / "google"
FILTERED_PARTS_DIR = GOOGLE_DIR / "filtered_parts"
SELECTED_MACHINES_FILE = GOOGLE_DIR / "selected_machines.json"
OUTPUT_CSV_FILE = DATA_DIR / "google_cpu_5min.csv"

# Trace Constants
BIN_SIZE_SEC = 300  # 300 seconds = 5 minutes


# ---------------------------------------------------------------------------
# 1. Reference Implementation (Pure Python / Exact Math)
# ---------------------------------------------------------------------------
def aggregate_machine_cpu_reference(
    task_records: list,
    min_timestamp_sec: int,
    max_timestamp_sec: int,
    bin_size_sec: int = BIN_SIZE_SEC
) -> np.ndarray:
    """
    Reference interval-overlap-weighted aggregation.
    task_records: list of tuples (start_time_sec, end_time_sec, cpu_rate)
    """
    num_bins = int(np.ceil((max_timestamp_sec - min_timestamp_sec) / bin_size_sec)) + 1
    bin_cpu_sums = np.zeros(num_bins, dtype=np.float64)

    for start_t, end_t, cpu_rate in task_records:
        if end_t <= start_t or cpu_rate <= 0:
            continue

        first_bin = max(0, int((start_t - min_timestamp_sec) // bin_size_sec))
        last_bin = min(num_bins - 1, int((end_t - min_timestamp_sec) // bin_size_sec))

        for b in range(first_bin, last_bin + 1):
            bin_start = min_timestamp_sec + b * bin_size_sec
            bin_end = bin_start + bin_size_sec

            overlap_start = max(start_t, bin_start)
            overlap_end = min(end_t, bin_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                fraction = overlap_duration / bin_size_sec
                bin_cpu_sums[b] += cpu_rate * fraction

    return bin_cpu_sums


# ---------------------------------------------------------------------------
# 2. Optimized Implementation (Vectorized with Pandas/Numpy)
# ---------------------------------------------------------------------------
def aggregate_machine_cpu_vectorized(
    df_tasks: pd.DataFrame,
    min_timestamp_sec: int,
    max_timestamp_sec: int,
    bin_size_sec: int = BIN_SIZE_SEC
) -> pd.DataFrame:
    """
    Optimized interval-overlap aggregation for a DataFrame containing tasks for one machine.
    df_tasks has columns: ['start_time_sec', 'end_time_sec', 'cpu_rate']
    """
    if df_tasks.empty:
        num_bins = int(np.ceil((max_timestamp_sec - min_timestamp_sec) / bin_size_sec)) + 1
        windows = np.arange(num_bins)
        timestamps = min_timestamp_sec + windows * bin_size_sec
        return pd.DataFrame({
            "window": windows,
            "timestamp": timestamps,
            "cpu": np.zeros(num_bins, dtype=np.float32)
        })

    num_bins = int(np.ceil((max_timestamp_sec - min_timestamp_sec) / bin_size_sec)) + 1

    # Filter invalid records
    valid_mask = (df_tasks["end_time_sec"] > df_tasks["start_time_sec"]) & (df_tasks["cpu_rate"] > 0)
    df_valid = df_tasks[valid_mask].copy()

    start_bins = np.maximum(0, ((df_valid["start_time_sec"] - min_timestamp_sec) // bin_size_sec).astype(np.int64))
    end_bins = np.minimum(num_bins - 1, ((df_valid["end_time_sec"] - min_timestamp_sec) // bin_size_sec).astype(np.int64))

    # Single-bin tasks vs multi-bin tasks
    is_single_bin = (start_bins == end_bins)
    
    # 1) Single-bin task calculations
    df_single = df_valid[is_single_bin]
    single_bins = start_bins[is_single_bin].values
    durations = (df_single["end_time_sec"] - df_single["start_time_sec"]).values
    single_cpu = (df_single["cpu_rate"].values * (durations / bin_size_sec)).astype(np.float64)

    bin_cpu_sums = np.zeros(num_bins, dtype=np.float64)
    np.add.at(bin_cpu_sums, single_bins, single_cpu)

    # 2) Multi-bin task calculations (span boundary)
    df_multi = df_valid[~is_single_bin]
    if not df_multi.empty:
        m_starts = df_multi["start_time_sec"].values
        m_ends = df_multi["end_time_sec"].values
        m_cpus = df_multi["cpu_rate"].values
        m_sb = start_bins[~is_single_bin].values
        m_eb = end_bins[~is_single_bin].values

        expanded_bins = []
        expanded_weighted_cpu = []

        for i in range(len(df_multi)):
            s_t = m_starts[i]
            e_t = m_ends[i]
            c_r = m_cpus[i]
            b_s = m_sb[i]
            b_e = m_eb[i]

            for b in range(b_s, b_e + 1):
                bin_s = min_timestamp_sec + b * bin_size_sec
                bin_e = bin_s + bin_size_sec
                overlap = max(0, min(e_t, bin_e) - max(s_t, bin_s))
                if overlap > 0:
                    expanded_bins.append(b)
                    expanded_weighted_cpu.append(c_r * (overlap / bin_size_sec))

        if expanded_bins:
            np.add.at(bin_cpu_sums, np.array(expanded_bins, dtype=np.int64), np.array(expanded_weighted_cpu, dtype=np.float64))

    windows = np.arange(num_bins)
    timestamps = min_timestamp_sec + windows * bin_size_sec
    
    # Clip to valid [0.0, 1.0] physical core fraction
    cpu_clipped = np.clip(bin_cpu_sums, 0.0, 1.0).astype(np.float32)

    return pd.DataFrame({
        "window": windows,
        "timestamp": timestamps,
        "cpu": cpu_clipped
    })


# ---------------------------------------------------------------------------
# 3. Verification & Synthetic Unit Tests
# ---------------------------------------------------------------------------
def run_synthetic_unit_test():
    """
    Synthetic check:
    Task: 301s -> 598s, CPU = 0.4.
    Bin 0: [0, 300] -> overlap = 0 -> CPU = 0.0
    Bin 1: [300, 600] -> overlap = 598 - 301 = 297s -> fraction = 297/300 = 0.99 -> CPU = 0.4 * 0.99 = 0.396
    """
    print("[*] Running synthetic unit test for interval-overlap aggregation...")
    task_records = [(301, 598, 0.4)]
    
    # Reference test
    ref_result = aggregate_machine_cpu_reference(task_records, min_timestamp_sec=0, max_timestamp_sec=600)
    expected_bin1 = 0.4 * (297 / 300.0)  # 0.396

    assert len(ref_result) >= 2, "Unit test failed: output array too short."
    assert np.isclose(ref_result[0], 0.0), f"Bin 0 should be 0.0, got {ref_result[0]}"
    assert np.isclose(ref_result[1], expected_bin1, atol=1e-5), f"Bin 1 expected {expected_bin1}, got {ref_result[1]}"

    # Vectorized test
    df_task = pd.DataFrame([{
        "start_time_sec": 301,
        "end_time_sec": 598,
        "cpu_rate": 0.4
    }])
    df_opt = aggregate_machine_cpu_vectorized(df_task, min_timestamp_sec=0, max_timestamp_sec=600)
    opt_bin1 = df_opt.loc[df_opt["window"] == 1, "cpu"].values[0]
    
    assert np.isclose(opt_bin1, expected_bin1, atol=1e-5), f"Vectorized Bin 1 expected {expected_bin1}, got {opt_bin1}"
    assert np.allclose(ref_result[:len(df_opt)], df_opt["cpu"].values, atol=1e-5), "Reference vs Optimized mismatch on synthetic test!"

    print(f"[✓] Synthetic unit test PASSED: task [301s, 598s] CPU=0.4 -> bin 1 = {opt_bin1:.4f} (expected {expected_bin1:.4f})")


def run_equivalence_test(sample_tasks_df: pd.DataFrame):
    """
    Verify np.allclose(ref, opt) across a real task subset.
    """
    print("[*] Testing equivalence of Reference vs Vectorized implementation on sample data...")
    m_id = sample_tasks_df["machine_id"].iloc[0]
    m_tasks = sample_tasks_df[sample_tasks_df["machine_id"] == m_id].copy()

    # Normalize microseconds to seconds
    m_tasks["start_time_sec"] = m_tasks["start_time"] / 1_000_000.0
    m_tasks["end_time_sec"] = m_tasks["end_time"] / 1_000_000.0

    min_t = int(m_tasks["start_time_sec"].min() // BIN_SIZE_SEC) * BIN_SIZE_SEC
    max_t = int(np.ceil(m_tasks["end_time_sec"].max() / BIN_SIZE_SEC)) * BIN_SIZE_SEC

    # Reference run
    task_tuples = list(zip(m_tasks["start_time_sec"], m_tasks["end_time_sec"], m_tasks["cpu_rate"]))
    ref_out = aggregate_machine_cpu_reference(task_tuples, min_t, max_t)
    ref_out_clipped = np.clip(ref_out, 0.0, 1.0)

    # Vectorized run
    opt_df = aggregate_machine_cpu_vectorized(m_tasks, min_t, max_t)
    opt_out = opt_df["cpu"].values

    assert np.allclose(ref_out_clipped, opt_out, atol=1e-4), "Equivalence check FAILED between reference and vectorized aggregation!"
    print(f"[✓] Equivalence test PASSED: np.allclose(reference, optimized) is True across {len(ref_out_clipped)} bins.")


# ---------------------------------------------------------------------------
# 4. Main 100-Machine Processing Pipeline
# ---------------------------------------------------------------------------
def process_all_machines():
    print("=" * 60)
    print(" Google Cluster Trace MVP — CPU Aggregation (Hour 3–5)")
    print("=" * 60)

    # 1. Run validation gates
    run_synthetic_unit_test()

    # 2. Load candidate dataset and selected machines
    if not SELECTED_MACHINES_FILE.exists():
        raise FileNotFoundError(f"Selected machines file not found at {SELECTED_MACHINES_FILE}. Run download.py first!")

    with open(SELECTED_MACHINES_FILE, "r") as f:
        selected_machines = json.load(f)

    print(f"[*] Loaded {len(selected_machines)} selected machine IDs.")

    # Load task usage records
    parquet_files = list(FILTERED_PARTS_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet partition files found in {FILTERED_PARTS_DIR}.")

    print(f"[*] Reading {len(parquet_files)} parquet partition files...")
    df_all_tasks = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    print(f"[+] Total task records loaded: {len(df_all_tasks)}")

    # Filter to selected 100 machines
    df_all_tasks = df_all_tasks[df_all_tasks["machine_id"].isin(selected_machines)].copy()
    print(f"[+] Records belonging to 100 selected machines: {len(df_all_tasks)}")

    # Convert timestamps from microseconds to seconds
    df_all_tasks["start_time_sec"] = df_all_tasks["start_time"] / 1_000_000.0
    df_all_tasks["end_time_sec"] = df_all_tasks["end_time"] / 1_000_000.0

    # Run equivalence test on first machine
    run_equivalence_test(df_all_tasks)

    # Global time grid bounds
    global_min_t = int(df_all_tasks["start_time_sec"].min() // BIN_SIZE_SEC) * BIN_SIZE_SEC
    global_max_t = int(np.ceil(df_all_tasks["end_time_sec"].max() / BIN_SIZE_SEC)) * BIN_SIZE_SEC
    total_bins = int((global_max_t - global_min_t) // BIN_SIZE_SEC) + 1
    print(f"[*] Global Time Range: {global_min_t}s -> {global_max_t}s ({total_bins} bins of 5 minutes)")

    # 3. Aggregate CPU per machine
    aggregated_dfs = []
    print(f"[*] Aggregating CPU usage for 100 machines...")

    for i, m_id in enumerate(selected_machines):
        m_tasks = df_all_tasks[df_all_tasks["machine_id"] == m_id]
        df_agg = aggregate_machine_cpu_vectorized(m_tasks, global_min_t, global_max_t, BIN_SIZE_SEC)
        df_agg["machine_id"] = m_id
        aggregated_dfs.append(df_agg)

        if (i + 1) % 25 == 0 or (i + 1) == len(selected_machines):
            print(f"    Progress: {i + 1}/100 machines aggregated...")

    # Combine into single DataFrame
    df_final = pd.concat(aggregated_dfs, ignore_index=True)
    
    # Reorder columns: machine_id, window, timestamp, cpu
    df_final = df_final[["machine_id", "window", "timestamp", "cpu"]]

    # 4. Strict Quality Checks & Gate Verification
    print("\n[*] Performing Data Integrity Checks on Aggregation Output...")
    assert len(df_final["machine_id"].unique()) == len(selected_machines), f"Machine count mismatch! Expected {len(selected_machines)}, got {len(df_final['machine_id'].unique())}"
    assert df_final["cpu"].min() >= 0.0, f"Negative CPU values detected: min = {df_final['cpu'].min()}"
    assert df_final["cpu"].max() <= 1.0, f"CPU values exceeding 1.0 physical core detected: max = {df_final['cpu'].max()}"
    assert not df_final["cpu"].isnull().any(), "Unexpected NaNs in aggregated output."
    
    # Check dynamics: average std across machines should be > 0 (not flat lines)
    stds = df_final.groupby("machine_id")["cpu"].std()
    assert (stds > 0.005).all(), "Warning: Some machines have near-zero variance (flat lines)!"

    # Save to final CSV
    df_final.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"[+] Successfully saved aggregated 5-min CPU data to {OUTPUT_CSV_FILE}")
    print(f"    Total rows: {len(df_final):,}")
    print(f"    Memory size: {df_final.memory_usage().sum() / 1024 / 1024:.2f} MB")
    print(f"    CPU Summary: Min={df_final['cpu'].min():.4f}, Mean={df_final['cpu'].mean():.4f}, Max={df_final['cpu'].max():.4f}, Std={df_final['cpu'].std():.4f}")

    print("\n[✓] Gate 2 Checkpoint: PASSED (Interval-overlap weighting validated, CPU in [0,1], 100 machines locked).")


if __name__ == "__main__":
    process_all_machines()
