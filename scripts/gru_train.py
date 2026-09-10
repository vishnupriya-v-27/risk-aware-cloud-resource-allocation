"""
Global GRU Forecaster Training and Inference Script
Part of the 16-20 Hour Google Cluster Trace MVP (Hour 7-9).

Architecture:
- Input Features (7): cpu, cpu_lag1, cpu_lag2, roll_mean_6, roll_std_6, hour_sin, hour_cos
- Sequence Length: 12 timesteps (1 hour lookback)
- GRU: hidden_size=32, num_layers=1
- Explicit Dropout: 0.2
- Linear: 32 -> 1
- Sigmoid Output: guarantees predicted CPU in [0.0, 1.0]

Training:
- Global training across all 100 machines
- Chronological ordering
- Optimizer: Adam(lr=1e-3), Loss: MSE
- Max epochs: 50, Early stopping: patience 10 on Val MSE
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
"""

import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
INPUT_PARQUET = DATA_DIR / "google_features_5min.parquet"
MODEL_SAVE_PATH = MODELS_DIR / "best_gru.pt"
PREDICTIONS_CSV = MODELS_DIR / "gru_predictions.csv"
METRICS_JSON = RESULTS_DIR / "gru_metrics.json"

# Fixed Seeds for Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SEQ_LEN = 12
BATCH_SIZE = 64
FEATURE_COLS = ["cpu", "cpu_lag1", "cpu_lag2", "roll_mean_6", "roll_std_6", "hour_sin", "hour_cos"]


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets, meta):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        self.meta = meta  # list of (machine_id, timestamp, split)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class GlobalGRUNetwork(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=32, dropout_prob=0.2):
        super(GlobalGRUNetwork, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Take the hidden state of the last timestep
        last_hidden = out[:, -1, :]
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        pred = self.sigmoid(logits)
        return pred


def create_sequences_per_machine(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Construct sliding window sequences per machine to respect boundaries.
    """
    train_seqs, train_tgts, train_meta = [], [], []
    val_seqs, val_tgts, val_meta = [], [], []
    test_seqs, test_tgts, test_meta = [], [], []

    machines = df["machine_id"].unique()
    for m_id in machines:
        m_df = df[df["machine_id"] == m_id].copy().reset_index(drop=True)
        feats = m_df[FEATURE_COLS].values
        targets = m_df["cpu"].values
        timestamps = m_df["timestamp"].values
        splits = m_df["split"].values

        for t in range(seq_len, len(m_df)):
            # Lookback window: t-seq_len to t-1
            seq = feats[t - seq_len : t]
            target = targets[t]
            meta = (m_id, timestamps[t], splits[t])

            sp = splits[t]
            if sp == "train":
                train_seqs.append(seq)
                train_tgts.append(target)
                train_meta.append(meta)
            elif sp == "val":
                val_seqs.append(seq)
                val_tgts.append(target)
                val_meta.append(meta)
            elif sp == "test":
                test_seqs.append(seq)
                test_tgts.append(target)
                test_meta.append(meta)

    return (
        TimeSeriesDataset(np.array(train_seqs), np.array(train_tgts), train_meta),
        TimeSeriesDataset(np.array(val_seqs), np.array(val_tgts), val_meta),
        TimeSeriesDataset(np.array(test_seqs), np.array(test_tgts), test_meta)
    )


def train_gru():
    print("=" * 60)
    print(" Google Cluster Trace MVP — Global GRU Forecaster (Hour 7–9)")
    print("=" * 60)

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input file {INPUT_PARQUET} not found. Run features.py first!")

    print(f"[*] Reading dataset from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    df.sort_values(by=["machine_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[*] Building sliding window sequences (lookback={SEQ_LEN} steps = 1 hour)...")
    train_dataset, val_dataset, test_dataset = create_sequences_per_machine(df, SEQ_LEN)

    print(f"[+] Dataset created:")
    print(f"    Train sequences: {len(train_dataset):,}")
    print(f"    Val sequences:   {len(val_dataset):,}")
    print(f"    Test sequences:  {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Using compute device: {device}")

    model = GlobalGRUNetwork(input_dim=len(FEATURE_COLS), hidden_dim=32, dropout_prob=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    max_epochs = 50
    early_stop_patience = 10
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"[*] Starting GRU training (max_epochs={max_epochs}, early_stop_patience={early_stop_patience})...")

    for epoch in range(1, max_epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(x_batch)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
            print(f"    Epoch {epoch:02d}/{max_epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[!] Early stopping triggered at epoch {epoch}. Best Val MSE: {best_val_loss:.5f}")
                break

    print(f"[+] Loaded best model weights from {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Generate predictions across full dataset
    def predict_split(dataset, loader, split_name):
        preds_list, acts_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                preds = model(x_batch).cpu().numpy().squeeze(-1)
                preds_list.extend(preds)
                acts_list.extend(y_batch.numpy().squeeze(-1))

        preds_arr = np.array(preds_list)
        acts_arr = np.array(acts_list)
        mae = float(np.mean(np.abs(acts_arr - preds_arr)))
        rmse = float(np.sqrt(np.mean((acts_arr - preds_arr) ** 2)))

        rows = []
        for i in range(len(dataset)):
            m_id, ts, sp = dataset.meta[i]
            rows.append({
                "machine_id": m_id,
                "timestamp": ts,
                "split": sp,
                "actual": float(acts_arr[i]),
                "predicted": float(preds_arr[i])
            })
        return rows, mae, rmse

    print(f"[*] Generating predictions on Train, Val, and Test splits...")
    train_rows, train_mae, train_rmse = predict_split(train_dataset, train_loader, "train")
    val_rows, val_mae, val_rmse = predict_split(val_dataset, val_loader, "val")
    test_rows, test_mae, test_rmse = predict_split(test_dataset, test_loader, "test")

    # Combine all predictions
    all_pred_rows = train_rows + val_rows + test_rows
    df_gru_preds = pd.DataFrame(all_pred_rows)
    df_gru_preds.sort_values(by=["machine_id", "timestamp"], inplace=True)
    df_gru_preds.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[+] Saved GRU predictions to {PREDICTIONS_CSV} ({len(df_gru_preds):,} rows).")

    # Save metrics
    metrics = {
        "model": "Global_GRU",
        "architecture": "GRU(input=7, hidden=32, dropout=0.2) -> Linear(32, 1) -> Sigmoid",
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # Load baseline metrics for comparison
    with open(RESULTS_DIR / "baseline_results.json", "r") as f:
        baseline_metrics = json.load(f)

    print("\n" + "=" * 65)
    print(" FORECASTING PERFORMANCE COMPARISON ON TEST SET (30,300 points)")
    print("=" * 65)
    print(f"  B0  Naive (Persistence):     MAE = {baseline_metrics['B0 (Naive)']['test_mae']:.4f} | RMSE = {baseline_metrics['B0 (Naive)']['test_rmse']:.4f}")
    print(f"  B0b Seasonal Naive (24h):   MAE = {baseline_metrics['B0b (Seasonal Naive)']['test_mae']:.4f} | RMSE = {baseline_metrics['B0b (Seasonal Naive)']['test_rmse']:.4f}")
    print(f"  B2  Holt-Winters (HW):       MAE = {baseline_metrics['B2 (Holt-Winters)']['test_mae']:.4f} | RMSE = {baseline_metrics['B2 (Holt-Winters)']['test_rmse']:.4f}")
    print(f"  B1  Global GRU:              MAE = {test_mae:.4f} | RMSE = {test_rmse:.4f}")
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    train_gru()
