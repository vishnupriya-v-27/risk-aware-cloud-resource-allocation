# PROJECT SPECIFICATION: Google Cluster Trace MVP (v2.0)
**Document Status:** FROZEN / LOCKED  
**Scope:** 16–20 Hour Google-Only MVP (100 Eligible Machines)  
**Experiment Hypothesis:** An adaptive uncertainty signal selector ($U^*$) dynamically choosing among model disagreement ($U_1$), recent forecast error EWMA ($U_2$), and local volatility ($U_3$) provides lower over-provisioning and better SLA compliance in cloud CPU resource allocation than any single fixed uncertainty signal or standard baseline.

---

## 1. Global Parameters & Reproducibility
- **Random Seed:** `42` (fixed for Python `random`, `numpy`, `torch`)
- **Dataset:** Google Cluster Trace 2011 (`clusterdata-2011-2`)
- **Target Metric:** CPU usage physical core fraction, normalized to $[0.0, 1.0]$.
- **Time Bin Resolution:** 5 minutes ($300$ seconds).
- **Time Horizon:** Full ~29 days trace ($\approx 8352$ bins per machine).
- **Cohort Size:** Exactly $100$ eligible machines.

---

## 2. Data Acquisition & Machine Eligibility Rules
1. **Candidate Sampling:** Sample candidate machine IDs from `machine_events` (or sampled parts) with `seed = 42`.
2. **Missingness & Usability Criteria:**
   - **Observed Zero:** Valid observation ($0.0$).
   - **No task record in 5-min window:** Missing (`NaN`).
   - **Internal gap $\le 2$ bins ($\le 10$ min):** Linear interpolation allowed.
   - **Internal gap $> 2$ bins:** Left as `NaN` (or marks segment breaks).
   - **Machine-level Missingness:** If total missing intervals $> 20\%$ after interpolation, **discard machine** and sample next candidate.
3. **Cohort Selection:** Selection loop runs until exactly 100 eligible machines meeting the $\le 20\%$ missing threshold are acquired.
4. **Artifact:** Output list of final IDs stored in `data/google/selected_machines.json`.

---

## 3. CPU Aggregation (Interval-Overlap-Weighted)
- **Aggregation Method:** For any task usage record spanning $[t_{\text{start}}, t_{\text{end}}]$ with average CPU usage $c$:
  - Overlap with 5-minute bin $[T, T+300]$ is $w = \max(0, \min(t_{\text{end}}, T+300) - \max(t_{\text{start}}, T))$.
  - Weighted contribution to bin is $c \times \frac{w}{300}$.
  - Machine CPU at bin $T$ is the sum of weighted contributions of all tasks active during that bin.
- **Reference vs Optimized Equivalence:** Vectorized aggregation must match pure reference implementation on synthetic benchmarks with `np.allclose(ref, opt) == True`.
- **Value Bounds:** Enforce $\text{CPU} \in [0.0, 1.0]$ with strict assertion (flag and cap any measurement anomalies).

---

## 4. Feature Engineering & Data Splitting
- **Chronological Split:** $70\%$ Train, $15\%$ Validation, $15\%$ Test.
  - **NO shuffling** across time.
- **7 GRU Input Features:**
  1. `cpu`: Current normalized CPU usage $y_t \in [0, 1]$.
  2. `cpu_lag1`: Usage at $t-1$ ($y_{t-1}$).
  3. `cpu_lag2`: Usage at $t-2$ ($y_{t-2}$).
  4. `roll_mean_6`: Rolling mean of previous 6 steps ($t-6$ to $t-1$, 30-min window).
  5. `roll_std_6`: Rolling standard deviation of previous 6 steps ($t-6$ to $t-1$).
  6. `hour_sin`: $\sin(2\pi \times \text{hour} / 24)$ (diurnal cycle).
  7. `hour_cos`: $\cos(2\pi \times \text{hour} / 24)$ (diurnal cycle).
- **Sequence Parameters:**
  - Input sequence length ($L$): $12$ timesteps (1 hour lookback).
  - Target: $y_{t}$ (1-step-ahead forecast, next 5 minutes).

---

## 5. Forecasting Models

### Baseline 0 (Naive / Persistence)
- Prediction: $\hat{y}_t = y_{t-1}$
- Calculated from continuous series (so test step 0 uses the final validation step).

### Baseline 0b (Seasonal Naive)
- Prediction: $\hat{y}_t = y_{t-288}$ ($288 \times 5\text{ min} = 24\text{ hours}$).
- Calculated from continuous series across split boundaries.

### Baseline 1 (Holt-Winters / Exponential Smoothing)
- Fitted per machine.
- Seasonality parameter grid search on validation MAE: $m \in \{48, 144, 288\}$ (4h, 12h, 24h).
- Refit best $m$ on Train + Validation before generating Test predictions.

### Baseline 2 (Global GRU)
- **Architecture:**
  - Input dimension: 7 features.
  - GRU Layer: `hidden_size = 32`, `num_layers = 1`.
  - Explicit Dropout: $p = 0.2$ applied to GRU output.
  - Linear Layer: $32 \to 1$.
  - Activation: Sigmoid (guarantees output $\in [0, 1]$).
- **Optimization:**
  - Optimizer: `Adam(lr=1e-3)`
  - Loss: Mean Squared Error (MSE)
  - Max Epochs: 50
  - Batch Size: 64
  - Early Stopping: Patience = 10 epochs on Validation Loss.
  - LR Scheduler: `ReduceLROnPlateau(factor=0.5, patience=5)`.

### Fusion Model
- Linear combination:
  $$\hat{y}_t = \alpha \hat{y}_{\text{HW}, t} + (1 - \alpha) \hat{y}_{\text{GRU}, t}$$
- Grid search $\alpha \in \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$ on Validation MAE.

---

## 6. Uncertainty Quantification Signals

All uncertainty signals are computed strictly causally without lookahead:

1. **$U_1$ (Model Disagreement):**
   $$U_{1, t} = |\hat{y}_{\text{HW}, t} - \hat{y}_{\text{GRU}, t}|$$

2. **$U_2$ (Recent Error EWMA):**
   $$e_{t-1} = |\hat{y}_{t-1} - y_{t-1}|$$
   $$U_{2, t} = \gamma e_{t-1} + (1 - \gamma) U_{2, t-1}, \quad \text{with } \gamma = 0.2$$
   *Boundary Rule:* At test timestep $0$, $U_{2, \text{test}}[0]$ is initialized using the final validation timestep error (not test actuals).

3. **$U_3$ (Local Volatility):**
   $$U_{3, t} = \text{std}(y_{t-6 : t-1})$$
   *Boundary Rule:* Computed on the continuous ground truth series spanning into previous split segments.

---

## 7. Adaptive Uncertainty Selector ($U^*$)
- **Mechanism:** Rolling Spearman rank correlation between each uncertainty signal $U_i$ and the ground truth absolute error $|y - \hat{y}|$ over a backward window of $N$ timesteps.
- **Window candidates:** $N \in \{20, 30, 50, 75, 100\}$.
- **Selection Criterion:** At timestep $t$, select signal $U^*_t = U_{k, t}$ where:
  $$k = \arg\max_{i \in \{1, 2, 3\}} \text{SpearmanCorr}\left(U_{i, t-N : t-1}, |y_{t-N : t-1} - \hat{y}_{t-N : t-1}|\right)$$
- Optimal window $N^*$ tuned on the validation set.

---

## 8. Resource Allocation & SLA Risk Buffer
- **Allocation Rule:**
  $$A_t = \min\left(1.0, \max\left(0.0, \hat{y}_t + \beta U_t\right)\right)$$
- **Buffer Multiplier Tuning:**
  - Candidates: $\beta \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0\}$.
  - Tuning objective: Select the **minimum $\beta$** that satisfies the target SLA on the validation set:
    $$\text{SLA Breach Rate}_{\text{val}}(\beta) \le 0.02 \quad (98\% \text{ Target SLA})$$
  - Grace Zone: $\delta = 0.005$ (CPU under-allocations $\le 0.005$ are tolerated without SLA breach penalty).

---

## 9. Evaluation Configurations & Metrics

### The 9 Standard Configurations:
1. `B0`: Naive (Persistence) + zero buffer
2. `B0b`: Seasonal Naive + zero buffer
3. `B1`: GRU + zero buffer
4. `B2`: Holt-Winters + zero buffer
5. `B3`: Hybrid Fusion + fixed buffer ($\beta \times \text{mean}(U)$)
6. `B4a`: Hybrid Fusion + $U_1$ buffer ($\beta U_1$)
7. `B4b`: Hybrid Fusion + $U_2$ buffer ($\beta U_2$)
8. `B4c`: Hybrid Fusion + $U_3$ buffer ($\beta U_3$)
9. `P`: Proposed Adaptive Hybrid Fusion + $U^*$ buffer ($\beta U^*$)

### Evaluation Metrics (Evaluated on identical test sets):
1. **MAE (Mean Absolute Error):** $\frac{1}{T} \sum |y_t - \hat{y}_t|$
2. **RMSE (Root Mean Square Error):** $\sqrt{\frac{1}{T} \sum (y_t - \hat{y}_t)^2}$
3. **SLA Breach Rate:** Fraction of timesteps where $y_t > A_t + \delta$ (where $\delta = 0.005$)
4. **Overshoot Rate:** Fraction of timesteps where $A_t > y_t$
5. **Grace Zone Rate:** Fraction of timesteps where $A_t < y_t \le A_t + \delta$
6. **Over-provisioning (Waste):** $\frac{1}{T} \sum \max(0, A_t - y_t)$
7. **Under-provisioning (Deficit):** $\frac{1}{T} \sum \max(0, y_t - A_t)$

---

## 10. Integrity & Leakage Checklist
- [ ] No future ground truth $y_{t+k}$ ($k \ge 0$) used in feature generation or prediction at $t$.
- [ ] Split boundary transitions seamlessly supply historical lags/windows from previous split without lookahead.
- [ ] All 9 models tested on the exact same timestamps and machine records.
- [ ] Aggregated CPU values strictly bounded in $[0.0, 1.0]$.
