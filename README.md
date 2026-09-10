# A Lightweight Risk-Aware Framework for Cost-Optimized Cloud Resource Allocation Using Hybrid Workload Prediction and Confidence Estimation

An empirical evaluation of a lightweight, estimator-agnostic, risk-aware cloud CPU resource allocation framework evaluated on Google Cluster Trace v2.

---

## 📌 Project Overview

Cloud resource autoscaling requires balancing SLA reliability (minimizing service degradation and under-allocation breaches) with resource efficiency (minimizing costly over-provisioning waste). This repository provides a complete, reproducible, and causal experimental framework that tests:

1. **Workload Forecasting:** Causal comparison between a lightweight Global Gated Recurrent Unit (GRU), Holt-Winters statistical modeling, and baseline persistence models.
2. **Heterogeneous Risk Sensing:** Extraction of three distinct dynamic uncertainty/risk proxies:
   - $U_1$: Model Disagreement ($|\hat{y}_{\text{HW}} - \hat{y}_{\text{GRU}}|$)
   - $U_2$: Recent Error Inertia ($\text{EWMA}_{\gamma=0.2}(|y_{t-1} - \hat{y}_{t-1}|)$)
   - $U_3$: Workload Volatility ($\text{std}(y_{t-6:t-1})$)
3. **Scale Harmonization (Method 1):** Validation Mean-Error Matching ($c_k = \overline{e}_{\text{val}} / \overline{u}_{k,\text{val}}$) to bring heterogeneous proxies onto a unified physical error scale without altering rank associations.
4. **Adaptive Reliability Selection:** Dynamic proxy selection via rolling Spearman rank correlation on historical error windows ($N^*=75$).
5. **Risk-Aware Resource Allocation:** Bounded allocation rule $A_t = \min(1.0, \max(0.0, \hat{y}_t + \beta^* \tilde{U}^*(t)))$ with validation-calibrated multiplier $\beta^*=3.00$ and grace tolerance $\delta=0.005$.

---

## 🔄 End-to-End Pipeline

```
Google Cluster Trace v2 (Task Usage)
  │
  ▼
5-Minute Overlap-Weighted CPU Aggregation (100 Machines × 2,018 Bins = 201,800 Rows)
  │
  ▼
Causal Feature Engineering (7 Features, 70/15/15 Chronological Split)
  │
  ▼
Forecasting Benchmarks (Naive, Seasonal Naive, Holt-Winters, Global GRU)
  │
  ▼
Model Fusion (Validation Optimization → α* = 0.00, GRU-based deployment)
  │
  ▼
Heterogeneous Risk Proxies (U1: Disagreement, U2: Error EWMA, U3: Volatility)
  │
  ▼
Validation Mean-Error Scale Calibration (c1 = 0.4285, c2 = 1.0025, c3 = 0.7751)
  │
  ▼
Adaptive Selector (Rolling Spearman Correlation on Past Errors, N* = 75)
  │
  ▼
Risk Buffer Multiplier Calibration (Validation Grid Sweep → β* = 3.00)
  │
  ▼
Test Set Resource Allocation & Pareto Evaluation (30,300 Timesteps, 9 Configurations)
```

---

## 📊 Dataset & Preprocessing

- **Dataset:** Google Cluster Trace v2 (`task_usage` & `machine_events`)
- **Machine Cohort:** Exactly 100 eligible machines selected deterministically (seed 42, missingness $\le 20\%$).
- **Aggregation:** Interval-overlap weighted CPU rate into continuous 5-minute bins ($2,018$ timesteps per machine).
- **Dataset Size:** Exactly $201,800$ observations with CPU utilization bounded strictly in $[0.0, 1.0]$.
- **Chronological Split (70/15/15):**
  - **Train:** 141,200 timesteps ($1,412$ per machine, $t=0 \dots 1411$)
  - **Validation:** 30,300 timesteps ($303$ per machine, $t=1412 \dots 1714$)
  - **Test:** 30,300 timesteps ($303$ per machine, $t=1715 \dots 2017$)

---

## 🔬 Empirical Benchmark Results

### 1. Workload Forecasting Performance (Test Split)

| Model | Model Description | Test MAE | Test RMSE |
| :--- | :--- | :---: | :---: |
| **Naive (Persistence)** | $y_{t-1}$ lag baseline | 0.0631 | 0.1080 |
| **Seasonal Naive (24h)** | $y_{t-288}$ 24-hour diurnal baseline | 0.1405 | 0.1882 |
| **Holt-Winters** | Multi-seasonality tuned ($m^* \in \{48, 144, 288\}$) | 0.1619 | 0.2107 |
| **Global GRU** | 7 causal features, hidden=32, dropout=0.20 | **0.0570** | **0.0943** |

*Note: Validation fusion grid search selected $\alpha^*=0.00$, confirming Global GRU as the deployed forecasting predictor. Holt-Winters is retained strictly as a secondary disagreement sensor for $U_1$.*

---

### 2. Resource Allocation & SLA Benchmark (30,300 Test Observations, $\beta^*=3.00$)

| Config | Strategy / Model Name | Buffer Strategy | SLA Breach (%) | Overshoot (%) | Waste (Cores) | Deficit (Cores) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | Naive (Persistence) | Zero Buffer | 42.75% | 53.08% | 0.0321 | 0.0310 |
| **B0b** | Seasonal Naive | Zero Buffer | 49.39% | 49.12% | 0.0686 | 0.0719 |
| **B1** | Global GRU | Zero Buffer | 39.86% | 56.45% | 0.0331 | 0.0239 |
| **B2** | Holt-Winters | Zero Buffer | 49.23% | 49.68% | 0.0853 | 0.0766 |
| **B3** | Fixed Mean Buffer | Fixed Scalar ($0.1659$) | **2.68%** | 97.24% | 0.1778 | **0.0027** |
| **B4a** | Calibrated Disagreement ($U_1$) | $3.0 \times \tilde{U}_1$ | 6.84% | 92.41% | 0.1981 | 0.0048 |
| **B4b** | Calibrated Error EWMA ($U_2$) | $3.0 \times \tilde{U}_2$ | **2.77%** | 96.94% | **0.1767** | 0.0031 |
| **B4c** | Calibrated Volatility ($U_3$) | $3.0 \times \tilde{U}_3$ | 4.20% | 95.29% | 0.1732 | 0.0038 |
| **P** | **Proposed Adaptive Calibrated** | **$3.0 \times \tilde{U}^*$ ($N^*=75$)** | **5.20%** | **94.26%** | **0.1796** | **0.0043** |

---

## 📈 Key Findings & Honest Scientific Assessment

1. **Scale Harmonization (Method 1):** Raw uncertainty proxies have disparate scales ($U_1 \approx 0.129, U_2 \approx 0.055, U_3 \approx 0.071$). Validation mean-error matching eliminated scale distortion and reduced over-provisioning waste by **$30.2\%$** ($0.1796$ vs. $0.2575$ cores) while preserving exact rank invariance ($0$ differing selector decisions).
2. **Comparative Strategy Performance:** On this evaluated Google Cluster Trace dataset, **calibrated Error EWMA ($B4b$, $\tilde{U}_2$) strictly dominates the proposed adaptive selector ($P$) across the entire Pareto frontier**, achieving a lower SLA breach rate ($2.77\%$ vs. $5.20\%$) and lower waste ($0.1767$ vs. $0.1796$).
3. **Selector Latency:** The rolling Spearman selector exhibits transition lag over its historical window ($N^*=75$ steps = 6.25h). During rapid burst transitions, switching to $U_1$ introduces higher breach penalties than maintaining continuous exponential smoothing ($U_2$).
4. **SLA Target:** The predefined validation SLA breach target of $\le 2.0\%$ was **not achieved** within the candidate multiplier grid $\beta \in [0.5, 3.0]$ (lowest validation breach was $5.62\%$ at $\beta^*=3.00$; test breach was $5.20\%$).

---

## ⚠️ Important Scope & Scientific Limitations

- **Dataset Scope:** Evaluated exclusively on a 100-machine sample of Google Cluster Trace v2. External multi-cloud traces (e.g., Alibaba) require separate evaluation.
- **Nature of Signals:** $U_1, U_2, U_3$ are heuristic risk proxies, not formal Bayesian confidence intervals or conformal prediction regions.
- **Forecaster Identity:** Fusion parameter optimization yielded $\alpha^*=0.00$; the deployed forecaster is GRU-only.
- **Performance Boundary:** The adaptive selector does not outperform fixed calibrated EWMA ($U_2$) on this trace under 5-minute sampling.

---

## 📁 Repository Structure

```
├── data/
│   ├── google/selected_machines.json    # 100 locked machine IDs
│   ├── google_cpu_5min.csv              # Aggregated 5-min CPU time-series
│   ├── google_features_5min.parquet     # 7 engineered features
│   └── split_info.json                  # Split indices & metadata
├── scripts/
│   ├── download.py                      # Data acquisition parser
│   ├── preprocess.py                    # Overlap-weighted aggregation
│   ├── features.py                      # Causal feature computation
│   ├── baselines_hw.py                  # Baseline persistence & Holt-Winters
│   ├── gru_train.py                     # PyTorch GRU training & checkpointing
│   ├── fusion.py                        # Validation fusion grid search
│   ├── uncertainty.py                   # Risk proxy calculation & calibration
│   ├── selector.py                      # Rolling Spearman adaptive selector
│   ├── allocation.py                    # Risk buffer calibration & allocation
│   ├── evaluate.py                      # 9-configuration benchmark & Pareto
│   ├── validate_integrity.py            # Automated 16-test integrity audit
│   └── generate_final_figures.py        # 300 DPI publication figure generator
├── models/
│   └── best_gru.pt                      # Serialized PyTorch GRU weights
├── uncertainty/
│   ├── calibration_factors.json         # Validation scaling constants (c1, c2, c3)
│   └── uncertainty.csv                  # Raw and calibrated proxy time-series
├── selector/
│   └── selector_results.csv             # Step-by-step proxy selection log
├── allocation/
│   └── allocation.csv                   # Step-by-step resource allocations
├── results/
│   ├── final_results_table.csv          # Complete benchmark metric table
│   ├── final_mvp_summary.json           # Machine-readable experiment summary
│   ├── pareto_results.csv               # Multi-beta Pareto sweep data
│   └── phase11_audit_report.json        # 16-test integrity verification report
├── figures/
│   ├── final_data_validation.png        # Figure 1: Workload traces
│   ├── final_prediction_comparison.png  # Figure 2: Forecast comparison
│   ├── final_uncertainty_signals.png    # Figure 3: Risk proxy dynamics
│   ├── final_selector_choices.png       # Figure 4: Selector choices timeline
│   ├── final_allocation_vs_actual.png   # Figure 5: Allocation vs actual
│   ├── final_results_comparison.png     # Figure 6: 9-config benchmark bar chart
│   └── final_pareto_frontier.png        # Figure 7: Multi-beta Pareto frontier
├── PROJECT_SPEC.md                      # Locked experimental specification
├── requirements.txt                     # Environment dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Pipeline (End-to-End)
```bash
python3 scripts/preprocess.py
python3 scripts/features.py
python3 scripts/baselines_hw.py
python3 scripts/gru_train.py
python3 scripts/fusion.py
python3 scripts/uncertainty.py
python3 scripts/selector.py
python3 scripts/allocation.py
python3 scripts/evaluate.py
python3 scripts/validate_integrity.py
python3 scripts/generate_final_figures.py
```
