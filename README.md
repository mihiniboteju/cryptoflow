# CryptoFlow

**Predicting Bitcoin price direction and volatility from hourly OHLCV data using classical machine learning and deep learning models.**

---

## Overview

CryptoFlow is an end-to-end machine learning project that frames Bitcoin (BTC/USDT) price prediction as two simultaneous tasks:

| Task | Target | Type |
|---|---|---|
| **Direction prediction** | Will the next hourly close be higher or lower? (`target_class`: 1 = Up, 0 = Down) | Binary classification |
| **Volatility prediction** | What will the next hourly Garman-Klass volatility be? (`target_reg`) | Regression |

The project covers the complete ML pipeline — data cleaning, feature engineering, preprocessing, model training from scratch, evaluation, and a diagnostic analysis of why models underperform on short-horizon crypto prediction.

---

## Dataset

| Property | Value |
|---|---|
| Source | Binance BTC/USDT hourly OHLCV candles |
| Raw file | `data/btc_1h_data_2018_to_2025.csv` |
| Date range | 2018-01-01 → 2026-02-02 |
| Total rows (clean) | 70,919 |
| Train / Test split | 60,281 train (≈85%) / 10,638 test (≈15%) |
| Split date | 2024-11-16 17:00 UTC (strict temporal split — no shuffling) |
| Class balance | 51.3% Up / 48.7% Down (naturally balanced) |

---

## Project Structure

```
cryptoflow/
│
├── Notebooks/
│   │
│   ├── ── Pipeline ──────────────────────────────────────────────────────────
│   ├── 1_data_cleaning_and_ground_truth.ipynb       Phase 1: Clean raw OHLCV, create targets
│   ├── 2_feature_engineering.ipynb                  Phase 2: Engineer 41 features
│   ├── 3_preprocessing_and_feature_selection.ipynb  Phase 3: Feature selection, train/test split
│   │
│   ├── ── Regression Models ─────────────────────────────────────────────────
│   ├── 4_regression_naive_baseline.ipynb            Persistence model baseline
│   ├── 4A_linear_regression.ipynb                   Single-feature linear regression (scratch)
│   ├── 4B_multiple_regression.ipynb                 Multi-feature linear regression (scratch)
│   ├── 4C_polynomial_regression.ipynb               Polynomial regression (scratch)
│   ├── 4D_lasso_regression.ipynb                    Lasso (L1 regularisation)
│   ├── 4E_ridge_regression.ipynb                    Ridge (L2 regularisation)
│   ├── 4F_random_forest.ipynb                       Random Forest regressor
│   ├── 4J_gradient_boosting_regression.ipynb        Gradient Boosting regressor
│   ├── 4Final_model_comparison_r2.ipynb             Regression results comparison
│   │
│   ├── ── Classification Models ─────────────────────────────────────────────
│   ├── 5_classification_naive_baseline.ipynb        Majority-class baseline
│   ├── 6_classification_logistic_regression.ipynb   Logistic Regression (scratch + sklearn)
│   ├── 7_classification_decision_tree.ipynb         Decision Tree (scratch + sklearn)
│   ├── 8_classification_random_forest.ipynb         Random Forest (scratch + sklearn)
│   ├── 9_classification_svm.ipynb                   SVM — linear scratch + RBF sklearn
│   ├── 10_dimensionality_reduction_classification.ipynb  PCA (scratch) + SVM/RF
│   ├── 11_k_means_clustering.ipynb                  K-Means (scratch) with majority-vote mapping
│   ├── 12_agglomerative_clustering.ipynb            Agglomerative clustering
│   ├── 13_perceptron_slp.ipynb                      Perceptron & Single-Layer Perceptron (scratch)
│   ├── 14_mlp_classification.ipynb                  MLP — scratch backprop + PyTorch
│   ├── 15_xgboost_classification.ipynb              XGBoost classifier
│   ├── 15_lstm_univariate_classification.ipynb      LSTM univariate
│   ├── 16_lstm_twostage_classification.ipynb        LSTM two-stage classification
│   ├── 17_lstm_multivariate_classification.ipynb    LSTM multivariate
│   ├── 18_cnn_lstm_boruta.ipynb                     CNN-LSTM with Boruta feature selection
│   │
│   ├── ── Analysis ──────────────────────────────────────────────────────────
│   ├── 19_classification_presentation.ipynb         Full results summary + all visualisations
│   └── 20_why_models_underperformed.ipynb           Root cause analysis (4 causes)
│
├── data/
│   ├── btc_1h_data_2018_to_2025.csv                 Raw Binance OHLCV input
│   ├── cleaned_data_with_dual_targets.csv            Phase 1 output
│   ├── engineered_features_dual_target.csv           Phase 2 output (41 features)
│   ├── final_training_data_classification_train.csv  Phase 3 output
│   ├── final_training_data_classification_test.csv   Phase 3 output
│   ├── final_training_data_regression_train.csv      Phase 3 output
│   ├── final_training_data_regression_test.csv       Phase 3 output
│   ├── selected_features_classification.json         Top 10 classification features
│   └── selected_features_regression.json             Top 10 regression features
│
├── results/
│   ├── classification_results.csv                    AUC, accuracy, F1 for all 16 classifiers
│   └── regression_results.csv                        RMSE, MAE, R² for all regression models
│
├── models/
│   ├── scaler_classification.pkl                     StandardScaler (fit on train only)
│   └── scaler_regression.pkl
│
├── plots/                                            Saved figures from notebooks
├── figures/
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Raw OHLCV (Binance)
        │
        ▼
1_data_cleaning_and_ground_truth.ipynb
  • Remove duplicates and gaps           → cleaned_data_with_dual_targets.csv
  • Create target_class and target_reg
        │
        ▼
2_feature_engineering.ipynb
  • 41 engineered features               → engineered_features_dual_target.csv
  • 6 categories (see Features section)
        │
        ▼
3_preprocessing_and_feature_selection.ipynb
  • 3-tier feature selection             → final_training_data_*.csv
  • Temporal 85/15 train/test split      → scaler_*.pkl, selected_features_*.json
        │
        ├──▶ Regression notebooks (4A – 4J)
        │
        └──▶ Classification notebooks (5 – 18)
                        │
                        ▼
              19_classification_presentation.ipynb
              20_why_models_underperformed.ipynb
```

---

## Engineered Features

41 features across 6 categories, all derived from raw OHLCV data:

| Category | Count | Features |
|---|---|---|
| **Volatility** | 6 | `gk_volatility`, `atr_14`, `bb_width_pct`, `close_vol_6h/12h/24h`, `atr_pct` |
| **Volume** | 4 | `taker_buy_ratio`, `volume_z_score`, `trade_intensity`, `volume_change_pct` |
| **Price-based** | 7 | `log_return_1h/3h/6h/12h`, `return_acceleration`, `hl_range_pct`, `close_position` |
| **Technical** | 7 | `rsi_14`, `macd`, `macd_signal`, `macd_histogram`, `stoch_k`, `stoch_d`, `ema_crossover` |
| **Lag features** | 10 | GK lags (1h/3h/6h), GK rolling averages (3h/6h), ATR lags (1h/3h), return lags (1h/3h/6h) |
| **Temporal** | 6 | `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `is_funding_hour`, `is_weekend` |

Raw OHLC price levels (`open`, `high`, `low`, `close`) and raw Binance metadata columns are excluded from model features — only engineered, normalised signals are used.

---

## Models & Results

### Classification (direction prediction, `target_class`)

All models use the same temporal train/test split. AUC is the primary metric.

| Model | AUC | Accuracy | F1 |
|---|---|---|---|
| Majority class baseline | 0.500 | 51.3% | 0.672 |
| Logistic Regression (scratch) | ~0.534 | ~52% | — |
| Decision Tree | ~0.512 | ~52% | — |
| Random Forest | ~0.536 | ~53% | — |
| SVM (RBF kernel) | ~0.537 | ~53% | — |
| PCA + SVM | ~0.530 | ~52% | — |
| K-Means Clustering | ~0.507 | ~51% | — |
| Agglomerative Clustering | ~0.505 | ~51% | — |
| Perceptron / SLP | ~0.520 | ~52% | — |
| MLP | ~0.538 | ~53% | — |
| **XGBoost** | **~0.544** | **~53.3%** | — |
| LSTM (univariate) | ~0.530 | ~52% | — |
| LSTM (multivariate) | ~0.533 | ~52% | — |
| CNN-LSTM | ~0.535 | ~53% | — |

> Full results with Precision, Recall, Specificity, and Confusion Matrices are in `19_classification_presentation.ipynb` and `results/classification_results.csv`.

### Regression (volatility prediction, `target_reg`)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Persistence baseline | — | — | — |
| Linear Regression | — | — | — |
| Multiple Regression | — | — | — |
| Polynomial Regression | — | — | — |
| Lasso / Ridge | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |

> Full results in `results/regression_results.csv`.

---

## Why Models Are Bounded Near AUC ≈ 0.54

`20_why_models_underperformed.ipynb` provides a quantified root cause analysis:

1. **BTC 1-hour returns are near-random** — Durbin-Watson = 2.016, ACF ≈ 0 at all 24 lags. Consistent with weak-form EMH (Fama, 1970) and confirmed by Bariviera (2017).

2. **Features have near-zero predictive power** — over 95% of engineered features have |Pearson r| < 0.05 with `target_class`. The strongest feature explains < 0.25% of target variance (r² < 0.0023).

3. **Non-stationarity / market regime changes** — rolling 30-day test AUC fluctuates between 0.43 and 0.57 across the 2018–2026 period, confirming a static model cannot generalise across bull, bear, and sideways regimes (López de Prado, 2018).

4. **Missing on-chain and derivatives data** — Omole & Enke (2024) achieved 82% accuracy on BTC using on-chain variables (exchange flows, active addresses) alongside OHLCV. Our best result is 53.3% — a ~30 percentage-point gap attributable to data availability, not model complexity.

---

## Setup

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/mihiniboteju/cryptoflow.git
cd cryptoflow

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `pandas-ta` | Technical indicator computation |
| `scikit-learn` | sklearn model implementations and preprocessing |
| `xgboost` | XGBoost classifier and regressor |
| `torch` | PyTorch (LSTM, MLP deep learning models) |
| `matplotlib`, `seaborn` | Visualisation |
| `statsmodels` | Durbin-Watson, ACF, statistical tests |

---

## Notebook Execution Order

Run notebooks in the order listed below. Each notebook saves its outputs to `data/` or `models/`, which subsequent notebooks expect as inputs.

```
1_data_cleaning_and_ground_truth.ipynb
2_feature_engineering.ipynb
3_preprocessing_and_feature_selection.ipynb
    ↓
4_regression_naive_baseline.ipynb   (then 4A → 4J)
5_classification_naive_baseline.ipynb  (then 6 → 18)
    ↓
19_classification_presentation.ipynb
20_why_models_underperformed.ipynb
```

> **Note:** Pre-processed data files are already included in `data/`. You can run any classification or regression notebook directly without re-running phases 1–3.

---

## Key Design Decisions

- **Strict temporal split** — no random shuffling. Training data ends 2024-11-16 17:00 UTC; test data begins immediately after. This prevents data leakage across the time boundary.
- **Scratch-first implementation** — every model (Logistic Regression, Decision Tree, SVM, K-Means, Perceptron, MLP, PCA) is implemented from scratch using NumPy before being confirmed with scikit-learn or PyTorch.
- **Leaky column exclusion** — `future_close`, `target_return`, `target_reg`, and `is_interpolated` are explicitly excluded from all feature sets to prevent look-ahead bias.
- **Dual-target framing** — regression (volatility) and classification (direction) are treated as independent tasks with separate feature sets, as the optimal features differ significantly between the two.

---

## References

- Fama, E. F. (1970). Efficient Capital Markets. *Journal of Finance*, 25(2), 383–417.
- Bariviera, A. F. (2017). The inefficiency of Bitcoin revisited. *Economics Letters*, 161, 1–4.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Omole, J. A., & Enke, D. (2024). Ensemble machine learning models for Bitcoin price prediction. *Expert Systems with Applications*, 238, 122160.
- Sezer, O. B. et al. (2020). Financial time series forecasting with deep learning. *Applied Soft Computing*, 90, 106181.

---

## Author

**Mihini Boteju** — BTC/USDT direction and volatility prediction using classical ML and deep learning.
