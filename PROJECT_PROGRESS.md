# CryptoFlow - Project Progress Summary

**Last Updated**: February 28, 2026  
**Dataset**: BTC/USDT Hourly (2018-01-01 → 2026-02-02)  
**Goal**: Predict next-hour Bitcoin **volatility** (regression) AND **price direction** (classification) simultaneously

---

## COMPLETED WORK

---

### Phase 1: Data Cleaning & Ground Truth Generation
**Notebook**: `1_data_cleaning_and_ground_truth.ipynb`  
**Output**: `data/cleaned_data_with_dual_targets.csv`

**What was done:**
- Loaded raw BTC/USDT hourly OHLCV data from Binance (2018–2025)
- Standardized column names to lowercase
- Removed duplicate timestamps
- Filtered out incomplete records (e.g., 2026-02-03 partial data)
- Detected and filled missing hours using forward/backward fill (temporal-aware)
- Flagged interpolated rows with `is_interpolated` column
- Computed **Garman-Klass (GK) volatility** for each hour using open/high/low/close
- Created two prediction targets:
  - `target_reg` = next hour's GK volatility (regression)
  - `target_class` = 1 if `close(t+1) > close(t)`, else 0 (classification)
- Validated class balance: **51.3% Up / 48.7% Down** (naturally balanced)
- Validated autocorrelation: GK volatility → next GK = **0.649 correlation**
- Validated inter-target independence: `Corr(target_reg, target_class) = -0.024`

**Key numbers:**
- Total clean rows: **70,919**
- Date range: 2018-01-01 → 2026-02-02

---

### Phase 2: Feature Engineering
**Notebook**: `2_feature_engineering.ipynb`  
**Input**: `data/cleaned_data_with_dual_targets.csv`  
**Output**: `data/engineered_features_dual_target.csv`

**What was done:**
- Engineered **40 features** across 6 categories from raw OHLCV data:

| # | Category | Count | Features |
|---|----------|-------|---------|
| 1 | **Volatility** | 6 | `atr_14`, `bb_width_pct`, `close_vol_6h`, `close_vol_12h`, `close_vol_24h`, `atr_pct` |
| 2 | **Volume** | 4 | `taker_buy_ratio`, `volume_z_score`, `trade_intensity`, `volume_change_pct` |
| 3 | **Price-Based** | 7 | `log_return_1h/3h/6h/12h`, `return_acceleration`, `hl_range_pct`, `close_position` |
| 4 | **Technical** | 7 | `rsi_14`, `macd`, `macd_signal`, `macd_histogram`, `stoch_k`, `stoch_d`, `ema_crossover` |
| 5 | **Lags** | 10 | GK lags (1h/3h/6h), GK averages (3h/6h), ATR lags (1h/3h), return lags (1h/3h/6h) |
| 6 | **Temporal** | 6 | `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `is_funding_hour`, `is_weekend` |

- Removed intermediate calculation columns (BB bands, volume stats)
- Replaced infinite values with NaN
- Computed Pearson & Spearman correlations for both targets
- Saved correlation plots to `plots/feature_correlations_pearson_spearman.png`

---

### Phase 3: Preprocessing & Feature Selection
**Notebook**: `3_preprocessing_and_feature_selection.ipynb`  
**Input**: `data/engineered_features_dual_target.csv`  
**Outputs**: See below

**What was done:**
- Loaded 40-feature dataset and handled NaN:
  - Dropped first 24 rows (insufficient rolling window history)
  - Forward fill (`ffill`) → Backward fill (`bfill`)
  - Result: **0 NaN values remaining**
- **Temporal Train/Test Split (85% / 15%)**:
  - Train: **60,299 rows**
  - Test: **10,642 rows**
- Applied **3-Tier Hybrid Feature Selection** for each task:

```
40 Features
    ↓ Tier 1: Pearson Correlation → Top 30
    ↓ Tier 2: Lasso / Logistic Lasso (L1) → 12–28 non-zero features
    ↓ Tier 3: RFE + XGBoost → Final Top 10
```

**Regression Top 10 Features** (for `target_reg` - volatility):
| Rank | Feature | Category |
|------|---------|----------|
| 1 | `hl_range_pct` | Price |
| 2 | `gk_avg_6h` | Lag |
| 3 | `gk_lag_6h` | Lag |
| 4 | `hour_cos` | Temporal |
| 5 | `gk_avg_3h` | Lag |
| 6 | `gk_lag_1h` | Lag |
| 7 | `trade_intensity` | Volume |
| 8 | `log_return_3h` | Price |
| 9 | `log_return_6h` | Price |
| 10 | `atr_lag_3h` | Lag |

**Classification Top 10 Features** (for `target_class` - direction):
| Rank | Feature | Category |
|------|---------|----------|
| 1 | `stoch_k` | Technical |
| 2 | `close_position` | Price |
| 3 | `return_lag_1h` | Lag |
| 4 | `log_return_1h` | Price |
| 5 | `log_return_12h` | Price |
| 6 | `volume_change_pct` | Volume |
| 7 | `log_return_3h` | Price |
| 8 | `hl_range_pct` | Price |
| 9 | `macd_signal` | Technical |
| 10 | `atr_lag_3h` | Lag |

**Key Insight**: Regression relies heavily on **volatility memory (lags)**. Classification relies on **price momentum (returns) and technical indicators**. Tasks need different features → dual-target approach validated.

- Standardized features using `StandardScaler` (fit on training set only)
- Saved feature importance plots to `plots/feature_importances_top10.png`

**Saved Artifacts:**
```
models/
  scaler_regression.pkl            ← StandardScaler for regression features
  scaler_classification.pkl        ← StandardScaler for classification features

data/
  final_training_data_regression_train.csv
  final_training_data_regression_test.csv
  final_training_data_classification_train.csv
  final_training_data_classification_test.csv

selected_features_regression.json       ← List of top 10 regression features
selected_features_classification.json   ← List of top 10 classification features
```

---

## REMAINING WORK

> **Critical Requirement**: Every model must be **built from scratch** using NumPy/manual math.  
> Scikit-learn / PyTorch implementations are allowed **alongside** scratch versions to confirm results, but cannot replace them.  
> This applies to **every single model** in both regression and classification.

---

### Phase 4: Naive Baseline
**Part of Notebook**: `4_model_training.ipynb` (first section)  
**Purpose**: Establish a minimum performance bar that all real models must beat

**Regression Baseline — Persistence Model:**
- Predict next-hour volatility = current-hour volatility: $\hat{y}_{t+1} = \text{gk\_volatility}_t$
- Compute RMSE, MAE, R² on the test set
- This is a strong baseline because GK autocorrelation = **0.649**

**Classification Baseline — Majority Class Classifier:**
- Always predict "Up" (class 1) since Up = 51.3% of labels
- Compute Accuracy, F1, ROC-AUC on the test set
- This gives ~51.3% accuracy for free — all models must exceed this

**Why it matters:** Without baselines, model results have no reference point. A model achieving 57% classification accuracy sounds weak — but "beats majority baseline by 6 points" is a meaningful claim.

---

### Phase 5: Model Training & Evaluation
**Planned Notebook**: `4_model_training.ipynb`  
**Input**: `data/final_training_data_*_train/test.csv`, scalers from `models/`

> For every model below: implement from scratch first, then confirm with sklearn/PyTorch.  
> Save both versions' predictions for comparison in the evaluation phase.

---

#### 5A — Regression Models (predict `target_reg` — next-hour GK volatility)

| # | Model | Notes |
|---|-------|-------|
| 1 | **Persistence Model** | Naive baseline — no training needed |
| 2 | **Linear Regression** | Single feature (e.g. `gk_lag_1h`) vs target — scratch: normal equations $w = (X^TX)^{-1}X^Ty$ |
| 3 | **Multiple Regression** | All 10 regression features — same normal equation, multivariate |
| 4 | **Polynomial Regression** | Add degree-2 terms (e.g. `gk_lag_1h²`) to multiple regression — scratch: manual feature expansion before applying normal equations |
| 5 | **Random Forest Regressor** | Ensemble of decision trees — scratch: build decision tree from scratch, then bag N trees |
| 6 | **XGBoost Regressor** *(extracurricular)* | Gradient boosting — use library version; optionally supplement with a simplified scratch boosting loop |
| 7 | **LSTM** *(extracurricular optional)* | If included: PyTorch implementation; captures temporal sequence beyond manual lag features |

> **Note on Linear vs Multiple Regression**: These must be presented as distinct models.  
> Linear = one predictor. Multiple = many predictors. Both use the same math but different input shapes — the project requires both separately.

> **Note on Polynomial Regression**: Degree should be chosen carefully (degree 2 recommended).  
> Higher degrees risk overfitting on 70k rows. Use train/validation loss curves to justify degree choice.

---

#### 5B — Classification Models (predict `target_class` — Up/Down)

| # | Model | Notes |
|---|-------|-------|
| 1 | **Majority Class** | Naive baseline — always predict 1 (Up) |
| 2 | **Logistic Regression** | Scratch: sigmoid + binary cross-entropy + gradient descent |
| 3 | **Decision Tree** | Scratch: Gini impurity splits, max_depth control to avoid overfitting |
| 4 | **Random Forest** | Scratch: bag N decision trees, majority vote |
| 5 | **Support Vector Machine (SVM)** | Scratch: linear SVM via gradient descent on hinge loss; sklearn RBF kernel for confirmation |
| 6 | **Dimensionality Reduction + RF/SVM** | Apply PCA (scratch: eigen-decomposition of covariance matrix) to 10 features → reduced dims → feed into RF or SVM |
| 7 | **K-Means Clustering** | See clustering notes below |
| 8 | **Agglomerative Clustering** | See clustering notes below |
| 9 | **Perceptron & Single-Layer Perceptron (SLP)** | Scratch: single neuron, step/sigmoid activation, manual weight update rule |
| 10 | **Multi-Layer Perceptron (MLP)** | Scratch: forward pass + backpropagation by hand; PyTorch version for confirmation |
| 11 | **XGBoost Classifier** *(extracurricular)* | Library version; compare against all scratch models |

---

#### Clustering — Special Handling Required

K-Means and Agglomerative Clustering are **unsupervised** — they do not use `target_class` labels during training. This creates a fundamental difference from every other model in the list and requires careful treatment.

**The core problem:**  
Clustering finds natural groupings in the feature space. There is no guarantee these groups align with Up/Down price direction. You cannot simply apply cluster labels as predictions and compute accuracy — the cluster numbers (0, 1, 2...) are arbitrary and have no inherent meaning.

**How to handle it correctly:**

*Step 1 — Cluster the training data (unsupervised, no labels used):*
```python
# K-Means example (scratch: manual centroid update loop)
# Run on classification features (unscaled or scaled — be consistent)
kmeans.fit(X_train_classification)
train_cluster_labels = kmeans.predict(X_train_classification)
```

*Step 2 — Map clusters to classes using training labels:*
After clustering, look at which real class (Up/Down) is the majority within each cluster. This mapping is learned from training data only.
```python
# For each cluster, find the majority real class
cluster_to_class = {}
for cluster_id in unique_clusters:
    mask = (train_cluster_labels == cluster_id)
    majority_class = mode(y_train[mask])
    cluster_to_class[cluster_id] = majority_class
```

*Step 3 — Apply mapping to test set:*
```python
test_cluster_labels = kmeans.predict(X_test_classification)
test_predictions = [cluster_to_class[c] for c in test_cluster_labels]
# Now compute accuracy, F1, etc. normally
```

**Additional things to watch out for:**

- **Number of clusters**: Use K=2 for direct Up/Down comparison. Optionally try K=3 or K=4 to find sub-regimes (e.g. low-vol-up, high-vol-up, low-vol-down, high-vol-down) and then merge into 2 classes.
- **Scaling**: K-Means is distance-based and therefore scale-sensitive. Use the saved `scaler_classification.pkl` — the same scaler used for all other classification models — for consistency and fairness.
- **Agglomerative Clustering** does not have a `predict()` method — it only assigns clusters to the data it was fit on. For the test set, assign each test point to the nearest training centroid (compute manually) or use a workaround like fitting on train+test and then separating results. Document whichever approach is used.
- **Expected performance**: Clustering will almost certainly underperform supervised models. This is expected and should be discussed — clustering is not optimizing for prediction accuracy, it is finding geometric structure in the data.
- **Interpretation angle**: The more valuable output from clustering may not be accuracy but rather the **characterization of regimes** — e.g., "Cluster 0 corresponds to low-volatility consolidation periods; Cluster 1 to high-momentum breakout hours." This aligns with the project's storytelling requirement.

---

#### 5C — Evaluation Metrics (Required for All Models)

All metrics must be computed on the **held-out test set only**.

**Regression metrics** (for models in 5A):
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error — primary metric |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |
| Loss curve | Training loss vs epoch (for iterative models: Polynomial, RF, LSTM) |
| Prediction vs Actual plot | Visual comparison of predicted vs real GK volatility over time |

**Classification metrics** (for models in 5B):
| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision | Of predicted Ups, how many were actually Up |
| Sensitivity (Recall) | Of actual Ups, how many were caught |
| **Specificity** | Of actual Downs, how many were correctly predicted Down — `TN / (TN + FP)` |
| **True Negative Rate** | Same as Specificity — explicitly report separately as required |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC & AUC | Full ROC curve + area under curve |
| Confusion Matrix | For every classifier — TP, TN, FP, FN table |
| Loss curve / Accuracy curve | Per epoch for iterative models (Logistic, SLP, MLP, LSTM) |
| Performance curve | Training vs validation metric over epochs where applicable |

> **Specificity and True Negative Rate are required** by the project rubric and are currently missing from the plan. For a balanced dataset (51.3/48.7), a model that ignores Down predictions entirely can still achieve ~51% accuracy — Specificity will expose this immediately.

---

### Phase 6: Hyperparameter Tuning
**Planned Notebook**: `5_hyperparameter_tuning.ipynb`

- [ ] Grid search / random search for RF and XGBoost hyperparameters
- [ ] Use **walk-forward cross-validation** (not standard k-fold) — respects temporal ordering:
  ```
  Fold 1: Train [1..n]   → Validate [n+1..n+k]
  Fold 2: Train [1..n+k] → Validate [n+k+1..n+2k]
  ...
  ```
- [ ] For MLP/LSTM: tune learning rate, hidden layer sizes, dropout, batch size
- [ ] For SVM: tune C and kernel type (linear vs RBF)
- [ ] For Polynomial Regression: tune degree (2–4) using validation loss
- [ ] Final model selection based on validation performance
- [ ] Re-evaluate tuned models against naive baseline

---

### Phase 7: Final Evaluation & Insights
**Planned Notebook**: `6_evaluation.ipynb`

- [ ] Final evaluation on held-out test set (covering Nov 2024 → Feb 2026)
- [ ] **Prediction vs Actual plots** for all regression models
- [ ] **Confusion matrix** for every classifier
- [ ] **ROC curves** — all classifiers on the same plot for comparison
- [ ] **Full metric comparison table** — Naive Baseline vs every model, side by side
- [ ] **Loss/accuracy curves** for all iterative models (Logistic, SLP, MLP, LSTM)
- [ ] Error analysis — when does each model fail? (e.g., during high-volatility regimes, during crashes)
- [ ] Clustering regime analysis — characterize what each cluster represents in market terms
- [ ] SHAP or feature importance on final best models
- [ ] **Combined signal analysis**: direction prediction + volatility prediction together  
  e.g., High-confidence Up + Low predicted volatility → strong buy signal

---

### Phase 8: Deployment / Production Pipeline
- [ ] Real-time feature engineering script (mirrors Phase 2 logic on new OHLCV data)
- [ ] Load saved scalers (`scaler_regression.pkl`, `scaler_classification.pkl`) + best models
- [ ] Apply same feature order as saved in `selected_features_*.json`
- [ ] Output: `{direction: 1, direction_prob: 0.63, volatility: 0.0045}` → trading signal
- [ ] API or script to connect to Binance live data feed

---

## 📁 Current Project Structure

```
cryptoflow/
│
├── 📓 Notebooks
│   ├── 1_data_cleaning_and_ground_truth.ipynb          Complete
│   ├── 2_feature_engineering.ipynb                     Complete
│   ├── 3_preprocessing_and_feature_selection.ipynb    Complete
│   ├── 4_model_training.ipynb                          Not started
│   ├── 5_hyperparameter_tuning.ipynb                    Not started
│   └── 6_evaluation.ipynb                              Not started
│
├── 📂 data/
│   ├── btc_1h_data_2018_to_2025.csv                    (raw input)
│   ├── cleaned_data_with_dual_targets.csv               (Phase 1 output)
│   ├── engineered_features_dual_target.csv              (Phase 2 output)
│   ├── final_training_data_regression_train.csv         (Phase 3 output)
│   ├── final_training_data_regression_test.csv          (Phase 3 output)
│   ├── final_training_data_classification_train.csv     (Phase 3 output)
│   └── final_training_data_classification_test.csv      (Phase 3 output)
│
├── 📂 models/
│   ├── scaler_regression.pkl                            (Phase 3 output)
│   └── scaler_classification.pkl                        (Phase 3 output)
│
├── 📂 plots/
│   ├── feature_correlations_pearson_spearman.png        (Phase 2 output)
│   └── feature_importances_top10.png                    (Phase 3 output)
│
├── selected_features_regression.json                    (Phase 3 output)
├── selected_features_classification.json                (Phase 3 output)
└── requirements.txt
```

---

## 📊 Key Stats at a Glance

| Item | Value |
|------|-------|
| Dataset | BTC/USDT Hourly |
| Total rows (clean) | 70,919 |
| Date range | 2018-01-01 → 2026-02-02 |
| Training rows | 60,299 (85%) |
| Test rows | 10,642 (15%) |
| Engineered features | 40 |
| Final features (regression) | 10 |
| Final features (classification) | 10 |
| Class balance | 51.3% Up / 48.7% Down |
| GK volatility autocorrelation | 0.649 |
| Inter-target correlation | -0.024 |

---

##  Model Checklist (Grading Reference)

### Regression
- [ ] Persistence baseline
- [ ] Linear Regression *(scratch + sklearn)*
- [ ] Multiple Regression *(scratch + sklearn)*
- [ ] Polynomial Regression *(scratch + sklearn)*
- [ ] Random Forest Regressor *(scratch + sklearn)*
- [ ] XGBoost Regressor *(extracurricular — library)*
- [ ] LSTM *(optional extracurricular — PyTorch)*

### Classification
- [ ] Majority class baseline
- [ ] Logistic Regression *(scratch + sklearn)*
- [ ] Decision Tree *(scratch + sklearn)*
- [ ] Random Forest *(scratch + sklearn)*
- [ ] SVM *(scratch linear + sklearn RBF)*
- [ ] PCA + RF or SVM *(PCA scratch: eigen-decomposition)*
- [ ] K-Means Clustering *(scratch + sklearn, with majority-vote mapping)*
- [ ] Agglomerative Clustering *(scratch + sklearn, with nearest-centroid test assignment)*
- [ ] Perceptron & SLP *(scratch)*
- [ ] MLP *(scratch backprop + PyTorch)*
- [ ] XGBoost Classifier *(extracurricular — library)*

### Evaluation (all models)
- [ ] RMSE, MAE, R² (regression)
- [ ] Loss curves / accuracy curves (iterative models)
- [ ] Confusion matrix (all classifiers)
- [ ] Accuracy, Precision, Sensitivity, **Specificity**, **True Negative Rate**, F1 (all classifiers)
- [ ] ROC & AUC (all classifiers)
- [ ] Prediction vs Actual plots (regression)
- [ ] Side-by-side comparison table (all models vs baseline)