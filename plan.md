# CryptoFlow Dual-Target Model: Architecture Plan
## Predicting Next-Hour Volatility & Candle Direction for BTC/USDT

**Dataset**: `btc_1h_data_2018_to_2025.csv` (8 years, Binance hourly OHLCV)  
**Targets**: 
1. **Regression**: Next-hour Garman-Klass (GK) Volatility Index
2. **Classification**: Next-hour Candle Direction (Binary: Up/Down)

**Modeling Paradigm**: Multi-Task Learning with Separate Feature Selection Pipelines

---

## Executive Summary

This pipeline transforms raw OHLCV data into a feature matrix optimized for **dual-task prediction**: volatility forecasting (regression) and directional movement (classification). The architecture consists of three sequential phases:

1. **Data Cleaning & Dual-Target Generation**: Handling of missing data and creation of two targets: `target_reg` (GK Volatility at T+1) and `target_class` (Binary candle direction at T+1)
2. **Feature Engineering**: Creation of 60-80 shared candidate features optimized for both regression and classification tasks
3. **Preprocessing & Task-Specific Feature Selection**: Scaling, transformation, and separate feature selection to identify "Top 10 for Regression" and "Top 10 for Classification"

The final output is a unified training dataset where each row represents a 1-hour interval with engineered features and **two target columns** for multi-task learning or separate model training.

---

## Phase 1: Data Cleaning & Dual-Target Generation

### Objectives
- Ensure temporal continuity (no missing hours)
- Generate ground truth volatility measure using Garman-Klass formula
- **Create dual predictive targets**: `target_reg` (volatility) and `target_class` (direction)

### 1.1 Missing Data Handling

**Challenge**: Exchange downtime, API failures, or delisting events can create gaps in hourly data.

**Solution Strategy**:
- **Price columns (Open, High, Low, Close)**: Linear interpolation
  - Rationale: Price movements are continuous processes; linear interpolation preserves trend momentum
  - Implementation: `df[['open', 'high', 'low', 'close']].interpolate(method='linear')`
  
- **Volume & Trade Count**: Zero-filling
  - Rationale: Missing hours represent periods of no trading activity
  - Implementation: `df[['volume', 'trade_count']].fillna(0)`

- **Validation**: Flag interpolated rows for downstream analysis
  - Create `is_interpolated` boolean column
  - Report % of data requiring interpolation

**Edge Cases**:
- If >5% consecutive hours are missing → consider that as a regime change boundary
- First/last rows with missing data → forward/backward fill as fallback

### 1.2 Garman-Klass Volatility Formula

**Mathematical Definition**:

The Garman-Klass (1980) estimator leverages intra-period High-Low range information:

$$
GK = \sqrt{\frac{1}{2}[\log(H/L)]^2 - (2\log2 - 1)[\log(C/O)]^2}
$$

Where:
- $H$ = High price during interval
- $L$ = Low price during interval
- $C$ = Close price at interval end
- $O$ = Open price at interval start

**Why GK over Simple Returns?**
- **Efficiency**: Uses 30% more information than close-to-close estimators
- **Range-Aware**: Captures intra-hour volatility (critical for crypto's wild swings)
- **No Bias**: Unbiased estimator under geometric Brownian motion assumptions

**Implementation Notes**:
- Handle zero/negative prices (crypto can have flash crashes)
- Annualization: Multiply by $\sqrt{365.25 \times 24}$ if needed (optional for hourly prediction)
- Outlier Detection: Flag GK values >3 standard deviations (possible data errors)

**Code Skeleton**:
```python
import numpy as np

def garman_klass_volatility(open_price, high_price, low_price, close_price):
    """
    Computes Garman-Klass volatility for a single interval.
    Returns NaN if any price is invalid.
    """
    if any(p <= 0 for p in [open_price, high_price, low_price, close_price]):
        return np.nan
    
    hl_ratio = np.log(high_price / low_price)
    co_ratio = np.log(close_price / open_price)
    
    gk = np.sqrt(0.5 * hl_ratio**2 - (2*np.log(2) - 1) * co_ratio**2)
    return gk

# Vectorized application
df['gk_volatility'] = df.apply(
    lambda row: garman_klass_volatility(row['open'], row['high'], row['low'], row['close']),
    axis=1
)
```

### 1.3 Dual-Target Variable Creation

**Objective**: Predict the *next* hour's volatility (regression) and candle direction (classification).

#### Target 1: Regression (Volatility)
```python
# Shift GK volatility backward by 1 row (shift=-1 means future)
df['target_reg'] = df['gk_volatility'].shift(-1)
```

#### Target 2: Binary Classification (Candle Direction)
```python
# Binary: 1 if next hour's close > current close, else 0
df['target_class'] = (df['close'].shift(-1) > df['close']).astype(int)
```

**Interpretation**:
- `target_class = 1`: Next candle closes **higher** (bullish)
- `target_class = 0`: Next candle closes **lower or equal** (bearish)

**Final Cleanup**:
```python
# Remove the last row (no future targets available)
df = df[:-1]
```

**Validation Checks**:
- No data leakage: Ensure targets use *only* future data (T+1)
- Class balance: Check distribution of `target_class` (expect ~50/50 in ranging markets, skewed in trending markets)
- Target correlation: Compute correlation between `target_reg` and `target_class` to understand volatility-direction relationship

### Phase 1 Deliverable
- **Notebook**: `1_data_cleaning_and_ground_truth.ipynb`
- **Output**: `cleaned_data_with_dual_targets.csv`
- **Columns Added**: `gk_volatility`, `target_reg`, `target_class`, `is_interpolated`

---

## Phase 2: Dual-Target Feature Engineering

### Objectives
- Generate 60-80 **shared** candidate features optimized for both regression and classification tasks
- Create **directional features** (RSI, MACD, log returns) for classification bias
- Create **volatility features** (ATR, BB width, price ranges) for regression strength
- Add **microstructure features** (taker buy ratio, volume z-score) for market dynamics
- Implement **memory lags** for both volatility and price returns to capture persistence
- Ensure **temporal alignment**: All features at time T predict targets at T+1

### Feature Categories Overview

| **Category** | **Features** | **Primary Task** | **Rationale** |
|--------------|--------------|------------------|---------------|
| **Directional** | RSI(14), MACD, Log Returns | Classification | Momentum/trend indicators predict up/down moves |
| **Volatility** | ATR(14), BB Width, HL Range % | Regression | Direct volatility proxies for GK prediction |
| **Microstructure** | Taker Buy Ratio, Volume Z-Score | Both | Liquidity & buying pressure affect both vol & direction |
| **Memory Lags** | GK lags (1h, 3h, 6h), Return lags | Both | Persistence in vol & mean-reversion in returns |
| **Temporal** | Hour of Day, Funding Hour, Weekend | Both | Seasonality affects both volatility & directional bias |

**Note**: All features are engineered for **both tasks** during Phase 2. Task-specific feature selection occurs in Phase 3.

### 2.1 Directional Features (Classification-Optimized)

**Purpose**: Capture momentum and trend strength for predicting next-hour candle direction.

#### Core Directional Indicators

1. **RSI (Relative Strength Index, 14-period)**
   ```python
   df['rsi_14'] = df.ta.rsi(length=14)
   ```
   - Range: [0, 100]
   - Interpretation: >70 = overbought (potential reversal down), <30 = oversold (potential reversal up)
   - Classification utility: Extreme values predict directional changes

2. **MACD (Moving Average Convergence Divergence)**
   ```python
   macd = df.ta.macd(fast=12, slow=26, signal=9)
   df['macd'] = macd['MACD_12_26_9']
   df['macd_signal'] = macd['MACDs_12_26_9']
   df['macd_histogram'] = macd['MACDh_12_26_9']
   ```
   - **MACD Line**: Fast EMA - Slow EMA (momentum strength)
   - **Signal Line**: 9-period EMA of MACD (trend confirmation)
   - **Histogram**: MACD - Signal (divergence magnitude)
   - Classification utility: Crossovers predict trend reversals; histogram slope indicates momentum acceleration

3. **Log Returns (Multi-Horizon)**
   ```python
   # 1-hour log return (most recent price change)
   df['log_return_1h'] = np.log(df['close'] / df['close'].shift(1))
   
   # 3-hour cumulative return (short-term trend)
   df['log_return_3h'] = np.log(df['close'] / df['close'].shift(3))
   
   # 6-hour cumulative return (medium-term momentum)
   df['log_return_6h'] = np.log(df['close'] / df['close'].shift(6))
   ```
   - **Why log returns?** Symmetric, time-additive, statistically well-behaved
   - Classification utility: Positive returns → bullish bias, negative → bearish bias

4. **Return Momentum (Acceleration)**
   ```python
   # Rate of change of returns (is momentum increasing?)
   df['return_acceleration'] = df['log_return_1h'] - df['log_return_1h'].shift(1)
   ```
   - Positive acceleration → strengthening trend (continuation signal)
   - Negative acceleration → weakening trend (reversal warning)

**Total Directional Features**: ~10-12

### 2.2 Volatility Features (Regression-Optimized)

**Purpose**: Capture price dispersion and range dynamics for predicting next-hour GK volatility.

#### Core Volatility Indicators

1. **ATR (Average True Range, 14-period)**
   ```python
   df['atr_14'] = df.ta.atr(length=14)
   ```
   - Measures average price movement (high-low range)
   - Regression utility: Direct volatility proxy; high ATR → high expected GK volatility
   - **Secondary**: Used for both primary GK lags AND ATR lags (different perspectives on range)

2. **Bollinger Band Width**
   ```python
   bbands = df.ta.bbands(length=20, std=2)
   df['bb_upper'] = bbands['BBU_20_2.0']
   df['bb_lower'] = bbands['BBL_20_2.0']
   df['bb_middle'] = bbands['BBM_20_2.0']
   
   # Width as % of middle band (normalized volatility)
   df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
   ```
   - BB width expands during high volatility, contracts during low volatility
   - Regression utility: Width correlates strongly with realized volatility

3. **High-Low Range Percentage**
   ```python
   # Intra-hour price range as % of open price
   df['hl_range_pct'] = (df['high'] - df['low']) / df['open'] * 100
   ```
   - Simple measure of intra-hour turbulence
   - Complements GK formula (which uses log ratios)

4. **Close-to-Close Volatility (Rolling Std)**
   ```python
   # 24-hour rolling standard deviation of log returns
   df['close_vol_24h'] = df['log_return_1h'].rolling(24).std()
   ```
   - Traditional volatility estimator (for comparison with GK)
   - Captures recent volatility regime

**Total Volatility Features**: ~8-10

### 2.5 Temporal DNA (Dual-Task Seasonality)

**Hypothesis**: Crypto markets exhibit hourly and weekly seasonality affecting BOTH volatility and directional bias.

**Features to Create**:

1. **Hour of Day** (0-23):
   ```python
   df['hour_of_day'] = df['open time'].dt.hour
   
   # Optional: Cyclical encoding for ML models
   df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
   df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
   ```
   - **Volatility impact**: US market open (14:00 UTC) → higher volatility
   - **Directional impact**: Asian session (00:00 UTC) → different trend behavior

2. **Day of Week** (0-6, Monday=0):
   ```python
   df['day_of_week'] = df['open time'].dt.dayofweek
   
   # Cyclical encoding
   df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
   df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
   ```
   - **Volatility impact**: Weekend liquidity drops → lower volatility
   - **Directional impact**: Monday momentum (weekend news accumulation)

3. **Is Funding Hour** (Binary - Crypto-Specific):
   ```python
   # Perpetual futures funding at 00:00, 08:00, 16:00 UTC on Binance
   df['is_funding_hour'] = df['hour_of_day'].isin([0, 8, 16]).astype(int)
   ```
   - **Volatility spike**: Traders adjust positions pre-funding
   - **Directional bias**: Funding rate affects sentiment

4. **Is Weekend** (Binary):
   ```python
   df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Saturday=5, Sunday=6
   ```

**Total Temporal Features**: ~6-8

### 2.3 Microstructure Features (Dual-Task Optimized)

**Purpose**: Quantify market energy, liquidity dynamics, and buying/selling pressure.

#### Core Microstructure Metrics

1. **Taker Buy Ratio (Buy Pressure)**
   ```python
   # Ratio of aggressive buy volume to total volume
   df['taker_buy_ratio'] = df['taker buy base asset volume'] / df['volume']
   ```
   - Range: [0, 1]
   - Interpretation:
     - >0.5 → Aggressive buying dominates (bullish pressure)
     - <0.5 → Aggressive selling dominates (bearish pressure)
   - **Dual utility**: 
     - Classification: Directional bias (high ratio → likely up candle)
     - Regression: Imbalance → potential volatility spike

2. **Volume Z-Score (Abnormal Activity Detection)**
   ```python
   # 24-hour rolling z-score of volume
   df['volume_z_score'] = (
       (df['volume'] - df['volume'].rolling(24).mean()) / 
       df['volume'].rolling(24).std()
   )
   ```
   - Interpretation:
     - >2 → Abnormal buying pressure (institutional activity)
     - <-2 → Liquidity drought
   - **Dual utility**: Volume spikes precede both directional moves and volatility increases

3. **Trade Count Intensity** (if available)
   ```python
   # Normalize trade count by 24-hour average
   df['trade_intensity'] = df['number of trades'] / df['number of trades'].rolling(24).mean()
   ```
   - Higher fragmentation → more volatility
   - Captures market participation level

4. **Quote Asset Volume Ratio**
   ```python
   # Quote volume (USDT) relative to base volume (BTC)
   df['quote_volume_ratio'] = df['quote asset volume'] / df['volume']
   ```
   - Proxy for average trade size (institutional vs. retail)

**Total Microstructure Features**: ~6-8

### 2.4 Memory Lags: Dual-Path Persistence

**Econometric Foundation**: 
- **Volatility** exhibits autocorrelation (GARCH principle)
- **Returns** exhibit short-term momentum and mean-reversion

#### Path 1: Volatility Lags (Primary - GK, Secondary - ATR)

**GK Volatility Lags**:
```python
# 1-hour lag (immediate past volatility)
df['gk_lag_1h'] = df['gk_volatility'].shift(1)

# 3-hour average (short-term volatility baseline)
df['gk_lag_3h_mean'] = df['gk_volatility'].rolling(3).mean().shift(1)

# 6-hour average (medium-term regime)
df['gk_lag_6h_mean'] = df['gk_volatility'].rolling(6).mean().shift(1)

# Volatility acceleration (is vol increasing or decreasing?)
df['gk_lag_acceleration'] = df['gk_lag_1h'] - df['gk_lag_3h_mean']
```

**ATR Lags (Secondary Perspective)**:
```python
# ATR provides alternative view of price range dynamics
df['atr_lag_1h'] = df['atr_14'].shift(1)
df['atr_lag_3h_mean'] = df['atr_14'].rolling(3).mean().shift(1)
```
- **Rationale**: GK uses log ratios (multiplicative), ATR uses absolute ranges (additive)
- Complementary features give model diverse volatility perspectives

#### Path 2: Return Lags (Momentum & Mean-Reversion)

```python
# 1-hour lagged return (immediate momentum)
df['return_lag_1h'] = df['log_return_1h'].shift(1)

# 3-hour lagged return (short-term trend)
df['return_lag_3h'] = df['log_return_3h'].shift(1)

# 6-hour lagged return (medium-term bias)
df['return_lag_6h'] = df['log_return_6h'].shift(1)

# Return reversal indicator (mean-reversion signal)
df['return_reversal_3h'] = -df['log_return_3h'].shift(1)  # Contrarian bet
```

**⚠️ CRITICAL - Data Leakage Prevention**:
```python
# Always shift by at least 1 to prevent leakage
# CORRECT:   df['feature_lag'] = df['feature'].shift(1)
# INCORRECT: df['feature_lag'] = df['feature']  # Uses current time T!
```

**Total Memory Features**: ~12-15

### 2.6 Additional Technical Indicators (Breadth & Confirmation)

**Purpose**: Add complementary signals for robustness.

```python
import pandas_ta as ta

# Trend strength
df['adx_14'] = df.ta.adx(length=14)['ADX_14']  # >25 = strong trend

# Volume confirmation
df['obv'] = df.ta.obv()  # On-Balance Volume (cumulative volume flow)
df['cmf_20'] = df.ta.cmf(length=20)  # Chaikin Money Flow

# Moving averages (trend baselines)
df['ema_9'] = df.ta.ema(length=9)
df['ema_21'] = df.ta.ema(length=21)
df['ema_crossover'] = (df['ema_9'] > df['ema_21']).astype(int)  # Bullish crossover

# Stochastic oscillator (momentum)
stoch = df.ta.stoch(k=14, d=3)
df['stoch_k'] = stoch['STOCHk_14_3_3']
df['stoch_d'] = stoch['STOCHd_14_3_3']
```

**Total Additional Features**: ~10-12

---

### Feature Engineering Summary

| **Category** | **Feature Count** | **Primary Task** |
|--------------|-------------------|------------------|
| Directional Features | 10-12 | Classification |
| Volatility Features | 8-10 | Regression |
| Microstructure | 6-8 | Both |
| Memory Lags | 12-15 | Both |
| Temporal DNA | 6-8 | Both |
| Additional Technical | 10-12 | Both |
| **TOTAL** | **60-80 features** | **Shared** |

### Phase 2 Deliverable
- **Notebook**: `2_feature_engineering.ipynb`
- **Output**: `engineered_features_dual_target.csv`
- **Structure**: `[60-80 feature columns] + [target_reg] + [target_class]`
- **Key Validations**:
  - No data leakage: All features at time T, targets at T+1
  - No NaN values after dropping first 24 rows (max lag window)
  - Class balance report for `target_class`
  - Feature-target correlation analysis

---

## Phase 3: Preprocessing & Task-Specific Feature Selection

### Objectives
- Transform skewed distributions (log, Box-Cox)
- Standardize all features to zero mean, unit variance
- **Separate feature selection**: Reduce 60-80 features → **Top 10 for Regression** + **Top 10 for Classification**
- Manage feature budget: Aim for ~20-30 features per task before final selection

### 3.1 Log Transformation for Skewed Features

**Candidates**: Volume, ATR, any unbounded positive indicators.

**Method**:
```python
from scipy.stats import skew

# Identify highly skewed features (|skewness| > 1)
skewed_features = [col for col in df.columns if abs(skew(df[col].dropna())) > 1]

# Apply log1p (log(1+x) to handle zeros)
for col in skewed_features:
    df[f'{col}_log'] = np.log1p(df[col])
    df.drop(col, axis=1, inplace=True)  # Replace original
```

**Why Log Transform?**
- Compresses extreme values (e.g., volume spikes during black swan events)
- Makes distributions closer to Gaussian (better for linear models)
- Ensures 2018 low-volume hours and 2025 high-volume hours are comparable

**Alternative**: Yeo-Johnson transformation (handles negative values, via `sklearn.preprocessing.PowerTransformer`)

### 3.2 Standardization (Z-Score Scaling)

**Method**: StandardScaler from scikit-learn

```python
from sklearn.preprocessing import StandardScaler

# Fit on training data only (prevent data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training stats
```

**Why Standardize?**
- Makes features with different units comparable (e.g., RSI [0,100] vs. ATR [0.001, 10])
- Essential for L1/L2 regularization (Lasso, Ridge) to work correctly
- Accelerates gradient descent in neural networks (if extended later)

**Alternative for Tree Models**: Tree-based models (Random Forest, XGBoost) don't require scaling, but we standardize anyway for Lasso phase.

### 3.3 Dual-Track Feature Selection

**Strategy**: Separate feature selection pipelines for regression and classification tasks.

#### Track 1: Regression Feature Selection (For `target_reg`)

**Tier 1: Correlation-Based Filtering**
```python
# Focus on features correlated with target_reg
reg_features = df.drop(['target_reg', 'target_class'], axis=1)
correlations = reg_features.corrwith(df['target_reg']).abs().sort_values(ascending=False)

# Keep top 30 features most correlated with volatility
top_30_reg = correlations.head(30).index.tolist()
print(f"Top 30 Regression Features:\n{correlations.head(30)}")
```

**Expected strong candidates**: `gk_lag_1h`, `atr_14`, `bb_width_pct`, `hl_range_pct`, `close_vol_24h`

**Tier 2: Lasso Regression (L1 Regularization)**
```python
from sklearn.linear_model import LassoCV

X_reg = df[top_30_reg]
y_reg = df['target_reg']

lasso_reg = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_reg.fit(X_reg_scaled, y_reg)

# Extract non-zero coefficients
lasso_coefs_reg = pd.Series(lasso_reg.coef_, index=top_30_reg)
selected_reg_20 = lasso_coefs_reg[lasso_coefs_reg != 0].index.tolist()

print(f"Lasso selected {len(selected_reg_20)} features for regression")
```

**Expected output**: ~20 features

**Tier 3: RFE with XGBoost (Final Top 10)**
```python
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

estimator_reg = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
selector_reg = RFE(estimator_reg, n_features_to_select=10, step=2)
selector_reg.fit(X_reg_scaled[selected_reg_20], y_reg)

top_10_reg = [selected_reg_20[i] for i in range(len(selected_reg_20)) if selector_reg.support_[i]]
print("Golden Top 10 for Regression:")
for i, feat in enumerate(top_10_reg, 1):
    print(f"{i}. {feat}")
```

---

#### Track 2: Classification Feature Selection (For `target_class`)

**Tier 1: Correlation-Based Filtering**
```python
# Focus on features correlated with target_class (directional movement)
correlations_class = reg_features.corrwith(df['target_class']).abs().sort_values(ascending=False)

# Keep top 30 features most correlated with direction
top_30_class = correlations_class.head(30).index.tolist()
print(f"Top 30 Classification Features:\n{correlations_class.head(30)}")
```

**Expected strong candidates**: `rsi_14`, `macd_histogram`, `log_return_1h`, `taker_buy_ratio`, `return_lag_1h`

**Tier 2: Logistic Regression with L1 (Lasso for Classification)**
```python
from sklearn.linear_model import LogisticRegressionCV

X_class = df[top_30_class]
y_class = df['target_class']

lasso_class = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', random_state=42)
lasso_class.fit(X_class_scaled, y_class)

# Extract non-zero coefficients
lasso_coefs_class = pd.Series(lasso_class.coef_[0], index=top_30_class)
selected_class_20 = lasso_coefs_class[lasso_coefs_class != 0].index.tolist()

print(f"Lasso selected {len(selected_class_20)} features for classification")
```

**Expected output**: ~20 features

**Tier 3: RFE with XGBoost Classifier (Final Top 10)**
```python
from xgboost import XGBClassifier

estimator_class = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False)
selector_class = RFE(estimator_class, n_features_to_select=10, step=2)
selector_class.fit(X_class_scaled[selected_class_20], y_class)

top_10_class = [selected_class_20[i] for i in range(len(selected_class_20)) if selector_class.support_[i]]
print("Golden Top 10 for Classification:")
for i, feat in enumerate(top_10_class, 1):
    print(f"{i}. {feat}")
```

---

#### Feature Overlap Analysis

```python
# Check how many features are shared between tasks
overlap = set(top_10_reg).intersection(set(top_10_class))
print(f"\nShared features between tasks: {len(overlap)}")
print(f"Overlapping features: {overlap}")

# Unique features per task
unique_reg = set(top_10_reg) - set(top_10_class)
unique_class = set(top_10_class) - set(top_10_reg)
print(f"Regression-only features: {unique_reg}")
print(f"Classification-only features: {unique_class}")
```

**Expected**: Some overlap (e.g., `gk_lag_1h`, `volume_z_score`), but most features are task-specific.

### 3.4 Feature Importance Visualization (Dual-Task)

**Regression Feature Importances**:
```python
import matplotlib.pyplot as plt

importances_reg = estimator_reg.feature_importances_
indices_reg = np.argsort(importances_reg)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Regression: Golden Top 10 Feature Importances")
plt.bar(range(10), importances_reg[indices_reg[:10]])
plt.xticks(range(10), [top_10_reg[i] for i in indices_reg[:10]], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('regression_feature_importance.png')
plt.show()
```

**Classification Feature Importances**:
```python
importances_class = estimator_class.feature_importances_
indices_class = np.argsort(importances_class)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Classification: Golden Top 10 Feature Importances")
plt.bar(range(10), importances_class[indices_class[:10]], color='green')
plt.xticks(range(10), [top_10_class[i] for i in indices_class[:10]], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('classification_feature_importance.png')
plt.show()
```

### Phase 3 Deliverable
- **Notebook**: `3_preprocessing_and_feature_selection.ipynb`
- **Outputs**: 
  - `final_training_data_regression.csv` (Top 10 regression features + `target_reg`)
  - `final_training_data_classification.csv` (Top 10 classification features + `target_class`)
  - `scaler_regression.pkl` (StandardScaler for regression features)
  - `scaler_classification.pkl` (StandardScaler for classification features)
  - `selected_features_regression.json` (list of Top 10 regression features)
  - `selected_features_classification.json` (list of Top 10 classification features)
  - Feature importance plots

---

## 4. Additional Enhancements & Best Practices

### 4.1 Temporal Train-Test Split

**Critical for Time Series**: Do NOT use random split!

```python
# Use first 85% for training, last 15% for testing
split_index = int(len(df) * 0.85)
train = df.iloc[:split_index]
test = df.iloc[split_index:]
```

**Why?**
- Prevents look-ahead bias
- Mimics real trading (train on past, predict future)

**Enhancement**: Use Walk-Forward Validation
- Train on Year 1-6, validate on Year 7, test on Year 8
- Retrain quarterly to adapt to regime changes

### 4.2 Handling Null Values Post-Feature Engineering

**Issue**: Rolling windows and lags create NaNs at the start of the dataset.

**Solution**:
```python
# Drop first 24 rows (max lookback period - though we use 6h max, be safe)
df = df.iloc[24:]

# Verify no NaNs remain
assert df.isnull().sum().sum() == 0, "Null values still exist!"
```

### 4.3 Feature Engineering Sanity Checks

**Validation Tests**:
1. **No Data Leakage**: Ensure all features use `.shift()` where appropriate
2. **No Infinite Values**: Check for division by zero in ratios
3. **Distribution Checks**: Plot histograms before/after transformations
4. **Target Correlation**: Compute Pearson correlation between each feature and **both targets**
   - Features with $|r| < 0.01$ for both targets are likely useless

### 4.4 Model Selection Strategy (Post-Feature Selection)

#### Regression Models (For `target_reg`)
1. **Linear Regression** (Baseline): Simple, interpretable
2. **Ridge Regression**: L2 regularization for stability
3. **Random Forest Regressor**: Captures non-linearities
4. **XGBoost Regressor**: State-of-art for tabular data
5. **LightGBM Regressor**: Faster alternative to XGBoost

#### Classification Models (For `target_class`)
1. **Logistic Regression** (Baseline): Probabilistic interpretation
2. **Random Forest Classifier**: Robust to outliers
3. **XGBoost Classifier**: State-of-art performance
4. **LightGBM Classifier**: Fast training on large datasets
5. **CatBoost Classifier**: Handles categorical features well (if any remain)

**Evaluation Metrics**:

**Regression**:
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Robust to outliers
- **R²** (Coefficient of Determination): Variance explained
- **MAPE** (Mean Absolute Percentage Error): Relative error

**Classification**:
- **Accuracy**: Overall correctness (baseline metric)
- **Precision/Recall**: For up/down class (if imbalanced)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (model's discriminative ability)
- **Confusion Matrix**: Detailed breakdown of predictions

### 4.5 Class Imbalance Handling (Phase 4 Modeling Consideration)

**Issue**: During trending markets, `target_class` can be imbalanced (e.g., 60% up candles in bull markets).

**Mitigation Strategies** (to implement in Phase 4):
1. **Class Weights**: Penalize minority class errors more heavily
   ```python
   XGBClassifier(scale_pos_weight=ratio_of_negatives_to_positives)
   ```

2. **SMOTE** (Synthetic Minority Oversampling):
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
   ```

3. **Threshold Tuning**: Instead of 0.5 probability cutoff, optimize threshold based on precision-recall curve

4. **Stratified Sampling**: Ensure train/test split maintains class distribution

**Documentation Note**: Class imbalance analysis should be included in the Phase 2 deliverable (class distribution report), but handling is deferred to Phase 4 modeling.

### 4.6 Explainability & Monitoring

**Post-Deployment** (Phase 5):
- Use SHAP values to explain individual predictions (both regression and classification)
- Monitor feature drift (distribution changes over time)
- Set up alerts:
  - Regression: When GK predictions deviate >2σ from realized volatility
  - Classification: When prediction accuracy drops below baseline (e.g., <55%)
- Track class distribution drift in production data

---

## Implementation Roadmap

| **Phase** | **Notebook** | **Estimated Time** | **Key Outputs** |
|-----------|--------------|-------------------|-----------------|
| 1 | `1_data_cleaning_and_dual_targets.ipynb` | 2-3 hours | `cleaned_data_with_dual_targets.csv` |
| 2 | `2_feature_engineering_dual_task.ipynb` | 5-7 hours | `engineered_features_dual_target.csv` (60-80 features + 2 targets) |
| 3 | `3_preprocessing_and_dual_selection.ipynb` | 4-5 hours | Separate datasets: Regression (Top 10) & Classification (Top 10) |
| 4 | `4_model_training_dual_task.ipynb` | 3-4 hours | Trained models (regression + classification), metrics, class imbalance handling |
| 5 | `5_backtesting_and_deployment.ipynb` | 3-4 hours | Walk-forward validation, production API, monitoring setup |

**Total Estimated Time**: 17-23 hours (distributed over 2-3 weeks)

---

## Dependencies & Environment Setup

**Required Libraries**:
```
pandas>=2.0.0
numpy>=1.24.0
pandas-ta>=0.3.14b
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
jupyter>=1.0.0
```

**Installation**:
```bash
pip install -r requirements.txt
```

**Hardware Recommendations**:
- **RAM**: 16GB minimum (8 years of hourly data ≈ 70,000 rows)
- **CPU**: Multi-core (XGBoost and RFE benefit from parallelization)
- **Storage**: 5GB for intermediate files

---

## Risk Factors & Limitations

1. **Regime Changes**: 2018-2025 spans multiple bull/bear cycles. Models may struggle during unprecedented events (e.g., FTX collapse).
   - **Mitigation**: Add regime detection features (funding rates, volatility regime shifts)

2. **Overfitting Risk**: 60-80 features on ~70k samples → risk of memorizing noise
   - **Mitigation**: Aggressive feature selection (Top 10 per task) + cross-validation + regularization

3. **Non-Stationarity**: Crypto volatility structure evolves (e.g., spot vs. derivatives dominance shift)
   - **Mitigation**: Periodic retraining (quarterly), walk-forward validation

4. **Class Imbalance**: `target_class` can be skewed during trending markets (60/40 or worse)
   - **Mitigation**: Class weighting, SMOTE, threshold optimization (Phase 4)

5. **Microstructure Noise**: Hourly data smooths out flash crashes and spoofing
   - **Trade-off**: Accept lower resolution for stability, focus on persistent patterns

6. **Survivorship Bias**: Binance data only (what about Bitfinex, Kraken price divergences?)
   - **Enhancement**: Future work could add multi-exchange correlation features

7. **Dual-Target Complexity**: Training separate models vs. multi-task learning requires careful evaluation
   - **Decision Point**: Phase 4 will compare:
     - **Approach A**: Separate models (XGBoost Regressor + XGBoost Classifier)
     - **Approach B**: Multi-task neural network (shared features, dual loss function)

---

## Conclusion

This **dual-target architecture** provides a comprehensive approach to crypto market prediction by addressing **two complementary dimensions**:

1. **Volatility Forecasting** (Regression): Predicts market turbulence for risk management
2. **Directional Prediction** (Classification): Predicts candle direction for trading signals

By combining:
- **Rigorous dual-target generation** (Phase 1)
- **Shared feature engineering with task-aware design** (Phase 2)
- **Separate feature selection pipelines** (Phase 3)

We transform raw OHLCV data into **two optimized feature matrices** (Top 10 each), ready for task-specific modeling. The resulting models will be:
- **Lean**: Only 10 features per task (no redundancy)
- **Interpretable**: Clear feature importances for both volatility and direction
- **Actionable**: Dual predictions enable combined strategies (e.g., "High vol + Up direction → Long with tight stops")

**Next Steps**: 
1. Update Phase 1 notebook to generate `target_reg` and `target_class`
2. Implement Phase 2 with all 60-80 features following the dual-task design
3. Execute dual-track feature selection in Phase 3
4. Compare separate vs. multi-task modeling approaches in Phase 4

---

**Questions or Modifications?** This plan is a living document. Adjust feature categories, lag windows, or selection thresholds based on empirical validation results.
