# CryptoFlow Regression Model: Architecture Plan
## Predicting Garman-Klass Volatility for 1-Hour BTC/USDT Intervals

**Author**: Quant Research Engineering Team  
**Dataset**: `btc_1h_data_2018_to_2025.csv` (8 years, Binance hourly OHLCV)  
**Target**: Next-hour Garman-Klass (GK) Volatility Index  
**Modeling Paradigm**: Supervised Regression with Advanced Feature Selection

---

## Executive Summary

This pipeline transforms raw OHLCV data into a curated feature matrix optimized for volatility forecasting. The architecture consists of three sequential phases:

1. **Data Cleaning & Ground Truth Generation**: Robust handling of missing data and implementation of the Garman-Klass volatility estimator
2. **Massive Feature Engineering**: Creation of 50+ candidate features spanning technical indicators, temporal patterns, and market microstructure
3. **Preprocessing & Multi-Tier Feature Selection**: Scaling, transformation, and systematic feature elimination to identify the "Golden Top 10"

The final output is a training-ready dataset where each row represents a 1-hour interval with engineered features and a ground-truth volatility target for the *next* hour.

---

## Phase 1: Data Cleaning & Ground Truth Generation

### Objectives
- Ensure temporal continuity (no missing hours)
- Generate ground truth volatility measure using Garman-Klass formula
- Create predictive target by forward-shifting volatility

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

### 1.3 Target Variable Creation

**Objective**: Predict the *next* hour's volatility.

**Implementation**:
```python
# Shift GK volatility backward by 1 row (shift=-1 means future)
df['target_gk_next_hour'] = df['gk_volatility'].shift(-1)

# Remove the last row (no future target available)
df = df[:-1]
```

**Validation Checks**:
- No data leakage: Ensure target is computed *only* from future data
- Target distribution: Plot histogram to check for normality (may need log transform later)
- Correlation: Verify that current GK ≠ future GK (otherwise trivial prediction)

### Phase 1 Deliverable
- **Notebook**: `1_data_cleaning_and_ground_truth.ipynb`
- **Output**: `cleaned_data_with_gk_target.csv`
- **Columns Added**: `gk_volatility`, `target_gk_next_hour`, `is_interpolated`

---

## Phase 2: Massive Feature Engineering

### Objectives
- Generate 50+ candidate features from technical analysis, temporal patterns, and market microstructure
- Create memory lags to capture volatility persistence
- Add domain-specific crypto features (funding hours, market energy)

### 2.1 Technical Indicators (20+ via pandas_ta)

**Library**: `pandas_ta` (https://github.com/twopirllc/pandas-ta)

**Categories & Rationale**:

| **Category**       | **Indicators**                              | **Why It Matters**                                      |
|--------------------|---------------------------------------------|---------------------------------------------------------|
| **Trend**          | EMA(9, 21, 50), SMA(200), VWMA              | Directional bias affects volatility clustering          |
| **Momentum**       | RSI(14), Stochastic, Williams %R, ROC       | Overbought/oversold regimes precede volatility spikes   |
| **Volatility**     | ATR(14), Bollinger Bands (20,2), Keltner    | Direct volatility proxies; BB width = realized vol      |
| **Volume**         | OBV, CMF, MFI, VWAP                         | Volume precedes price; smart money signals              |
| **Oscillators**    | MACD, PPO, AO                               | Divergences flag regime transitions                     |
| **Strength**       | ADX, Aroon, Supertrend                      | Trend strength correlates with volatility sustainability|

**Implementation Strategy**:
```python
import pandas_ta as ta

# Apply all indicators at once
df.ta.strategy("all", append=True)

# Or selective application
df.ta.ema(length=9, append=True)
df.ta.rsi(length=14, append=True)
df.ta.bbands(length=20, std=2, append=True)
df.ta.atr(length=14, append=True)
df.ta.macd(append=True)
# ... (continue for 20+ indicators)
```

**Normalization Note**: Some indicators are unbounded (e.g., MACD) while others are ratio-bound (RSI ∈ [0,100]). Standardization in Phase 3 will handle this.

### 2.2 Temporal DNA

**Hypothesis**: Crypto markets exhibit hourly and weekly seasonality.

**Features to Create**:

1. **Hour of Day** (0-23):
   ```python
   df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
   ```
   - Rationale: US market open (14:00 UTC), Asian session (00:00 UTC) show distinct volatility
   - Encoding: Consider cyclical encoding → `sin(2π·hour/24)` and `cos(2π·hour/24)`

2. **Day of Week** (0-6, Monday=0):
   ```python
   df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
   ```
   - Rationale: Weekend liquidity drops, Monday momentum
   - Encoding: Cyclical encoding or one-hot (test both)

3. **Is Funding Hour** (Binary):
   ```python
   df['is_funding_hour'] = df['hour_of_day'].isin([0, 8, 16]).astype(int)
   ```
   - Rationale: Perpetual futures funding happens 3x daily on Binance (00:00, 08:00, 16:00 UTC)
   - Impact: Volatility spikes as traders adjust positions to avoid funding fees

**Enhancement Suggestion**:
- Add `is_weekend` (Saturday/Sunday)
- Add `is_month_end` (last 3 days of month → institutional rebalancing)
- Add `days_since_epoch` (linear time trend to capture bull/bear macro cycles)

### 2.3 Market Energy Metrics

**Concept**: Quantify market "stress" and liquidity dynamics.

**Features**:

1. **Volume Z-Score** (Rolling 24h window):
   ```python
   df['volume_z_score'] = (
       (df['volume'] - df['volume'].rolling(24).mean()) / 
       df['volume'].rolling(24).std()
   )
   ```
   - Interpretation: >2 → abnormal buying pressure; <-2 → liquidity drought

2. **Trade Count** (if available in dataset):
   - Higher trade count → market fragmentation → higher volatility
   - Normalize per hour

3. **Taker Buy Ratio** (if available):
   ```python
   df['taker_buy_ratio'] = df['taker_buy_volume'] / df['volume']
   ```
   - Ratio >0.5 → aggressive buying (bullish pressure)
   - Deviation from 0.5 correlates with directional volatility

**Fallback** (if trade-level data unavailable):
- Compute `volume_to_volatility_ratio = volume / gk_volatility` (liquidity proxy)
- Add `price_range_pct = (high - low) / open` (intra-hour turbulence)

### 2.4 Memory Lags: Volatility Persistence

**Econometric Fact**: Volatility exhibits autocorrelation (GARCH models exploit this).

**Features**:
```python
# Previous hour's GK volatility
df['gk_lag_1h'] = df['gk_volatility'].shift(1)

# 3-hour average (short-term momentum)
df['gk_lag_3h_mean'] = df['gk_volatility'].rolling(3).mean().shift(1)

# 24-hour average (daily baseline)
df['gk_lag_24h_mean'] = df['gk_volatility'].rolling(24).mean().shift(1)

# Volatility trend (is it accelerating?)
df['gk_lag_diff'] = df['gk_lag_1h'] - df['gk_lag_3h_mean']
```

**Critical**: Always shift by 1 to avoid data leakage! The model must not see the current hour's GK when predicting the next hour.

**Additional Lag Ideas**:
- Exponential moving average of volatility (EWMA with α=0.94, RiskMetrics standard)
- Max/Min GK in past 24h (range-bound volatility)
- Volatility of volatility (rolling std of GK over 24h)

### 2.5 Price Momentum & Returns

**Features**:
```python
# Hourly log returns
df['log_return_1h'] = np.log(df['close'] / df['close'].shift(1))

# Multi-horizon returns
df['log_return_6h'] = np.log(df['close'] / df['close'].shift(6))
df['log_return_24h'] = np.log(df['close'] / df['close'].shift(24))

# Return acceleration
df['return_acceleration'] = df['log_return_1h'] - df['log_return_1h'].shift(1)
```

**Why Log Returns?**
- Symmetric around zero
- Time-additive: $r_{t,t+2} = r_{t,t+1} + r_{t+1,t+2}$
- Better for statistical modeling (closer to normal distribution)

### Phase 2 Deliverable
- **Notebook**: `2_feature_engineering.ipynb`
- **Output**: `engineered_features.csv`
- **Feature Count**: ~60-80 columns (depends on pandas_ta strategy)

---

## Phase 3: Preprocessing & Multi-Tier Feature Selection

### Objectives
- Transform skewed distributions (log, Box-Cox)
- Standardize all features to zero mean, unit variance
- Reduce dimensionality from 60-80 features → **Golden Top 10**

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

### 3.3 Feature Selection: Three-Tier Filtration

#### Tier 1: Correlation-Based Filtering (Remove Redundancy)

**Problem**: Many technical indicators are highly correlated (e.g., EMA(9) and EMA(21) ≈ 0.95 correlation).

**Solution**:
1. Compute correlation matrix of all features
2. Visualize with heatmap (seaborn)
3. Remove one feature from each pair where $|r| > 0.90$

**Algorithm**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
corr_matrix = df.corr().abs()

# Plot heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# Remove highly correlated features
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.90)]
df.drop(to_drop, axis=1, inplace=True)

print(f"Removed {len(to_drop)} highly correlated features")
```

**Expected Output**: Reduce from 60-80 features → ~40-50 features

#### Tier 2: Embedded Method (Lasso Regression)

**Method**: L1 Regularization (Lasso) shrinks irrelevant coefficients to exactly zero.

**Implementation**:
```python
from sklearn.linear_model import LassoCV

# Use cross-validated alpha selection
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Extract non-zero coefficients
lasso_coefs = pd.Series(lasso.coef_, index=X_train.columns)
selected_features = lasso_coefs[lasso_coefs != 0].index.tolist()

print(f"Lasso selected {len(selected_features)} features")
print(lasso_coefs[lasso_coefs != 0].sort_values())
```

**Why Lasso?**
- Automatic feature selection (unlike Ridge which shrinks but doesn't eliminate)
- Interpretable: Non-zero coefficients = relevant features
- Fast computation (linear model)

**Hyperparameter**: `alpha` (regularization strength)
- Use `LassoCV` to automatically find optimal alpha via cross-validation
- Higher alpha → more aggressive feature elimination

**Expected Output**: Reduce to ~20-30 features

#### Tier 3: Wrapper Method (Recursive Feature Elimination)

**Method**: Recursively train model, remove weakest feature, repeat.

**Implementation**:
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
# Or use XGBoost for better performance
from xgboost import XGBRegressor

# Base estimator
estimator = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# RFE to select top 10 features
selector = RFE(estimator, n_features_to_select=10, step=1)
selector.fit(X_train_scaled, y_train)

# Get selected features
selected_features = X_train.columns[selector.support_].tolist()
print("Golden Top 10 Features:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i}. {feat}")
```

**Why RFE with Tree Models?**
- **Random Forest**: Captures non-linear interactions, robust to outliers
- **XGBoost**: State-of-art for tabular data, built-in feature importance
- **Recursive Elimination**: Accounts for multivariate relationships (not just univariate correlation)

**Computational Cost**: RFE is expensive (trains n_features models). Optimize by:
- Use `step=5` (remove 5 features at a time) for faster convergence
- Pre-filter to top 30 features before RFE

**Expected Output**: **Golden Top 10** features

### 3.4 Feature Importance Visualization

**Post-Selection Analysis**:
```python
import matplotlib.pyplot as plt

# Get feature importances from final model
importances = estimator.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Golden Top 10)")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), [selected_features[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
```

### Phase 3 Deliverable
- **Notebook**: `3_preprocessing_and_feature_selection.ipynb`
- **Output**: 
  - `final_training_data.csv` (only Golden Top 10 + target)
  - `scaler.pkl` (saved StandardScaler for production)
  - `selected_features.json` (list of top 10 features)

---

## Additional Enhancements & Best Practices

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
# Drop first 24 rows (max lookback period)
df = df.iloc[24:]

# Verify no NaNs remain
assert df.isnull().sum().sum() == 0, "Null values still exist!"
```

### 4.3 Feature Engineering Sanity Checks

**Validation Tests**:
1. **No Data Leakage**: Ensure all features use `.shift()` where appropriate
2. **No Infinite Values**: Check for division by zero in ratios
3. **Distribution Checks**: Plot histograms before/after transformations
4. **Target Correlation**: Compute Pearson correlation between each feature and target
   - Features with $|r| < 0.01$ are likely useless

### 4.4 Model Selection Strategy (Post-Feature Selection)

**Recommended Models to Test**:
1. **Linear Regression** (Baseline): Simple, interpretable
2. **Ridge Regression**: L2 regularization for stability
3. **Random Forest**: Captures non-linearities
4. **XGBoost/LightGBM**: State-of-art for tabular data
5. **LSTM** (Advanced): If temporal dependencies are strong

**Evaluation Metrics**:
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Robust to outliers
- **R²** (Coefficient of Determination): Variance explained
- **MAPE** (Mean Absolute Percentage Error): Relative error

### 4.5 Explainability & Monitoring

**Post-Deployment**:
- Use SHAP values to explain individual predictions
- Monitor feature drift (distribution changes over time)
- Set up alerts for when GK predictions deviate >2σ from realized volatility

---

## Implementation Roadmap

| **Phase** | **Notebook**                              | **Estimated Time** | **Key Outputs**                          |
|-----------|-------------------------------------------|--------------------|------------------------------------------|
| 1         | `1_data_cleaning_and_ground_truth.ipynb`  | 2-3 hours          | `cleaned_data_with_gk_target.csv`        |
| 2         | `2_feature_engineering.ipynb`             | 4-6 hours          | `engineered_features.csv` (60-80 cols)   |
| 3         | `3_preprocessing_and_feature_selection.ipynb` | 3-4 hours      | `final_training_data.csv` (Golden Top 10)|
| 4         | `4_model_training_and_evaluation.ipynb`   | 2-3 hours          | Trained models, performance metrics      |
| 5         | `5_backtesting_and_deployment.ipynb`      | 3-4 hours          | Walk-forward validation, production API  |

**Total Estimated Time**: 14-20 hours (distributed over 1-2 weeks)

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

1. **Regime Changes**: 2018-2025 spans multiple bull/bear cycles. Model may struggle during unprecedented events (e.g., FTX collapse).
   - **Mitigation**: Add regime detection features (VIX, funding rates)

2. **Overfitting**: 60-80 features on 70k samples → risk of memorizing noise.
   - **Mitigation**: Aggressive feature selection + cross-validation

3. **Non-Stationarity**: Crypto volatility structure evolves (e.g., spot vs. derivatives dominance).
   - **Mitigation**: Periodic retraining (quarterly)

4. **Microstructure Noise**: Hourly data smooths out flash crashes and spoofing.
   - **Trade-off**: Accept lower resolution for stability

5. **Survivorship Bias**: Binance data only (what about Bitfinex, Kraken?).
   - **Enhancement**: Add multi-exchange correlation features

---

## Conclusion

This architecture provides a robust, systematic approach to volatility forecasting. By combining:
- **Rigorous data cleaning** (Phase 1)
- **Domain-driven feature engineering** (Phase 2)
- **Multi-tier feature selection** (Phase 3)

We transform raw OHLCV data into a lean, high-signal feature matrix optimized for machine learning. The resulting **Golden Top 10** features will be statistically independent, maximally informative, and ready for production deployment.

**Next Step**: Generate Phase 1 notebook (`1_data_cleaning_and_ground_truth.ipynb`) and begin implementation.

---

**Questions or Modifications?** This plan is a living document. Adjust parameters (lookback windows, indicator choices, feature count) based on empirical validation results.
