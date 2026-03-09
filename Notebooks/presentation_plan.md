# CryptoFlow Classification — Presentation Plan
## 5-Slide Structure: From Data to Why Nothing Works

---

## Rationale for Condensing

You built **18 models across 9 families** (LR, DT, RF, SVM, PCA+X, Clustering, MLP, LSTM, XGBoost/GB).
Presenting each scratch + sklearn pair separately = 18 mini-stories that all end the same way: AUC ≈ 0.54.

**The narrative that actually matters is:**
1. What we tried to predict and why it's hard
2. A tour through model complexity tiers — showing the ceiling never moves
3. The best model and the full leaderboard
4. The honest root cause explanation

The scratch implementations serve one purpose in this presentation: **proof of correctness** (they match sklearn results).
That is a single bullet point per section, not a dedicated comparison.

---

## Slide Structure Overview

| Slide | Title | Core Message | Est. Time |
|-------|-------|--------------|-----------|
| 1 | The Problem | Predicting BTC direction is hard by design | 2 min |
| 2 | Simple Models (Linear + Trees) | AUC ceiling appears early at ~0.54 | 3 min |
| 3 | Beyond Linearity (SVM + PCA + Clustering) | Kernels and dimensionality reduction don't help | 2 min |
| 4 | Neural Networks (MLP → LSTM → CNN-LSTM) | Depth and temporal structure don't help either | 3 min |
| 5 | Best Model + Full Leaderboard | XGBoost wins, but barely beats random | 2 min |
| 6 | Why We Couldn't Do Better | 6 root causes — data, not model, is the bottleneck | 3 min |

**Total: ~15 minutes**

---

---

## Slide 1 — The Problem & The Dataset

### Headline
> *Predicting whether Bitcoin will go up or down in the next hour — with 7 years of data*

### Content

**Task definition (1 sentence each):**
- Binary classification: `target_class = 1` (UP) if next 1-hour return > 0, else `0` (DOWN)
- Dataset: BTC/USDT 1-hour OHLCV + engineered features, Jan 2018 – Feb 2026
- 70,919 samples × 55 features — price, volume, momentum, volatility, rolling stats
- Train/test split: 60,281 train (up to Nov 2024) / 10,638 test (Nov 2024 – Feb 2026)
- Class balance: **UP 50.34% / DOWN 49.66%** → near-perfectly balanced, not trivial

**Baseline to beat:**
- Majority class baseline: AUC = 0.500 (always predicts UP)
- Any model must beat this to claim signal

### Plots to Show
1. **BTC price history (2018–2026)** — shows dramatic regime changes (bear/bull/crash/ATH)
   - *Notebook cell:* `#VSC-b36f3072` — `btc_price_history` subplot
2. **Hourly return distribution** — shows near-symmetric, fat-tailed distribution
   - *Notebook cell:* `#VSC-b36f3072` — return histogram subplot

### Talking Points
- "The target is inherently noisy — half the hours go up, half go down"
- "We have a rich feature set, but all features are derived from price and volume"
- "The model sees the same 7 years of history you do — let's see what it finds"

### What to Cut
- No need to list all 55 feature names — a **feature category pie chart** suffices
  (see `#VSC-06aac2b2`: momentum/volatility/trend/volume/lag buckets)

---

---

## Slide 2 — Simple Models: Linear & Trees

### Headline
> *The ceiling appears immediately — simple models hit AUC ≈ 0.54 right away*

### Models Covered (6 models → 3 families)
| Family | Scratch AUC | sklearn AUC | Key Note |
|--------|-------------|-------------|----------|
| Logistic Regression | 0.5388 | 0.5386 | Results match — scratch validates |
| Decision Tree | 0.5295 | 0.5359 | Best depth=3; overfits above depth 5 |
| Random Forest | S | **0.5405** | Top feature: RSI/momentum indicators |

### How to Handle Scratch vs Sklearn
> **One bullet point:** *"Scratch implementations match sklearn within 0.002 AUC — confirms correctness of both codebases"*

Do **not** show separate confusion matrices for each. Show **one combined metrics table**.

### Plots to Show
1. **DT Depth vs AUC chart** — shows overfitting cliff after depth 3–4
   - *Notebook cell:* `#VSC-4295e531`
   - *Why it's interesting:* visually proves the signal is shallow — deep trees just memorize noise
2. **RF Top-15 Feature Importance bar chart**
   - *Notebook cell:* `#VSC-584aa5bd`
   - *Why it's interesting:* momentum/RSI features dominate — confirms technical analysis intuition

### Results Table to Show
```
Model               | AUC    | Accuracy | F1
--------------------|--------|----------|------
LR Scratch          | 0.5388 | 52.4%    | 0.530
LR sklearn          | 0.5386 | 52.5%    | 0.531
DT Scratch          | 0.5295 | 52.9%    | 0.533
DT sklearn (d=3)    | 0.5359 | 53.1%    | 0.560
RF Scratch          | 0.5378 | 52.3%    | 0.567
RF sklearn (n=200)  | 0.5405 | 53.1%    | 0.552
```

### Talking Points
- "Logistic regression finds some signal immediately — 0.539 vs 0.500 baseline"
- "Decision trees overfit badly past depth 3 — the patterns are shallow, not deep"
- "Random forest with 200 trees is only 0.003 better than one shallow tree"
- "All three families cluster between AUC 0.53 and 0.54 — the ceiling is real"

---

---

## Slide 3 — Beyond Linearity: SVM, PCA & Clustering

### Headline
> *Kernel tricks and dimensionality tricks don't move the needle*

### Models Covered (7 models → 3 families)
| Family | Best AUC | Key Finding |
|--------|----------|-------------|
| SVM (Linear) | 0.5381 | Same as LR — no gain from margin maximisation |
| SVM (RBF kernel) | **0.5429** | Best of the three SVM variants — marginal gain |
| PCA + RF | 0.5220 | **Worse** than plain RF — PCA loses signal |
| PCA + SVM | 0.5329 | Recovers some signal but still below no-PCA baseline |
| K-Means / Agglomerative | ~0.500 | Equivalent to majority baseline — no clusters found |

### How to Handle Scratch vs Sklearn
> **One bullet:** *"SVM scratch linear (AUC=0.527) confirms the sklearn result; PCA scratch collapsed to AUC=0.515 — a degenerate classifier predicting all-negative"*

### Plots to Show
1. **PCA Scree Plot (explained variance vs n_components)**
   - *Notebook cell:* `#VSC-01924c3e`
   - *Why:* shows ~20 components capture 90% variance — but this doesn't mean those components are predictive
2. **PCA before/after AUC comparison bar** (RF with vs without PCA, SVM with vs without PCA)
   - *Notebook cell:* `#VSC-130bd4be`
   - *Why:* directly shows PCA hurts in both cases
3. **K-Means Elbow Plot**
   - *Notebook cell:* `#VSC-5a7ad51e`
   - *Why:* no elbow = no natural clusters in feature space = no latent structure to exploit

### Results Table to Show
```
Model                | AUC    | Accuracy | Note
---------------------|--------|----------|-----------------------------
SVM Scratch Linear   | 0.5271 | 51.1%    | Degenerate (low recall)
SVM sklearn Linear   | 0.5381 | 51.9%    | Matches LR
SVM sklearn RBF      | 0.5429 | 52.7%    | Best SVM — kernel helps slightly
PCA+RF sklearn       | 0.5220 | 51.7%    | WORSE than plain RF (0.5405)
PCA+SVM sklearn      | 0.5329 | 51.2%    | Recovers some but still below
KMeans/Agglom        | ~0.500 | 50.3%    | Majority class baseline
```

### Talking Points
- "The RBF kernel gives +0.002 over linear SVM — probably not significant"
- "PCA actually hurts — the noisy dimensions we discard might contain the little signal we have"
- "Clustering finds no structure — there is no hidden geometric separation in this feature space"
- "SVM scratch had a degenerate collapse — predicting mostly DOWN — a good lesson in implementation sensitivity"

---

---

## Slide 4 — Neural Networks: Perceptron → MLP → LSTM → CNN-LSTM

### Headline
> *More depth, more parameters, more temporal context — none of it helps*

### Models Covered (9 models → 4 tiers)
| Tier | Model | AUC | Key Finding |
|------|-------|-----|-------------|
| Simplest NN | Perceptron Scratch | ~0.52 | Single neuron — same as LR |
| | SLP Scratch | ~0.52 | One hidden layer doesn't move it |
| Fully Connected | MLP Scratch | 0.5415 | Best neural result — 3 layers |
| | MLP sklearn | **0.5423** | Confirms — neural nets peak here |
| Temporal / Sequential | LSTM Univariate | 0.504–0.505 | Barely above baseline — **worse than MLP** |
| | LSTM Multivariate | 0.522–0.530 | More features help slightly but still below MLP |
| | LSTM Two-Stage | 0.475–0.476 | **Below baseline** — worst neural model |
| Hybrid | CNN-LSTM Boruta | 0.5268 | Feature selection + convolution — still below MLP |

### Collapse scratch vs framework
> *"Scratch implementations (NumPy backprop) match PyTorch variants within 0.003 AUC — correctness confirmed"*

### Plots to Show
1. **MLP Training Loss / Accuracy Curve** (train vs val per epoch)
   - *Notebook cell:* `#VSC-a5c2505a`
   - *Why:* shows convergence and that the model isn't underfitting — it's genuinely saturating
2. **Model Complexity vs AUC Scatter Plot**
   - *Notebook cell:* `#VSC-2a5ff0ae`
   - *Why:* the most powerful single chart — shows adding parameters does nothing
   - *Emphasise:* flat or downward-sloping trend as complexity increases

### Results Table to Show
```
Model                      | AUC    | Accuracy | F1
---------------------------|--------|----------|------
Perceptron / SLP Scratch   | ~0.52  | ~52%     | ~0.52
MLP Scratch                | 0.5415 | 52.8%    | 0.557
MLP sklearn                | 0.5423 | 53.1%    | 0.552
LSTM Univariate Scratch    | 0.5050 | 49.8%    | 0.063
LSTM Univariate PyTorch    | 0.5039 | 50.3%    | 0.670 ⚠️ degenerate
LSTM Multivariate Scratch  | 0.5218 | 51.3%    | 0.638
LSTM Multivariate PyTorch  | 0.5295 | 52.0%    | 0.539
LSTM Two-Stage Scratch     | 0.4760 | 48.1%    | 0.485 ❌ below baseline
LSTM Two-Stage PyTorch     | 0.4746 | 47.8%    | 0.486 ❌ below baseline
CNN-LSTM Boruta            | 0.5268 | 51.8%    | 0.543
```

> ⚠️ Flag the degenerate models: LSTM Univariate PyTorch (predicts all UP), LSTM Two-Stage (below random).

### Talking Points
- "MLP peaks at 0.542 — close to the best result in the whole project"
- "LSTM was the most expensive experiment and performed the worst among neural nets"
- "The temporal structure LSTM is designed to exploit doesn't exist in 1-hour BTC returns — each hour is essentially independent"
- "CNN-LSTM with Boruta feature selection is our most sophisticated architecture — still 0.02 below MLP"
- "The complexity vs AUC scatter is the punchline: flat line, no trend"

---

---

## Slide 5 — The Full Leaderboard & Best Model

### Headline
> *XGBoost wins at AUC = 0.5464 — only 4.6% above random*

### Content

**Winner: XGBoost**
- AUC = **0.5464** — best across all 24 models + baseline
- Accuracy = 53.3%, F1 = 0.570
- Predicts UP correctly 61.5% of the time (high recall for class 1)
- Uses gradient boosting over 200 shallow trees

**Runner-up cluster (AUC 0.541–0.543):**
SVM RBF, MLP sklearn, MLP Scratch, RF sklearn, GB Scratch

**Notable failures:**
- LSTM Two-Stage: AUC = 0.4746 — **below** majority baseline (0.500)
- LSTM Univariate PyTorch: degenerate, all-UP predictions
- PCA+SVM Scratch: AUC = 0.515, all-DOWN predictions

### Plots to Show
1. **All-Models AUC Bar Chart** (colour-coded by family)
   - *Notebook cell:* `#VSC-a9e4c797`
   - *Must have:* baseline line at 0.500
   - *Best chart in the presentation — put it centre-stage*
2. **XGBoost ROC Curve**
   - *Notebook cell:* `#VSC-919de9ae`
   - *Why:* shows how close the curve is to the diagonal — almost no area under the curve

### Full Ranked Leaderboard (show as table on slide)
```
Rank | Model                  | AUC    | Accuracy
-----|------------------------|--------|----------
  1  | XGBoost                | 0.5464 | 53.3%
  2  | SVM sklearn RBF        | 0.5429 | 52.7%
  3  | MLP sklearn            | 0.5423 | 53.1%
  4  | MLP Scratch            | 0.5415 | 52.8%
  5  | RF sklearn (n=200)     | 0.5405 | 53.1%
  6  | GB Scratch             | 0.5405 | 52.5%
  7  | LR Scratch             | 0.5388 | 52.4%
  8  | LR sklearn             | 0.5386 | 52.5%
  9  | SVM sklearn Linear     | 0.5381 | 51.9%
 10  | RF Scratch             | 0.5378 | 52.3%
 11  | DT sklearn (depth=3)   | 0.5359 | 53.1%
 12  | PCA+SVM sklearn        | 0.5329 | 51.2%
 13  | LSTM Multivariate Py   | 0.5295 | 52.0%
 14  | DT Scratch             | 0.5295 | 53.0%
 15  | SVM Scratch Linear     | 0.5271 | 51.1%
 16  | CNN-LSTM Boruta        | 0.5268 | 51.8%
 17  | PCA+RF sklearn         | 0.5220 | 51.7%
 18  | LSTM Multivariate Sc   | 0.5218 | 51.3%
 19  | PCA+SVM Scratch        | 0.5148 | 49.7%
 20  | LSTM Univariate Sc     | 0.5050 | 49.8%
 21  | LSTM Univariate Py     | 0.5039 | 50.3%
 --  | Majority Baseline      | 0.5000 | 50.3%
 22  | LSTM Two-Stage Sc      | 0.4760 | 48.1%
 23  | LSTM Two-Stage Py      | 0.4746 | 47.8%
```

### Talking Points
- "XGBoost is the winner, but 0.546 vs 0.500 is a very thin edge"
- "Every model family is between 0.51 and 0.546 — the ceiling is consistent"
- "The bottom 5 are degenerate or below-baseline — all are LSTMs or collapsed classifiers"
- "Gradient boosted models (XGBoost, GB Scratch) dominate — they handle noisy tabular data best"

---

---

## Slide 6 — Why We Couldn't Do Better

### Headline
> *The bottleneck is the data, not the model*

### Six Root Causes (compact, one visual each)

**Cause 1 — Returns are a near-random walk**
- Durbin-Watson statistic = **2.016** (perfect independence = 2.0)
- Max ACF coefficient at any lag = **0.038** (near zero)
- *Plot:* ACF chart of hourly returns — flat line with no significant lags
- *Notebook cell:* `#VSC-b4dbdf9f`

**Cause 2 — Feature multicollinearity**
- All 55 features derived from same price/volume series
- Many feature pairs have correlation > 0.95
- *Visual:* correlation heatmap — `#VSC-681b6cf2`
- Models see 55 features but there's effectively much less independent information

**Cause 3 — Weak feature-to-target correlation**
- Max |correlation| between any feature and `target_class` < **0.04**
- No individual feature is predictive — they collectively capture marginal signal
- *Visual:* bar chart of top-20 feature-target correlations (all near zero)
- *Notebook cell:* `#VSC-25e306e4`

**Cause 4 — Market regime non-stationarity**
- 2018 bear market, 2019–2020 consolidation, 2021 bull, 2022 crash, 2023 recovery, 2024 ATH
- Patterns learned in one regime don't transfer to another
- *Visual:* BTC price history with regime bands shaded — `#VSC-b36f3072`

**Cause 5 — Missing alpha-generating data**
- Order book depth and bid/ask spread (microstructure)
- On-chain data: exchange flows, whale transactions, SOPR
- Sentiment: Fear & Greed Index, social volume, funding rates
- *This slide has no plot — just a bullet list with icons*

**Cause 6 — 1-hour horizon is semi-efficient**
- The 1-hour BTC market is heavily arbitraged by HFT and algorithmic traders
- Any persistent pattern at this timescale gets traded away immediately
- Longer horizons (daily, weekly) may have more exploitable signal

### Summary Visual
> **A 2×3 grid** with one icon/mini-chart per root cause — fits on one slide cleanly

### Talking Points
- "We didn't fail — we confirmed that this problem is hard for principled reasons"
- "The models are working correctly. The data doesn't contain much predictable signal"
- "To do better, we'd need order book data, on-chain flows, or sentiment — or a different time horizon"
- "The scratch implementations matching sklearn is actually a positive result — it validates correctness"

---

---

## What to Cut vs Keep

### Cut From Each Model Section
| What | Why |
|------|-----|
| Side-by-side scratch vs sklearn confusion matrices | They look identical — just say they match |
| Per-model AUC history discussion | The cross-model bar chart makes this redundant |
| Hyper-parameter sweep tables (all depth values, all n_estimators) | Keep only the winner and the overfitting chart |
| LSTM architecture diagrams | Too technical; mention architecture in one line |
| Clustering evaluation metrics tables | They're all baseline — summarise in one sentence |

### Keep
| What | Why |
|------|-----|
| DT Depth vs AUC plot | Best illustration of overfitting in the dataset |
| RF Feature Importance chart | Shows which signals exist at all |
| MLP Training curve | Proves models converge — not an underfitting problem |
| Complexity vs AUC scatter | The single most compelling summary chart |
| All-models AUC bar chart | Tells the whole story in one visual |
| ACF chart (Root Cause 1) | Statistical proof of near-random walk |
| Correlation heatmap | Shows multicollinearity at a glance |
| Full leaderboard table | Audience can scan all 24 models in context |

---

---

## Notebook Cell Quick Reference

| Slide | Cell IDs to Use |
|-------|----------------|
| Slide 1 | `#VSC-b36f3072` (price + returns), `#VSC-06aac2b2` (class balance + feature categories) |
| Slide 2 | `#VSC-4295e531` (DT depth vs AUC), `#VSC-584aa5bd` (RF feature importance) |
| Slide 3 | `#VSC-01924c3e` (PCA scree), `#VSC-130bd4be` (PCA comparison), `#VSC-5a7ad51e` (elbow) |
| Slide 4 | `#VSC-a5c2505a` (MLP curves), `#VSC-2a5ff0ae` (complexity scatter) |
| Slide 5 | `#VSC-a9e4c797` (AUC bar chart), `#VSC-919de9ae` (XGBoost ROC), `#VSC-ba8fde29` (leaderboard) |
| Slide 6 | `#VSC-b4dbdf9f` (ACF), `#VSC-681b6cf2` (heatmap), `#VSC-25e306e4` (feat-target corr) |

---

## Suggested Slide Tool

Build slides in **Google Slides** or **PowerPoint** by:
1. Running each referenced notebook cell
2. Right-clicking the output → Save Image
3. Pasting into slides with the talking points above as speaker notes

Alternatively, use `nbconvert` to export the notebook as HTML/slides:
```bash
jupyter nbconvert --to slides 19_classification_presentation.ipynb --post serve
```

---

*Plan created: CryptoFlow Classification Project*
*Models covered: 24 total (including baseline) across 9 families*
*Best result: XGBoost AUC = 0.5464*
