# S&P 500 Index Forecasting

**Machine learning applied to stock market price prediction. Comparing 5 models (MLP, LSTM, CNN, SVM, Random Forest) on S&P 500 data.**

---

We try to address the following question: Can we forecast S&P 500 price movements using machine learning? If so, which model works best?

**Short answer:** LSTM edges out MLP slightly, but CNN vastly outperforms feedforward networks on individual equity predictions.

---

## Key results

### S&P 500 index forecasting (MLP vs LSTM)

| Model | MAE | MSE | RMSE |
|-------|-----|-----|------|
| **LSTM** | 7.88 | 78.41 | 8.85 |
| **MLP** | 8.71 | 79.89 | 8.94 |

**Finding:** LSTM performs marginally better, but differences are small (~10% improvement in MAE).

### Individual equity forecasting (MLP vs CNN)

| Model | MAE | MSE | RMSE |
|-------|-----|-----|------|
| **CNN** | 16.42 | 320.54 | 17.90 |
| **MLP** | 599.99 | 360,050.77 | 600.04 |

**Finding:** CNN dominates MLP by a massive margin (97% error reduction). MLP struggles with equity price volatility.

### Benchmark of tested models

| Model | MAE | MSE | RMSE |
|-------|-----|-----|------|
| **LSTM** | 7.88 | 78.41 | 8.85 |
| **MLP** | 8.71 | 79.89 | 8.94 |
| **SVM** | 18.94 | 437.27 | 20.91 |
| **Gradient Boosting** | 28.90 | 1000.96 | 31.64 |
| **Random Forest** | 26.77 | 1017.43 | 31.90 |

**Finding:** Neural networks (LSTM, MLP) outperform tree-based models on S&P 500 index-level data.

---

## Data

- **Source:** Yahoo Finance / S&P 500 historical data
- **Observations:** 41,265 daily price points
- **Index performance:** 2360 → 2480 (+5.08% over period)
- **Price profile:** Uptrend with narrow drawdowns (0–3%)
- **Return distribution:** Leptokurtic (extreme events present, but centered at zero)

### Train-test split

- **Training window:** 250 observations (for initial models)
- **Full dataset:** 41,265 observations split into training and testing
- **Normalization:** Z-score with ±3σ clipping

---

## Models tested

### 1. **Multilayer Perceptron (MLP)**

**Architecture:**
- 3 fully-connected layers (dense)
- ReLU activation (introduces non-linearity)
- Dropout: 25% (prevents overfitting)
- Output: 1 neuron (single price prediction)
- Training: 10 epochs, batch size 16

**Design philosophy:** Capture non-linear patterns in price data through stacked layers.

### 2. **LSTM (Long Short-Term Memory)**

**Architecture:**
- Layer 1: 128 neurons, returns sequences (for stacking)
- Layer 2: 64 neurons, does not return sequences
- Dropout: 25%
- Dense layers: 25 neurons (ReLU) → 1 neuron (output)
- Training: 10 epochs, batch size 16

**Design philosophy:** Capture long-range temporal dependencies (memory) in price sequences.

### 3. **CNN (Convolutional Neural Network)**

**Architecture:**
- Conv1D layers: 2 layers with 64 filters each, kernel size 3
- Max pooling: 1D max pooling with pool size 2
- Dropout: 25%
- Flatten layer + dense output
- Training: 10 epochs, batch size 16

**Design philosophy:** Capture local patterns in time-series sequences via convolution.

### 4. **SVM (Support Vector Regression)**

- Kernel: RBF (radial basis function)
- C parameter: 1.0 (complexity-generalization trade-off)

### 5. **Gradient Boosting**

- 50 sequential trees
- Learning rate: 0.1
- Max depth: 3

### 6. **Random Forest**

- 50 decision trees
- Max depth: variable
- Reduces overfitting through ensemble averaging

---

## Methodology

### Data preparation

1. **Loading:** Import 41,265 daily S&P 500 observations
2. **Cleaning:** Remove missing values, handle outliers
3. **Feature engineering:** Create lagged price features
4. **Normalization:** Z-score scaling for neural networks

### Evaluation metrics

**MAE (Mean Absolute Error):** Average absolute prediction error (lower is better)

**MSE (Mean Squared Error):** Penalizes larger errors more heavily

**RMSE (Root Mean Squared Error):** Square root of MSE; in original price units

---

## Key insights

### 1. **LSTM slightly outperforms MLP on index data**

- LSTM MAE: 7.88 vs. MLP MAE: 8.71 (~10% improvement)
- Both models capture overall price trends visually
- Differences marginal, suggesting index-level trends are relatively smooth

**Reason behind outperformance:** Ability to capture temporal dependencies in sequences.

### 2. **CNN strong outperformance over MLP on individual stocks**

- CNN MAE: 16.42 vs. MLP MAE: 599.99 (97% error reduction)
- MLP completely fails on equity volatility
- CNN's convolutional filters capture local price patterns

**Reason behind outperformance:** Convolutional filters naturally detect local temporal patterns; MLP treats all features equally.

### 3. **Neural Networks outperforms Tree based methods on index**

- LSTM/MLP outperform Random Forest, Gradient Boosting, SVM
- Tree-based methods show MAE > 26 (vs. LSTM 7.88)

**Why:** Index price follows relatively smooth trends; tree-based methods overfit to noise.

### 4. **Increasing epochs doesn't always help**

MLP trained for 30 epochs (vs. 10) showed *worse* performance:
- 10 epochs MAE: 599.99
- 30 epochs MAE: 624.05 (overfitting)

**Lesson:** More training = overfitting on equity data. Early stopping matters.

---

## Limitations

1. **Single market snapshot:** Data covers one period only; different market conditions may yield different results
2. **No macro features:** Models don't include interest rates, volatility index, or Fed policy
3. **Uptrend bias:** Index was in uptrend during period; downtrends may show different patterns
4. **Equity-level struggles:** MLP completely failed on individual equities (CNN needed for that)
5. **Prediction horizon:** Forecasting 1-step ahead; multi-step forecasting untested

---

## Running the Code

### Prerequisites

```bash
pip install -r requirements.txt
```

---

## Key Takeaways

**LSTM edges MLP on index forecasting** (capture temporal patterns)  
**CNN dominates on equity volatility** (convolutional filters > dense layers)  
**Neural networks beat tree-based methods on smooth data** (index level)  
**More epochs ≠ better performance** (overfitting risk on equity data)  
**Single model choice matters significantly** (30x error difference between best/worst)

---

## Future Work

- Add macro features (Fed Funds, VIX, credit spreads)
- Test ensemble methods (combine LSTM + CNN)
- Multi-step forecasting (predict 5-10 days ahead)
- Regime detection (separate models for up/down markets)
- Apply to other indices (Russell 2000, DAX, Nikkei)

---

## References

- Course: Machine Learning (M2 Risk and Asset Management), Université Paris-Saclay
- Supervisor: Abdelkader Bousabaa
- Data: Yahoo Finance (S&P 500 historical data)

---

## Authors

**Youssef LOURAOUI** — youssef.louraoui@essec.edu  
**François PEDEBOY** — francois.pedeboy@etud.univ-evry.fr

---

**Status:** Academic project (M2 GRA coursework) | **Date:** March 2024 | **Reproducibility:** Full notebooks + PDF (data is heavy in size). Shared upon request. 