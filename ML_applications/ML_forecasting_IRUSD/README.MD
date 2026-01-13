# Machine Learning Failures in Time Series Analysis

**Why simple models beat neural networks on USD rates and FX forecasting. A case study in regime-shift robustness.**

---

## The central paradox

**Conventional wisdom:** Deep learning = optimal for financial time series  
**Our finding:** Simple ensemble methods beat neural networks by 66–83%

| Model | Complexity | MAE (USD 3M rates) | Performance |
|-------|-----------|-------------------|-------------|
| Random Forest | High | 1.215 | **#1 (Best)** |
| Gradient Boosting | High | 1.517 | #2 |
| SVM | Medium | 3.193 | #3 |
| MLP | Medium-High | 3.443 | #4 |
| LSTM | Medium-High | 7.027 | #5 (Worst) |

**Key insight:** LSTM (most complex, theoretically best for sequences) is **5.8x worse** than Random Forest (simpler, non-parametric).

---

## The central question

**Can machine learning predict interest rates and currency levels?**

**Industry belief:**
- Neural networks capture non-linearity automatically
- Deep learning is state-of-the-art

**Our answer:**
- Simple models > neural nets in crisis periods
- Fed policy shifts destroy all predictions equally
- **Implication:** ML models fail catastrophically during regime shifts

---

## Data & experimental setup

### Dataset

- **Time period:** 2014–2024 (2,608 daily observations)
- **Targets:** 
  - USD 3M deposit rates (USD3MD=)
  - USD FX index (=USD)
- **Training data:** 1,825 observations (70%) — 2014–2020 (pre-pandemic + early outbreak)
- **Test data:** 783 observations (30%) — 2020–2024 (Fed tightening regime shift)
- **Normalization:** Z-score with ±3σ threshold

### Features

- **9 currency pairs:** GBP, EUR, JPY, CHF, AUD, NZD, CAD, SEK, NOK
- **3M deposit rates:** For each currency
- **Total features:** 20
- **Dimensionality:** First 5 variables explain 95% of variance (PCA)

### Key challenge

**Train on one regime, test on a completely different regime:**

- Training: 2014–2020 (low volatility, rates near zero)
- Testing: 2020–2024 (Fed tightening, rates 0% → 5.5%)

This is realistic: models must survive policy shocks.

---

## Models tested

### Neural Networks

**MLP (Multilayer Perceptron)**
- Architecture: 3-layer perceptron
- Dropout: 25%
- Epochs: 10
- Batch size: 16

**LSTM (Long Short-Term Memory)**
- Sequence-to-sequence architecture
- Lookback window: 30 observations
- Dropout: 25%
- Epochs: 10

### Tree-Based + SVM

**Random Forest**
- 50 trees
- Max depth: 20
- Min samples: 5

**Gradient Boosting**
- Sequential boosting
- Learning rate: 0.1
- Depth: 5

**SVM (Support Vector Regression)**
- Kernel: RBF
- Gamma: auto
- C: 1.0

---

## Results

### USD 3M rate forecasting

| Model | MAE | RMSE | Rank |
|-------|-----|------|------|
| **Random Forest** | 1.215 | 2.194 | **#1** |
| **Gradient Boosting** | 1.517 | 2.564 | #2 |
| **SVM** | 3.193 | 3.861 | #3 |
| **MLP** | 3.443 | 3.865 | #4 |
| **LSTM** | 7.027 | 8.634 | #5 |

**LSTM shock:** Predicting −700bp when actual is +400bp (on 10bp move scale)

### USD FX Index Forecasting

| Model | MAE | RMSE | Rank |
|-------|-----|------|------|
| **Random Forest** | 1.354 | 2.194 | **#1** |
| **Gradient Boosting** | 1.609 | 2.348 | #2 |
| **LSTM** | 2.507 | 3.049 | #3 |
| **MLP** | 2.505 | 3.049 | #4 |
| **SVM** | 19.838 | 19.838 | #5 |

**Pattern holds:** Tree-based methods dominate (though gap smaller than rates). SVM fails completely on FX.

---

## Why models failed

### Non stationarity

**Training regime (2014–2020):**
- Rates: 0.25% ± 3% (trending, bounded range)
- Models learned: "rates stay low"

**Test regime (2020–2024):**
- Rates: 0% → 5.5% (trending, non-stationary)
- Problem: Models encountered regime they never saw

**Result:** All neural networks extrapolate confidently into wrong regime.

### Exogenous policy shift

**Federal Reserve tightening (March 2022):**
- Event: 0% → 4.25% (50bp hikes × 9)
- Magnitude: Sharp, unprecedented in training set

**Problem:**
- Fed policy NOT in feature set
- Models trained on historical correlations only
- Fed tightening is exogenous = breaks all learned patterns

**Result:** No model captures the 4% sharp increase accurately.

### Insufficient crisis history

- Training data includes: 2018 taper tantrum, 2020 pandemic
- Missing: 2008 financial crisis, 1990s tightening cycles
- Models have never seen rate shock of this magnitude

---

## Why RF outperforms (3 advantages)

### Non-parametric (No distributional assumption)

**Neural Networks:**
- Assume data follows learned distribution
- Parameters capture training dynamics
- Regime shift → parameters become invalid
- Result: Extrapolate confidently into wrong regime

**Random Forest:**
- No distributional assumptions
- Simply averages similar historical trees
- New regime → use closest trees (conservative)
- Result: Don't extrapolate, stay flat

**Consequence:** When Fed tightens, RF says "no opinion" (stays flat). LSTM says "I learned the pattern" (confidently wrong).

### Local learning (No global extrapolation)

**LSTM extrapolation error:**

When rates jump from 0.5% → 4.5%:
- LSTM learned 0–0.5% pattern, extrapolates to 4.5% = massive error
- RF: No trees trained at 4.5%, limits prediction (conservative)

**The "curse of extrapolation":**
- Neural nets extrapolate confidently
- Trees conservatively refuse to extrapolate

### Automatic feature selection (Adaptive to regime)

**Neural Networks:**
- All 20 features same weight across entire dataset
- Learned weights fixed (cannot adapt to regime change)
- Fed policy absent from features = disaster

**Random Forest:**
- Each tree uses subset of features
- Different trees prioritize different features
- High-volatility regime uses different features
- Result: Automatic adaptation to regime

---

## The regime shift (visual evidence)

**Federal Reserve tightening, March 2022:**

| Observation | Actual Rates | RF Prediction | LSTM Prediction |
|-------------|--------------|---------------|-----------------|
| 100 (pre-shock) | 0.25% | 0.26% | 0.24% |
| 150 (transition) | 1.50% | 1.45% | 1.10% |
| **200 (peak shock)** | **4.25%** | **3.80%** | **−0.50%** |

**Finding:** RF stayed closer to reality; LSTM confidently predicted negative rates (impossible).

---

## Key insights for practitioners

### Insight 1: Use simple models for regime-switching markets

**Problem:** Financial time series have exogenous shocks (Fed policy, geopolitics, earnings surprises)

**Solution:**
- Use Random Forest + volatility switching rules
- Detect VIX > 30 → reduce weight on ML predictions
- Accept that models fail in true crises

### Insight 2: Add macro indicators or accept conservatism

**Why LSTM failed:**
- Zero Fed-related features in model
- No CRB Commodity Index, no VIX

**Two options:**
1. Add macro features (Fed Funds Rate, VIX, credit spreads) → LSTM improves dramatically
2. Stick with Random Forest → works without them (just conservative)

**Proposed features to add:**
- Federal Funds Rate (direct policy)
- 2-Year Yield (market expectations)
- VIX (risk-off trigger for regime shifts)
- Credit spreads (financial conditions)
- Unemployment rate (Fed's dual mandate)

### Insight 3: Ensemble Methods > Single Model

**Best approach:**
- Combine Random Forest + Gradient Boosting with calibrated weighting
- Use volatility regime to weight models dynamically
- Reduces single-model catastrophe risk

---

## Limitations

1. **Missing features:** No macro indicators (Fed Funds, VIX, CRB, credit spreads)
   - Adding these would improve LSTM significantly

2. **Limited crisis diversity:** Only 1 regime shift tested (Fed tightening 2022–2024)
   - Unknown: Does RF work better for other shocks (2008, geopolitics)?

3. **Hyperparameter tuning:** RF used defaults (50 trees)
   - Intensive tuning might improve all models (or introduce overfitting)

4. **Single horizon:** 1-step-ahead forecasting only
   - Multi-step (5–10 day) forecasting untested

---

## Future research directions

### Direction 1: Feature Engineering

**Hypothesis:** Add Fed Funds Rate + VIX + Credit Spreads → LSTM improves to RF-level

**Experiment:**
- Retrain LSTM with macroeconomic features
- Compare performance vs. baseline

**Expected result:** LSTM MAE drops from 7.0 to 2.5–3.0 range

### Direction 2: Adaptive models

**Idea:** Build model that detects regime shifts automatically

**Approach:**
- Train model on detecting high-volatility (VIX > 30)
- Switch between LSTM (captures trends) ↔ RF (ignores noise)
- Test on 2008, 2011, 2020 crises

### Direction 3: Ensemble learning

**Combine strengths of both:**
- LSTM captures trends in normal periods
- RF ignores noise in crisis periods
- Use volatility regime to weight dynamically

**Hypothesis:** Ensemble beats either model alone

---

## Practical recommendations

### For Traders using LSTM/MLP for rates

1. Add Fed policy detector (alert when CRB > 90th percentile)
2. Override predictions during crises
3. Reduce position sizes when model confidence is high but market unusual

### For Traders using Random Forest

1. Trust predictions in quiet periods (rate range-bound)
2. Remember RF won't capture Fed shocks (accept 50–100bp forecast error)
3. Combine with fundamental view (Fed-watching)

### For USD FX specifically

- RF performance best, but SVM collapsed
- **Lesson:** Not all ML models generalizable across assets
- **Recommendation:** Test each model on SPECIFIC asset before deploying

---

## Bottom line

### Four takeaways

1. **Random Forest outperforms LSTM by 66–83%** on financial time series with regime shifts
2. **Neural networks fail catastrophically** during exogenous policy changes (Fed tightening)
3. **Simple non-parametric methods** (RF, GBoost) more robust than deep learning to market shocks
4. **For rates/FX forecasting:** Use RF unless you have macro indicators to feed LSTM

### For the Industry

The ML hype cycle has oversold neural networks for finance:
- Simpler, less sexy methods often work better
- Add macro features or accept conservative predictions
- Don't deploy single-model strategies in production

---

## References

### Academic Papers

- Nelson, D. B. (1991). "Conditional Heteroskedasticity in Asset Returns." *Econometrica*, 59(2), 347–370.
- Fama, E. F., & French, K. R. (2015). "A five-factor asset pricing model." *Journal of Financial Economics*, 116(1), 1–22.

### Data Sources

- **G20 FX + Rates:** Refinitiv Eikon

---

## Authors

**Youssef LOURAOUI**  
Université Paris-Saclay, M2 Risk and Asset Management  
youssef.louraoui@essec.edu

---

## Citation

```bibtex
@repository{louraoui2025ml_failures,
  author = {Louraoui, Youssef},
  title = {Machine Learning Failures in Financial Time Series},
  year = {2025},
  url = {https://github.com/Gimkhana/ML-Failures-FinTS}
}
```

---

**Status:** Research-ready | **Date:** November 2025 | **Reproducibility:** Full pipeline + data included