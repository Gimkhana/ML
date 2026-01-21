**Machine Learning Failures in Time Series Analysis**

Why Random Forest beats LSTM by 66–83% on interest rate and FX forecasting.

I tested five machine learning models (Random Forest, Gradient Boosting, SVM, MLP, LSTM) predicting USD 3M deposit rates and the USD FX index across a 10-year dataset (2014–2024). 
The core insight: I trained all models on calm markets (2014–2020, rates near zero) and tested them on the Fed's aggressive tightening cycle (2020–2024, rates 0% → 5.5%). This is realistic—models face regime shifts they've never seen. Random Forest crushed every neural network. LSTM, theoretically best for sequence prediction, was 5.8x worse than Random Forest on rates (MAE 7.0 vs. 1.2) -> This matters because most quant teams and trading desks default to LSTM for financial forecasting. They shouldn't.

Why it happened: Neural networks are function approximators that extrapolate confidently beyond their training range. 
- When the Fed raised rates from 0.5% to 4.5%, LSTM had never seen rates above 2% so it made up an answer (predicting negative rates, which is impossible). 
- Random Forest, on the other hand, is non-parametric: it averages historical decision trees and simply says "no opinion" when it encounters a regime it wasn't trained on -> Being conservative beats being wrong. 
- The second killer was architecture: my LSTM had zero macro indicators (Fed Funds Rate, VIX, credit spreads). The Fed's tightening was exogenous to the model—learned patterns became worthless overnight.

Mechanically, three things favored Random Forest: 
- First, it makes no distributional assumptions about data. It learns locally, not globally, so extreme regimes don't break it. 
- Second, it automatically adapts feature importance across different volatility regimes different decision trees use different subsets of the 20 features depending on market conditions. Neural networks lock in fixed weights, which is fine until the market regime changes. 
- Third, it refuses to extrapolate. When rates hit 4.5% and no tree was trained at that level, Random Forest conservatively stays flat rather than confidently guessing.

Possible modifications discussed: 
- The cleanest fix is feature engineering. Adding Fed Funds Rate, 2-Year Yield, VIX, and credit spreads would likely bring LSTM performance in line with Random Forest. - Build an adaptive ensemble, use LSTM for normal periods (it captures trends better) and Random Forest during crises (it's robust). But the honest takeaway is that for financial time series with exogenous shocks, you don't need deep learning -> Simple tree based methods work better. I have published the full code, presentation (latex presentation compiled in PDF), hyperparameters, and reproducible pipeline on GitHub. The data is from Refinitiv; results are clean.

Data source: Refinitiv Eikon (2014–2024)
Status: Research-ready | Full reproducibility | November 2025
