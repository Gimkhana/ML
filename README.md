### Machine Learning (ML) Forecasting: An Application to Urals Loading Prediction 

The project aim to answer the following question: can we build a generalizable ML model that works across normal AND crisis periods, using only economically independent variables?

**Insights**:
- Wide gap between the models tested (0.521 → 1.371 mbbl/d deviations in MAE) MLP outperforms LSTM by 25%. 
- By removing multicollinear features, MLP now beats LSTM by 58%
- MLP (3 layers) outperforms LSTM despite simpler architecture, because it trains on independent, interpretable features.

**Dataset**: 
- 234 weekly observations (2017-2021)
- Target: Urals crude loading volumes (mb/d)
- Training: 164 observations (70%)
- Testing: 70 observations (30%)
- Normalization: Z-score (+/− 3 σ threshold)
- Source: We tested two dfferrent set of features in the ML model using Reﬁnitiv Eikon, Petrologistics datasets

**Technologies**: Jupyter Notebook (Python script) 
**Methods**: ML techniques

### Extension work -> ML forecasting with synthetic data: An application to Urals loading 

High ﬁdelity synthetic commodity data that preserves critical market characteristics and offers an alternative solution for data generation without breaching proprietary strategies. Case study using multivariate distribution (based on mean features and covariance to derive the synthetic dataset → projection for usage with copula modeling). 

**Insights**:
- 96% statistical similarity between original and synthetic data → crisis simulation, ML model
development, and Basel III compliance without exposing proprietary strategies.
- Multiple layer of statistical testing confirm a good fit of the generated data to the original dataset

**Technologies**: Jupyter Notebook (Python script) 
**Methods**: Machine Learning (ML), Multivariate distribution, Data generation
**Data**: Urals loading, Brent futures (LCOc1), MOEX index (Russian equity index), Brent crack spread (2017-2021, Refinitiv Eikon)
