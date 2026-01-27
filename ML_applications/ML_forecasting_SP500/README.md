### **S&P 500 Index Forecasting** 

Can machine learning predict stock prices? Testing 5 models to find out.

We tested five machine learning models (LSTM, MLP, CNN, SVM, Random Forest) trying to forecast S&P 500 daily prices using 41,265 observations of historical data. 
- The quick takeaway: LSTM outperforms at predicting the overall index, but only barely. It's 10% better than a simple neural network (MLP).
- More importantly, when we tried the same models on individual stocks, CNN outperforms everything else. MLP fails catastrophically on single-stock volatility (97% error worse than CNN).
- Why does this matter? Because most investors testing "AI stock prediction" pick one model and deploy it everywhere.
- The reality is different: models work for different problems (index tend to have smooth trend data vs. individual stocks noisy, volatile data) require completely different approaches.

Findings: 

- LSTM's edge on the S&P 500 index is that it has "memory": it remembers price sequences and learns temporal patterns.
- MLP treats each data point independently, which works fine for a smooth uptrend but struggles with volatility.
- Tree-based models (Random Forest, Gradient Boosting) came in dead last on index forecasting because they overfit to noise in the trend.
- On individual equities, CNN's convolutional layer (which scan for local price patterns) outperformed everything.
- MLP generated absurd predictions when facing stock-level volatility.
- The lesson: Simple smoothness favors LSTM; complexity and sudden moves favor CNN. Interestingly, more training (30 epochs vs. 10) made MLP worse, suggesting overfitting is the real enemy on equity data.

The data we used:
- 41,265 daily S&P 500 price points spanning a period where the index rose from 2360 to 2480 (+5.08%), mostly in an uptrend with shallow drawdowns (0–3%). 
- We trained all models on historical data and tested on held-out observations. T
- The index behaved nicely: relatively smooth trends without major shocks (unlike 2008 or March 2020). This uptrend bias matters because our results might flip during market corrections. We normalized prices to be comparable across models and evaluated using MAE (average prediction error), which is intuitive.

Where we fell short: We didn't include market-wide variables like the Fed's interest rates, the VIX (volatility index), or credit spreads -> things that actually move stock prices. So our models learned historical patterns only, not why those patterns exist. We only tested one time period (uptrend), so we don't know if CNN's dominance on equities holds during crashes. We also only predicted one day ahead; nobody trades on 1-day-ahead predictions in real money. And honestly, we built this as a coursework project, not a trading system. The S&P 500 data we used isn't proprietary, but individual equity data is large. Notebooks and code are on GitHub, raw data available **on request**.

Bottom line: Yes, machine learning can forecast stock prices slightly better than random guessing. But not by much. LSTM beats MLP by 10% on the index, which is noise. The headline finding (CNN dominates MLP on individual stocks by 97%) is only because MLP was the wrong tool. When you test enough models, one will look brilliant. Deploy it to live trading, and market conditions shift. The only reliable pattern here is that smooth index trends favor memory-based models (LSTM), while noisy volatility requires pattern detection (CNN). Add real macro features and regime detection, and maybe you have something tradeable. This project should be treated as academic: interesting enough for a portfolio, but not investment advice.

Data: 41,265 daily S&P 500 observations (data on request)
Status: Academic coursework | March 2024
