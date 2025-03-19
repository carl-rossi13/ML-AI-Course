### Jupyter Notebook Link - https://github.com/carl-rossi13/ML-AI-Course/blob/main/Capstone-Final/capstone%20(2).ipynb 

## Problem Statement
The objective of this research is to determine the effectiveness of machine learning models in predicting stock price movements to generate profitable trading strategies. Specifically, this study develops and evaluates Random Forest, XGBoost, and LGBM models for Apple Inc. (AAPL) to assess their predictive capabilities. Model performance is measured using conventional error metrics (MAE/MSE) and a secondary performance metric, cumulative return, which is benchmarked against a traditional buy-and-hold strategy. This research aims to identify whether machine learning-driven trading models can outperform passive investment strategies in terms of profitability.

## Model Outcomes
Each model is designed to predict the next day's high and low for the target asset, providing critical and actionable inputs for calculating risk-reward ratios for potential long (buy) and short (sell) trades. These forecasts enable informed decision-making, improving trade entry strategies and overall risk management.

## Data Acquisition
AAPL was selected as the target asset due to its extensive historical data spanning over 40 years and its market significance in terms of capitalization, liquidity, and ETF weighting. Data from 2008 to the present was utilized to account for evolving market conditions. AAPL's strong correlations with key indices—S&P 500 (SPY), NASDAQ (QQQ), and the U.S. dollar currency index (DX-Y.NYB)—further support its suitability for analysis. All data was sourced from Yahoo Finance (yfinance python library), with the dataset containing 4,316 rows from January 1, 2008, to March 1, 2025, prior to processing.

## Data Preprocessing and Target Definition
Each of our 4 assets ('AAPL', 'SPY', 'QQQ', 'DX-Y.NYB) contains 5 columns, for a total of 20 columns in the dataset. We begin by labeling the columns for easier use, drop columns that will not be used, then inspect and handle rows with nan, inf, or zero values in the resulting df. 
To achieve our objective of creating a profitable strategy, this, we first define our target variables as the next period's high and low prices:

df['target_high'] = df['AAPL_High'].shift(-1)
df['target_low'] = df['AAPL_Low'].shift(-1)

Next, we express these targets as potential returns from the current closing price:

df['target_long_returns'] = df['target_high'] - df['AAPL_Close']
df['target_short_returns'] = df['target_low'] - df['AAPL_Close']

By normalizing the target relative to the current price, we eliminate price scale dependencies and focus solely on the magnitude of potential returns, allowing for a clearer comparison across different stock prices and market conditions. This ensures that our strategy is optimized for maximizing returns rather than being biased by absolute price levels.
Furthermore, by centering the targets around zero as seen in Plot 1.4, we hope to improve model performance for the Random Forest, LightGBM, and XGBoost models by stabilizing gradient updates, reducing numerical instability, and ensuring more balanced error distribution. This transformation may allow our tree-based models to make more efficient splits, leading to better generalization and predictive accuracy, especially when dealing with skewed or heavy-tailed distributions like the original targets.

### Plot 1.4 Distribution of Target High and Low vs Long Returns and Short Returns
This plot depicts the original target variables (on the left) and the normalized target variables (on the right).

## Feature Engineering
Feature engineering incorporates both technical and macroeconomic market analysis to enhance predictive capability. Technical indicators are designed to capture market psychology, based on the premise that price movements reflect collective human behavior. For example, a volume-anchored moving average is employed to assess trend strength—where increasing volume supports trend formation, while declining volume during a trend may signal continuation or consolidation. This approach helps identify potential trend shifts and stability, improving the model’s ability to recognize market dynamics.
[Plot 2.1]

## Modeling
Three composite machine learning models—Random Forest, XGBoost, and LGBM—were developed and optimized using random search for hyperparameter tuning. Each composite model consists of two separate models: one predicting the next day's high and another predicting the low.  The primary evaluation metric for all models is Mean Squared Error (MSE), as minimizing large errors is critical in financial contexts where significant deviations can impact portfolio size. However, Mean Average Error (MAE) is also tracked and analyzed, as well as the secondary metric ‘cumulative return’
To ensure robustness in a dynamic market environment, walk-forward optimization (WFO) was implemented using TimeSeriesSplit, allowing the models to adapt to evolving patterns while preventing data leakage. The most recent 1% of data (approximately 40 trading days) was reserved for live trading simulation and evaluation, ensuring an unbiased assessment of real-world performance. Due to computational limitations, random search was used instead of grid search, as it efficiently explores the hyperparameter space while significantly reducing the time and resources required for optimization.

## Model Evaluation
Plots 3.2, 4.2, and 5.2 illustrate each model’s Mean Squared Error (MSE) at each Walk-Forward Optimization (WFO) step, with the average performance represented by a horizontal reference line. Notably, all models exhibit a sharp increase in MSE at time step 6, regardless of hyperparameters or step size. Given the time-series nature of the target variable, the inherent volatility of financial markets, and the abrupt nature of the spike, this suggests the presence of concept drift—a phenomenon where the statistical properties of the target variable change over time, leading to degraded model performance.
Concept drift is particularly common in financial markets due to shifts in macroeconomic conditions, evolving investor sentiment, regime changes (such as bull or bear markets), and unexpected external shocks (e.g., policy changes, earnings reports, or geopolitical events). Further supporting this hypothesis, Plots 3.3, 3.4, 4.3, 4.4, 5.3, and 5.4 display the residuals for each model, showing a noticeable ballooning of residuals in later WFO steps and within the live test reserve. This pattern indicates that as the market environment changes, the models struggle to generalize, further confirming that concept drift is likely impacting predictive accuracy.

## Results & Findings

### Residuals vs. Target Returns Summary

LightGBM had the lowest residuals overall (Y1: 0.586, Y2: 0.610), followed by Random Forest (Y1: 0.581, Y2: 0.658) and XGBoost (Y1: 0.595, Y2: 0.628).
Given the median target long return (0.204) and short return (-0.164), all models' residuals exceed the central tendency of target returns, indicating significant forecast error.
Live Testing Performance (40 Periods, ~1% of Data)

Residuals increased sharply in live testing, suggesting concept drift or poor adaptability to new market conditions.
Random Forest had the lowest live residual for long trades (Y1: 2.528), while XGBoost performed best for short trades (Y2: 2.454).
All models struggled, with live residuals exceeding the standard deviation of target returns, highlighting limitations in generalization.

Key Takeaway
Each model forecasts one period ahead, not the full 40-period live test. Ideally, models should be retrained daily to better adapt to market shifts.

### Model Performance Summary
Since models were optimized based on MSE, LightGBM achieved the best overall performance with the lowest MSE for both Y1 (1.469) and Y2 (1.558), followed closely by XGBoost. Random Forest had the highest MSE, indicating slightly weaker predictive accuracy.
For MAE, LightGBM also had the lowest error for Y2 (0.610), while Random Forest performed best for Y1 (0.581). However, differences in MAE across models are minimal, suggesting similar predictive capabilities.

Key Takeaway
LightGBM emerged as the best model in terms of MSE, aligning with the optimization goal, while Random Forest had the lowest absolute error for long trades.

### Trading Strategy Performance Summary
While all models underperformed relative to the target return distribution, they still outperformed the benchmark AAPL buy-and-hold strategy (-2.74%), demonstrating the effectiveness of an active trading approach. XGBoost achieved the highest cumulative return (16.28%), followed by Random Forest (12.81%) and LightGBM (12.07%).
Despite their suboptimal predictive accuracy, all models successfully exploited short-term price movements better than passive investing, reinforcing the potential value of machine learning-driven trading strategies in volatile markets.

Key Takeaway
XGBoost is the preferred model because it delivered the highest cumulative return (16.28%), significantly outperforming both LightGBM (12.07%) and Random Forest (12.81%). While LightGBM had the lowest MSE and residuals, XGBoost’s error metrics were comparable across both all WFO steps and live testing, yet it translated those forecasts into substantially higher profitability. This suggests that while LightGBM minimized prediction errors, XGBoost was more effective in capitalizing on short-term price movements, making it the superior choice for practical trading applications.

## Going Forward
To improve model adaptability and mitigate the effects of concept drift, future iterations should incorporate adaptive mechanisms that adjust to evolving market conditions. One potential enhancement is a dynamic WFO step size, where the retraining window expands or contracts based on drift detection metrics such as sudden spikes in residuals or MSE. By dynamically adjusting the training schedule, the models can better adapt to regime changes, ensuring more stable and reliable predictions over time.

