# EUA Carbon Price Forecasting

Causal forecasting of EU ETS Phase IV carbon allowance futures prices using
macro energy variables. Combines econometric validation (ADF, Granger causality,
VAR) with machine learning (XGBoost) on 2021–2025 daily data.

## Research Question
Can observable macro variables — gas-coal switching spreads, renewable generation
share, weather — provide statistically significant predictive information for
EUA price returns beyond the random walk? And does a causal ML model outperform
an ARIMAX benchmark?

## Variables
| Variable | Source | Causal Hypothesis |
|---|---|---|
| TTF–Coal Spread | Yahoo Finance | Switching signal: high gas → more coal → more EUAs needed |
| Renewable Share | ENTSO-E | High renewables → less fossil burn → EUA demand falls |
| HDD/CDD | Open-Meteo | Temperature → energy demand → ETS compliance pressure |
| Policy Dummies | Manual | Regulatory shocks produce discontinuous price jumps |

## Methodology & Leakage Protection
To ensure strict real-world forecasting validity, this pipeline enforces a rigorous **Temporal Isolation Rule**. All contemporaneous target-derived features (such as `log_eua` or current-day returns) are explicitly dropped during feature selection. Input signals are lag-shifted ($t-1$, $t-5$, $t-10$) to prevent *Look-Ahead Bias*, forcing the model to predict true forward horizons rather than merely copying current-day market states.

## Results

### Feature Importance Analysis
The chart below shows the average gain contributions of our engineered features after resolving temporal data leakage. Autoregressive anchors dominate structural baselines, while fundamental weather load and fuel costs dictate the residual splits.

![XGBoost Feature Importance](figures/feature_importance.png)

### Out-of-Sample Performance
Our model's backtested predictions are plotted against actual historical prices across the holdout validation set.

![Out-of-Sample Forecast Evaluation](figures/oos_forecast.png)

## Key Findings

- **Autoregressive Anchoring:** The previous day's closing price (`eua_lag_1`) is the single most dominant feature by gain contribution (~196.39), acting as the core baseline anchor for daily price setting. 
- **Fundamental Load Signals:** Temperature extremes—specifically cooling load pressures (`CDD`) and heating load proxies (`HDD`)—rank as the highest-impact external macroeconomic features, outpacing raw grid generation shares.
- **Fuel-Switching & Economics:** Global commodity pricing fundamentals (`coal_price` and the `gas_coal_spread`) show persistent, statistically significant importance when navigating mid-term structural trend shifts.
- **Market Efficiency Constraints:** While the un-leaked XGBoost framework successfully maps macro trends and structural shifts, directional accuracy hovers near 55%. This marginal outperformance over a naive random walk is heavily consistent with weak-form market efficiency in liquid carbon markets.
- **Policy Shocks:** Quantifiable regulatory changes and policy interventions (`policy_shock`) provide key discontinuous directional adjustments that pure market pricing indicators fail to capture on their own.