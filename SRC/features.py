# src/features.py
import pandas as pd
import numpy as np

def build_feature_matrix(eua, ttf, coal, gen, weather, policy_events):
    """
    Merge all sources on business-day index, engineer features.
    """
    df = eua.copy()
    df = df.join(ttf, how="left")
    df = df.join(coal, how="left")
    df = df.join(gen[["renewable_share", "fossil_share"]], how="left")
    df = df.join(weather[["HDD", "CDD", "temperature_2m_mean"]], how="left")

    # Forward-fill weekends/holidays in commodity prices
    df = df.ffill().dropna()

    # --- Derived features ---

    # Gas-coal spread (the key switching signal)
    # Gas price in EUR/MWh, coal price in USD/tonne → convert coal
    # Rough conversion: 1 tonne coal ≈ 6.978 MWh (thermal), HHV basis
    # Adjust by EUR/USD rate or use raw ratio as a proxy
    df["gas_coal_spread"] = df["ttf_gas"] - (df["coal_price"] / 6.978)

    # Lagged EUA price (autoregressive component)
    for lag in [1, 5, 10, 21]:  # 1d, 1w, 2w, 1m
        df[f"eua_lag_{lag}"] = df["eua_price"].shift(lag)

    # Lagged gas-coal spread
    df["spread_lag_1"] = df["gas_coal_spread"].shift(1)
    df["spread_lag_5"] = df["gas_coal_spread"].shift(5)

    # Rolling volatility of EUA price (21-day)
    df["eua_vol_21d"] = df["eua_price"].pct_change().rolling(21).std()

    # Log price (for stationarity transformation)
    df["log_eua"] = np.log(df["eua_price"])
    df["d_log_eua"] = df["log_eua"].diff()  # log returns = approximately % change

    # Month and season dummies (for MSR auction seasonality)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_q1"] = (df["quarter"] == 1).astype(int)  # compliance season

    # Policy event window dummies
    events = pd.read_csv("data/processed/policy_events.csv", parse_dates=["date"])
    df["policy_shock"] = 0
    for _, row in events.iterrows():
        window = pd.date_range(row["date"] - pd.Timedelta(days=5),
                               row["date"] + pd.Timedelta(days=5), freq="B")
        df.loc[df.index.isin(window), "policy_shock"] = 1

    df = df.dropna()
    return df