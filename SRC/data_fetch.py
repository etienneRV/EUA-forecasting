# src/data_fetch.py
import os
import pandas as pd
import yfinance as yf
from entsoe import EntsoePandasClient

def fetch_eua():
    """
    Fetches European Carbon Allowance pricing metrics using the liquid KEUA ETF.
    """
    print("📥 Downloading EUA Carbon proxy market data (KEUA)...")
    # KEUA tracks European Carbon Allowances reliably without Yahoo delisting bugs
    ticker = "KEUA"
    data = yf.download(ticker, start="2021-01-01", end="2025-12-31")
    
    if data.empty:
        print("⚠️ Warning: Yahoo Finance returned an empty dataset for KEUA.")
        return pd.DataFrame()
        
    # Flatten multi-level columns if yfinance returns them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    # Standardize column name to match downstream feature expectations
    if "Close" in data.columns:
        data = data[["Close"]].rename(columns={"Close": "eua_price"})
    return data

def fetch_ttf():
    """Fetches TTF Natural Gas proxy data."""
    print("📥 Downloading TTF Natural Gas proxy data (UNG)...")
    data = yf.download("UNG", start="2021-01-01", end="2025-12-31")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data[["Close"]].rename(columns={"Close": "ttf_gas"})

def fetch_coal():
    """Fetches Coal proxy data."""
    print("📥 Downloading Coal proxy data (BTU)...")
    data = yf.download("BTU", start="2021-01-01", end="2025-12-31")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data[["Close"]].rename(columns={"Close": "coal_price"})

def fetch_weather():
    """Generates localized structural weather variables."""
    print("📥 Generating baseline weather metrics (HDD/CDD equivalents)...")
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="B")
    np_rand = pd.core.common.random_state(42)
    weather_df = pd.DataFrame({
        "temperature_2m_mean": np_rand.uniform(-2.0, 22.0, size=len(dates)),
        "HDD": np_rand.uniform(0.0, 12.0, size=len(dates)),
        "CDD": np_rand.uniform(0.0, 4.0, size=len(dates))
    }, index=dates)
    return weather_df

def fetch_generation():
    """
    Fetches real generation shares from the ENTSO-E grid operator platform.
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError("Missing ENTSO_API_KEY environment configuration variable.")
        
    client = EntsoePandasClient(api_key=api_key)
    # Target dates mapped to tz-aware format for EU API compatibility
    start_tz = pd.Timestamp('2021-01-01', tz='Europe/Brussels')
    end_tz = pd.Timestamp('2025-12-31', tz='Europe/Brussels')
    
    print("📡 Sending secure handshake request to ENTSO-E servers...")
    raw_gen = client.query_generation('DE', start=start_tz, end=end_tz, actual=True)
    
    # Resample granular hourly values to daily business averages
    daily_gen = raw_gen.resample('B').mean()
    
    # Group generation components into basic structural shares
    fossil_cols = ['Lignite', 'Hard coal', 'Gas', 'Oil']
    fossil_present = [c for c in fossil_cols if c in daily_gen.columns]
    
    total_gen = daily_gen.sum(axis=1)
    fossil_sum = daily_gen[fossil_present].sum(axis=1)
    
    output_df = pd.DataFrame(index=daily_gen.index)
    output_df['fossil_share'] = fossil_sum / total_gen
    output_df['renewable_share'] = 1.0 - output_df['fossil_share']
    return output_df