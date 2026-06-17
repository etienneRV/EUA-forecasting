import os
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()

def fetch_eua(start="2021-01-01", end="2025-12-31"):
    eua = yf.download("^ICEEUA", start=start, end=end)
    eua = eua[["Close"]].rename(columns={"Close": "eua_price"})
    eua.index = pd.to_datetime(eua.index)
    return eua.dropna()

def fetch_ttf(start="2021-01-01", end="2025-12-31"):
    ttf = yf.download("TTF=F", start=start, end=end)
    ttf = ttf[["Close"]].rename(columns={"Close": "ttf_gas"})
    return ttf.dropna()

def fetch_coal(start="2021-01-01", end="2025-12-31"):
    coal = yf.download("MTF=F", start=start, end=end)
    coal = coal[["Close"]].rename(columns={"Close": "coal_price"})
    return coal.dropna()

def fetch_generation(start="20210101", end="20251231", country="DE"):
    client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
    start_ts = pd.Timestamp(start, tz="Europe/Brussels")
    end_ts   = pd.Timestamp(end,   tz="Europe/Brussels")
    gen = client.query_generation(country, start=start_ts, end=end_ts)
    keep_cols = [c for c in gen.columns if any(x in str(c) for x in ["Coal", "Gas", "Wind", "Solar", "Nuclear", "Lignite"])]
    gen_daily = gen[keep_cols].resample("D").mean()
    renewable = [c for c in gen_daily.columns if any(x in str(c) for x in ["Wind", "Solar"])]
    fossil    = [c for c in gen_daily.columns if any(x in str(c) for x in ["Coal", "Gas", "Lignite"])]
    gen_daily["renewable_share"] = gen_daily[renewable].sum(axis=1) / gen_daily[keep_cols].sum(axis=1)
    gen_daily["fossil_share"] = gen_daily[fossil].sum(axis=1) / gen_daily[keep_cols].sum(axis=1)
    return gen_daily

def fetch_weather(lat=51.5, lon=10.0, start="2021-01-01", end="2025-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min", "timezone": "Europe/Berlin"
    }
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    BASE_TEMP = 15.5
    df["HDD"] = (BASE_TEMP - df["temperature_2m_mean"]).clip(lower=0)
    df["CDD"] = (df["temperature_2m_mean"] - BASE_TEMP).clip(lower=0)
    return df[["temperature_2m_mean", "HDD", "CDD"]]