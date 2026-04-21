from datetime import datetime

import pandas as pd
import pytz
import requests
import yfinance as yf

from config import ISIN_MAP

UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1SkNWNFkiLCJqdGkiOiI2OWUyNGQ5ZmZiOTk1NzJjN2Q3NjM5OWQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3NjQzODY4NywiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc2NDYzMjAwfQ.HvuoST01AEtMCwaGn-MyswNY1Jbmu3_PvOLO7P_d4ag"

# Cache for tracking failed symbols in current session (avoid redundant Upstox API calls)
_failed_symbols = set()

def allow_trade_time():
    now = datetime.now(pytz.timezone('Asia/Kolkata')).time()
    start = datetime.strptime("09:25", "%H:%M").time()
    end = datetime.strptime("14:30", "%H:%M").time()
    return start <= now <= end


def get_ohlc(symbol, interval="5m", lookback=100):
    """Fetch OHLC candles for a symbol and interval."""
    df = yf.download(tickers=symbol, period="7d", interval=interval)
    df = df.tail(lookback)
    df = df.reset_index()
    return df


def get_data(symbol, interval, period):
    """Fetch data from Upstox API.
    Note: only 1minute and 30minute intervals are supported by Upstox.
    Period is converted to lookback for 1-minute candles:
    - "5d" = 360 candles (optimized for speed: 5 days at 1-min, market open ~72 min/day)
    - "6mo" = 2000 candles (sufficient for model training without being too slow)
    """
    try:
        # Map period to lookback (optimized for speed)
        period_map = {
            "5d": 360,      # ~5 days of intraday data
            "5day": 360,
            "1d": 72,       # ~1 day of intraday data
            "1day": 72,
            "6mo": 2000,    # Sufficient for ML model training
            "1mo": 400,
        }
        lookback = period_map.get(period, 100)
        
        # Map common interval names to Upstox supported ones
        interval_map = {
            "5m": "1minute",
            "5min": "1minute", 
            "1m": "1minute",
            "1min": "1minute",
            "30m": "30minute",
            "30min": "30minute",
            "1d": "1minute",
            "1h": "1minute",
            "1minute": "1minute",
            "30minute": "30minute"
        }
        mapped_interval = interval_map.get(interval, "1minute")
        df = fetch_upstox_ohlc(symbol, mapped_interval, lookback=lookback)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna()
        return df
    except Exception as e:
        return pd.DataFrame()


def nifty_trend():
    df = get_data("^NSEI", "5m", "5d")
    if df.empty or len(df) < 20:
        return 0
    last_close = df["Close"].iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    if pd.isna(last_close) or pd.isna(ma20):
        return 0
    return 1 if last_close > ma20 else -1
