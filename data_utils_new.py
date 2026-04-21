from datetime import datetime

import pandas as pd
import pytz
import requests

UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1SkNWNFkiLCJqdGkiOiI2OWUyNGQ5ZmZiOTk1NzJjN2Q3NjM5OWQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3NjQzODY4NywiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc2NDYzMjAwfQ.HvuoST01AEtMCwaGn-MyswNY1Jbmu3_PvOLO7P_d4ag"

def allow_trade_time():
    return True

def fetch_upstox_ohlc(symbol, interval="5m", lookback=100):
    """Fetch OHLC candles from Upstox API with proper error handling."""
    try:
        if ".NS" in symbol:
            symbol = symbol.replace(".NS", "")
        instrument_key = f"NSE_EQ|{symbol}"
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval}"
        headers = {"Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
        params = {"from": "2026-04-01", "to": "2026-04-17"}
        
        resp = requests.get(url, headers=headers, params=params)
        response_data = resp.json()
        
        # Check HTTP status
        if resp.status_code != 200:
            print(f"[ERROR] {symbol}: HTTP {resp.status_code}")
            return pd.DataFrame()
        
        # Check if response has data structure
        if "data" not in response_data:
            print(f"[ERROR] {symbol}: No 'data' in response")
            return pd.DataFrame()
        
        # Get candles
        candles = response_data.get("data", {}).get("candles", [])
        if not candles:
            print(f"[WARNING] {symbol}: No candles returned")
            return pd.DataFrame()
        
        # Build DataFrame
        df = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        print(f"✓ {symbol}: {len(df)} candles fetched")
        return df.tail(lookback)
    except KeyError as e:
        print(f"[ERROR] {symbol}: Missing key {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return pd.DataFrame()

def get_ohlc(symbol, interval="5m", lookback=100):
    """Fetch OHLC candles for a symbol and interval from Upstox."""
    return fetch_upstox_ohlc(symbol, interval, lookback)

def get_data(symbol, interval, period):
    """Fetch data from Upstox API instead of yfinance."""
    try:
        df = fetch_upstox_ohlc(symbol, interval, lookback=100)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error in get_data for {symbol}: {e}")
        return pd.DataFrame()

def nifty_trend(interval="5m"):
    """Get NIFTY 50 trend."""
    return fetch_upstox_ohlc("NIFTY 50", interval, lookback=10)
