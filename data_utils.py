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
    # Try yfinance as it's more reliable
    try:
        df = yf.download(tickers=symbol, period="7d", interval=interval, progress=False)
        if df is not None and not df.empty:
            df = df.tail(lookback)
            df = df.reset_index()
            return df
    except:
        pass
    return pd.DataFrame()


def fetch_upstox_ohlc(symbol, interval="1minute", lookback=100):
    """Fetch OHLC from Upstox with proper error handling."""
    global _failed_symbols
    
    try:
        clean_symbol = symbol.replace(".NS", "") if ".NS" in symbol else symbol
        
        # Map ^NSEI to NSEI
        if clean_symbol == "^NSEI":
            clean_symbol = "NSEI"
        
        # Skip if already failed
        if clean_symbol in _failed_symbols:
            return pd.DataFrame()
        
        # Get ISIN code
        isin_code = ISIN_MAP.get(clean_symbol)
        if not isin_code:
            _failed_symbols.add(clean_symbol)
            return pd.DataFrame()
        
        # Determine instrument type
        if isin_code == "NIFTY50":
            instrument_key = f"NSE_INDEX|Nifty 50"  # Correct format for Nifty
        else:
            instrument_key = f"NSE_EQ|{isin_code}"
        
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval}"
        headers = {"Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
        # Fetch last 7 days of data
        import datetime
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        params = {"from": start_date, "to": end_date}
        
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        
        if resp.status_code != 200:
            _failed_symbols.add(clean_symbol)
            return pd.DataFrame()
        
        response_data = resp.json()
        candles = response_data.get("data", {}).get("candles", [])
        
        if not candles:
            _failed_symbols.add(clean_symbol)
            return pd.DataFrame()
        
        # Build DataFrame with 7 columns (Datetime, O, H, L, C, Volume, OI)
        df = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"])
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        
        return df.tail(lookback)
    except Exception as e:
        print(f"DEBUG fetch_upstox_ohlc error for {symbol}: {e}")
        return pd.DataFrame()



def get_data(symbol, interval, period):
    """Fetch data from Upstox API with yfinance fallback."""
    try:
        # Map period to lookback
        period_map = {
            "5d": 360,
            "5day": 360,
            "1d": 72,
            "1day": 72,
            "6mo": 500,
            "1mo": 400,
        }
        lookback = period_map.get(period, 100)
        
        # Map interval
        interval_map = {
            "5m": "1minute",
            "5min": "1minute", 
            "1m": "1minute",
            "1min": "1minute",
            "30m": "30minute",
            "30min": "30minute",
            "1d": "1minute",
            "1h": "1minute",
        }
        mapped_interval = interval_map.get(interval, "1minute")
        
        # Upstox only - no fallback to yfinance
        df = fetch_upstox_ohlc(symbol, mapped_interval, lookback=lookback)
        return df.dropna() if (df is not None and not df.empty) else pd.DataFrame()
    except Exception as e:
        # Last resort - try yfinance
        try:
            df = yf.download(symbol, period="7d", interval="1m", progress=False)
            return df.reset_index().tail(100) if df is not None and not df.empty else pd.DataFrame()
        except:
            return pd.DataFrame()


def nifty_trend():
    """Get NIFTY trend with fallback."""
    df = get_data("^NSEI", "5m", "5d")
    if df.empty or len(df) < 20:
        return 0  # Default to neutral if no data
    try:
        last_close = df["Close"].iloc[-1]
        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        if pd.isna(last_close) or pd.isna(ma20):
            return 0
        return 1 if last_close > ma20 else -1
    except:
        return 0
