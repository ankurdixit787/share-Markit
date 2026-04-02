from datetime import datetime

import pandas as pd
import pytz
import yfinance as yf


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
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
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
