import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_utils import get_data
from indicators import rsi, macd


def train(symbol):
    # Use available intraday data from last 7 days (Upstox limitation)
    df = get_data(symbol, "5m", "5d")
    print(f"DEBUG: {symbol} data rows: {len(df)}")
    if df.empty or len(df) < 30:  # Reduced from 50 to 30 for 5-day data
        print(f"⚠️  No data for {symbol}")
        return None

    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    # Use simple momentum target: price up in next 5 candles
    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
    df = df.dropna()
    if df.empty or len(df) < 20:  # Reduced minimum for training
        print(f"⚠️  Insufficient data for {symbol}")
        return None

    X = df[["RSI", "MACD"]]
    y = df["Target"]
    if len(X) < 10:
        return None
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    print(f"✅ Model Ready: {symbol}")
    return model


def predict(model, df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df = df.dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    X = pd.DataFrame({"RSI": [float(last["RSI"])], "MACD": [float(last["MACD"])]})
    return model.predict_proba(X)[0][1]
