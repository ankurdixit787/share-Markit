import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_utils import get_data
from indicators import rsi, macd


def train(symbol):
    df = get_data(symbol, "1d", "6mo")
    if df.empty:
        return None

    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    df = df.dropna()
    if df.empty:
        return None

    X = df[["RSI", "MACD"]]
    y = df["Target"]
    model = RandomForestClassifier(n_estimators=50)
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
