import pandas as pd


def rsi(close):
    print("📈 RSI")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14).mean() / loss.ewm(alpha=1/14).mean()
    return 100 - (100 / (1 + rs))


def macd(close):
    print("📈 MACD")
    return close.ewm(span=12).mean() - close.ewm(span=26).mean()


def atr(df):
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def vwap(df):
    return (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()


def is_retest_sell(df):
    if len(df) < 20:
        return False

    c1 = df.iloc[-1]
    c2 = df.iloc[-2]
    c3 = df.iloc[-3]
    vwap_val = vwap(df).iloc[-1]

    return (
        c3["Close"] < vwap_val
        and c2["High"] >= vwap_val
        and c1["Close"] < c2["Low"]
    )


def is_retest_buy(df):
    if len(df) < 20:
        return False

    c1 = df.iloc[-1]
    c2 = df.iloc[-2]
    c3 = df.iloc[-3]
    vwap_val = vwap(df).iloc[-1]

    return (
        c3["Close"] > vwap_val
        and c2["Low"] <= vwap_val
        and c1["Close"] > c2["High"]
    )
