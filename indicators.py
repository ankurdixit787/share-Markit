import pandas as pd


def rsi(close):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14).mean() / loss.ewm(alpha=1/14).mean()
    return 100 - (100 / (1 + rs))


def macd(close):
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
    if len(df) < 26:
        return False

    c1 = df.iloc[-1]
    c2 = df.iloc[-2]
    c3 = df.iloc[-3]
    vwap_val = vwap(df).iloc[-1].item() if hasattr(vwap(df).iloc[-1], 'item') else vwap(df).iloc[-1]

    # Volume confirmation: retest candle volume > 20-period average
    vol_ok = c1["Volume"] > df["Volume"].rolling(20).mean().iloc[-1].item()
    # RSI confirmation: RSI < 50 (bearish)
    rsi_val = rsi(df["Close"]).iloc[-1].item() if hasattr(rsi(df["Close"]).iloc[-1], 'item') else rsi(df["Close"]).iloc[-1]
    rsi_ok = rsi_val < 50
    # MACD confirmation: MACD < 0 (bearish)
    macd_val = macd(df["Close"]).iloc[-1].item() if hasattr(macd(df["Close"]).iloc[-1], 'item') else macd(df["Close"]).iloc[-1]
    macd_ok = macd_val < 0

    return (
        c3["Close"] < vwap_val
        and c2["High"] >= vwap_val
        and c1["Close"] < c2["Low"]
        and vol_ok
        and rsi_ok
        and macd_ok
    )


def is_retest_buy(df):
    if len(df) < 26:
        return False

    c1 = df.iloc[-1]
    c2 = df.iloc[-2]
    c3 = df.iloc[-3]
    vwap_val = vwap(df).iloc[-1].item() if hasattr(vwap(df).iloc[-1], 'item') else vwap(df).iloc[-1]

    # Volume confirmation: retest candle volume > 20-period average
    vol_ok = c1["Volume"] > df["Volume"].rolling(20).mean().iloc[-1].item()
    # RSI confirmation: RSI > 50 (bullish)
    rsi_val = rsi(df["Close"]).iloc[-1].item() if hasattr(rsi(df["Close"]).iloc[-1], 'item') else rsi(df["Close"]).iloc[-1]
    rsi_ok = rsi_val > 50
    # MACD confirmation: MACD > 0 (bullish)
    macd_val = macd(df["Close"]).iloc[-1].item() if hasattr(macd(df["Close"]).iloc[-1], 'item') else macd(df["Close"]).iloc[-1]
    macd_ok = macd_val > 0

    return (
        c3["Close"] > vwap_val
        and abs(c2["Low"] - vwap_val) <= 3
        and c1["Close"] > c2["High"]
        and vol_ok
        and rsi_ok
        and macd_ok
    )
