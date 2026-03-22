import time
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import requests

# ---------- TELEGRAM ----------
BOT_TOKEN = "8747551982:AAGlQW_Cll2xtV21e2gAo1bI-CnEqxf2vOI"
CHAT_ID = "5909464423"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
        print("📲 Telegram Sent")
    except Exception as e:
        print("❌ Telegram Error:", e)

# ---------- DATA ----------
def get_data(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"❌ Failed download for {symbol} ({interval}, {period}): {e}")
        return pd.DataFrame()

# ---------- INDICATORS ----------
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/period).mean() / loss.ewm(alpha=1/period).mean()
    return 100 - (100/(1+rs))

def macd(close):
    return close.ewm(span=12).mean() - close.ewm(span=26).mean()

def atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def volume_spike(df):
    avg_vol = df["Volume"].rolling(20).mean()
    return df["Volume"].iloc[-1] > avg_vol.iloc[-1] * 1.5

# ---------- NEWS ----------
def news_sentiment(symbol):
    try:
        news = yf.Ticker(symbol).news
        if not news:
            return 0
        score, count = 0, 0
        for n in news[:5]:
            title = n.get("title","") if isinstance(n,dict) else str(n)
            if title:
                score += TextBlob(title).sentiment.polarity
                count += 1
        return score/count if count else 0
    except:
        return 0

# ---------- AI MODEL ----------
def train(symbol):
    df = get_data(symbol,"1d","6mo")
    if df.empty:
        return None
    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    df = df.dropna()
    X = df[["RSI","MACD"]]
    y = df["Target"]
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X,y)
    return model

def predict(model, df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df = df.dropna()
    if df.empty or model is None:
        return None
    last = df.iloc[-1]
    X = pd.DataFrame({"RSI":[float(last["RSI"])],"MACD":[float(last["MACD"])]})
    return model.predict_proba(X)[0][1]

# ---------- MARKET TREND ----------
def market_trend(symbol="NIFTYBEES.NS"):
    df = get_data(symbol,"1h","60d")
    if df.empty:
        return 0
    ema50 = df["Close"].ewm(span=50).mean()
    ema200 = df["Close"].ewm(span=200).mean()
    return 1 if ema50.iloc[-1] > ema200.iloc[-1] else -1

# ---------- SIGNAL LOGIC ----------
last_alert = {}

def run(stocks):
    print("🔥 Training Models...")
    models = {s: train(s) for s in stocks}
    print("✅ System Ready\n")

    while True:
        m_trend = market_trend()
        for s in stocks:
            try:
                # Intraday max 50-60 days
                df_5m = get_data(s,"5m","50d")
                df_15m = get_data(s,"15m","50d")
                df_1h = get_data(s,"1h","60d")

                if df_5m.empty or len(df_5m) < 50:
                    continue

                price = float(df_5m["Close"].iloc[-1])
                prev_high = float(df_5m["High"].rolling(20).max().iloc[-2])
                prev_low = float(df_5m["Low"].rolling(20).min().iloc[-2])

                prob = predict(models[s], df_5m)
                if prob is None:
                    continue

                rsi_val = float(rsi(df_5m["Close"]).iloc[-1])
                macd_val = float(macd(df_5m["Close"]).iloc[-1])
                news = news_sentiment(s)
                atr_val = float(atr(df_5m).iloc[-1])
                vol_ok = volume_spike(df_5m)

                # Multi-timeframe
                rsi_15 = float(rsi(df_15m["Close"]).iloc[-1]) if not df_15m.empty else rsi_val
                macd_15 = float(macd(df_15m["Close"]).iloc[-1]) if not df_15m.empty else macd_val
                rsi_1h = float(rsi(df_1h["Close"]).iloc[-1]) if not df_1h.empty else rsi_val
                macd_1h = float(macd(df_1h["Close"]).iloc[-1]) if not df_1h.empty else macd_val

                # -------- BUY SIGNAL --------
                if (prob > 0.80 and price > prev_high and rsi_val>60 and macd_val>0
                    and vol_ok and m_trend==1
                    and rsi_15>55 and macd_15>0 and rsi_1h>50 and macd_1h>0
                    and news>=0):
                    if last_alert.get(s)!="BUY":
                        msg=f"""
                    🚀 *STRONG BUY SIGNAL*
                    Stock: {s}
                    Price: {price:.2f}
                    🎯 Target: {price + 2*atr_val:.2f}
                    🛑 SL: {price - atr_val:.2f}
                    📊 AI: {prob:.2f}
                    📈 RSI: {rsi_val:.2f} / 15m: {rsi_15:.2f} / 1h: {rsi_1h:.2f}
                    📈 MACD: {macd_val:.2f} / 15m: {macd_15:.2f} / 1h: {macd_1h:.2f}
                    📰 News: {news:.2f}
                    📊 Volume Spike: {vol_ok}
                    📊 Market Trend: {'UP' if m_trend==1 else 'DOWN'}
                    """
                        print(msg)
                        send_telegram(msg)
                        last_alert[s]="BUY"

                # -------- SELL SIGNAL --------
                elif (prob < 0.20 and price < prev_low and rsi_val<40 and macd_val<0
                      and vol_ok and m_trend==-1
                      and rsi_15<45 and macd_15<0 and rsi_1h<50 and macd_1h<0
                      and news<=0):
                    if last_alert.get(s)!="SELL":
                        msg=f"""
                        🔻 *STRONG SELL SIGNAL*
                        Stock: {s}
                        Price: {price:.2f}
                        🎯 Target: {price - 2*atr_val:.2f}
                        🛑 SL: {price + atr_val:.2f}
                        📊 AI: {prob:.2f}
                        📉 RSI: {rsi_val:.2f} / 15m: {rsi_15:.2f} / 1h: {rsi_1h:.2f}
                        📉 MACD: {macd_val:.2f} / 15m: {macd_15:.2f} / 1h: {macd_1h:.2f}
                        📰 News: {news:.2f}
                        📊 Volume Spike: {vol_ok}
                        📊 Market Trend: {'UP' if m_trend==1 else 'DOWN'}
                        """
                        print(msg)
                        send_telegram(msg)
                        last_alert[s]="SELL"

                else:
                    print(f"{s}: No strong trade")

            except Exception as e:
                print(f"❌ Error {s}: {e}")

        time.sleep(20)

# ---------- RUN ----------
if __name__ == "__main__":
    run(["NIFTYBEES.NS","RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS"])