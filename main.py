import time
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import requests
from datetime import datetime
import pytz
import os
import json

# ---------- TELEGRAM ----------
BOT_TOKEN = "8747551982:AAGlQW_Cll2xtV21e2gAo1bI-CnEqxf2vOI"
CHAT_ID = "5909464423"

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except:
        pass

# ---------- FILE STORAGE ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "trades.json")

def load_trades():
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return []

def save_trades(data):
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

trade_log = load_trades()

# ---------- STOCKS ----------
stocks = [
"RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
"HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
"LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
"TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","POWERGRID.NS","NTPC.NS",
"BAJFINANCE.NS","BAJAJFINSV.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
"ADANIENT.NS","ADANIPORTS.NS","JSWSTEEL.NS","TATASTEEL.NS","GRASIM.NS",
"DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","EICHERMOT.NS","HEROMOTOCO.NS",
"ONGC.NS","COALINDIA.NS","BPCL.NS","INDUSINDBK.NS","BRITANNIA.NS",
"APOLLOHOSP.NS","BAJAJ-AUTO.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIPRULI.NS",
"DABUR.NS","GODREJCP.NS","PIDILITIND.NS","TATACONSUM.NS","M&M.NS","UPL.NS","SHREECEM.NS","AMBUJACEM.NS",
"ACC.NS","VEDL.NS","SIEMENS.NS","ABB.NS","BHEL.NS",
"HAL.NS","BEL.NS","GAIL.NS","IOC.NS","TORNTPHARM.NS",
"LUPIN.NS","AUROPHARMA.NS","ICICIGI.NS","HAVELLS.NS","VOLTAS.NS",
"COLPAL.NS","MARICO.NS","BERGEPAINT.NS","SRF.NS",
"MPHASIS.NS","NAUKRI.NS","PAYTM.NS",
"POLYCAB.NS","INDIGO.NS","DLF.NS","OBEROIRLTY.NS",
"PNB.NS","BANKBARODA.NS","CANBK.NS","UNIONBANK.NS",
"IDFCFIRSTB.NS","FEDERALBNK.NS","RBLBANK.NS"
]

# ---------- DATA ----------
def get_data(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

# ---------- INDICATORS ----------
def rsi(close):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14).mean() / loss.ewm(alpha=1/14).mean()
    return 100 - (100/(1+rs))

def macd(close):
    return close.ewm(span=12).mean() - close.ewm(span=26).mean()

def atr(df):
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

# ---------- NEWS ----------
def news_sentiment(symbol):
    try:
        news = yf.Ticker(symbol).news
        if not news:
            return 0
        score, count = 0, 0
        for n in news[:3]:
            title = n.get("title","") if isinstance(n,dict) else str(n)
            if title:
                score += TextBlob(title).sentiment.polarity
                count += 1
        return score/count if count else 0
    except:
        return 0

# ---------- AI ----------
def train(symbol):
    df = get_data(symbol,"1d","6mo")
    if df.empty:
        return None

    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    df = df.dropna()

    if df.empty:
        return None

    X = df[["RSI","MACD"]]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=80)
    model.fit(X,y)
    return model

def predict(model, df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df = df.dropna()

    if df.empty:
        return None

    last = df.iloc[-1]
    X = pd.DataFrame({
        "RSI":[float(last["RSI"])],
        "MACD":[float(last["MACD"])]
    })

    return model.predict_proba(X)[0][1]

# ---------- NIFTY TREND ----------
def nifty_trend():
    df = get_data("^NSEI","5m","5d")
    if df.empty:
        return 0
    price = df["Close"].iloc[-1]
    ma = df["Close"].rolling(20).mean().iloc[-1]
    return 1 if price > ma else -1

# ---------- DAY-END REPORT ----------
def send_day_end_report():
    total = len(trade_log)
    if total == 0:
        msg = "📊 DAY END REPORT\nNo trades today."
    else:
        msg = f"📊 DAY END REPORT\nTotal Trades: {total}\n\n"
        for t in trade_log:
            msg += f"{t['symbol']} | {t['type']} | ₹{t['price']}\n"
    send_telegram(msg)

# ---------- MAIN ----------
last_alert = {}
report_sent = False

def run():
    global trade_log, report_sent
    print("🔥 Training Models...")
    models = {}

    for s in stocks:
        m = train(s)
        if m is not None:
            models[s] = m

    print("✅ System Ready\n")

    ist = pytz.timezone("Asia/Kolkata")

    while True:
        now = datetime.now(ist)
        hour = now.hour
        minute = now.minute

        # MARKET TIME FILTER
        if not ((hour > 9 or (hour == 9 and minute >= 20)) and (hour < 15 or (hour == 15 and minute <= 15))):
            print("⏸ Market Closed")

            # 🎯 EXACT 3:30 REPORT
            if not report_sent and hour == 15 and minute == 30:
                send_day_end_report()
                report_sent = True

            time.sleep(60)
            continue

        nifty = nifty_trend()

        for s in models:
            try:
                df = get_data(s,"5m","5d")
                if df.empty or len(df) < 50:
                    continue

                price = float(df["Close"].iloc[-1])
                prev_high = df["High"].rolling(20).max().iloc[-2]
                prev_low = df["Low"].rolling(20).min().iloc[-2]

                prob = predict(models[s], df)
                if prob is None:
                    continue

                rsi_val = float(rsi(df["Close"]).iloc[-1])
                atr_val = float(atr(df).iloc[-1])
                vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]
                trend = 1 if price > df["Close"].rolling(20).mean().iloc[-1] else -1
                news = news_sentiment(s)

                score_buy = 0
                score_sell = 0

                if prob > 0.7: score_buy += 1
                if rsi_val > 55: score_buy += 1
                if vol_ratio > 1.2: score_buy += 1
                if trend == 1: score_buy += 1
                if price > prev_high: score_buy += 1
                if news >= 0: score_buy += 1

                if prob < 0.3: score_sell += 1
                if rsi_val < 45: score_sell += 1
                if vol_ratio > 1.2: score_sell += 1
                if trend == -1: score_sell += 1
                if price < prev_low: score_sell += 1
                if news <= 0: score_sell += 1

                # NIFTY FILTER
                if nifty == 1:
                    score_sell = 0
                elif nifty == -1:
                    score_buy = 0

                target_buy = price + 2 * atr_val
                sl_buy = price - atr_val

                target_sell = price - 2 * atr_val
                sl_sell = price + atr_val

                accuracy = int(prob * 100)

                if score_buy >= 5 and last_alert.get(s) != "BUY":
                    msg = f"""🚀💀 VERY STRONG BUY
{s}
Price: {price:.2f}

🎯 Target: {target_buy:.2f}
🛑 SL: {sl_buy:.2f}

📊 Accuracy: {accuracy}%
"""
                    print(msg)
                    send_telegram(msg)
                    trade_log.append({"symbol": s, "price": price, "type": "BUY", "time": now.strftime("%H:%M")})
                    save_trades(trade_log)
                    last_alert[s] = "BUY"

                elif score_sell >= 5 and last_alert.get(s) != "SELL":
                    msg = f"""🔻💀 VERY STRONG SELL
{s}
Price: {price:.2f}

🎯 Target: {target_sell:.2f}
🛑 SL: {sl_sell:.2f}

📊 Accuracy: {accuracy}%
"""
                    print(msg)
                    send_telegram(msg)
                    trade_log.append({"symbol": s, "price": price, "type": "SELL", "time": now.strftime("%H:%M")})
                    save_trades(trade_log)
                    last_alert[s] = "SELL"

                else:
                    print(f"{s}: No trade")

            except Exception as e:
                print(f"❌ Error {s}: {e}")

        time.sleep(20)

# ---------- RUN ----------
if __name__ == "__main__":
    run()