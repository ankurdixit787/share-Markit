import time
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import requests
from datetime import datetime

# ---------- TELEGRAM ----------
BOT_TOKEN = "8747551982:AAGlQW_Cll2xtV21e2gAo1bI-CnEqxf2vOI"
CHAT_ID = "5909464423"


def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except:
        pass

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

# ---------- TRACKING ----------
last_alert = {}
trade_log = []
notification_count = 0
report_sent = False

# ---------- REPORT ----------
def day_end_report():
    total = len(trade_log)
    profit = sum(1 for t in trade_log if t["result"] == "TARGET")
    loss = sum(1 for t in trade_log if t["result"] == "SL")

    report = f"📊 DAY END REPORT\n\n"
    report += f"Total Trades: {total}\n"
    report += f"Profit Trades: {profit} ✅\n"
    report += f"Loss Trades: {loss} ❌\n\n"

    for t in trade_log:
        status = "⚪"
        if t["result"] == "TARGET":
            status = "✅"
        elif t["result"] == "SL":
            status = "❌"

        report += f"{status} {t['symbol']} | {t['type']} | Score:{t['score']}\n"

    send_telegram(report)

# ---------- MAIN ----------
def run():
    global notification_count, report_sent

    print("🔥 Training Models...")
    models = {}

    for s in stocks:
        print(f"Training {s}...")
        m = train(s)
        if m is not None:
            models[s] = m

    print("✅ System Ready\n")
    send_telegram("✅ Bot Started")

    while True:
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        # DAILY RESET
        if hour == 9 and minute < 10:
            trade_log.clear()
            report_sent = False
            last_alert.clear()
            notification_count = 0

        # MARKET CLOSED BLOCK + REPORT FIX
        if not ((hour > 9 or (hour == 9 and minute >= 20)) and (hour < 15 or (hour == 15 and minute <= 15))):
            print("⏸ Market Closed")

            # ✅ REPORT FIX
            if not report_sent and (hour > 15 or (hour == 15 and minute >= 20)):
                send_telegram("📊 Sending Day-End Report...")
                day_end_report()
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

                if nifty == 1:
                    score_sell = 0
                elif nifty == -1:
                    score_buy = 0

                accuracy = int(prob * 100)

                # ✅ 90% FILTER
                if accuracy < 90:
                    continue

                # ✅ REPEAT ALERT BLOCK
                if last_alert.get(s) == "BUY" or last_alert.get(s) == "SELL":
                    continue

                if score_buy >= 5:
                    send_telegram(f"🚀 VERY STRONG BUY\n{s}\nPrice: {price}\nAccuracy: {accuracy}%")
                    last_alert[s] = "BUY"
                    notification_count += 1

                elif score_sell >= 5:
                    send_telegram(f"🔻 VERY STRONG SELL\n{s}\nPrice: {price}\nAccuracy: {accuracy}%")
                    last_alert[s] = "SELL"
                    notification_count += 1

            except Exception as e:
                print(f"Error {s}: {e}")

        time.sleep(20)

# ---------- RUN ----------
if __name__ == "__main__":
    run()