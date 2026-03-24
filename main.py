import time
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests
from datetime import datetime
import pytz
import os
import json

# ---------- TELEGRAM ----------
BOT_TOKEN = "PASTE_TOKEN"
CHAT_ID = "PASTE_CHAT_ID"

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)

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
        print(f"✅ Saved Trades: {len(data)}")
    except Exception as e:
        print("Save Error:", e)

trade_log = load_trades()

# ---------- STOCKS ----------
stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]

# ---------- DATA ----------
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
        print("Data Error:", e)
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

    model = RandomForestClassifier(n_estimators=50)
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
        "RSI": [float(last["RSI"])],
        "MACD": [float(last["MACD"])]
    })

    return model.predict_proba(X)[0][1]

# ---------- MAIN ----------
def run():
    global trade_log

    print("🔥 Training Models...")
    models = {}

    for s in stocks:
        print(f"Training: {s}")
        m = train(s)
        if m:
            models[s] = m

    print("✅ System Ready")
    
    last_alert = {}
    report_sent = False

    while True:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        hour = now.hour
        minute = now.minute

        print(f"\n🕒 Time: {hour}:{minute}")

        # ---------- MARKET CLOSED ----------
        if not ((hour > 9 or (hour == 9 and minute >= 20)) and (hour < 15 or (hour == 15 and minute <= 15))):

            print("⏸ Market Closed")

            # 🎯 EXACT 3:20 REPORT
            if not report_sent and hour == 15 and minute == 20:

                print("📊 Sending Report")

                total = len(trade_log)

                msg = "📊 DAY END REPORT\n\n"
                msg += f"Total Trades: {total}\n\n"

                for t in trade_log:
                    msg += f"{t['symbol']} | {t['type']} | ₹{t['price']}\n"

                send_telegram(msg)
                report_sent = True

            # 🔁 BACKUP
            elif not report_sent and hour == 15 and minute > 20:

                print("📊 Backup Report")

                total = len(trade_log)

                msg = "📊 DAY END REPORT\n\n"
                msg += f"Total Trades: {total}\n\n"

                for t in trade_log:
                    msg += f"{t['symbol']} | {t['type']} | ₹{t['price']}\n"

                send_telegram(msg)
                report_sent = True

            time.sleep(20)
            continue

        # ---------- MARKET OPEN ----------
        for s in models:
            try:
                print(f"Checking: {s}")

                df = get_data(s,"5m","5d")
                if df.empty or len(df) < 50:
                    continue

                price = df["Close"].iloc[-1]
                if isinstance(price, pd.Series):
                    price = price.iloc[0]
                price = float(price)

                prob = predict(models[s], df)
                if prob is None:
                    continue

                accuracy = int(prob * 100)

                print(f"{s} Accuracy: {accuracy}")

                # 🎯 ONLY 90%+
                if accuracy >= 90 and last_alert.get(s) != "BUY":

                    send_telegram(f"🚀 STRONG BUY\n{s}\nPrice: {price}\nAccuracy: {accuracy}%")

                    trade_log.append({
                        "symbol": s,
                        "price": price,
                        "type": "BUY",
                        "time": now.strftime("%H:%M")
                    })

                    save_trades(trade_log)

                    last_alert[s] = "BUY"

            except Exception as e:
                print("Error:", e)

        time.sleep(15)

# ---------- RUN ----------
if __name__ == "__main__":
    run()