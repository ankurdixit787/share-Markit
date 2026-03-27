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
import yfinance as yf
import pandas as pd


HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-26",  # Shri Ram Navami
    "2026-03-31",  # Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Eid
    "2026-06-26",  # Moharram
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-10",  # Diwali Balipratipada
]

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# ---------- TELEGRAM ----------
BOT_TOKEN = "8747551982:AAGlQW_Cll2xtV21e2gAo1bI-CnEqxf2vOI"
CHAT_ID = "5909464423"

def get_ohlc(symbol, interval="5m", lookback=100):
    """
    OHLC data fetch karega given symbol aur interval ke liye
    interval options: '1m','5m','15m','30m','1h','1d'
    lookback = kitne candles chahiye
    """
    df = yf.download(tickers=symbol, period="7d", interval=interval)
    df = df.tail(lookback)
    df = df.reset_index()
    return df

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        print("📨 Telegram sent")
    except Exception as e:
        print("⚠️ Telegram Error:", e)

# ---------- FILE STORAGE ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "trades.json")

def load_trades():
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE,"r") as f:
                return json.load(f)
    except:
        pass
    return []

def save_trades(data):
    try:
        with open(LOG_FILE,"w") as f:
            json.dump(data,f)
        print(f"✅ Saved Trades: {len(data)}")
    except Exception as e:
        print("⚠️ Save Error:", e)

trade_log = load_trades()

# ---------- STOCKS ----------
stocks = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
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
"IDFCFIRSTB.NS","FEDERALBNK.NS","RBLBANK.NS"]

# ---------- DATA ----------
def get_data(symbol, interval, period):
    try:
        print(f"📊 Fetching {symbol}")
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df is None or df.empty:
            print("❌ Data empty")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        print("✅ Data Ready")
        return df
    except Exception as e:
        print("❌ Data Error:", e)
        return pd.DataFrame()

# ---------- INDICATORS ----------
def rsi(close):
    print("📈 RSI")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14).mean() / loss.ewm(alpha=1/14).mean()
    return 100 - (100/(1+rs))

def macd(close):
    print("📈 MACD")
    return close.ewm(span=12).mean() - close.ewm(span=26).mean()

def atr(df):
    tr = pd.concat([
        df["High"]-df["Low"],
        abs(df["High"]-df["Close"].shift()),
        abs(df["Low"]-df["Close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def vwap(df):
    return (df['Close']*df['Volume']).cumsum()/df['Volume'].cumsum()

# ---------- NEWS SENTIMENT ----------
def news_sentiment(symbol):
    try:
        news = yf.Ticker(symbol).news
        if not news:
            return 0
        score, count = 0, 0
        for n in news[:3]:
            title = n.get("title","") if isinstance(n, dict) else str(n)
            if title:
                score += TextBlob(title).sentiment.polarity
                count += 1
        return score/count if count else 0
    except:
        return 0

# ---------- AI ----------
def train(symbol):
    df = get_data(symbol,"1d","6mo")
    if df.empty: return None
    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    df = df.dropna()
    if df.empty: return None
    X = df[["RSI","MACD"]]
    y = df["Target"]
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X,y)
    print(f"✅ Model Ready: {symbol}")
    return model

def predict(model, df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["MACD"] = macd(df["Close"])
    df = df.dropna()
    if df.empty: return None
    last = df.iloc[-1]
    X = pd.DataFrame({"RSI":[float(last["RSI"])],"MACD":[float(last["MACD"])]})
    return model.predict_proba(X)[0][1]

# ---------- NIFTY TREND ----------
def nifty_trend():
    df = get_data("^NSEI","5m","5d")
    if df.empty or len(df)<20: return 0
    last_close = df["Close"].iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    if pd.isna(last_close) or pd.isna(ma20): return 0
    return 1 if last_close>ma20 else -1

# ---------- MAIN ----------
last_alert = {}

def run():
    global trade_log
    print("🚀 TOP 1 NUMBER BOT STARTED")
    print("🔥 Training Models...")
    models = {}
    for s in stocks:
        m = train(s)
        if m: models[s] = m
    print("✅ System Ready\n")

    ist = pytz.timezone('Asia/Kolkata')

    while True:
        now = datetime.now(ist)
        hour, minute = now.hour, now.minute
        print(f"\n🕒 Time: {hour}:{minute}")

        today_str = datetime.now(ist).strftime("%Y-%m-%d")

        # Market filter (comment out for testing)
        if (today_str in HOLIDAYS_2026) or not ((hour > 9 or (hour == 9 and minute >= 15)) and (hour < 15 or (hour == 15 and minute <= 30))):
            print("⏸ Market Closed or Holiday")
            if hour == 15 and minute >= 30:
                print("📊 Sending Daily Report")
                msg = generate_daily_report()
                send_telegram(msg)
            time.sleep(60)
            continue

        nifty = nifty_trend()

        for s in models:
            try:
                df = get_data(s, "5m", "5d")
                if df.empty or len(df) < 20:
                    continue

                # ---------- SAFE INDICATORS ----------
                price = df["Close"].iloc[-1].item()
                ma20 = df["Close"].rolling(20).mean().iloc[-1].item()
                last_high = df["High"].rolling(20).max().iloc[-2].item()
                last_low = df["Low"].rolling(20).min().iloc[-2].item()
                vol_ratio = df["Volume"].iloc[-1].item() / df["Volume"].rolling(20).mean().iloc[-1].item()

                rsi_val = rsi(df["Close"]).iloc[-1].item()
                atr_val = atr(df).iloc[-1].item()
                vwap_val = vwap(df).iloc[-1].item()

                macd_series = macd(df["Close"])
                macd_val = macd_series.iloc[-1].item() if not macd_series.empty else 0.0

                roc_series = df["Close"].pct_change()
                roc_val = roc_series.iloc[-1].item() if not roc_series.empty else 0.0

                news_score = news_sentiment(s)
                prob = predict(models[s], df)
                if prob is None:
                    continue

                # ---------- BACKBONE CHECK ----------
                backbone_score = 0
                if (price > ma20) and (nifty == 1): backbone_score += 1
                if (rsi_val > 60) and (macd_val > 0): backbone_score += 1
                if (price > vwap_val) and (vol_ratio > 1.5): backbone_score += 1
                df_5m = get_ohlc(symbol=s, interval="5m", lookback=50)
                ma20_5m = df_5m["Close"].rolling(20).mean().iloc[-1].item()
                df_1h = get_ohlc(symbol=s, interval="1h", lookback=50)
                ma20_1h = df_1h["Close"].rolling(20).mean().iloc[-1].item()
                if (price > ma20_5m) and (price > ma20_1h): backbone_score += 1

              # ---------- BUY BLOCK ----------
                if backbone_score == 3:
                    print(f"{GREEN}{s} | Backbone: PASS")
                    score = 0
                    cond_details = []

                    # 1. AI filter
                    if prob > 0.75:
                        score += 1; cond_details.append("AI✔"); print("AI filter passed")

                    # 2. Breakout filter
                    if price > last_high:
                        score += 1; cond_details.append("BREAKOUT✔"); print("Breakout filter passed")

                    # 3. News sentiment filter
                    if news_score > 0.2:
                        score += 1; cond_details.append("NEWS✔"); print("News filter passed")

                    # 4. Bollinger band filter
                    upper_band = float(df["Close"].rolling(20).mean().iloc[-1] + 2 * df["Close"].rolling(20).std().iloc[-1])
                    if price > upper_band:
                        score += 1; cond_details.append("BOLL✔"); print("Bollinger filter passed")

                    # 5. ROC filter
                    if roc_val > 0:
                        score += 1; cond_details.append("ROC✔"); print("ROC filter passed")

                    # 6. ATR filter
                    if abs(price - df["Close"].iloc[-2]) > 1.5 * atr_val:
                        score += 1; cond_details.append("ATR✔"); print("ATR filter passed")

                    # 7. Extra timeframe confirm (15m MA check)
                    ma_15m = df["Close"].rolling(15).mean().iloc[-1]
                    if price > ma_15m:
                        score += 1; cond_details.append("TF15m✔"); print("15m timeframe confirm passed")

                    # 8. Volume spike confirm
                    vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]
                    if vol_ratio > 2:
                        score += 1; cond_details.append("VOL✔"); print("Volume spike filter passed")

                    print(f"{s} | BUY Score: {score}/8 | {' | '.join(cond_details)} | Price: {price:.2f}")

                    target = price + 2 * atr_val
                    sl = price - atr_val

                    if score >= 5 and last_alert.get(s) != "BUY":
                        msg = (
                            f"🚀 BUY ALERT: {s}\n"
                            f"Price: {price:.2f}\n"
                            f"Target: {target:.2f}\n"
                            f"Stop Loss: {sl:.2f}\n"
                            f"Score: {score}/8\n"
                            f"{' | '.join(cond_details)}\n"
                            f"Time: {now.strftime('%H:%M')}"
                        )
                        send_telegram(msg)
                        trade_log.append({"symbol": s, "price": price, "type": "BUY", "time": now.strftime("%H:%M")})
                        save_trades(trade_log)
                        last_alert[s] = "BUY"
                else:
                    print(f"{RED}{s} | Backbone: FAIL | BUY Score: 0/8 | Price: {price:.2f}")
                    # ---------- SELL BLOCK ----------
                if backbone_score == 3:
                    print(f"{GREEN}{s} | Backbone: PASS")
                    score = 0
                    cond_details = []

                    # 1. AI filter
                    if prob < 0.25:
                        score += 1; cond_details.append("AI✔"); print("AI filter passed")

                    # 2. Breakdown filter
                    if price < last_low:
                        score += 1; cond_details.append("BREAKDOWN✔"); print("Breakdown filter passed")

                    # 3. News sentiment filter
                    if news_score < -0.2:
                        score += 1; cond_details.append("NEWS✔"); print("News filter passed")

                    # 4. Bollinger band filter
                    lower_band = float(df["Close"].rolling(20).mean().iloc[-1] - 2 * df["Close"].rolling(20).std().iloc[-1])
                    if price < lower_band:
                        score += 1; cond_details.append("BOLL✔"); print("Bollinger filter passed")

                    # 5. ROC filter
                    if roc_val < 0:
                        score += 1; cond_details.append("ROC✔"); print("ROC filter passed")

                    # 6. ATR filter
                    if abs(df["Close"].iloc[-2] - price) > 1.5 * atr_val:
                        score += 1; cond_details.append("ATR✔"); print("ATR filter passed")

                    # 7. Extra timeframe confirm (15m MA check)
                    ma_15m = df["Close"].rolling(15).mean().iloc[-1]
                    if price < ma_15m:
                        score += 1; cond_details.append("TF15m✔"); print("15m timeframe confirm passed")

                    # 8. Volume spike confirm
                    vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]
                    if vol_ratio > 2:
                        score += 1; cond_details.append("VOL✔"); print("Volume spike filter passed")

                    print(f"{RED}{s} | SELL Score: {score}/8 | {' | '.join(cond_details)} | Price: {price:.2f}")

                    target = price - 2 * atr_val
                    sl = price + atr_val

                    if score >= 5 and last_alert.get(s) != "SELL":
                        msg = (
                            f"⚠️ SELL ALERT: {s}\n"
                            f"Price: {price:.2f}\n"
                            f"Target: {target:.2f}\n"
                            f"Stop Loss: {sl:.2f}\n"
                            f"Score: {score}/8\n"
                            f"{' | '.join(cond_details)}\n"
                            f"Time: {now.strftime('%H:%M')}"
                        )
                        send_telegram(msg)
                        trade_log.append({"symbol": s, "price": price, "type": "SELL", "time": now.strftime("%H:%M")})
                        save_trades(trade_log)
                        last_alert[s] = "SELL"
                else:
                     print(f"{s} | Backbone: FAIL | SELL Score: 0/8 | Price: {price:.2f}")
            except Exception as e:
                     print(f"❌ MAIN ERROR: {e}")

        time.sleep(20)
        
def generate_daily_report():
    # Load trade log
    with open("trade_log.json", "r") as f:
        trades = json.load(f)

    today = datetime.now().strftime("%d %B %Y")

    # Summary counts
    buy_count = sum(1 for t in trades if t["type"] == "BUY")
    sell_count = sum(1 for t in trades if t["type"] == "SELL")
    total = len(trades)

    # Performance metrics
    profitable = sum(1 for t in trades if t.get("status") == "TARGET")
    stoploss = sum(1 for t in trades if t.get("status") == "STOPLOSS")
    win_rate = (profitable / total * 100) if total > 0 else 0

    # Highlights (last 5 trades only)
    highlights = []
    for t in trades[-5:]:
        highlights.append(
            f"- {t['symbol']} {t['type']} @ ₹{t['price']:.2f} ({t.get('status','Active')})"
        )

    # Report text
    report = (
        f"📊 Daily Report – {today}\n\n"
        f"🔢 Summary:\n"
        f"- Total Alerts: {total}\n"
        f"- BUY: {buy_count}\n"
        f"- SELL: {sell_count}\n"
        f"- ✅ Profitable: {profitable}\n"
        f"- ❌ Stop Loss: {stoploss}\n"
        f"- Win Rate: {win_rate:.0f}%\n\n"
        f"🔎 Highlights (Last 5 Trades):\n" + "\n".join(highlights)
    )

    return report

# Example: send at 3:30 PM
if datetime.now().strftime("%H:%M") == "15:30":
    msg = generate_daily_report()
    send_telegram(msg)

if __name__=="__main__":
    run()