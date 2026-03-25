import time
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import requests
from datetime import datetime
import pytz
import threading

# ---------- TELEGRAM ----------
BOT_TOKEN = "PASTE_YOUR_TOKEN"
CHAT_ID = "PASTE_YOUR_CHAT_ID"

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print(f"[Telegram] send failed: {e}")

# ---------- STOCKS ----------
stocks = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
          "BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS","TITAN.NS"]

# ---------- MEMORY STORAGE ----------
trade_log = []
lock = threading.Lock()
last_alert = {}

# ---------- DATA & INDICATORS ----------
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

def rsi(close):
    try:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.ewm(alpha=1/14).mean() / loss.ewm(alpha=1/14).mean()
        return 100 - (100/(1+rs))
    except:
        return pd.Series([50]*len(close))

def macd(close):
    try:
        return close.ewm(span=12).mean() - close.ewm(span=26).mean()
    except:
        return pd.Series([0]*len(close))

def atr(df):
    try:
        tr = pd.concat([
            df["High"] - df["Low"],
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(14).mean()
    except:
        return pd.Series([1]*len(df))

def news_sentiment(symbol):
    try:
        news = yf.Ticker(symbol).news
        if not news: return 0
        score,count = 0,0
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
    try:
        df = get_data(symbol,"1d","6mo")
        if df.empty: return None
        df["RSI"] = rsi(df["Close"])
        df["MACD"] = macd(df["Close"])
        df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
        df = df.dropna()
        if df.empty: return None
        X = df[["RSI","MACD"]]
        y = df["Target"]
        model = RandomForestClassifier(n_estimators=80)
        model.fit(X,y)
        print(f"[{symbol}] Model trained")
        return model
    except:
        print(f"[{symbol}] Training failed")
        return None

def predict(model, df):
    try:
        df = df.copy()
        df["RSI"] = rsi(df["Close"])
        df["MACD"] = macd(df["Close"])
        df = df.dropna()
        if df.empty: return None
        last = df.iloc[-1]
        X = pd.DataFrame({"RSI":[float(last["RSI"])],"MACD":[float(last["MACD"])]})
        return model.predict_proba(X)[0][1]
    except:
        return None

def nifty_trend():
    try:
        df = get_data("^NSEI","5m","5d")
        if df.empty: return 0
        price = df["Close"].iloc[-1]
        ma = df["Close"].rolling(20).mean().iloc[-1]
        return 1 if price>ma else -1
    except:
        return 0

# ---------- DAILY REPORT ----------
def day_end_report():
    try:
        with lock:
            if not trade_log:
                send_telegram("📊 DAY END REPORT\nNo trades today.")
                return
            total = len(trade_log)
            profit = sum(1 for t in trade_log if t.get("result")=="TARGET")
            loss = sum(1 for t in trade_log if t.get("result")=="SL")
            net_pl = sum(t.get("pl",0) for t in trade_log)
            msg=f"📊 DAY END REPORT\nTotal Trades: {total}\nProfit: {profit} ✅\nLoss: {loss} ❌\nNet P/L: ₹{net_pl}\n\n"
            for t in trade_log:
                msg+=f"{t['symbol']} | Entry: ₹{t['price']} | Target: ₹{t['target']} | SL: ₹{t['sl']} | Score: {t['score']}/6 | Accuracy: {t.get('accuracy',0)}% | Result: {t.get('result','OPEN')} | P/L: ₹{t.get('pl',0)}\n"
            send_telegram(msg)
    except:
        pass

# ---------- ALERT LOOP ----------
def alert_loop(symbol, model):
    ist = pytz.timezone("Asia/Kolkata")
    while True:
        try:
            now=datetime.now(ist)
            hour,minute=now.hour,now.minute
            market_open=(hour>9 or (hour==9 and minute>=20)) and (hour<15 or (hour==15 and minute<=15))
            if not market_open:
                time.sleep(10)
                continue
            df=get_data(symbol,"5m","5d")
            if df.empty or len(df)<50:
                time.sleep(5)
                continue
            price=float(df["Close"].iloc[-1])
            prob=predict(model,df)
            if prob is None: time.sleep(5); continue
            accuracy=int(prob*100)
            atr_val=float(atr(df).iloc[-1])
            target=round(price+1.5*atr_val,2)
            sl=round(price-atr_val,2)
            entry=round(price,2)
            score=6 if accuracy>=90 else 0
            with lock:
                # Alert only if accuracy >=90 and not sent yet
                if accuracy>=90 and last_alert.get(symbol)!="BUY":
                    msg=f"📈 BUY SIGNAL\nStock: {symbol}\nEntry: ₹{entry}\nTarget: ₹{target}\nSL: ₹{sl}\nScore: {score}/6\nAccuracy: {accuracy}%"
                    send_telegram(msg)
                    trade_log.append({"symbol":symbol,"price":entry,"target":target,"sl":sl,"score":score,"type":"BUY","result":"OPEN","pl":0,"accuracy":accuracy})
                    last_alert[symbol]="BUY"
                # Auto-update P/L
                for t in trade_log:
                    if t["result"]=="OPEN" and t["symbol"]==symbol:
                        if price>=t["target"]:
                            t["result"]="TARGET"; t["pl"]=round(t["target"]-t["price"],2)
                        elif price<=t["sl"]:
                            t["result"]="SL"; t["pl"]=round(t["sl"]-t["price"],2)
        except:
            time.sleep(5)
        time.sleep(5)

# ---------- RUN ----------
def run():
    print("🔥 Training Models...")
    models={}
    for s in stocks:
        m=train(s)
        if m:
            models[s]=m
    print("✅ System Ready")
    send_telegram("✅ Bot Started")
    # Start alert threads
    for s in models:
        t=threading.Thread(target=alert_loop,args=(s,models[s]))
        t.daemon=True
        t.start()
    # Day-end report monitor
    ist=pytz.timezone("Asia/Kolkata")
    report_sent=False
    reset_done=False
    while True:
        try:
            now=datetime.now(ist)
            hour,minute=now.hour,now.minute
            if not reset_done and hour==9 and minute<10:
                with lock:
                    trade_log.clear()
                    last_alert.clear()
                    report_sent=False
                reset_done=True
            if not report_sent and hour==15 and minute>=20:
                day_end_report()
                report_sent=True
        except:
            pass
        time.sleep(10)

if __name__=="__main__":
    run()