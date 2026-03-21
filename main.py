import time
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---------------- EMAIL ALERT ----------------
def send_alert(message):
    sender = "ankurdixitd@gmail.com"
    receiver = "ankurdixitd@gmail.com"
    msg = MIMEText(message)
    msg["Subject"] = "Stock Alert"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, "ajar rptk oque jncz")  # App Password
            server.sendmail(sender, receiver, msg.as_string())
        print("Alert sent:", message)
    except Exception as e:
        print("Error sending alert:", e)

# ---------------- INDICATORS ----------------
def calculate_rsi(prices, period=14):
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short=12, long=26, signal=9):
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema = prices.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger(prices, window=20):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, lower

# ---------------- AI MODEL (TRAINING) ----------------
def train_ai_model():
    # Historical Sensex data for training
    data = yf.download("^BSESN", period="6mo", interval="1d")
    data["RSI"] = calculate_rsi(data["Close"])
    macd, signal_line = calculate_macd(data["Close"])
    data["MACD"] = macd
    data["MACD_Signal"] = signal_line
    upper, lower = calculate_bollinger(data["Close"])
    data["Upper"] = upper
    data["Lower"] = lower

    # Label: 1 = BUY, 0 = SELL (simple rule for training)
    data["Label"] = np.where((data["Close"] > data["Close"].shift(1)), 1, 0)

    features = data[["Close", "RSI", "MACD", "MACD_Signal", "Upper", "Lower"]].fillna(0)
    labels = data["Label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model

# ---------------- MAIN ALERT SYSTEM ----------------
def run_alert_system(stock_symbol="^BSESN"):
    ai_model = train_ai_model()

    while True:
        data = yf.download(stock_symbol, period="1d", interval="5m")

        price = data["Close"].iloc[-1].item()
        avg = data["Close"].rolling(window=20).mean().iloc[-1].item()
        rsi = calculate_rsi(data["Close"]).iloc[-1].item()
        macd, signal_line = calculate_macd(data["Close"])
        macd_val = macd.iloc[-1].item()
        signal_val = signal_line.iloc[-1].item()
        upper, lower = calculate_bollinger(data["Close"])
        upper_val = upper.iloc[-1].item()
        lower_val = lower.iloc[-1].item()

        print(f"Price: {price:.2f}, Avg: {avg:.2f}, RSI: {rsi:.2f}, MACD: {macd_val:.2f}, Signal: {signal_val:.2f}")

        # AI Prediction
        features = np.array([[price, rsi, macd_val, signal_val, upper_val, lower_val]])
        ai_prob = ai_model.predict_proba(features)[0][1]  # Probability of BUY

        # Strong BUY condition
        if (price > avg) and (rsi > 60) and (macd_val > signal_val) and (price > upper_val) and (ai_prob > 0.8):
            send_alert(f"{stock_symbol}: STRONG BUY! "
                       f"Price {price:.2f} > Avg {avg:.2f}, RSI={rsi:.2f}, MACD Bullish, Bollinger Breakout, "
                       f"AI Confidence={ai_prob:.2f}")

        # Strong SELL condition
        elif (price < avg) and (rsi < 40) and (macd_val < signal_val) and (price < lower_val) and (ai_prob < 0.2):
            send_alert(f"{stock_symbol}: STRONG SELL! "
                       f"Price {price:.2f} < Avg {avg:.2f}, RSI={rsi:.2f}, MACD Bearish, Bollinger Breakdown, "
                       f"AI Confidence={1-ai_prob:.2f}")

        else:
            print("No strong AI signal, skipping notification.")

        time.sleep(60)

if __name__ == "__main__":
    run_alert_system("^BSESN")
