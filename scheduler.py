import time
from datetime import datetime

import pytz

from config import HOLIDAYS_2026, RED
try:
    from config import ALWAYS_RUN
except ImportError:
    ALWAYS_RUN = False
from data_utils import allow_trade_time, get_data, nifty_trend
from engine import evaluate_symbol
from report import generate_daily_report
from runner import build_models
from storage import save_trades, trade_log
from telegram import send_telegram

last_alert = {}
last_report_date = None


def is_market_open(now: datetime) -> bool:
    # MARKET HOURS BYPASS - testing 24/7
    return True
    

def update_trade_statuses(trade_log, now: datetime):
    updated = False
    for trade in trade_log:
        current_status = trade.get("status")
        if current_status not in [None, "Open", "Active"]:
            continue

        side = trade.get("side") or trade.get("type")
        if side not in ["BUY", "SELL"]:
            continue

        if "sl" not in trade or "target" not in trade:
            continue

        try:
            df = get_data(trade["symbol"], "5m", "5d")
            if df.empty or len(df) < 20:
                continue
            price = df["Close"].iloc[-1]
        except Exception:
            continue

        new_status = None
        if side == "BUY":
            if price >= float(trade["target"]):
                new_status = "Profitable"
            elif price <= float(trade["sl"]):
                new_status = "Stop Loss"
        else:
            if price <= float(trade["target"]):
                new_status = "Profitable"
            elif price >= float(trade["sl"]):
                new_status = "Stop Loss"

        if new_status and new_status != current_status:
            trade["status"] = new_status
            trade["close_date"] = now.strftime("%Y-%m-%d")
            trade["close_time"] = now.strftime("%H:%M")
            updated = True
            print(
                f"STATUS UPDATE: {trade['symbol']} {side} -> {new_status} "
                f"price={price:.2f} sl={trade['sl']} target={trade['target']}"
            )

    if updated:
        save_trades(trade_log)


def run():
    global last_report_date
    print(" TOP 1 NUMBER BOT STARTED")
    print(" Training Models...")

    models = build_models()
    print(" System Ready\n")

    ist = pytz.timezone('Asia/Kolkata')
    if datetime.now(ist).strftime("%H:%M") == "15:30":
        msg = generate_daily_report(trade_log)
        send_telegram(msg)

    while True:
        now = datetime.now(ist)
        hour, minute = now.hour, now.minute
        print(f"\n Time: {hour}:{minute}")

        today_str = now.strftime("%Y-%m-%d")
        if today_str in HOLIDAYS_2026 or not is_market_open(now):
            print(" Market Closed or Holiday")
            if hour == 15 and minute >= 30 and last_report_date != today_str:
                print(" Sending Daily Report")
                msg = generate_daily_report(trade_log)
                send_telegram(msg)
                last_report_date = today_str
            time.sleep(60)
            continue

        if not allow_trade_time():
            print(" Time Filter Blocked")
            time.sleep(20)
            continue

        nifty = nifty_trend()

        update_trade_statuses(trade_log, now)

        for symbol, model in models.items():
            try:
                df = get_data(symbol, "5m", "5d")
                if df.empty or len(df) < 20:
                    continue

                actions = evaluate_symbol(symbol, model, df, nifty, last_alert.get(symbol), now)
                for action in actions:
                    trade_log.append(action["trade"])
                    send_telegram(action["msg"])
                    save_trades(trade_log)
                    last_alert[symbol] = action["side"]
            except Exception as e:
                print(f" MAIN ERROR: {e}")

        time.sleep(5)  # Reduced from 20 to 5 seconds for faster alert detection
