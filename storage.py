import json

from config import LOG_FILE


def load_trades():
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_trades(data):
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(data, f)
        print(f"✅ Saved Trades: {len(data)}")
    except Exception as e:
        print("⚠️ Save Error:", e)


trade_log = load_trades()
