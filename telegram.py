import requests

from config import BOT_TOKEN, CHAT_ID


def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        print("📨 Telegram sent")
    except Exception as e:
        print("⚠️ Telegram Error:", e)
