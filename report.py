from datetime import datetime


def generate_daily_report(trade_log):
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        today_trades = [t for t in trade_log if t.get("date") == today]
        total_alerts = len(today_trades)
        buy_count = sum(
            1 for t in today_trades if (t.get("side") or t.get("type")) == "BUY"
        )
        sell_count = sum(
            1 for t in today_trades if (t.get("side") or t.get("type")) == "SELL"
        )

        profitable = sum(1 for t in today_trades if t.get("status") == "Profitable")
        stop_loss = sum(1 for t in today_trades if t.get("status") == "Stop Loss")
        active = sum(1 for t in today_trades if t.get("status") == "Active")
        open_count = sum(1 for t in today_trades if not t.get("status") or t.get("status") == "Open")

        closed_trades = profitable + stop_loss
        if closed_trades > 0:
            win_rate = f"{profitable / closed_trades * 100:.1f}%"
        else:
            win_rate = "No closed trades yet"

        report = (
            f"📊 Daily Report – {datetime.now().strftime('%d %B %Y')}\n"
            f"{'-'*40}\n"
            f"🔢 Summary:\n"
            f"- Total Alerts: {total_alerts}\n"
            f"- BUY: {buy_count}\n"
            f"- SELL: {sell_count}\n"
            f"- ✅ Profitable: {profitable}\n"
            f"- ❌ Stop Loss: {stop_loss}\n"
            f"- 📂 Active: {active}\n"
            f"- 🟡 Open / No Status: {open_count}\n"
            f"- Win Rate: {win_rate}\n"
        )

        return report
    except Exception as e:
        return f"❌ Error generating report: {e}"
