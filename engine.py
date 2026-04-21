ENABLE_RETEST_ALERT = False  # Set to False to disable breakout+retest only alerts (disabled - false alerts issue)
from typing import List
from datetime import datetime

from ai_utils import predict
from telegram import send_telegram
from alerts import (
    build_buy_message,
    build_sell_message,
    calculate_buy_backbone,
    calculate_buy_score,
    calculate_sell_backbone,
    calculate_sell_score,
)
from data_utils import get_ohlc
from indicators import atr, is_retest_buy, is_retest_sell, macd, rsi, vwap
from news import news_sentiment


def evaluate_symbol(symbol: str, model, df, nifty: int, last_alert_side: str, now: datetime) -> List[dict]:
    # --- Big News Alert Logic ---
    try:
        import yfinance as yf
        news = yf.Ticker(symbol).news
        important_keywords = [
            "crash", "ban", "record high", "all-time high", "scam", "merger", "acquisition", "policy change",
            "fraud", "default", "bankruptcy", "fire", "strike", "investigation", "raid", "tax", "fine", "penalty",
            "court", "lawsuit", "regulation", "approval", "deal", "partnership", "IPO", "listing", "delisting"
        ]
        big_news_found = False
        big_news_title = ""
        for n in news[:5]:
            title = n.get("title", "") if isinstance(n, dict) else str(n)
            if title:
                title_lower = title.lower()
                if any(k in title_lower for k in important_keywords):
                    big_news_found = True
                    big_news_title = title
                    break
        news_score = news_sentiment(symbol)
        if big_news_found or news_score > 0.7 or news_score < -0.7:
            if news_score > 0.3:
                suggestion = "BUY"
            elif news_score < -0.3:
                suggestion = "SELL"
            else:
                suggestion = "WAIT"
            msg = (
                f"📰 BIG NEWS ALERT: {symbol}\n"
                f"Title: {big_news_title if big_news_found else 'Sentiment Spike'}\n"
                f"Sentiment: {news_score:.2f}\n"
                f"Suggestion: {suggestion}\n"
                f"Time: {now.strftime('%H:%M')}"
            )
            send_telegram(msg)
    except Exception:
        pass

    # --- Flag Pattern Detection removed as per user request ---
    # --- Breakout + Retest Only Alert Logic ---
    
    # Initialize actions list FIRST (before any use)
    actions = []
    
    if ENABLE_RETEST_ALERT and len(df) > 22:
        # BUY: Breakout candle, then retest candle
        last_high = df["High"].rolling(20).max().iloc[-3]
        breakout_candle = df.iloc[-2]
        retest_candle = df.iloc[-1]
        # Breakout: previous candle closes above last_high, previous-1 closes below
        breakout = (
            breakout_candle["Close"] > last_high and
            df["Close"].iloc[-3] < last_high
        )
        # Retest: current candle low <= last_high and close > last_high
        retest = (
            retest_candle["Low"] <= last_high and
            retest_candle["Close"] > last_high
        )
        if breakout and retest:
            entry_price = retest_candle["Close"]
            sl = entry_price - 1.5 * atr(df).iloc[-1]
            target = entry_price + 2 * atr(df).iloc[-1]
            actions.append({
                "side": "BUY",
                "trade": {
                    "symbol": symbol,
                    "side": "BUY",
                    "entry": entry_price,
                    "sl": sl,
                    "target": target,
                    "status": "Active",
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M"),
                },
                "msg": f"🚀 BREAKOUT+RETEST BUY ALERT: {symbol}\nPrice: {entry_price:.2f}\nTarget: {target:.2f}\nStop Loss: {sl:.2f}\nBreakout: {last_high:.2f}\nTime: {now.strftime('%H:%M')}\n(Score/Backbone ignored)",
            })

        # SELL: Breakdown candle, then retest candle
        last_low = df["Low"].rolling(20).min().iloc[-3]
        breakdown = (
            breakout_candle["Close"] < last_low and
            df["Close"].iloc[-3] > last_low
        )
        retest_sell = (
            retest_candle["High"] >= last_low and
            retest_candle["Close"] < last_low
        )
        if breakdown and retest_sell:
            entry_price = retest_candle["Close"]
            sl = entry_price + atr(df).iloc[-1]
            target = entry_price - 2 * atr(df).iloc[-1]
            actions.append({
                "side": "SELL",
                "trade": {
                    "symbol": symbol,
                    "side": "SELL",
                    "entry": entry_price,
                    "sl": sl,
                    "target": target,
                    "status": "Active",
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M"),
                },
                "msg": f"⚠️ BREAKDOWN+RETEST SELL ALERT: {symbol}\nPrice: {entry_price:.2f}\nTarget: {target:.2f}\nStop Loss: {sl:.2f}\nBreakdown: {last_low:.2f}\nTime: {now.strftime('%H:%M')}\n(Score/Backbone ignored)",
            })


    # Return empty if not enough data for analysis
    if df.empty or len(df) < 20:
        return []



    # Latest close price, high, and low
    close_price = df["Close"].iloc[-1].item()
    high_price = df["High"].iloc[-1].item()
    low_price = df["Low"].iloc[-1].item()
    # 20-period moving average
    ma20 = df["Close"].rolling(20).mean().iloc[-1].item()
    # Volume ratio (current/average)
    vol_ratio = df["Volume"].iloc[-1].item() / df["Volume"].rolling(20).mean().iloc[-1].item()


    # Technical indicators
    rsi_val = rsi(df["Close"]).iloc[-1].item()  # Relative Strength Index
    atr_val = atr(df).iloc[-1].item()            # Average True Range
    vwap_val = vwap(df).iloc[-1].item()          # Volume Weighted Average Price


    # MACD (Moving Average Convergence Divergence)
    macd_series = macd(df["Close"])
    macd_val = macd_series.iloc[-1].item() if not macd_series.empty else 0.0

    # ROC (Rate of Change)
    roc_series = df["Close"].pct_change()
    roc_val = roc_series.iloc[-1].item() if not roc_series.empty else 0.0


    # News sentiment score and AI model probability
    news_score = news_sentiment(symbol)
    prob = predict(model, df)
    if prob is None:
        return []  # Skip if model can't predict


    # Multi-timeframe moving averages (derived from main 1-minute data)
    # Instead of 3 separate API calls, calculate from existing df
    try:
        # 5-minute MA: aggregate every 5 candles
        df_5m_agg = df.iloc[::5].copy() if len(df) >= 5 else df
        ma20_5m = df_5m_agg["Close"].rolling(20).mean().iloc[-1].item() if len(df_5m_agg) >= 20 else ma20
    except:
        ma20_5m = ma20  # Fallback to main MA if aggregation fails
    
    try:
        # 1-hour MA: aggregate every 60 candles (60 min candles = 1 hour)
        df_1h_agg = df.iloc[::60].copy() if len(df) >= 60 else df
        ma20_1h = df_1h_agg["Close"].rolling(20).mean().iloc[-1].item() if len(df_1h_agg) >= 20 else ma20
    except:
        ma20_1h = ma20  # Fallback to main MA if aggregation fails

    # Calculate backbone score for BUY setup (core filters)
    buy_backbone_score, buy_backbone_details = calculate_buy_backbone(
        high_price,
        ma20,
        vwap_val,
        vol_ratio,
        ma20_5m,
        ma20_1h,
        nifty,
        rsi_val,
        macd_val,
    )
    print("\n========== BUY CHECK ==========")
    print(f"Symbol: {symbol}")
    print(f"🦴 Buy Backbone Score: {buy_backbone_score}/4")
    print(f"Details: {buy_backbone_details}")

    # Calculate optional filters score for BUY setup
    buy_score, buy_cond_details = calculate_buy_score(
        high_price,
        df,
        news_score,
        prob,
        roc_val,
    )
    print(f"⭐ Buy Optional Score: {buy_score}")
    print(f"Details: {buy_cond_details}")
    print("==============================\n")

    # BUY signal: At least 2 backbone and 2 score filters pass, all 3 conditions True, and last alert not BUY
    candle_ok = df["Close"].iloc[-1] > df["Open"].iloc[-1]
    volume_ok = df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1]
    retest_ok = is_retest_buy(df)
    # Retest is now optional: alert triggers even if retest_ok is False
    if buy_backbone_score == 4 and buy_score >= 4 and last_alert_side != "BUY" and candle_ok and volume_ok:
        entry_price = high_price
        sl = entry_price - 1.5 * atr_val
        target = entry_price + 2 * atr_val
        actions.append({
            "side": "BUY",
            "trade": {
                "symbol": symbol,
                "side": "BUY",
                "entry": entry_price,
                "sl": sl,
                "target": target,
                "status": "Active",
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M"),
            },
            "msg": build_buy_message(
                symbol,
                entry_price,
                target,
                sl,
                candle_ok,
                volume_ok,
                retest_ok,
                buy_backbone_score,
                buy_backbone_details,
                buy_score,
                buy_cond_details,
                now,
            ),
        })

    # Calculate backbone score for SELL setup (core filters)
    sell_backbone_score, sell_backbone_details = calculate_sell_backbone(
        low_price,
        ma20,
        vwap_val,
        vol_ratio,
        ma20_5m,
        ma20_1h,
        nifty,
        rsi_val,
        macd_val,
    )
    print("========== SELL CHECK ==========")
    print(f"Symbol: {symbol}")
    print(f"🦴 Sell Backbone Score: {sell_backbone_score}/4")
    print(f"Details: {sell_backbone_details}")

    # Calculate optional filters score for SELL setup
    sell_score, sell_cond_details = calculate_sell_score(
        low_price,
        df,
        news_score,
        prob,
        roc_val,
    )
    print(f"⭐ Sell Optional Score: {sell_score}")
    print(f"Details: {sell_cond_details}")
    print("==============================\n")

    # SELL signal: At least 2 backbone and 2 score filters pass, all 3 conditions True, and last alert not SELL
    candle_ok_sell = df["Close"].iloc[-1] > df["Open"].iloc[-1]
    volume_ok_sell = df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1]
    retest_ok_sell = is_retest_sell(df)
    # Retest is now optional: alert triggers even if retest_ok_sell is False
    if sell_backbone_score == 4 and sell_score >= 4 and last_alert_side != "SELL" and candle_ok_sell and volume_ok_sell:
        entry_price = low_price
        target = entry_price - 2 * atr_val
        sl = entry_price + atr_val
        actions.append({
            "side": "SELL",
            "trade": {
                "symbol": symbol,
                "side": "SELL",
                "entry": entry_price,
                "sl": sl,
                "target": target,
                "status": "Active",
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M"),
            },
            "msg": build_sell_message(
                symbol,
                entry_price,
                target,
                sl,
                candle_ok_sell,
                volume_ok_sell,
                retest_ok_sell,
                sell_backbone_score,
                sell_backbone_details,
                sell_score,
                sell_cond_details,
                now,
            ),
        })

    # Return all generated trade actions (buy/sell signals)
    return actions


def check_pcr_drop(index_name, prev_pcr, curr_pcr):
    """
    Checks PCR drop for Nifty/Sensex and sends notification if drop crosses thresholds.
    index_name: 'Nifty' or 'Sensex'
    prev_pcr: Previous PCR value (float)
    curr_pcr: Current PCR value (float)
    """
    if prev_pcr == 0:
        return  # Avoid division by zero
    drop_pct = ((prev_pcr - curr_pcr) / prev_pcr) * 100
    thresholds = [15, 30, 40]
    for threshold in thresholds:
        if drop_pct >= threshold:
            suggestion = "Consider BUY (bullish signal)"  # PCR drop usually means call activity increases
            send_telegram(f"{index_name} PCR down {int(drop_pct)}% 📉\nCall option activity increases. Threshold: {threshold}%\n{suggestion}")
            break  # Only notify for the highest threshold crossed


def check_pcr_alerts(index_name, prev_pcr, curr_pcr):
    """
    Checks PCR drop/rise for Nifty/Sensex and sends notification if change crosses thresholds.
    index_name: 'Nifty' or 'Sensex'
    prev_pcr: Previous PCR value (float)
    curr_pcr: Current PCR value (float)
    """
    if prev_pcr == 0:
        return  # Avoid division by zero
    change_pct = ((curr_pcr - prev_pcr) / prev_pcr) * 100
    thresholds = [15, 30, 40]
    # Bullish: PCR drops
    if change_pct <= 0:
        for threshold in thresholds:
            if abs(change_pct) >= threshold:
                suggestion = "Consider BUY (bullish signal)"  # PCR drop means call activity increases
                send_telegram(f"{index_name} PCR down {int(abs(change_pct))}% 📉\nCall option activity increases. Threshold: {threshold}%\n{suggestion}")
                break
    # Bearish: PCR rises
    else:
        for threshold in thresholds:
            if change_pct >= threshold:
                suggestion = "Consider SELL (bearish signal)"  # PCR rise means put activity increases
                send_telegram(f"{index_name} PCR up {int(change_pct)}% 📈\nPut option activity increases. Threshold: {threshold}%\n{suggestion}")
                break


# Example PCR values (replace with real data source)
    prev_nifty_pcr = 1.2  # Replace with actual previous value
    curr_nifty_pcr = 0.9  # Replace with actual current value
    prev_sensex_pcr = 1.1  # Replace with actual previous value
    curr_sensex_pcr = 0.8  # Replace with actual current value

    # Check PCR drop for both indices before buy/sell logic
    check_pcr_drop("Nifty", prev_nifty_pcr, curr_nifty_pcr)
    check_pcr_drop("Sensex", prev_sensex_pcr, curr_sensex_pcr)
