ENABLE_RETEST_ALERT = True  # Set to False to disable breakout+retest only alerts
from typing import List
from datetime import datetime

from ai_utils import predict
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
    # --- Breakout + Retest Only Alert Logic ---
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


    # Multi-timeframe moving averages
    df_5m = get_ohlc(symbol=symbol, interval="5m", lookback=50)
    ma20_5m = df_5m["Close"].rolling(20).mean().iloc[-1].item()
    df_1h = get_ohlc(symbol=symbol, interval="1h", lookback=50)
    ma20_1h = df_1h["Close"].rolling(20).mean().iloc[-1].item()


    # List to collect trade actions (buy/sell signals)
    actions = []

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
        atr_val,
    )
    print(f"⭐ Buy Optional Score: {buy_score}")
    print(f"Details: {buy_cond_details}")
    print("==============================\n")

    # BUY signal: All backbone filters pass, enough optional filters, and last alert not BUY
    if buy_backbone_score == 4 and buy_score >= 4 and last_alert_side != "BUY":
        entry_price = high_price
        sl = entry_price - 1.5 * atr_val
        target = entry_price + 2 * atr_val
        candle_ok = df["Close"].iloc[-1] > df["Open"].iloc[-1]
        volume_ok = df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1]
        retest_ok = is_retest_buy(df)

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
        atr_val,
    )
    print(f"⭐ Sell Optional Score: {sell_score}")
    print(f"Details: {sell_cond_details}")
    print("==============================\n")

    # SELL signal: All backbone filters pass, enough optional filters, and last alert not SELL
    if sell_backbone_score == 4 and sell_score >= 4 and last_alert_side != "SELL":
        entry_price = low_price
        target = entry_price - 2 * atr_val
        sl = entry_price + atr_val
        candle_ok = df["Close"].iloc[-1] > df["Open"].iloc[-1]
        volume_ok = df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1]
        retest_ok = is_retest_sell(df)

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
                candle_ok,
                volume_ok,
                retest_ok,
                sell_backbone_score,
                sell_backbone_details,
                sell_score,
                sell_cond_details,
                now,
            ),
        })

    # Return all generated trade actions (buy/sell signals)
    return actions
