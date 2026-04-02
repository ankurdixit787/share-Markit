from typing import List, Tuple

from indicators import is_retest_buy, is_retest_sell


def calculate_buy_backbone(price, ma20, vwap_val, vol_ratio, ma20_5m, ma20_1h, nifty, rsi_val, macd_val):
    score = 0
    details: List[str] = []

    if (price > ma20) and (nifty == 1):
        score += 1
        details.append("MA20+NIFTY✔")
    if (rsi_val > 60) and (macd_val > 0):
        score += 1
        details.append("RSI+MACD✔")
    if (price > vwap_val) and (vol_ratio > 1.5):
        score += 1
        details.append("VWAP+VOL✔")
    if (price > ma20_5m) and (price > ma20_1h):
        score += 1
        details.append("MultiTF✔")

    return score, details


def calculate_sell_backbone(price, ma20, vwap_val, vol_ratio, ma20_5m, ma20_1h, nifty, rsi_val, macd_val):
    score = 0
    details: List[str] = []

    if (price < ma20) and (nifty == -1):
        score += 1
        details.append("MA20+NIFTY✔")
    if (rsi_val < 40) and (macd_val < 0):
        score += 1
        details.append("RSI+MACD✔")
    if (price < vwap_val) and (vol_ratio > 1.5):
        score += 1
        details.append("VWAP+VOL✔")
    if (price < ma20_5m) and (price < ma20_1h):
        score += 1
        details.append("MultiTF✔")

    return score, details


def calculate_buy_score(price, df, news_score, prob, roc_val):
    score = 0
    cond_details: List[str] = []

    # 1. AI model probability filter
    # If model probability is strong (>0.75), add to score
    if prob > 0.75:
        score += 1
        cond_details.append("AI✔")


    # 3. News sentiment filter
    # If news_score is positive (>0.2), add to score
    if news_score > 0.2:
        score += 1
        cond_details.append("NEWS✔")


    # 5. ROC (Rate of Change) filter
    # If ROC is positive, add to score
    if roc_val > 0:
        score += 1
        cond_details.append("ROC✔")

    # 7. 15-min Moving Average filter
    # If price is above 15-bar moving average, add to score
    ma_15m = df["Close"].rolling(15).mean().iloc[-1]
    if price > ma_15m:
        score += 1
        cond_details.append("TF15m✔")

    # 8. Volume filter
    # If current volume is 20% higher than 20-bar average, add to score
    vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]
    if vol_ratio > 1.2:
        score += 1
        cond_details.append("VOL✔")

    return score, cond_details


def calculate_sell_score(price, df, news_score, prob, roc_val):
    score = 0
    cond_details: List[str] = []

    if prob < 0.25:
        score += 1
        cond_details.append("AI✔")


    if news_score < -0.2:
        score += 1
        cond_details.append("NEWS✔")


    if roc_val < 0:
        score += 1
        cond_details.append("ROC✔")


    ma_15m = df["Close"].rolling(15).mean().iloc[-1]
    if price < ma_15m:
        score += 1
        cond_details.append("TF15m✔")

    vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]
    if vol_ratio > 1.2:
        score += 1
        cond_details.append("VOL✔")

    return score, cond_details


def build_buy_message(symbol, price, target, sl, candle_ok, volume_ok, retest_ok, backbone_score, backbone_details, score, cond_details, now):
    return (
        f"🚀 BUY ALERT: {symbol}\n"
        f"Price: {price:.2f}\n"
        f"Target: {target:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"Candle{'✔' if candle_ok else '❌'} | "
        f"Volume{'✔' if volume_ok else '❌'} | "
        f"Retest{'✔' if retest_ok else '❌'}\n"
        f"Backbone: {backbone_score}/4 | {' | '.join(backbone_details)}\n"
        f"Score: {score}/5 | {' | '.join(cond_details)}\n"
        f"Time: {now.strftime('%H:%M')}"
    )


def build_sell_message(symbol, price, target, sl, candle_ok, volume_ok, retest_ok, backbone_score, backbone_details, score, cond_details, now):
    return (
        f"⚠️ SELL ALERT: {symbol}\n"
        f"Price: {price:.2f}\n"
        f"Target: {target:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"Candle{'✔' if candle_ok else '❌'} | "
        f"Volume{'✔' if volume_ok else '❌'} | "
        f"Retest{'✔' if retest_ok else '❌'}\n"
        f"Backbone: {backbone_score}/4 | {' | '.join(backbone_details)}\n"
        f"Score: {score}/5 | {' | '.join(cond_details)}\n"
        f"Time: {now.strftime('%H:%M')}"
    )
