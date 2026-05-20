from textblob import TextBlob
import yfinance as yf


def news_sentiment(symbol):
    try:
        news = yf.Ticker(symbol).news
        if not news:
            return 0

        score, count = 0, 0
        for n in news[:3]:
            title = n.get("title", "") if isinstance(n, dict) else str(n)
            if title:
                score += TextBlob(title).sentiment.polarity
                count += 1

        return score / count if count else 0
    except Exception:
        return 0
