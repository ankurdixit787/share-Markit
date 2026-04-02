from ai_utils import train
from config import stocks


def build_models():
    models = {}
    for symbol in stocks:
        model = train(symbol)
        if model:
            models[symbol] = model
    return models
