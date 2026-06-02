"""
bot.py — Professional Sniper Signal Bot
=======================================

Timeframe:
  D1  → asosiy trend (EMA50/200)
  H1  → trend tasdiqi (EMA20/50)
  M15 → entry signal

Entry:
  EMA9/21 Cross
  + MACD Histogram
  + Stochastic
  + Volume Confirmation

TP:
  H1 swing level
  D1 swing level

SL:
  So'nggi swing low/high

Risk:
  Minimum R/R = 1.5
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_KEY       = os.environ["TWELVE_DATA_KEY"]

SYMBOLS = [
    {"symbol": "XAU/USD", "name": "Oltin", "type": "forex", "digits": 2},
    {"symbol": "BTC/USD", "name": "Bitcoin", "type": "crypto", "digits": 1},
]

# ==========================================================
# INDICATORS
# ==========================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd_histogram(close: pd.Series) -> pd.Series:
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd - signal


def stochastic(df: pd.DataFrame, period: int = 14):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()

    k = 100 * ((df["close"] - low_min) / (high_max - low_min))
    d = k.rolling(3).mean()

    return float(k.iloc[-1]), float(d.iloc[-1])


def swing_high(df: pd.DataFrame, n: int = 10) -> float:
    return float(df["high"].iloc[-n:].max())


def swing_low(df: pd.DataFrame, n: int = 10) -> float:
    return float(df["low"].iloc[-n:].min())


def prev_swing_high(df: pd.DataFrame, n: int = 30) -> float:
    return float(df["high"].iloc[-n:-1].max())


def prev_swing_low(df: pd.DataFrame, n: int = 30) -> float:
    return float(df["low"].iloc[-n:-1].min())


def volume_confirmation(df: pd.DataFrame) -> bool:
    if "volume" not in df.columns:
        return True

    try:
        return df["volume"].iloc[-1] > df["volume"].tail(20).mean()
    except:
        return True


# ==========================================================
# API
# ==========================================================

_last_call = 0

def _wait():
    global _last_call
    gap = time.time() - _last_call
    if gap < 8:
        time.sleep(8 - gap)
    _last_call = time.time()


def get_price(symbol: str):
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/price",
            params={"symbol": symbol, "apikey": TWELVE_KEY},
            timeout=10
        ).json()
        return float(r["price"]) if "price" in r else None
    except:
        return None


def get_candles(symbol: str, interval: str, size: int = 100):
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": size,
                "apikey": TWELVE_KEY
            },
            timeout=15
        ).json()

        if "values" not in r:
            return None

        df = pd.DataFrame(r["values"]).iloc[::-1]

        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.dropna().reset_index(drop=True)

    except:
        return None


# ==========================================================
# ANALYZE
# ==========================================================

def analyze(item: dict):

    symbol = item["symbol"]
    digits = item["digits"]

    price = get_price(symbol)
    if price is None:
        return None

    # D1 TREND
    d1 = get_candles(symbol, "1day", 220)
    if d1 is None:
        return None

    ema50 = ema(d1["close"], 50).iloc[-1]
    ema200 = ema(d1["close"], 200).iloc[-1]

    if ema50 > ema200:
        d1_trend = "BUY"
    elif ema50 < ema200:
        d1_trend = "SELL"
    else:
        return None

    # H1 CONFIRM
    h1 = get_candles(symbol, "1h", 100)
    if h1 is None:
        return None

    if d1_trend == "BUY" and ema(h1["close"], 20).iloc[-1] <= ema(h1["close"], 50).iloc[-1]:
        return None

    if d1_trend == "SELL" and ema(h1["close"], 20).iloc[-1] >= ema(h1["close"], 50).iloc[-1]:
        return None

    # M15 ENTRY
    m15 = get_candles(symbol, "15min", 80)
    if m15 is None:
        return None

    ema9 = ema(m15["close"], 9)
    ema21 = ema(m15["close"], 21)

    cross_up = ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] >= ema21.iloc[-1]
    cross_down = ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] <= ema21.iloc[-1]

    action = d1_trend

    if action == "BUY" and not cross_up:
        return None
    if action == "SELL" and not cross_down:
        return None

    # MACD + STOCH + VOLUME
    hist = macd_histogram(m15["close"])
    k, d = stochastic(m15)

    macd_bull = hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]
    macd_bear = hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]

    stoch_bull = k > d and k < 30
    stoch_bear = k < d and k > 70

    vol_ok = volume_confirmation(m15)

    if action == "BUY":
        if not (macd_bull and stoch_bull and vol_ok):
            return None
    else:
        if not (macd_bear and stoch_bear and vol_ok):
            return None

    # SL
    if action == "BUY":
        sl = swing_low(m15) * 0.999
    else:
        sl = swing_high(m15) * 1.001

    sl = round(sl, digits)

    # TP
    if action == "BUY":
        tp1 = prev_swing_high(h1)
        tp2 = prev_swing_high(d1)
    else:
        tp1 = prev_swing_low(h1)
        tp2 = prev_swing_low(d1)

    tp1 = round(tp1, digits)
    tp2 = round(tp2, digits)

    sl_dist = abs(price - sl)
    rr1 = abs(tp1 - price) / sl_dist if sl_dist else 0

    if rr1 < 1.5:
        return None

    return {
        **item,
        "action": action,
        "price": round(price, digits),
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr1": round(rr1, 2),
        "macd": float(hist.iloc[-1]),
        "stoch_k": k,
        "stoch_d": d,
        "d1_trend": d1_trend
    }


# ==========================================================
# TELEGRAM
# ==========================================================

def send(msg):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }
    )


def format_msg(s):

    e = "🟢" if s["action"] == "BUY" else "🔴"

    return (
        f"{e} <b>{s['symbol']} — {s['action']}</b>\n\n"
        f"Entry: {s['price']}\n"
        f"TP1: {s['tp1']} | TP2: {s['tp2']}\n"
        f"SL: {s['sl']}\n\n"
        f"MACD: {round(s['macd'],4)}\n"
        f"Stoch: {round(s['stoch_k'],1)} / {round(s['stoch_d'],1)}\n"
        f"D1: {s['d1_trend']}"
    )


# ==========================================================
# MAIN LOOP
# ==========================================================

def main():

    send("🔍 Bot Started")

    signals = []

    for s in SYMBOLS:
        res = analyze(s)
        if res:
            signals.append(res)

    if signals:
        for s in signals:
            send(format_msg(s))
            time.sleep(1)
    else:
        send("No signal")

if __name__ == "__main__":

    while True:
        try:
            main()
        except Exception as e:
            print(e)

        time.sleep(4 * 60 * 60)