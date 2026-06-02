# """ bot.py — Sniper Signal Bot

Timeframe:
  D1  → asosiy trend (EMA50/200)
  H1  → oraliq trend (EMA20/50)
  M15 → entry signal (EMA9/21 cross + MACD + Stochastic + Volume)

Entry:
  EMA cross + MACD + Stochastic + Volume tasdiqi

TP:
  keyingi swing high/low

SL:
  so'nggi swing low/high
"""

import os, time, requests, pandas as pd
from datetime import datetime

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_KEY       = os.environ["TWELVE_DATA_KEY"]

SYMBOLS = [
    {"symbol": "XAU/USD", "name": "Oltin",   "type": "forex",  "digits": 2},
    {"symbol": "BTC/USD", "name": "Bitcoin", "type": "crypto", "digits": 1},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def macd_histogram(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def stochastic(df, period=14):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()

    k = 100 * ((df["close"] - low_min) / (high_max - low_min))
    d = k.rolling(3).mean()

    return float(k.iloc[-1]), float(d.iloc[-1])


def swing_high(df: pd.DataFrame, n: int = 5) -> float:
    return float(df["high"].iloc[-n:].max())


def swing_low(df: pd.DataFrame, n: int = 5) -> float:
    return float(df["low"].iloc[-n:].min())


def prev_swing_high(df: pd.DataFrame, n: int = 20) -> float:
    return float(df["high"].iloc[-n:-1].max())


def prev_swing_low(df: pd.DataFrame, n: int = 20) -> float:
    return float(df["low"].iloc[-n:-1].min())


def volume_ok(df):
    if "volume" not in df.columns:
        return True
    return df["volume"].iloc[-1] > df["volume"].tail(20).mean()


# ── API ───────────────────────────────────────────────────────────────────────

_last = 0

def _wait():
    global _last
    gap = time.time() - _last
    if gap < 8:
        time.sleep(8 - gap)
    _last = time.time()


def get_price(symbol: str) -> float | None:
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


def get_candles(symbol: str, interval: str, size: int = 100) -> pd.DataFrame | None:
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": symbol, "interval": interval,
                    "outputsize": size, "apikey": TWELVE_KEY},
            timeout=15
        ).json()

        if "values" not in r:
            return None

        df = pd.DataFrame(r["values"]).iloc[::-1].reset_index(drop=True)
        for c in ["open","high","low","close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.dropna().reset_index(drop=True)

    except:
        return None


# ── Tahlil ────────────────────────────────────────────────────────────────────

def analyze(item: dict) -> dict | None:
    sym    = item["symbol"]
    digits = item["digits"]

    price = get_price(sym)
    if price is None:
        return None

    # D1 trend
    d1 = get_candles(sym, "1day", 220)
    if d1 is None or len(d1) < 200:
        return None

    e50_d1  = float(ema(d1["close"], 50).iloc[-1])
    e200_d1 = float(ema(d1["close"], 200).iloc[-1])

    if e50_d1 > e200_d1 * 1.001:
        d1_trend = "BUY"
    elif e50_d1 < e200_d1 * 0.999:
        d1_trend = "SELL"
    else:
        return None

    # H1 confirm
    h1 = get_candles(sym, "1h", 60)
    if h1 is None:
        return None

    e20_h1 = float(ema(h1["close"], 20).iloc[-1])
    e50_h1 = float(ema(h1["close"], 50).iloc[-1])

    if d1_trend == "BUY" and e20_h1 < e50_h1:
        return None
    if d1_trend == "SELL" and e20_h1 > e50_h1:
        return None

    # M15 entry
    m15 = get_candles(sym, "15min", 60)
    if m15 is None:
        return None

    e9  = ema(m15["close"], 9)
    e21 = ema(m15["close"], 21)

    cross_up = e9.iloc[-2] < e21.iloc[-2] and e9.iloc[-1] >= e21.iloc[-1]
    cross_down = e9.iloc[-2] > e21.iloc[-2] and e9.iloc[-1] <= e21.iloc[-1]

    action = d1_trend

    if action == "BUY" and not cross_up:
        return None
    if action == "SELL" and not cross_down:
        return None

    # MACD + STOCH + VOLUME
    hist = macd_histogram(m15["close"])
    k, d = stochastic(m15)

    macd_buy = hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]
    macd_sell = hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]

    stoch_buy = k > d and k < 30
    stoch_sell = k < d and k > 70

    vol = volume_ok(m15)

    if action == "BUY":
        if not (macd_buy and stoch_buy and vol):
            return None
    else:
        if not (macd_sell and stoch_sell and vol):
            return None

    # SL
    if action == "BUY":
        sl = round(swing_low(m15, n=10) - (price * 0.001), digits)
    else:
        sl = round(swing_high(m15, n=10) + (price * 0.001), digits)

    sl_dist = abs(price - sl)
    if sl_dist == 0:
        return None

    # TP
    if action == "BUY":
        tp1 = round(prev_swing_high(h1, n=30), digits)
        tp2 = round(prev_swing_high(d1, n=20), digits)
    else:
        tp1 = round(prev_swing_low(h1, n=30), digits)
        tp2 = round(prev_swing_low(d1, n=20), digits)

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
        "volume_ok": vol,
        "d1_trend": d1_trend
    }


# ── Telegram ──────────────────────────────────────────────────────────────────

def format_msg(s: dict) -> str:
    e = "🟢" if s["action"] == "BUY" else "🔴"
    act = "SOTIB OL" if s["action"] == "BUY" else "SOT"
    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    d = s["digits"]

    return (
        f"{e} <b>{s['symbol']} — {act}</b>\n"
        f"<i>{s['name']}</i>\n\n"
        f"💰 Entry: <b>{s['price']:.{d}f}</b>\n"
        f"🎯 TP1: <b>{s['tp1']:.{d}f}</b> (R/R 1:{s['rr1']})\n"
        f"🎯 TP2: <b>{s['tp2']:.{d}f}</b>\n"
        f"🛑 SL: <b>{s['sl']:.{d}f}</b>\n\n"
        f"📊 MACD: {round(s['macd'],4)}\n"
        f"📈 Stoch: {round(s['stoch_k'],1)} / {round(s['stoch_d'],1)}\n"
        f"📦 Volume: {'OK' if s['volume_ok'] else 'NO'}\n"
        f"📉 D1 Trend: {s['d1_trend']}\n"
    )


def send(msg: str):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────

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
        main()
        time.sleep(4 * 60 * 60)