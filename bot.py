"""
bot.py — XAU/USD Sniper Scalping Bot
======================================
Tahlil:  H4 (trend) + H1 (zona)
Entry:   M15 + M5 (EMA9/21 crossover)
Signal:  EMA9/21 + RSI + MACD line
SL:      0.7 × ATR (M15)
TP1:     H1 swing high/low
TP2:     H4 swing high/low
TP3:     H4 keyingi kuchli level
Cron:    0 */4 * * * python3 bot.py
"""

import os, time, requests, pandas as pd
from datetime import datetime

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_KEY       = os.environ["TWELVE_DATA_KEY"]

SYMBOL = "XAU/USD"
DIGITS = 2

# =========================
# Indikatorlar
# =========================

def ema_list(values: list, period: int) -> list:
    """JS dagi EMA logikasi (list)."""
    k = 2 / (period + 1)
    result = [values[0]]
    for i in range(1, len(values)):
        result.append(values[i] * k + result[i - 1] * (1 - k))
    return result

def ema(s: pd.Series, p: int) -> pd.Series:
    """Pandas Series uchun EMA (swing/ATR uchun)."""
    return s.ewm(span=p, adjust=False).mean()

def rsi(values: list, period: int = 14) -> float:
    """JS dagi RSI logikasi."""
    gains, losses = [], []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period or 1
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, p: int = 14) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(p).mean().dropna()
    return float(val.iloc[-1]) if len(val) > 0 else 1.0

def swing_highs(df: pd.DataFrame, n: int = 3) -> list:
    levels = []
    for i in range(n, len(df) - n):
        h = df["high"].iloc[i]
        if all(h > df["high"].iloc[i-j] for j in range(1, n+1)) and \
           all(h > df["high"].iloc[i+j] for j in range(1, n+1)):
            levels.append(h)
    return sorted(set(round(x, DIGITS) for x in levels))

def swing_lows(df: pd.DataFrame, n: int = 3) -> list:
    levels = []
    for i in range(n, len(df) - n):
        l = df["low"].iloc[i]
        if all(l < df["low"].iloc[i-j] for j in range(1, n+1)) and \
           all(l < df["low"].iloc[i+j] for j in range(1, n+1)):
            levels.append(l)
    return sorted(set(round(x, DIGITS) for x in levels))

def next_level_above(levels: list, price: float):
    above = [l for l in levels if l > price * 1.0005]
    return min(above) if above else None

def next_level_below(levels: list, price: float):
    below = [l for l in levels if l < price * 0.9995]
    return max(below) if below else None

# =========================
# Twelve Data API
# =========================

_last = 0

def _wait():
    global _last
    gap = time.time() - _last
    if gap < 8:
        time.sleep(8 - gap)
    _last = time.time()

def get_price() -> float | None:
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/price",
            params={"symbol": SYMBOL, "apikey": TWELVE_KEY},
            timeout=10
        ).json()
        return float(r["price"]) if "price" in r else None
    except:
        return None

def get_candles(interval: str, size: int) -> pd.DataFrame | None:
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol":     SYMBOL,
                "interval":   interval,
                "outputsize": size,
                "apikey":     TWELVE_KEY,
            },
            timeout=15
        ).json()
        if "values" not in r:
            print(f"  [{interval}] {r.get('message','?')}")
            return None
        df = pd.DataFrame(r["values"]).iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    except Exception as e:
        print(f"  [{interval}] {e}")
        return None

# =========================
# Tahlil
# =========================

def analyze() -> dict | None:

    price = get_price()
    if price is None:
        print("  Narx olinmadi")
        return None
    print(f"  Narx: {price}")

    # ── H4: trend ma'lumotlari (swing TP uchun) ───────────────────────────────
    h4 = get_candles("4h", 250)
    if h4 is None or len(h4) < 50:
        print("  H4 yetarli emas")
        return None

    # ── H1: swing TP1 uchun ───────────────────────────────────────────────────
    h1 = get_candles("1h", 100)
    if h1 is None or len(h1) < 50:
        print("  H1 yetarli emas")
        return None

    # ── M15: asosiy signal hisoblash ──────────────────────────────────────────
    m15 = get_candles("15min", 100)
    if m15 is None or len(m15) < 30:
        print("  M15 yetarli emas")
        return None

    close_prices = m15["close"].tolist()
    last_price   = close_prices[-1]

    # EMA9 / EMA21
    e9  = ema_list(close_prices, 9)
    e21 = ema_list(close_prices, 21)
    last_ema9  = e9[-1]
    last_ema21 = e21[-1]

    # RSI
    last_rsi = rsi(close_prices)

    # MACD line
    e12       = ema_list(close_prices, 12)
    e26       = ema_list(close_prices, 26)
    macd_line = e12[-1] - e26[-1]

    # ── Signal sharti (JS kodidan aynan) ──────────────────────────────────────
    if last_ema9 > last_ema21 and last_rsi < 70 and macd_line > 0:
        trend = "BUY"
    elif last_ema9 < last_ema21 and last_rsi > 30 and macd_line < 0:
        trend = "SELL"
    else:
        print(f"  Signal yo'q | RSI:{last_rsi:.1f} MACD:{macd_line:.2f}")
        return None

    # ── ATR → SL ──────────────────────────────────────────────────────────────
    atr_val = atr(m15, 14)
    sl_dist = round(atr_val * 0.7, DIGITS)

    sl = round(last_price - sl_dist, DIGITS) if trend == "BUY" \
         else round(last_price + sl_dist, DIGITS)

    # ── Swing levellardan TP ──────────────────────────────────────────────────
    h1_highs = swing_highs(h1, n=3)
    h1_lows  = swing_lows(h1,  n=3)
    h4_highs = swing_highs(h4, n=3)
    h4_lows  = swing_lows(h4,  n=3)

    if trend == "BUY":
        tp1 = next_level_above(h1_highs, last_price)
        tp2 = next_level_above(h4_highs, last_price)
        h4_above = sorted([l for l in h4_highs if l > last_price * 1.0005])
        tp3 = h4_above[1] if len(h4_above) >= 2 else None

        if tp1 is None: tp1 = round(last_price + atr_val * 2.0, DIGITS)
        if tp2 is None: tp2 = round(last_price + atr_val * 3.0, DIGITS)
        if tp3 is None: tp3 = round(last_price + atr_val * 4.0, DIGITS)

        tp1 = min(tp1, tp2, tp3)
        tp3 = max(tp1, tp2, tp3)
        tp2 = sorted([tp1, tp2, tp3])[1]
    else:
        tp1 = next_level_below(h1_lows, last_price)
        tp2 = next_level_below(h4_lows, last_price)
        h4_below = sorted([l for l in h4_lows if l < last_price * 0.9995], reverse=True)
        tp3 = h4_below[1] if len(h4_below) >= 2 else None

        if tp1 is None: tp1 = round(last_price - atr_val * 2.0, DIGITS)
        if tp2 is None: tp2 = round(last_price - atr_val * 3.0, DIGITS)
        if tp3 is None: tp3 = round(last_price - atr_val * 4.0, DIGITS)

        tp1 = max(tp1, tp2, tp3)
        tp3 = min(tp1, tp2, tp3)
        tp2 = sorted([tp1, tp2, tp3], reverse=True)[1]

    def rr(tp):
        return round(abs(tp - last_price) / sl_dist, 1) if sl_dist > 0 else 0

    return {
        "action":   trend,
        "price":    round(last_price, DIGITS),
        "sl":       sl,
        "tp1":      tp1,
        "tp2":      tp2,
        "tp3":      tp3,
        "rr1":      rr(tp1),
        "rr2":      rr(tp2),
        "rr3":      rr(tp3),
        "atr":      round(atr_val, DIGITS),
        "rsi":      round(last_rsi, 2),
        "macd":     round(macd_line, 2),
    }

# =========================
# Telegram
# =========================

def format_msg(s: dict) -> str:
    e   = "🟢" if s["action"] == "BUY" else "🔴"
    act = "SOTIB OL" if s["action"] == "BUY" else "SOT"
    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    tr  = "📈 Uptrend" if s["action"] == "BUY" else "📉 Downtrend"

    return (
        f"{e} <b>XAU/USD — {act}</b> 🥇\n"
        f"<i>Oltin</i>\n\n"
        f"💰 Entry:  <b>{s['price']:.2f}</b>\n"
        f"🎯 TP1:   <b>{s['tp1']:.2f}</b>  (1:{s['rr1']}R)\n"
        f"🎯 TP2:   <b>{s['tp2']:.2f}</b>  (1:{s['rr2']}R)\n"
        f"🎯 TP3:   <b>{s['tp3']:.2f}</b>  (1:{s['rr3']}R)\n"
        f"🛑 SL:    <b>{s['sl']:.2f}</b>  (0.7×ATR: {s['atr']})\n\n"
        f"✅ {tr}\n"
        f"✅ EMA9/21 kesishdi\n"
        f"✅ RSI: {s['rsi']} | MACD: {s['macd']}\n\n"
        f"⏰ {now}\n"
        f"⚠️ Risk: 1-2%"
    )

def send(msg: str):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        ).json()
        print("  ✓ Yuborildi" if r.get("ok") else f"  ✗ {r}")
    except Exception as e:
        print(f"  ✗ {e}")

# =========================
# Main
# =========================

def main():
    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    print(f"\n{'='*40}\nXAU/USD Bot: {now}\n{'='*40}")

    try:
        res = analyze()
        if res:
            print(
                f"\n✓ {res['action']} | "
                f"Entry:{res['price']} | "
                f"TP1:{res['tp1']} TP2:{res['tp2']} TP3:{res['tp3']} | "
                f"SL:{res['sl']}"
            )
            send(format_msg(res))
        else:
            now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
            send(
                f"📊 <b>XAU/USD — {now}</b>\n\n"
                f"Signal yo'q.\n"
                f"⏰ Keyingi tekshiruv 4 soatdan so'ng."
            )
    except Exception as e:
        print(f"XATO: {e}")
        import traceback; traceback.print_exc()

    print("\nTugadi.")

if __name__ == "__main__":
    main()
