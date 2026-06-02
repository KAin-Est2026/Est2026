"""
bot.py — Sniper Signal Bot
==========================
Timeframe:
  D1  → asosiy trend (EMA50/200)
  H1  → oraliq trend (EMA20/50)
  M15 → entry signal (EMA9/21 cross + RSI)

Entry: EMA cross + RSI tasdiqi
TP:    keyingi swing high/low
SL:    so'nggi swing low/high
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

def rsi(s: pd.Series, p: int = 14) -> float:
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    rs = g / l.replace(0, 1e-10)
    return float((100 - 100 / (1 + rs)).iloc[-1])

def swing_high(df: pd.DataFrame, n: int = 5) -> float:
    """So'nggi n bardagi eng yuqori high"""
    return float(df["high"].iloc[-n:].max())

def swing_low(df: pd.DataFrame, n: int = 5) -> float:
    """So'nggi n bardagi eng past low"""
    return float(df["low"].iloc[-n:].min())

def prev_swing_high(df: pd.DataFrame, n: int = 20) -> float:
    """n bar ichidagi eng yuqori swing (TP uchun)"""
    return float(df["high"].iloc[-n:-1].max())

def prev_swing_low(df: pd.DataFrame, n: int = 20) -> float:
    """n bar ichidagi eng past swing (TP uchun)"""
    return float(df["low"].iloc[-n:-1].min())

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
            print(f"  [{symbol}/{interval}] {r.get('message','?')}")
            return None
        df = pd.DataFrame(r["values"]).iloc[::-1].reset_index(drop=True)
        for c in ["open","high","low","close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        print(f"  [{symbol}/{interval}] {e}")
        return None

# ── Tahlil ────────────────────────────────────────────────────────────────────

def analyze(item: dict) -> dict | None:
    sym    = item["symbol"]
    digits = item["digits"]

    # 1. Real narx
    price = get_price(sym)
    if price is None:
        print(f"  [{sym}] narx olinmadi")
        return None
    print(f"  [{sym}] Narx: {price}")

    # 2. D1 — asosiy trend
    d1 = get_candles(sym, "1day", 220)
    if d1 is None or len(d1) < 200:
        print(f"  [{sym}] D1 yetarli emas")
        return None

    e50_d1  = float(ema(d1["close"], 50).iloc[-1])
    e200_d1 = float(ema(d1["close"], 200).iloc[-1])

    if e50_d1 > e200_d1 * 1.001:
        d1_trend = "BUY"
    elif e50_d1 < e200_d1 * 0.999:
        d1_trend = "SELL"
    else:
        print(f"  [{sym}] D1 trend aniq emas")
        return None  # Trend yo'q — signal yo'q

    # 3. H1 — oraliq trend tasdiqi
    h1 = get_candles(sym, "1h", 60)
    if h1 is None or len(h1) < 50:
        print(f"  [{sym}] H1 yetarli emas")
        return None

    e20_h1 = float(ema(h1["close"], 20).iloc[-1])
    e50_h1 = float(ema(h1["close"], 50).iloc[-1])

    if d1_trend == "BUY"  and e20_h1 < e50_h1:
        print(f"  [{sym}] H1 trend D1 ga zid")
        return None
    if d1_trend == "SELL" and e20_h1 > e50_h1:
        print(f"  [{sym}] H1 trend D1 ga zid")
        return None

    # 4. M15 — entry signal
    m15 = get_candles(sym, "15min", 60)
    if m15 is None or len(m15) < 30:
        print(f"  [{sym}] M15 yetarli emas")
        return None

    e9  = ema(m15["close"], 9)
    e21 = ema(m15["close"], 21)

    # Crossover: so'nggi 3 barda
    cross_up   = any(
        e9.iloc[i-1] < e21.iloc[i-1] and e9.iloc[i] >= e21.iloc[i]
        for i in range(-3, 0)
    )
    cross_down = any(
        e9.iloc[i-1] > e21.iloc[i-1] and e9.iloc[i] <= e21.iloc[i]
        for i in range(-3, 0)
    )

    # Trend yo'nalishida crossover bo'lishi kerak
    if d1_trend == "BUY"  and not cross_up:
        print(f"  [{sym}] BUY cross yo'q")
        return None
    if d1_trend == "SELL" and not cross_down:
        print(f"  [{sym}] SELL cross yo'q")
        return None

    action = d1_trend

    # 5. RSI tasdiqi
    rsi_val = rsi(m15["close"], 14)

    if action == "BUY":
        # RSI 30-65 oralig'ida bo'lishi kerak (oversold/normal)
        if rsi_val > 70:
            print(f"  [{sym}] RSI overbought ({rsi_val:.1f}) — BUY o'tkazildi")
            return None
        rsi_ok = rsi_val < 60  # Ideal: 30-55
    else:
        # RSI 35-70 oralig'ida bo'lishi kerak
        if rsi_val < 30:
            print(f"  [{sym}] RSI oversold ({rsi_val:.1f}) — SELL o'tkazildi")
            return None
        rsi_ok = rsi_val > 40  # Ideal: 45-70

    # 6. SL — so'nggi swing
    if action == "BUY":
        sl = round(swing_low(m15, n=10) - (price * 0.001), digits)
    else:
        sl = round(swing_high(m15, n=10) + (price * 0.001), digits)

    sl_dist = abs(price - sl)
    if sl_dist == 0:
        return None

    # 7. TP — keyingi kuchli swing level
    if action == "BUY":
        # H1 dagi so'nggi swing high — TP
        tp1 = round(prev_swing_high(h1, n=30), digits)
        tp2 = round(prev_swing_high(d1, n=20), digits)

        # TP1 entry dan yuqori bo'lishi kerak
        if tp1 <= price:
            tp1 = round(price + sl_dist * 2, digits)
        if tp2 <= tp1:
            tp2 = round(price + sl_dist * 3, digits)
    else:
        tp1 = round(prev_swing_low(h1, n=30), digits)
        tp2 = round(prev_swing_low(d1, n=20), digits)

        if tp1 >= price:
            tp1 = round(price - sl_dist * 2, digits)
        if tp2 >= tp1:
            tp2 = round(price - sl_dist * 3, digits)

    rr1 = round(abs(tp1 - price) / sl_dist, 2)
    rr2 = round(abs(tp2 - price) / sl_dist, 2)

    # Minimum R/R 1.5
    if rr1 < 1.5:
        print(f"  [{sym}] R/R yetarli emas: {rr1}")
        return None

    return {
        **item,
        "action":   action,
        "price":    round(price, digits),
        "sl":       sl,
        "tp1":      tp1,
        "tp2":      tp2,
        "rr1":      rr1,
        "rr2":      rr2,
        "rsi":      round(rsi_val, 1),
        "d1_trend": d1_trend,
        "rsi_ok":   rsi_ok,
    }

# ── Telegram ──────────────────────────────────────────────────────────────────

def format_msg(s: dict) -> str:
    e    = "🟢" if s["action"] == "BUY" else "🔴"
    act  = "SOTIB OL" if s["action"] == "BUY" else "SOT"
    now  = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    d    = s["digits"]
    te   = "💱" if s["type"] == "forex" else "🪙"
    rsi_icon = "✅" if s["rsi_ok"] else "⚠️"

    return (
        f"{e} <b>{s['symbol']} — {act}</b> {te}\n"
        f"<i>{s['name']}</i>\n\n"
        f"💰 Entry: <b>{s['price']:.{d}f}</b>\n"
        f"🎯 TP1:   <b>{s['tp1']:.{d}f}</b>  (R/R 1:{s['rr1']})\n"
        f"🎯 TP2:   <b>{s['tp2']:.{d}f}</b>  (R/R 1:{s['rr2']})\n"
        f"🛑 SL:    <b>{s['sl']:.{d}f}</b>\n\n"
        f"📊 RSI: {s['rsi']} {rsi_icon}\n"
        f"📈 D1 Trend: {s['d1_trend']}\n"
        f"🕐 Entry TF: M15 EMA cross\n\n"
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    print(f"\n{'='*45}\nBot: {now}\n{'='*45}")

    send(f"🔍 <b>Tahlil — {now}</b>\nXAU/USD va BTC/USD...")

    signals = []
    for item in SYMBOLS:
        print(f"\n[{item['symbol']}]")
        try:
            res = analyze(item)
            if res:
                signals.append(res)
                print(f"  ✓ {res['action']} | Entry:{res['price']} TP1:{res['tp1']} SL:{res['sl']} R/R:{res['rr1']}")
            else:
                print("  — Signal yo'q")
        except Exception as e:
            print(f"  XATO: {e}")
            import traceback; traceback.print_exc()

    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")

    if signals:
        for s in signals:
            send(format_msg(s))
            time.sleep(0.5)
    else:
        send(
            f"📊 <b>{now}</b>\n\n"
            f"Signal yo'q.\n"
            f"Sabab: trend yoki M15 cross sharti bajarilmadi.\n"
            f"⏰ Keyingi tekshiruv 1 soatdan so'ng."
        )

    print("\nTugadi.")

if __name__ == "__main__":
    main()
