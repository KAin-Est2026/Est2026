"""
bot.py — XAU/USD Sniper Scalping Bot (v2 — kuchaytirilgan filtrlar)
======================================================================
Tahlil:  H4 (trend + ADX kuch) + H1 (zona)
Entry:   M15 VA M5 (ikkalasi ham EMA9/21 crossoverni tasdiqlashi shart)
Filter:  RSI zona + Engulfing/Pin Bar + MACD histogram
Yangi:   ADX(14) H4 trend kuchi filtri (ADX<20 = flet, signal yo'q)
Yangi:   Minimal RR filtri (RR1 < 1.0 bo'lsa signal yuborilmaydi)
Yangi:   Signal cooldown (bir xil signal 4 soat ichida takrorlanmaydi)
SL:      0.7 x ATR (M15)
TP1:     H1 swing / Fib 1.272 dan yaqinrog'i (min ATR masofasi bilan)
TP2:     H4 swing / Fib 1.618 dan mantiqiyrog'i
TP3:     H4 keyingi kuchli level (min ATR spacing bilan)
Cron:    0 */4 * * * python3 bot.py

MUHIM ESLATMA: Hech qanday texnik indikator kombinatsiyasi 96-99% aniqlik
bera olmaydi. Bu versiya signal SIFATINI oshiradi (kamroq, lekin sifatli
signal), win-rate'ni emas — bozorda bunday kafolat yo'q. Har doim
risk-menejmentga rioya qiling.
"""

import os, json, time, requests, pandas as pd
from datetime import datetime, timedelta

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_KEY       = os.environ["TWELVE_DATA_KEY"]

SYMBOL = "XAU/USD"
DIGITS = 2

STATE_FILE = "bot_state.json"
COOLDOWN_HOURS = 4          # bir xil yo'nalishdagi signal shu soat ichida qayta yuborilmaydi
COOLDOWN_PRICE_PCT = 0.003  # yoki narx 0.3% dan ko'p siljisa, cooldown'ni bekor qiladi
MIN_RR1 = 1.0                # TP1 gacha minimal risk/reward, shundan past bo'lsa signal yo'q
MIN_ADX_H4 = 20              # H4 trend kuchi minimal ADX qiymati

# =========================
# Indikatorlar
# =========================

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(p).mean()
    loss  = (-delta.clip(upper=0)).rolling(p).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, p: int = 14) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(p).mean().dropna()
    return float(val.iloc[-1]) if len(val) > 0 else 1.0

def adx(df: pd.DataFrame, p: int = 14) -> float:
    """H4 trend kuchini o'lchash uchun. ADX < 20 = zaif/flet trend."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up_move > down_move) & (up_move > 0)]    = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_s = tr.ewm(alpha=1/p, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/p, adjust=False).mean()  / atr_s.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/p, adjust=False).mean() / atr_s.replace(0, 1e-10))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx_s = dx.ewm(alpha=1/p, adjust=False).mean().dropna()
    return float(adx_s.iloc[-1]) if len(adx_s) > 0 else 0.0

def engulfing(df: pd.DataFrame, direction: str) -> bool:
    o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]
    o2, c2 = df["open"].iloc[-1], df["close"].iloc[-1]
    if direction == "BUY":
        return c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1
    else:
        return c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1

def pin_bar(df: pd.DataFrame, direction: str) -> bool:
    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    h = df["high"].iloc[-1]
    l = df["low"].iloc[-1]
    body  = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    if body == 0:
        return False
    if direction == "BUY":
        return lower >= body * 2 and upper <= body * 0.5
    else:
        return upper >= body * 2 and lower <= body * 0.5

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

def last_impulse(df: pd.DataFrame, direction: str):
    """Fibonacci extension uchun oxirgi swing low->high (yoki aksincha) ni topadi."""
    highs = swing_highs(df, n=2)
    lows  = swing_lows(df, n=2)
    if not highs or not lows:
        return None
    if direction == "BUY":
        lo = max([l for l in lows if l < df["close"].iloc[-1]], default=None)
        hi = max(highs) if highs else None
        if lo is None or hi is None or hi <= lo:
            return None
        return lo, hi
    else:
        hi = min([h for h in highs if h > df["close"].iloc[-1]], default=None)
        lo = min(lows) if lows else None
        if hi is None or lo is None or lo >= hi:
            return None
        return hi, lo

def fib_extension(df: pd.DataFrame, direction: str, ratio: float):
    imp = last_impulse(df, direction)
    if imp is None:
        return None
    a, b = imp  # a=start, b=end of impulse
    dist = abs(b - a)
    if direction == "BUY":
        return round(b + dist * (ratio - 1), DIGITS)
    else:
        return round(b - dist * (ratio - 1), DIGITS)

# =========================
# Cooldown / state (bir xil signalni takrorlamaslik)
# =========================

def load_state() -> dict:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"  [state] saqlanmadi: {e}")

def in_cooldown(state: dict, action: str, price: float) -> bool:
    last = state.get("last_signal")
    if not last:
        return False
    if last.get("action") != action:
        return False
    try:
        last_time = datetime.fromisoformat(last["time"])
    except Exception:
        return False
    if datetime.utcnow() - last_time > timedelta(hours=COOLDOWN_HOURS):
        return False
    last_price = last.get("price", price)
    if last_price == 0:
        return False
    moved_pct = abs(price - last_price) / last_price
    if moved_pct > COOLDOWN_PRICE_PCT:
        return False
    return True  # hali cooldown ichida va narx sezilarli siljimagan

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
    except Exception:
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

    # ── H4: asosiy trend + kuch (ADX) ───────────────────────────────────────
    h4 = get_candles("4h", 250)
    if h4 is None or len(h4) < 200:
        print("  H4 yetarli emas")
        return None

    e50_h4  = float(ema(h4["close"], 50).iloc[-1])
    e200_h4 = float(ema(h4["close"], 200).iloc[-1])

    if e50_h4 > e200_h4 * 1.001:
        trend = "BUY"
    elif e50_h4 < e200_h4 * 0.999:
        trend = "SELL"
    else:
        print("  H4 trend aniq emas")
        return None

    adx_h4 = adx(h4, 14)
    if adx_h4 < MIN_ADX_H4:
        print(f"  H4 ADX zaif: {adx_h4:.1f} (<{MIN_ADX_H4}) — flet bozor, signal yo'q")
        return None
    print(f"  H4 trend: {trend} | ADX: {adx_h4:.1f} ✓")

    # ── H1: zona tasdiqi ──────────────────────────────────────────────────────
    h1 = get_candles("1h", 100)
    if h1 is None or len(h1) < 50:
        print("  H1 yetarli emas")
        return None

    e50_h1   = float(ema(h1["close"], 50).iloc[-1])
    price_h1 = float(h1["close"].iloc[-1])

    if trend == "BUY"  and price_h1 < e50_h1:
        print("  H1: narx EMA50 ostida — BUY o'tkazildi")
        return None
    if trend == "SELL" and price_h1 > e50_h1:
        print("  H1: narx EMA50 ustida — SELL o'tkazildi")
        return None

    # ── M15 va M5: ikkalasi ham tasdiqlashi SHART ────────────────────────────
    m15 = get_candles("15min", 60)
    if m15 is None or len(m15) < 40:
        print("  M15 yetarli emas")
        return None

    e9_m15  = ema(m15["close"], 9)
    e21_m15 = ema(m15["close"], 21)

    cross_up_m15 = any(
        e9_m15.iloc[i-1] < e21_m15.iloc[i-1] and e9_m15.iloc[i] >= e21_m15.iloc[i]
        for i in range(-5, 0)
    )
    cross_down_m15 = any(
        e9_m15.iloc[i-1] > e21_m15.iloc[i-1] and e9_m15.iloc[i] <= e21_m15.iloc[i]
        for i in range(-5, 0)
    )

    m5 = get_candles("5min", 60)
    if m5 is None or len(m5) < 30:
        print("  M5 yetarli emas")
        return None

    e9_m5  = ema(m5["close"], 9)
    e21_m5 = ema(m5["close"], 21)

    cross_up_m5 = any(
        e9_m5.iloc[i-1] < e21_m5.iloc[i-1] and e9_m5.iloc[i] >= e21_m5.iloc[i]
        for i in range(-4, 0)
    )
    cross_down_m5 = any(
        e9_m5.iloc[i-1] > e21_m5.iloc[i-1] and e9_m5.iloc[i] <= e21_m5.iloc[i]
        for i in range(-4, 0)
    )

    m15_ok = cross_up_m15 if trend == "BUY" else cross_down_m15
    m5_ok  = cross_up_m5  if trend == "BUY" else cross_down_m5

    # AVVAL: "or" edi (bittasi yetarli). ENDI: ikkalasi ham tasdiqlashi shart.
    if not (m15_ok and m5_ok):
        print("  M15 VA M5 ikkalasi ham tasdiqlamadi — signal yo'q")
        return None

    entry_tf = "M15+M5"
    entry_df = m5  # sniper entry uchun eng qisqa TF candle patterni

    # ── RSI filtri (M15) ──────────────────────────────────────────────────────
    rsi_val = float(rsi(m15["close"]).iloc[-1])
    if trend == "BUY"  and rsi_val > 65:
        print(f"  RSI overbought: {rsi_val:.1f} — BUY o'tkazildi")
        return None
    if trend == "SELL" and rsi_val < 35:
        print(f"  RSI oversold: {rsi_val:.1f} — SELL o'tkazildi")
        return None
    print(f"  RSI: {rsi_val:.1f} ✓")

    # ── Candle pattern tasdiqi (M5, chunki entry_tf = M5) ────────────────────
    eng = engulfing(entry_df, trend)
    pin = pin_bar(entry_df, trend)
    if not eng and not pin:
        print("  Candle pattern yo'q (engulfing/pin bar)")
        return None
    pattern = "Engulfing" if eng else "Pin Bar"
    print(f"  Pattern: {pattern} ✓")

    # ── MACD histogram tasdiqi (M15) ──────────────────────────────────────────
    macd_s = ema(m15["close"], 12) - ema(m15["close"], 26)
    hist   = macd_s - ema(macd_s, 9)
    h_now  = float(hist.iloc[-1])
    h_prv  = float(hist.iloc[-2])

    if trend == "BUY"  and not (h_now > 0 or h_now > h_prv):
        print("  MACD BUY tasdiqlamadi")
        return None
    if trend == "SELL" and not (h_now < 0 or h_now < h_prv):
        print("  MACD SELL tasdiqlamadi")
        return None

    last_price = float(m15["close"].iloc[-1])

    # ── ATR → SL ──────────────────────────────────────────────────────────────
    atr_val = atr(m15, 14)
    sl_dist = round(atr_val * 0.7, DIGITS)

    sl = round(last_price - sl_dist, DIGITS) if trend == "BUY" \
         else round(last_price + sl_dist, DIGITS)

    # ── TP: swing level + Fibonacci extension kombinatsiyasi ─────────────────
    h1_highs = swing_highs(h1, n=3)
    h1_lows  = swing_lows(h1,  n=3)
    h4_highs = swing_highs(h4, n=3)
    h4_lows  = swing_lows(h4,  n=3)

    min_gap = max(atr_val * 0.5, sl_dist * 0.5)  # TP'lar orasidagi minimal masofa

    if trend == "BUY":
        struct_tp1 = next_level_above(h1_highs, last_price)
        fib_tp1    = fib_extension(h1, "BUY", 1.272)
        candidates1 = [x for x in [struct_tp1, fib_tp1] if x is not None and x - last_price >= atr_val * 1.0]
        tp1 = min(candidates1) if candidates1 else round(last_price + atr_val * 1.5, DIGITS)

        struct_tp2 = next_level_above(h4_highs, last_price)
        fib_tp2    = fib_extension(h1, "BUY", 1.618)
        candidates2 = [x for x in [struct_tp2, fib_tp2] if x is not None and x - tp1 >= min_gap]
        tp2 = min(candidates2) if candidates2 else round(tp1 + max(atr_val * 1.5, min_gap), DIGITS)

        h4_above = sorted([l for l in h4_highs if l > tp2 + min_gap])
        tp3 = h4_above[0] if h4_above else round(tp2 + max(atr_val * 1.5, min_gap), DIGITS)

        tp1, tp2, tp3 = sorted([tp1, tp2, tp3])
    else:
        struct_tp1 = next_level_below(h1_lows, last_price)
        fib_tp1    = fib_extension(h1, "SELL", 1.272)
        candidates1 = [x for x in [struct_tp1, fib_tp1] if x is not None and last_price - x >= atr_val * 1.0]
        tp1 = max(candidates1) if candidates1 else round(last_price - atr_val * 1.5, DIGITS)

        struct_tp2 = next_level_below(h4_lows, last_price)
        fib_tp2    = fib_extension(h1, "SELL", 1.618)
        candidates2 = [x for x in [struct_tp2, fib_tp2] if x is not None and tp1 - x >= min_gap]
        tp2 = max(candidates2) if candidates2 else round(tp1 - max(atr_val * 1.5, min_gap), DIGITS)

        h4_below = sorted([l for l in h4_lows if l < tp2 - min_gap], reverse=True)
        tp3 = h4_below[0] if h4_below else round(tp2 - max(atr_val * 1.5, min_gap), DIGITS)

        tp1, tp2, tp3 = sorted([tp1, tp2, tp3], reverse=True)

    def rr(tp):
        return round(abs(tp - last_price) / sl_dist, 1) if sl_dist > 0 else 0

    rr1 = rr(tp1)

    # ── Minimal RR filtri ──────────────────────────────────────────────────────
    if rr1 < MIN_RR1:
        print(f"  RR1 juda past: {rr1} (<{MIN_RR1}) — signal foydasiz, o'tkazildi")
        return None

    return {
        "action":   trend,
        "price":    round(last_price, DIGITS),
        "sl":       sl,
        "tp1":      round(tp1, DIGITS),
        "tp2":      round(tp2, DIGITS),
        "tp3":      round(tp3, DIGITS),
        "rr1":      rr1,
        "rr2":      rr(tp2),
        "rr3":      rr(tp3),
        "atr":      round(atr_val, DIGITS),
        "adx":      round(adx_h4, 1),
        "entry_tf": entry_tf,
        "pattern":  pattern,
        "rsi":      round(rsi_val, 1),
        "macd":     round(h_now, 4),
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
        f"✅ H4 {tr} | ADX: {s['adx']}\n"
        f"✅ H1 narx EMA50 {'ustida' if s['action']=='BUY' else 'ostida'}\n"
        f"✅ {s['entry_tf']} EMA9/21 kesdi (ikkalasi ham)\n"
        f"✅ RSI: {s['rsi']} | MACD: {s['macd']}\n"
        f"✅ Pattern: {s['pattern']}\n\n"
        f"⏰ {now}\n"
        f"⚠️ Risk: 1-2% | Bu signal emas, kafolat emas — o'z tahlilingiz bilan tasdiqlang"
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

    state = load_state()

    try:
        res = analyze()
        if res:
            if in_cooldown(state, res["action"], res["price"]):
                print(f"  Cooldown: {res['action']} signal {COOLDOWN_HOURS} soat ichida allaqachon yuborilgan, o'tkazildi")
            else:
                print(
                    f"\n✓ {res['action']} | "
                    f"Entry:{res['price']} | "
                    f"TP1:{res['tp1']} TP2:{res['tp2']} TP3:{res['tp3']} | "
                    f"SL:{res['sl']} | RR1:{res['rr1']} | {res['pattern']}"
                )
                send(format_msg(res))
                state["last_signal"] = {
                    "action": res["action"],
                    "price": res["price"],
                    "time": datetime.utcnow().isoformat(),
                }
                save_state(state)
        else:
            now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
            send(
                f"📊 <b>XAU/USD — {now}</b>\n\n"
                f"Signal yo'q (filtrlar o'tmadi).\n"
                f"⏰ Keyingi tekshiruv 4 soatdan so'ng."
            )
    except Exception as e:
        print(f"XATO: {e}")
        import traceback; traceback.print_exc()

    print("\nTugadi.")

if __name__ == "__main__":
    main()