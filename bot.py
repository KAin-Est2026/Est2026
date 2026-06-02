"""
bot.py — Sniper Scalping Bot
Har 3-4 soatda ishga tushadi (cron: 0 */4 * * *)

Timeframe:
  D1  → asosiy trend
  H1  → oraliq trend
  M15 → entry

Indikatorlar:
  EMA 50/200 (D1 trend)
  EMA 20/50  (H1 trend)
  EMA 9/21   (M15 crossover)
  MACD       (momentum)
  Stochastic (entry tasdiqi)
  Volume     (kuch tasdiqi)

TP: keyingi swing high/low
SL: so'nggi swing low/high
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

# ── Indikatorlar ──────────────────────────────────────────────────────────────

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def macd_hist(s: pd.Series) -> pd.Series:
    m = ema(s, 12) - ema(s, 26)
    return m - ema(m, 9)

def stochastic(df: pd.DataFrame, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    k_line = 100 * (df["close"] - lo) / (hi - lo).replace(0, 1e-10)
    return k_line, k_line.rolling(d).mean()

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

def get_candles(symbol: str, interval: str, size: int) -> pd.DataFrame | None:
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol":     symbol,
                "interval":   interval,
                "outputsize": size,
                "apikey":     TWELVE_KEY,
            },
            timeout=15
        ).json()
        if "values" not in r:
            print(f"  [{symbol}/{interval}] {r.get('message','?')}")
            return None
        df = pd.DataFrame(r["values"]).iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(
            df["volume"] if "volume" in df.columns else 0,
            errors="coerce"
        ).fillna(0)
        return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    except Exception as e:
        print(f"  [{symbol}/{interval}] {e}")
        return None

# ── Tahlil ────────────────────────────────────────────────────────────────────

def analyze(item: dict) -> dict | None:
    sym    = item["symbol"]
    digits = item["digits"]

    # ── Real narx ─────────────────────────────────────────────────────────────
    price = get_price(sym)
    if price is None:
        print(f"  [{sym}] narx olinmadi")
        return None
    print(f"  [{sym}] Narx: {price}")

    # ── D1: asosiy trend ──────────────────────────────────────────────────────
    d1 = get_candles(sym, "1day", 220)
    if d1 is None or len(d1) < 200:
        print(f"  [{sym}] D1 yetarli emas")
        return None

    e50_d1  = float(ema(d1["close"], 50).iloc[-1])
    e200_d1 = float(ema(d1["close"], 200).iloc[-1])

    if e50_d1 > e200_d1 * 1.001:
        trend = "BUY"
    elif e50_d1 < e200_d1 * 0.999:
        trend = "SELL"
    else:
        print(f"  [{sym}] D1 trend aniq emas")
        return None

    # ── H1: oraliq trend ──────────────────────────────────────────────────────
    h1 = get_candles(sym, "1h", 60)
    if h1 is None or len(h1) < 50:
        print(f"  [{sym}] H1 yetarli emas")
        return None

    e20_h1 = float(ema(h1["close"], 20).iloc[-1])
    e50_h1 = float(ema(h1["close"], 50).iloc[-1])

    if trend == "BUY"  and e20_h1 < e50_h1:
        print(f"  [{sym}] H1 D1 ga zid")
        return None
    if trend == "SELL" and e20_h1 > e50_h1:
        print(f"  [{sym}] H1 D1 ga zid")
        return None

    # ── M15: entry signali ────────────────────────────────────────────────────
    m15 = get_candles(sym, "15min", 60)
    if m15 is None or len(m15) < 40:
        print(f"  [{sym}] M15 yetarli emas")
        return None

    e9  = ema(m15["close"], 9)
    e21 = ema(m15["close"], 21)

    # EMA crossover — so'nggi 3 barda
    cross_up = any(
        e9.iloc[i-1] < e21.iloc[i-1] and e9.iloc[i] >= e21.iloc[i]
        for i in range(-3, 0)
    )
    cross_down = any(
        e9.iloc[i-1] > e21.iloc[i-1] and e9.iloc[i] <= e21.iloc[i]
        for i in range(-3, 0)
    )

    if trend == "BUY"  and not cross_up:
        print(f"  [{sym}] M15 BUY cross yo'q")
        return None
    if trend == "SELL" and not cross_down:
        print(f"  [{sym}] M15 SELL cross yo'q")
        return None

    # ── MACD tasdiqi ──────────────────────────────────────────────────────────
    hist   = macd_hist(m15["close"])
    h_now  = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2])

    if trend == "BUY":
        macd_ok = h_now > 0 or (h_now > h_prev)
    else:
        macd_ok = h_now < 0 or (h_now < h_prev)

    if not macd_ok:
        print(f"  [{sym}] MACD tasdiqlamadi")
        return None

    # ── Stochastic tasdiqi ────────────────────────────────────────────────────
    k_line, d_line = stochastic(m15)
    k_now  = float(k_line.iloc[-1])
    d_now  = float(d_line.iloc[-1])
    k_prev = float(k_line.iloc[-2])
    d_prev = float(d_line.iloc[-2])

    if trend == "BUY":
        # %K pastdan yuqoriga %D ni kesdi va overbought emas
        stoch_ok = (k_prev <= d_prev and k_now > d_now) and k_now < 80
    else:
        # %K yuqoridan pastga %D ni kesdi va oversold emas
        stoch_ok = (k_prev >= d_prev and k_now < d_now) and k_now > 20

    if not stoch_ok:
        print(f"  [{sym}] Stochastic tasdiqlamadi (K={k_now:.1f})")
        return None

    # ── Volume tasdiqi ────────────────────────────────────────────────────────
    vol_now  = float(m15["volume"].iloc[-1])
    vol_avg  = float(m15["volume"].iloc[-20:].mean())
    vol_ok   = vol_avg == 0 or vol_now >= vol_avg * 0.8

    if not vol_ok:
        print(f"  [{sym}] Volume juda past")
        return None

    vol_x = f"{vol_now/vol_avg:.1f}x" if vol_avg > 0 else "—"

    # ── SL ────────────────────────────────────────────────────────────────────
    if trend == "BUY":
        sl = round(float(m15["low"].iloc[-15:].min()) - price * 0.0003, digits)
    else:
        sl = round(float(m15["high"].iloc[-15:].max()) + price * 0.0003, digits)

    sl_dist = abs(price - sl)
    if sl_dist == 0:
        return None

    # ── TP — swing level ──────────────────────────────────────────────────────
    if trend == "BUY":
        tp1_raw = float(h1["high"].iloc[-30:].max())
        tp2_raw = float(d1["high"].iloc[-20:].max())
        tp1 = round(tp1_raw if tp1_raw > price * 1.001 else price + sl_dist * 2.0, digits)
        tp2 = round(tp2_raw if tp2_raw > tp1  else price + sl_dist * 3.0, digits)
    else:
        tp1_raw = float(h1["low"].iloc[-30:].min())
        tp2_raw = float(d1["low"].iloc[-20:].min())
        tp1 = round(tp1_raw if tp1_raw < price * 0.999 else price - sl_dist * 2.0, digits)
        tp2 = round(tp2_raw if tp2_raw < tp1  else price - sl_dist * 3.0, digits)

    rr1 = round(abs(tp1 - price) / sl_dist, 2)
    rr2 = round(abs(tp2 - price) / sl_dist, 2)

    if rr1 < 1.5:
        print(f"  [{sym}] R/R yetarli emas: {rr1}")
        return None

    return {
        **item,
        "action": trend,
        "price":  round(price, digits),
        "sl":     sl,
        "tp1":    tp1,
        "tp2":    tp2,
        "rr1":    rr1,
        "rr2":    rr2,
        "k_now":  round(k_now, 1),
        "d_now":  round(d_now, 1),
        "macd_h": round(h_now, 4),
        "vol_x":  vol_x,
    }

# ── Telegram ──────────────────────────────────────────────────────────────────

def format_msg(s: dict) -> str:
    e   = "🟢" if s["action"] == "BUY" else "🔴"
    act = "SOTIB OL" if s["action"] == "BUY" else "SOT"
    te  = "💱" if s["type"] == "forex" else "🪙"
    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    d   = s["digits"]

    return (
        f"{e} <b>{s['symbol']} — {act}</b> {te}\n"
        f"<i>{s['name']}</i>\n\n"
        f"💰 Entry:  <b>{s['price']:.{d}f}</b>\n"
        f"🎯 TP1:   <b>{s['tp1']:.{d}f}</b>  (1:{s['rr1']})\n"
        f"🎯 TP2:   <b>{s['tp2']:.{d}f}</b>  (1:{s['rr2']})\n"
        f"🛑 SL:    <b>{s['sl']:.{d}f}</b>\n\n"
        f"✅ D1 trend: {'Uptrend' if s['action']=='BUY' else 'Downtrend'}\n"
        f"✅ H1 trend mos\n"
        f"✅ M15 EMA9/21 kesdi\n"
        f"✅ MACD: {s['macd_h']}\n"
        f"✅ Stoch: K={s['k_now']} D={s['d_now']}\n"
        f"✅ Volume: {s['vol_x']}\n\n"
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

    signals = []

    for item in SYMBOLS:
        print(f"\n[{item['symbol']}]")
        try:
            res = analyze(item)
            if res:
                signals.append(res)
                print(f"  ✓ {res['action']} | Entry:{res['price']} TP1:{res['tp1']} SL:{res['sl']} R/R:1:{res['rr1']}")
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
            f"Hozircha signal yo'q.\n"
            f"⏰ Keyingi tekshiruv 4 soatdan so'ng."
        )

    print("\nTugadi.")


if __name__ == "__main__":
    main()
