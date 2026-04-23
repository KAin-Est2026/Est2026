"""
Professional Signal Bot — 5 tasdiqlash tizimi
=============================================
Strategiya:
  1. D1  — asosiy trend (EMA50 > EMA200 = uptrend)
  2. H4  — oraliq trend tasdiqi (EMA20 > EMA50)
  3. H1  — kirish vaqti (EMA9 x EMA21 crossover)
  4. RSI — 30-70 chegarasidan qaytish (divergence)
  5. MACD — histogram musbat/manfiy o'tish
  6. Bollinger — narx band chegarasidan qaytish
  7. Volume — o'rtachadan yuqori hajm (tasdiqlash)

Signal faqat 5/7 yoki 7/7 shart bajarilganda yuboriladi.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime

# ── Sozlamalar ────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_DATA_KEY  = os.environ["TWELVE_DATA_KEY"]

# Minimum tasdiqlash soni (2 dan kam bo'lsa signal berilmaydi)
MIN_CONFIRMATIONS = 2

SYMBOLS = [
    {"symbol": "XAU/USD", "name": "Oltin",   "type": "forex",  "pip": 0.10},
    {"symbol": "BTC/USD", "name": "Bitcoin", "type": "crypto", "pip": 10},
]

# TP/SL koeffitsientlari bozor turiga qarab (ATR asosida)
TP_SL = {
    "forex":  {"tp1": 1.5, "tp2": 2.5, "sl": 1.0},
    "crypto": {"tp1": 2.0, "tp2": 3.5, "sl": 1.2},
    "stock":  {"tp1": 1.8, "tp2": 3.0, "sl": 1.0},
    "index":  {"tp1": 1.5, "tp2": 2.5, "sl": 1.0},
}


# ── Yordamchi funksiyalar ─────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger(series: pd.Series, period=20, std=2):
    mid = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    return upper, mid, lower

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Twelve Data API ───────────────────────────────────────────────────────────

def get_candles(symbol: str, interval: str, outputsize: int = 100) -> pd.DataFrame | None:
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params={
            "symbol":     symbol,
            "interval":   interval,
            "outputsize": outputsize,
            "apikey":     TWELVE_DATA_KEY,
        }, timeout=15)
        data = r.json()
        if data.get("status") == "error" or "values" not in data:
            print(f"  [{symbol}/{interval}] API xato: {data.get('message','?')}")
            return None
        df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        print(f"  [{symbol}/{interval}] So'rov xato: {e}")
        return None


# ── Ko'p tasdiqlash tahlili ───────────────────────────────────────────────────

def analyze_symbol(item: dict) -> dict | None:
    symbol = item["symbol"]
    confirmations = []
    direction_votes = {"BUY": 0, "SELL": 0}

    # ── 1. D1 TREND (EMA50 vs EMA200) ─────────────────────────────────────────
    d1 = get_candles(symbol, "1day", 220)
    time.sleep(0.2)
    if d1 is not None and len(d1) >= 200:
        e50_d1  = ema(d1["close"], 50).iloc[-1]
        e200_d1 = ema(d1["close"], 200).iloc[-1]
        if e50_d1 > e200_d1 * 1.001:
            confirmations.append(("D1 trend", "BUY", "EMA50 > EMA200 — Uptrend"))
            direction_votes["BUY"] += 2  # Og'irlik: 2 (muhim tasdiqlash)
        elif e50_d1 < e200_d1 * 0.999:
            confirmations.append(("D1 trend", "SELL", "EMA50 < EMA200 — Downtrend"))
            direction_votes["SELL"] += 2

    # ── 2. H4 TREND (EMA20 vs EMA50) ──────────────────────────────────────────
    time.sleep(65)   # API daqiqalik limitdan o'tmasligi uchun
    h4 = get_candles(symbol, "4h", 60)
    if h4 is not None and len(h4) >= 50:
        e20_h4 = ema(h4["close"], 20).iloc[-1]
        e50_h4 = ema(h4["close"], 50).iloc[-1]
        if e20_h4 > e50_h4:
            confirmations.append(("H4 trend", "BUY", "EMA20 > EMA50"))
            direction_votes["BUY"] += 1
        elif e20_h4 < e50_h4:
            confirmations.append(("H4 trend", "SELL", "EMA20 < EMA50"))
            direction_votes["SELL"] += 1

    # ── 3. H1 EMA CROSSOVER (kirish signali) ──────────────────────────────────
    time.sleep(65)   # API daqiqalik limitdan o'tmasligi uchun
    h1 = get_candles(symbol, "1h", 50)
    if h1 is not None and len(h1) >= 30:
        e9_h1  = ema(h1["close"], 9)
        e21_h1 = ema(h1["close"], 21)
        # Crossover: oldingi barda kesishish bo'lganmi?
        cross_up   = e9_h1.iloc[-2] < e21_h1.iloc[-2] and e9_h1.iloc[-1] > e21_h1.iloc[-1]
        cross_down = e9_h1.iloc[-2] > e21_h1.iloc[-2] and e9_h1.iloc[-1] < e21_h1.iloc[-1]
        if cross_up:
            confirmations.append(("H1 EMA cross", "BUY", "EMA9 yuqoriga kesdi"))
            direction_votes["BUY"] += 2
        elif cross_down:
            confirmations.append(("H1 EMA cross", "SELL", "EMA9 pastga kesdi"))
            direction_votes["SELL"] += 2

    # ── 4. RSI (H1) ────────────────────────────────────────────────────────────
    if h1 is not None and len(h1) >= 20:
        rsi_h1 = rsi(h1["close"], 14)
        rsi_val = rsi_h1.iloc[-1]
        rsi_prev = rsi_h1.iloc[-2]
        # Oversold dan qaytish
        if rsi_prev < 35 and rsi_val > rsi_prev:
            confirmations.append(("RSI", "BUY", f"RSI oversold dan qaytdi ({rsi_val:.1f})"))
            direction_votes["BUY"] += 1
        # Overbought dan qaytish
        elif rsi_prev > 65 and rsi_val < rsi_prev:
            confirmations.append(("RSI", "SELL", f"RSI overbought dan qaytdi ({rsi_val:.1f})"))
            direction_votes["SELL"] += 1
        # Markaziy zona (40-60) — trend kuchli
        elif 45 <= rsi_val <= 60:
            if direction_votes["BUY"] >= direction_votes["SELL"]:
                confirmations.append(("RSI zona", "BUY", f"RSI kuchli zona ({rsi_val:.1f})"))
                direction_votes["BUY"] += 1
        elif 40 <= rsi_val <= 55:
            if direction_votes["SELL"] >= direction_votes["BUY"]:
                confirmations.append(("RSI zona", "SELL", f"RSI kuchli zona ({rsi_val:.1f})"))
                direction_votes["SELL"] += 1

    # ── 5. MACD (H1) ───────────────────────────────────────────────────────────
    if h1 is not None and len(h1) >= 35:
        _, _, hist = macd(h1["close"])
        hist_prev = hist.iloc[-2]
        hist_curr = hist.iloc[-1]
        # Histogram musbatga o'tdi
        if hist_prev < 0 and hist_curr > 0:
            confirmations.append(("MACD", "BUY", "Histogram musbatga o'tdi"))
            direction_votes["BUY"] += 2
        # Histogram manfiyga o'tdi
        elif hist_prev > 0 and hist_curr < 0:
            confirmations.append(("MACD", "SELL", "Histogram manfiyga o'tdi"))
            direction_votes["SELL"] += 2
        # Kuchayib bormoqda
        elif hist_curr > hist_prev > 0:
            confirmations.append(("MACD kuch", "BUY", "Histogram kuchaymoqda"))
            direction_votes["BUY"] += 1
        elif hist_curr < hist_prev < 0:
            confirmations.append(("MACD kuch", "SELL", "Histogram kuchaymoqda"))
            direction_votes["SELL"] += 1

    # ── 6. BOLLINGER BANDS (H1) ───────────────────────────────────────────────
    if h1 is not None and len(h1) >= 25:
        bb_up, bb_mid, bb_low = bollinger(h1["close"])
        price_now  = h1["close"].iloc[-1]
        price_prev = h1["close"].iloc[-2]
        bb_up_v  = bb_up.iloc[-1]
        bb_low_v = bb_low.iloc[-1]
        bb_mid_v = bb_mid.iloc[-1]
        # Pastki band dan qaytish (BUY)
        if price_prev < bb_low_v and price_now > bb_low_v:
            confirmations.append(("Bollinger", "BUY", "Pastki band dan qaytdi"))
            direction_votes["BUY"] += 1
        # Yuqori band dan qaytish (SELL)
        elif price_prev > bb_up_v and price_now < bb_up_v:
            confirmations.append(("Bollinger", "SELL", "Yuqori band dan qaytdi"))
            direction_votes["SELL"] += 1
        # Narx o'rta chiziq ustida (trend tasdiqi)
        elif price_now > bb_mid_v:
            confirmations.append(("BB mid", "BUY", "Narx o'rta chiziq ustida"))
            direction_votes["BUY"] += 1
        elif price_now < bb_mid_v:
            confirmations.append(("BB mid", "SELL", "Narx o'rta chiziq ostida"))
            direction_votes["SELL"] += 1

    # ── 7. VOLUME (H1) — O'rtachadan yuqori hajm ──────────────────────────────
    if h1 is not None and "volume" in h1.columns and len(h1) >= 20:
        vol_curr = h1["volume"].iloc[-1]
        vol_avg  = h1["volume"].iloc[-20:].mean()
        if vol_curr > vol_avg * 1.3:
            # Hajm katta — dominant yo'nalishni tasdiqlaydi
            dominant = "BUY" if direction_votes["BUY"] >= direction_votes["SELL"] else "SELL"
            confirmations.append(("Volume", dominant, f"Hajm o'rtachadan {vol_curr/vol_avg:.1f}x katta"))
            direction_votes[dominant] += 1

    # ── Yakuniy qaror ──────────────────────────────────────────────────────────
    buy_score  = direction_votes["BUY"]
    sell_score = direction_votes["SELL"]
    total_conf = len(confirmations)

    # Dominant yo'nalish
    if buy_score > sell_score:
        action = "BUY"
        score  = buy_score
    elif sell_score > buy_score:
        action = "SELL"
        score  = sell_score
    else:
        return None  # Teng — signal yo'q

    # Faqat bir yo'nalish uchun tasdiqlashlarni hisoblash
    action_confs = [c for c in confirmations if c[1] == action]
    if len(action_confs) < MIN_CONFIRMATIONS or score < 3:
        print(f"  {symbol}: zaif signal ({len(action_confs)} tasdiqlash, score={score}) — o'tkazib yuborildi")
        return None

    # ── ATR asosida TP/SL hisoblash ────────────────────────────────────────────
    curr_price = h1["close"].iloc[-1] if h1 is not None else 0
    atr_val = atr(h1).iloc[-1] if h1 is not None else curr_price * 0.005
    coef = TP_SL.get(item["type"], TP_SL["forex"])

    if action == "BUY":
        tp1 = round(curr_price + atr_val * coef["tp1"], 5)
        tp2 = round(curr_price + atr_val * coef["tp2"], 5)
        sl  = round(curr_price - atr_val * coef["sl"],  5)
    else:
        tp1 = round(curr_price - atr_val * coef["tp1"], 5)
        tp2 = round(curr_price - atr_val * coef["tp2"], 5)
        sl  = round(curr_price + atr_val * coef["sl"],  5)

    rr = abs(tp1 - curr_price) / abs(sl - curr_price) if sl != curr_price else 0

    return {
        **item,
        "action":        action,
        "price":         round(curr_price, 5),
        "tp1":           tp1,
        "tp2":           tp2,
        "sl":            sl,
        "rr":            round(rr, 2),
        "score":         score,
        "confirmations": action_confs,
        "rsi_val":       round(rsi(h1["close"], 14).iloc[-1], 1) if h1 is not None else 0,
        "atr":           round(atr_val, 5),
    }


# ── Telegram xabari ───────────────────────────────────────────────────────────

STRENGTH = {(5,6): "●●●○○", (6,7): "●●●●○", (7,99): "●●●●●"}

def get_strength_bar(score: int) -> str:
    for (lo, hi), bar in STRENGTH.items():
        if lo <= score < hi:
            return bar
    return "●●●●●"

TYPE_EMOJI = {"forex": "💱", "crypto": "🪙", "stock": "📈"}

def format_signal(s: dict) -> str:
    e        = "🟢" if s["action"] == "BUY" else "🔴"
    act_uz   = "SOTIB OL" if s["action"] == "BUY" else "SOT"
    t_emoji  = TYPE_EMOJI.get(s["type"], "📊")
    strength = get_strength_bar(s["score"])
    now      = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")

    conf_lines = "\n".join(f"  ✅ {c[0]}: {c[2]}" for c in s["confirmations"])

    return (
        f"{e} <b>{s['symbol']} — {act_uz}</b> {t_emoji}\n"
        f"<i>{s['name']}</i>\n"
        f"💪 Kuch: {strength} ({s['score']} ball)\n\n"
        f"💰 <b>Kirish:</b> <code>{s['price']}</code>\n"
        f"🎯 <b>TP1:</b>    <code>{s['tp1']}</code>\n"
        f"🎯 <b>TP2:</b>    <code>{s['tp2']}</code>\n"
        f"🛑 <b>SL:</b>     <code>{s['sl']}</code>\n"
        f"⚖️ <b>R/R:</b>    1 : {s['rr']}\n\n"
        f"📊 <b>Tasdiqlashlar ({len(s['confirmations'])} ta):</b>\n"
        f"{conf_lines}\n\n"
        f"📈 RSI: {s['rsi_val']} | ATR: {s['atr']}\n"
        f"⏰ {now}\n\n"
        f"⚠️ <i>Tahlil asosida. Risk: kapitalning 1-2%.</i>"
    )


def send_telegram(msg: str):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
        timeout=10
    )


# ── Asosiy ────────────────────────────────────────────────────────────────────

def main():
    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M")
    print(f"\n{'='*50}")
    print(f"Bot ishga tushdi: {now} UTC")
    print(f"{'='*50}")

    signals = []

    # Boshlanganda ham xabar yuborsin
    now_str = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    send_telegram(f"🔍 <b>Tahlil boshlandi — {now_str}</b>\nXAU/USD va BTC/USD tekshirilmoqda...")

    for item in SYMBOLS:
        print(f"\n[{item['symbol']}] tahlil boshlanmoqda...")
        try:
            result = analyze_symbol(item)
            if result:
                signals.append(result)
                print(f"  SIGNAL: {result['action']} | Score: {result['score']} | R/R: {result['rr']}")
            else:
                print(f"  Signal yo'q yoki zaif")
        except Exception as e:
            print(f"  XATO: {e}")
        time.sleep(0.3)

    print(f"\n{'='*50}")
    print(f"Natija: {len(signals)} ta signal topildi")
    print(f"{'='*50}")

    now_str = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")

    if signals:
        signals.sort(key=lambda x: x["score"], reverse=True)
        header = (
            f"📡 <b>TAHLIL NATIJASI — {now_str}</b>\n"
            f"Topilgan signal: <b>{len(signals)} ta</b>\n"
            f"{'─'*28}"
        )
        send_telegram(header)
        time.sleep(0.3)
        for s in signals:
            send_telegram(format_signal(s))
            time.sleep(0.3)
    else:
        send_telegram(
            f"📊 <b>Tahlil natijasi — {now_str}</b>\n\n"
            f"• XAU/USD — signal yo'q\n"
            f"• BTC/USD — signal yo'q\n\n"
            f"Bozor hozir kutish holatida.\n"
            f"⏰ Keyingi tahlil 5 soatdan so'ng."
        )

    print("Tugadi.")

if __name__ == "__main__":
    main()
