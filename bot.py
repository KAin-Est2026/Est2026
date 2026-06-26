import os
import time
import requests
import pandas as pd
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
import logging

# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# =========================
# Config
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
SYMBOLS   = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]   # Kerakli symbollarni shu yerga qo'shing
BINANCE_API = "https://api.binance.com/api/v3/klines"
INTERVAL  = "1h"
LIMIT     = 100
DIGITS    = 2


# =========================
# Indikatorlar
# =========================

def ema_series(s: pd.Series, period: int) -> pd.Series:
    """Pandas Series uchun EMA."""
    return s.ewm(span=period, adjust=False).mean()


def ema(values: list, period: int) -> list:
    """List uchun EMA (asl logika saqlandi)."""
    k = 2 / (period + 1)
    ema_array = [values[0]]
    for i in range(1, len(values)):
        ema_array.append(values[i] * k + ema_array[i - 1] * (1 - k))
    return ema_array


def rsi(values: list, period: int = 14) -> float:
    """RSI — oxirgi qiymat (asl logika saqlandi)."""
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


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(period).mean().dropna()
    return float(val.iloc[-1]) if len(val) > 0 else 1.0


def macd_hist(s: pd.Series) -> pd.Series:
    """MACD histogram."""
    m = ema_series(s, 12) - ema_series(s, 26)
    return m - ema_series(m, 9)


def swing_highs(df: pd.DataFrame, n: int = 3) -> list:
    """Swing high levellar."""
    levels = []
    for i in range(n, len(df) - n):
        h = df["high"].iloc[i]
        if all(h > df["high"].iloc[i - j] for j in range(1, n + 1)) and \
           all(h > df["high"].iloc[i + j] for j in range(1, n + 1)):
            levels.append(h)
    return sorted(set(round(x, DIGITS) for x in levels))


def swing_lows(df: pd.DataFrame, n: int = 3) -> list:
    """Swing low levellar."""
    levels = []
    for i in range(n, len(df) - n):
        l = df["low"].iloc[i]
        if all(l < df["low"].iloc[i - j] for j in range(1, n + 1)) and \
           all(l < df["low"].iloc[i + j] for j in range(1, n + 1)):
            levels.append(l)
    return sorted(set(round(x, DIGITS) for x in levels))


def next_level_above(levels: list, price: float):
    above = [l for l in levels if l > price * 1.0005]
    return min(above) if above else None


def next_level_below(levels: list, price: float):
    below = [l for l in levels if l < price * 0.9995]
    return max(below) if below else None


# =========================
# Binance API
# =========================

def fetch_klines(symbol: str, interval: str = INTERVAL, limit: int = LIMIT):
    """Binance API dan OHLC DataFrame olish."""
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(BINANCE_API, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "qav", "trades",
            "tbbav", "tbqav", "ignore"
        ])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    except Exception as e:
        logging.error(f"[{symbol}] fetch_klines xatosi: {e}")
        return None


# =========================
# Tahlil
# =========================

def analyze(symbol: str) -> str:
    # ── H4: asosiy trend ──────────────────────────────────────────────────────
    h4 = fetch_klines(symbol, "4h", 250)
    if h4 is None or len(h4) < 200:
        return f"*{symbol}*\n❌ H4 ma'lumot yetarli emas"

    e50_h4  = float(ema_series(h4["close"], 50).iloc[-1])
    e200_h4 = float(ema_series(h4["close"], 200).iloc[-1])

    if e50_h4 > e200_h4 * 1.001:
        trend = "BUY"
    elif e50_h4 < e200_h4 * 0.999:
        trend = "SELL"
    else:
        return f"*{symbol}*\nSignal: ⚪ NO SIGNAL\nH4 trend aniq emas"

    # ── H1: zona tasdiqi ──────────────────────────────────────────────────────
    h1 = fetch_klines(symbol, "1h", 100)
    if h1 is None or len(h1) < 50:
        return f"*{symbol}*\n❌ H1 ma'lumot yetarli emas"

    e50_h1   = float(ema_series(h1["close"], 50).iloc[-1])
    price_h1 = float(h1["close"].iloc[-1])

    if trend == "BUY"  and price_h1 < e50_h1:
        return f"*{symbol}*\nSignal: ⚪ NO SIGNAL\nH1: narx EMA50 ostida"
    if trend == "SELL" and price_h1 > e50_h1:
        return f"*{symbol}*\nSignal: ⚪ NO SIGNAL\nH1: narx EMA50 ustida"

    # ── M15: entry ────────────────────────────────────────────────────────────
    m15 = fetch_klines(symbol, "15m", 60)
    if m15 is None or len(m15) < 40:
        return f"*{symbol}*\n❌ M15 ma'lumot yetarli emas"

    e9_m15  = ema_series(m15["close"], 9)
    e21_m15 = ema_series(m15["close"], 21)

    cross_up_m15 = any(
        e9_m15.iloc[i-1] < e21_m15.iloc[i-1] and e9_m15.iloc[i] >= e21_m15.iloc[i]
        for i in range(-5, 0)
    )
    cross_down_m15 = any(
        e9_m15.iloc[i-1] > e21_m15.iloc[i-1] and e9_m15.iloc[i] <= e21_m15.iloc[i]
        for i in range(-5, 0)
    )

    # ── M5: sniper entry ──────────────────────────────────────────────────────
    m5 = fetch_klines(symbol, "5m", 60)
    if m5 is None or len(m5) < 30:
        return f"*{symbol}*\n❌ M5 ma'lumot yetarli emas"

    e9_m5  = ema_series(m5["close"], 9)
    e21_m5 = ema_series(m5["close"], 21)

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

    if not m15_ok and not m5_ok:
        return f"*{symbol}*\nSignal: ⚪ NO SIGNAL\nM15 va M5 cross yo'q"

    entry_tf = "M15" if m15_ok else "M5"

    # ── MACD tasdiqi (M15) ────────────────────────────────────────────────────
    hist  = macd_hist(m15["close"])
    h_now = float(hist.iloc[-1])
    h_prv = float(hist.iloc[-2])

    if trend == "BUY"  and not (h_now > 0 or h_now > h_prv):
        return f"*{symbol}*\nSignal: ⚪ NO SIGNAL\nMACD BUY tasdiqlamadi"
    if trend == "SELL" and not (h_now < 0 or h_now < h_prv):
        return f"*{symbol}*\nSignal: ⚪ NO SIGNAL\nMACD SELL tasdiqlamadi"

    # ── Narx, ATR, SL ─────────────────────────────────────────────────────────
    last_price = float(m15["close"].iloc[-1])
    atr_val    = atr(m15, 14)
    sl_dist    = round(atr_val * 0.7, DIGITS)

    sl = round(last_price - sl_dist, DIGITS) if trend == "BUY" else round(last_price + sl_dist, DIGITS)

    # ── TP: swing levellardan ─────────────────────────────────────────────────
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

        tp1, tp2, tp3 = sorted([tp1, tp2, tp3])
    else:
        tp1 = next_level_below(h1_lows, last_price)
        tp2 = next_level_below(h4_lows, last_price)
        h4_below = sorted([l for l in h4_lows if l < last_price * 0.9995], reverse=True)
        tp3 = h4_below[1] if len(h4_below) >= 2 else None

        if tp1 is None: tp1 = round(last_price - atr_val * 2.0, DIGITS)
        if tp2 is None: tp2 = round(last_price - atr_val * 3.0, DIGITS)
        if tp3 is None: tp3 = round(last_price - atr_val * 4.0, DIGITS)

        tp1, tp2, tp3 = sorted([tp1, tp2, tp3], reverse=True)

    def rr(tp):
        return round(abs(tp - last_price) / sl_dist, 1) if sl_dist > 0 else 0

    emoji = "📈" if trend == "BUY" else "📉"
    tr_txt = "Uptrend" if trend == "BUY" else "Downtrend"

    return (
        f"{emoji} *{symbol} — {trend}*\n\n"
        f"💰 Entry:  `{last_price:.{DIGITS}f}`\n"
        f"🎯 TP1:   `{tp1:.{DIGITS}f}`  (1:{rr(tp1)}R)\n"
        f"🎯 TP2:   `{tp2:.{DIGITS}f}`  (1:{rr(tp2)}R)\n"
        f"🎯 TP3:   `{tp3:.{DIGITS}f}`  (1:{rr(tp3)}R)\n"
        f"🛑 SL:    `{sl:.{DIGITS}f}`  (0.7×ATR: {atr_val:.{DIGITS}f})\n\n"
        f"✅ H4 {tr_txt}\n"
        f"✅ H1 narx EMA50 {'ustida' if trend == 'BUY' else 'ostida'}\n"
        f"✅ {entry_tf} EMA9/21 kesdi\n"
        f"✅ MACD hist: {h_now:.4f}\n"
        f"⚠️ Risk: 1-2%"
    )


# =========================
# Telegram Handler
# =========================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    results = []
    for symbol in SYMBOLS:
        try:
            results.append(analyze(symbol))
        except Exception as e:
            results.append(f"*{symbol}*\n❌ Xatolik: {e}")

    now = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    header = f"📊 *Bozor tahlili — {now}*\n\n"

    await update.message.reply_text(
        header + "\n\n".join(results),
        parse_mode="Markdown"
    )


# =========================
# Main
# =========================

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logging.info("Bot ishga tushdi...")
    app.run_polling()


if __name__ == "__main__":
    main()
