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
    df = fetch_klines(symbol, INTERVAL, LIMIT)
    if df is None or len(df) < 30:
        return f"*{symbol}*\n❌ Ma'lumot yetarli emas"

    close_prices = df["close"].tolist()
    last_price   = close_prices[-1]

    # ── Asl logika (o'zgartirilmagan) ────────────────────────────────────────
    e9  = ema(close_prices, 9)
    e21 = ema(close_prices, 21)
    last_ema9  = e9[-1]
    last_ema21 = e21[-1]

    last_rsi = rsi(close_prices)

    e12 = ema(close_prices, 12)
    e26 = ema(close_prices, 26)
    macd_line = e12[-1] - e26[-1]

    if last_ema9 > last_ema21 and last_rsi < 70 and macd_line > 0:
        signal = "BUY"
    elif last_ema9 < last_ema21 and last_rsi > 30 and macd_line < 0:
        signal = "SELL"
    else:
        return (
            f"*{symbol}*\n"
            f"Signal: ⚪ NO SIGNAL\n"
            f"Price: {last_price:.{DIGITS}f}\n"
            f"RSI: {last_rsi:.2f} | MACD: {macd_line:.2f}"
        )

    # ── Qo'shimcha hisob-kitoblar ─────────────────────────────────────────────
    atr_val  = atr(df, 14)
    sl_dist  = round(atr_val * 0.7, DIGITS)

    if signal == "BUY":
        sl = round(last_price - sl_dist, DIGITS)
    else:
        sl = round(last_price + sl_dist, DIGITS)

    # Swing levellardan TP
    highs = swing_highs(df, n=3)
    lows  = swing_lows(df,  n=3)

    if signal == "BUY":
        tp1 = next_level_above(highs, last_price)
        tp2_candidates = sorted([l for l in highs if l > last_price * 1.0005])
        tp2 = tp2_candidates[1] if len(tp2_candidates) >= 2 else None
        tp3 = tp2_candidates[2] if len(tp2_candidates) >= 3 else None

        if tp1 is None: tp1 = round(last_price + atr_val * 2.0, DIGITS)
        if tp2 is None: tp2 = round(last_price + atr_val * 3.0, DIGITS)
        if tp3 is None: tp3 = round(last_price + atr_val * 4.0, DIGITS)

        tp1, tp2, tp3 = sorted([tp1, tp2, tp3])

    else:
        tp1 = next_level_below(lows, last_price)
        tp2_candidates = sorted([l for l in lows if l < last_price * 0.9995], reverse=True)
        tp2 = tp2_candidates[1] if len(tp2_candidates) >= 2 else None
        tp3 = tp2_candidates[2] if len(tp2_candidates) >= 3 else None

        if tp1 is None: tp1 = round(last_price - atr_val * 2.0, DIGITS)
        if tp2 is None: tp2 = round(last_price - atr_val * 3.0, DIGITS)
        if tp3 is None: tp3 = round(last_price - atr_val * 4.0, DIGITS)

        tp1, tp2, tp3 = sorted([tp1, tp2, tp3], reverse=True)

    def rr(tp):
        return round(abs(tp - last_price) / sl_dist, 1) if sl_dist > 0 else 0

    # MACD histogram (qo'shimcha ma'lumot)
    hist_val = float(macd_hist(df["close"]).iloc[-1])

    emoji  = "📈" if signal == "BUY" else "📉"

    return (
        f"{emoji} *{symbol} — {signal}*\n\n"
        f"💰 Entry:  `{last_price:.{DIGITS}f}`\n"
        f"🎯 TP1:   `{tp1:.{DIGITS}f}`  (1:{rr(tp1)}R)\n"
        f"🎯 TP2:   `{tp2:.{DIGITS}f}`  (1:{rr(tp2)}R)\n"
        f"🎯 TP3:   `{tp3:.{DIGITS}f}`  (1:{rr(tp3)}R)\n"
        f"🛑 SL:    `{sl:.{DIGITS}f}`  (0.7×ATR: {atr_val:.{DIGITS}f})\n\n"
        f"RSI: {last_rsi:.2f} | MACD line: {macd_line:.2f} | MACD hist: {hist_val:.4f}\n"
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
