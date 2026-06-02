"""
Sniper Scalping Signal Bot
==========================
Smart Money + Klassik S/R birlashtirilgan:

Smart Money:
  - Order Block (OB): so'nggi impuls oldidagi qarama-qarshi shamdon
  - Fair Value Gap (FVG): narx tez o'tib ketgan bo'sh zona
  - Liquidity Sweep: swing high/low ni sindirib qaytish
  - BOS (Break of Structure): tuzilma buzilishi

Klassik:
  - Swing High / Swing Low S/R levellar
  - EMA trend filtri (D1, H4)
  - RSI divergence
  - Volume tasdiqi

Scalping logikasi:
  - Timeframe: H1 signal, M15 entry
  - Entry: faqat OB yoki FVG ichida + S/R tasdiqi
  - TP: keyingi kuchli S/R level (ATR emas)
  - SL: OB/FVG ning narigi tomoni
  - Min R/R: 1:2
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ── Sozlamalar ────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_DATA_KEY  = os.environ["TWELVE_DATA_KEY"]

SYMBOLS = [
    {"symbol": "XAU/USD", "name": "Oltin",   "type": "forex",  "digits": 2},
    {"symbol": "BTC/USD", "name": "Bitcoin", "type": "crypto", "digits": 1},
]

MIN_RR       = 2.0   # Minimum Risk/Reward
MIN_SCORE    = 5     # Minimum signal kuchi
API_INTERVAL = 8     # Twelve Data free: 8s/so'rov

# ── API ───────────────────────────────────────────────────────────────────────
_last_req = 0

def _wait():
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < API_INTERVAL:
        time.sleep(API_INTERVAL - elapsed)
    _last_req = time.time()

def get_price(symbol: str) -> float | None:
    """Real-time narx"""
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/price",
            params={"symbol": symbol, "apikey": TWELVE_DATA_KEY},
            timeout=10
        )
        d = r.json()
        if "price" in d:
            return float(d["price"])
        print(f"  [{symbol}] price xato: {d.get('message','?')}")
        return None
    except Exception as e:
        print(f"  [{symbol}] price so'rov xato: {e}")
        return None

def get_candles(symbol: str, interval: str, size: int = 100) -> pd.DataFrame | None:
    _wait()
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol, "interval": interval,
                "outputsize": size, "apikey": TWELVE_DATA_KEY
            },
            timeout=15
        )
        d = r.json()
        if d.get("status") == "error" or "values" not in d:
            print(f"  [{symbol}/{interval}] xato: {d.get('message','?')}")
            return None
        df = pd.DataFrame(d["values"]).iloc[::-1].reset_index(drop=True)
        for col in ["open","high","low","close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    except Exception as e:
        print(f"  [{symbol}/{interval}] so'rov xato: {e}")
        return None

# ── Texnik funksiyalar ────────────────────────────────────────────────────────

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, 1e-10)))

def atr(df: pd.DataFrame, p: int = 14) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    v = tr.rolling(p).mean().dropna()
    return float(v.iloc[-1]) if len(v) > 0 else float(df["close"].iloc[-1]) * 0.003

# ── Smart Money funksiyalar ───────────────────────────────────────────────────

def find_swing_levels(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Swing High va Swing Low larni topish.
    Har bir bar uchun chap va o'ng tomondagi N bar bilan solishtirish.
    """
    highs = []
    lows  = []
    n = 3  # har tomondan 3 bar

    for i in range(n, len(df) - n):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        # Swing High: ikki tomonidagi barlardan baland
        if all(h >= df["high"].iloc[i-j] for j in range(1, n+1)) and \
           all(h >= df["high"].iloc[i+j] for j in range(1, n+1)):
            highs.append({"idx": i, "price": h})
        # Swing Low: ikki tomonidagi barlardan past
        if all(l <= df["low"].iloc[i-j] for j in range(1, n+1)) and \
           all(l <= df["low"].iloc[i+j] for j in range(1, n+1)):
            lows.append({"idx": i, "price": l})

    # So'nggi lookback ichidagilarni olish
    recent_highs = [x for x in highs if x["idx"] >= len(df) - lookback]
    recent_lows  = [x for x in lows  if x["idx"] >= len(df) - lookback]

    return {
        "highs": sorted(recent_highs, key=lambda x: x["price"], reverse=True),
        "lows":  sorted(recent_lows,  key=lambda x: x["price"])
    }

def find_order_blocks(df: pd.DataFrame, direction: str) -> list:
    """
    Order Block topish:
    BUY OB  = impuls UP oldidagi oxirgi bearish shamdon
    SELL OB = impuls DOWN oldidagi oxirgi bullish shamdon
    """
    obs = []
    # So'nggi 30 bardagi OBlarni topamiz
    for i in range(2, min(30, len(df) - 1)):
        idx = len(df) - 1 - i
        if idx < 1:
            break

        candle    = df.iloc[idx]
        next_c    = df.iloc[idx + 1]
        next_next = df.iloc[idx + 2] if idx + 2 < len(df) else None

        if direction == "BUY":
            # Bearish shamdon + keyin kuchli bullish impuls
            is_bearish = candle["close"] < candle["open"]
            if next_next is not None:
                impulse_up = (next_c["close"] > candle["high"] or
                              (next_next is not None and next_next["close"] > candle["high"]))
                if is_bearish and impulse_up:
                    obs.append({
                        "top":    candle["open"],   # bearish OB top = open
                        "bottom": candle["close"],  # bearish OB bottom = close
                        "idx":    idx
                    })
        else:  # SELL
            # Bullish shamdon + keyin kuchli bearish impuls
            is_bullish = candle["close"] > candle["open"]
            if next_next is not None:
                impulse_down = (next_c["close"] < candle["low"] or
                                (next_next is not None and next_next["close"] < candle["low"]))
                if is_bullish and impulse_down:
                    obs.append({
                        "top":    candle["close"],  # bullish OB top = close
                        "bottom": candle["open"],   # bullish OB bottom = open
                        "idx":    idx
                    })

    return obs

def find_fvg(df: pd.DataFrame, direction: str) -> list:
    """
    Fair Value Gap (Imbalance):
    BUY FVG:  candle[i-1].high < candle[i+1].low  → gap yuqoriga
    SELL FVG: candle[i-1].low  > candle[i+1].high → gap pastga
    """
    fvgs = []
    for i in range(1, min(25, len(df) - 1)):
        idx = len(df) - 1 - i
        if idx < 1 or idx + 1 >= len(df):
            continue
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]
        nxt  = df.iloc[idx + 1]

        if direction == "BUY":
            # Yuqoriga FVG: oldingi high < keyingi low
            if prev["high"] < nxt["low"]:
                fvgs.append({
                    "top":    nxt["low"],
                    "bottom": prev["high"],
                    "mid":    (nxt["low"] + prev["high"]) / 2,
                    "idx":    idx
                })
        else:
            # Pastga FVG: oldingi low > keyingi high
            if prev["low"] > nxt["high"]:
                fvgs.append({
                    "top":    prev["low"],
                    "bottom": nxt["high"],
                    "mid":    (prev["low"] + nxt["high"]) / 2,
                    "idx":    idx
                })

    return fvgs

def check_liquidity_sweep(df: pd.DataFrame, swing_levels: dict, direction: str) -> dict | None:
    """
    Liquidity Sweep:
    BUY:  narx swing low ni sindirib (pastga o'tib) keyin yuqoriga qaytdi
    SELL: narx swing high ni sindirib (yuqoriga o'tib) keyin pastga qaytdi
    """
    if not swing_levels["lows"] and not swing_levels["highs"]:
        return None

    last_bar  = df.iloc[-1]
    prev_bar  = df.iloc[-2]

    if direction == "BUY" and swing_levels["lows"]:
        # Eng yaqin swing low
        nearest_low = swing_levels["lows"][0]["price"]
        # Wick pastga sindirib, close yuqorida
        swept = (last_bar["low"] < nearest_low and last_bar["close"] > nearest_low)
        if not swept:
            swept = (prev_bar["low"] < nearest_low and prev_bar["close"] > nearest_low)
        if swept:
            return {"level": nearest_low, "type": "Sell-side liquidity sweep"}

    elif direction == "SELL" and swing_levels["highs"]:
        nearest_high = swing_levels["highs"][0]["price"]
        swept = (last_bar["high"] > nearest_high and last_bar["close"] < nearest_high)
        if not swept:
            swept = (prev_bar["high"] > nearest_high and prev_bar["close"] < nearest_high)
        if swept:
            return {"level": nearest_high, "type": "Buy-side liquidity sweep"}

    return None

def check_bos(df: pd.DataFrame, swing_levels: dict, direction: str) -> bool:
    """
    Break of Structure:
    BUY:  so'nggi swing high sindirildi (bullish BOS)
    SELL: so'nggi swing low sindirildi (bearish BOS)
    """
    if len(df) < 5:
        return False

    close_now = df["close"].iloc[-1]

    if direction == "BUY" and swing_levels["highs"]:
        # So'nggi swing high dan yuqoriga o'tish
        recent_high = swing_levels["highs"][0]["price"]
        return close_now > recent_high

    elif direction == "SELL" and swing_levels["lows"]:
        recent_low = swing_levels["lows"][0]["price"]
        return close_now < recent_low

    return False

def find_tp_level(swing_levels: dict, direction: str, entry: float, min_rr: float, sl: float) -> float | None:
    """
    TP = keyingi kuchli S/R level (ATR emas, haqiqiy level)
    """
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return None

    if direction == "BUY":
        # Entry dan yuqoridagi swing highlar
        candidates = [x["price"] for x in swing_levels["highs"]
                      if x["price"] > entry * 1.001]
        candidates.sort()
        for lvl in candidates:
            rr = (lvl - entry) / sl_dist
            if rr >= min_rr:
                return round(lvl, 2)
    else:
        # Entry dan pastdagi swing lowlar
        candidates = [x["price"] for x in swing_levels["lows"]
                      if x["price"] < entry * 0.999]
        candidates.sort(reverse=True)
        for lvl in candidates:
            rr = (entry - lvl) / sl_dist
            if rr >= min_rr:
                return round(lvl, 2)

    return None

# ── Asosiy tahlil ─────────────────────────────────────────────────────────────

def analyze(item: dict) -> dict | None:
    symbol = item["symbol"]
    digits = item["digits"]
    confirmations  = []
    direction_votes = {"BUY": 0, "SELL": 0}

    # ── Real narx ─────────────────────────────────────────────────────────────
    print(f"  [{symbol}] Real narx...")
    price = get_price(symbol)
    if price is None:
        return None
    print(f"  [{symbol}] Narx: {price}")

    # ── D1: Asosiy trend (EMA 50/200) ─────────────────────────────────────────
    print(f"  [{symbol}] D1 trend...")
    d1 = get_candles(symbol, "1day", 220)
    d1_trend = "NEUTRAL"
    if d1 is not None and len(d1) >= 200:
        e50  = float(ema(d1["close"], 50).iloc[-1])
        e200 = float(ema(d1["close"], 200).iloc[-1])
        if e50 > e200 * 1.001:
            d1_trend = "BUY"
            confirmations.append(("D1 Trend", "BUY", f"EMA50 > EMA200 — Uptrend"))
            direction_votes["BUY"] += 2
        elif e50 < e200 * 0.999:
            d1_trend = "SELL"
            confirmations.append(("D1 Trend", "SELL", f"EMA50 < EMA200 — Downtrend"))
            direction_votes["SELL"] += 2

    # ── H4: Oraliq trend ──────────────────────────────────────────────────────
    print(f"  [{symbol}] H4 trend...")
    h4 = get_candles(symbol, "4h", 60)
    if h4 is not None and len(h4) >= 50:
        e20 = float(ema(h4["close"], 20).iloc[-1])
        e50_h4 = float(ema(h4["close"], 50).iloc[-1])
        if e20 > e50_h4:
            confirmations.append(("H4 Trend", "BUY", "EMA20 > EMA50"))
            direction_votes["BUY"] += 1
        else:
            confirmations.append(("H4 Trend", "SELL", "EMA20 < EMA50"))
            direction_votes["SELL"] += 1

    # ── H1: Smart Money tahlil ────────────────────────────────────────────────
    print(f"  [{symbol}] H1 Smart Money...")
    h1 = get_candles(symbol, "1h", 100)
    if h1 is None or len(h1) < 50:
        print(f"  [{symbol}] H1 ma'lumot yetarli emas")
        return None

    # Swing levels (H1)
    swings_h1 = find_swing_levels(h1, lookback=30)

    # Dominant yo'nalishni aniqlash (shu paytgacha)
    temp_dir = "BUY" if direction_votes["BUY"] >= direction_votes["SELL"] else "SELL"

    # Order Block
    obs = find_order_blocks(h1, temp_dir)
    ob_zone = None
    if obs:
        ob = obs[0]  # eng yaqin OB
        # Narx OB ichidami?
        if ob["bottom"] <= price <= ob["top"]:
            ob_zone = ob
            confirmations.append(("Order Block", temp_dir,
                f"Narx OB ichida ({ob['bottom']:.{digits}f}–{ob['top']:.{digits}f})"))
            direction_votes[temp_dir] += 3
        elif abs(price - ob["top"]) / price < 0.003:
            ob_zone = ob
            confirmations.append(("Order Block", temp_dir,
                f"Narx OB yaqinida ({ob['bottom']:.{digits}f}–{ob['top']:.{digits}f})"))
            direction_votes[temp_dir] += 2

    # FVG
    fvgs = find_fvg(h1, temp_dir)
    fvg_zone = None
    if fvgs:
        fvg = fvgs[0]
        if fvg["bottom"] <= price <= fvg["top"]:
            fvg_zone = fvg
            confirmations.append(("FVG", temp_dir,
                f"Narx FVG ichida ({fvg['bottom']:.{digits}f}–{fvg['top']:.{digits}f})"))
            direction_votes[temp_dir] += 2
        elif abs(price - fvg["mid"]) / price < 0.005:
            fvg_zone = fvg
            confirmations.append(("FVG", temp_dir,
                f"Narx FVG yaqinida (mid: {fvg['mid']:.{digits}f})"))
            direction_votes[temp_dir] += 1

    # Liquidity Sweep
    sweep = check_liquidity_sweep(h1, swings_h1, temp_dir)
    if sweep:
        confirmations.append(("Liquidity Sweep", temp_dir,
            f"{sweep['type']} — {sweep['level']:.{digits}f}"))
        direction_votes[temp_dir] += 3

    # BOS (Break of Structure)
    if check_bos(h1, swings_h1, temp_dir):
        confirmations.append(("BOS", temp_dir, "Tuzilma buzildi — trend tasdiqlandi"))
        direction_votes[temp_dir] += 2

    # RSI
    rsi_s   = rsi(h1["close"], 14)
    rsi_val = float(rsi_s.iloc[-1])
    rsi_prv = float(rsi_s.iloc[-2])
    if rsi_val < 30:
        confirmations.append(("RSI", "BUY",  f"Oversold: {rsi_val:.1f}"))
        direction_votes["BUY"] += 2
    elif rsi_val > 70:
        confirmations.append(("RSI", "SELL", f"Overbought: {rsi_val:.1f}"))
        direction_votes["SELL"] += 2
    elif rsi_prv < 35 and rsi_val > rsi_prv:
        confirmations.append(("RSI", "BUY",  f"Oversold qaytish: {rsi_val:.1f}"))
        direction_votes["BUY"] += 1
    elif rsi_prv > 65 and rsi_val < rsi_prv:
        confirmations.append(("RSI", "SELL", f"Overbought qaytish: {rsi_val:.1f}"))
        direction_votes["SELL"] += 1
    elif 50 < rsi_val < 65:
        confirmations.append(("RSI", "BUY",  f"Bullish zona: {rsi_val:.1f}"))
        direction_votes["BUY"] += 1
    elif 35 < rsi_val < 50:
        confirmations.append(("RSI", "SELL", f"Bearish zona: {rsi_val:.1f}"))
        direction_votes["SELL"] += 1

    # Volume tasdiqi
    if h1["volume"].sum() > 0:
        vol_now = float(h1["volume"].iloc[-1])
        vol_avg = float(h1["volume"].iloc[-20:].mean())
        if vol_avg > 0 and vol_now > vol_avg * 1.3:
            dom = "BUY" if direction_votes["BUY"] >= direction_votes["SELL"] else "SELL"
            confirmations.append(("Volume", dom,
                f"Hajm {vol_now/vol_avg:.1f}x yuqori — kuchli harakat"))
            direction_votes[dom] += 1

    # ── Yakuniy yo'nalish ──────────────────────────────────────────────────────
    buy_sc  = direction_votes["BUY"]
    sell_sc = direction_votes["SELL"]

    if buy_sc > sell_sc:
        action = "BUY"
        score  = buy_sc
    elif sell_sc > buy_sc:
        action = "SELL"
        score  = sell_sc
    else:
        return None

    # Faqat shu yo'nalishdagi tasdiqlashlar
    act_confs = [c for c in confirmations if c[1] == action]
    if score < MIN_SCORE or len(act_confs) < 3:
        print(f"  [{symbol}] Zaif: score={score}, conf={len(act_confs)}")
        return None

    # ── Entry zone (OB yoki FVG markazi) ──────────────────────────────────────
    zone = ob_zone or fvg_zone
    if zone:
        entry_zone_top    = round(zone["top"],    digits)
        entry_zone_bottom = round(zone["bottom"], digits)
        # Optimal entry: zone ning 50%-si
        entry_price = round((zone["top"] + zone["bottom"]) / 2, digits)
    else:
        # Zona topilmasa real narxni ishlatish
        entry_price       = round(price, digits)
        entry_zone_top    = round(price, digits)
        entry_zone_bottom = round(price, digits)

    # ── SL: OB/FVG ning narigi tomoni ─────────────────────────────────────────
    atr_val = atr(h1)
    if zone:
        if action == "BUY":
            sl = round(zone["bottom"] - atr_val * 0.3, digits)  # OB pastidan bir oz past
        else:
            sl = round(zone["top"] + atr_val * 0.3, digits)     # OB tepasidan bir oz yuqori
    else:
        sl = round(price - atr_val * 0.8, digits) if action == "BUY" else \
             round(price + atr_val * 0.8, digits)

    # ── TP: Keyingi kuchli S/R level ──────────────────────────────────────────
    # H4 swing levels (kattaroq timeframe = kuchliroq level)
    swings_h4 = find_swing_levels(h4, lookback=20) if h4 is not None else swings_h1

    tp1 = find_tp_level(swings_h4, action, entry_price, MIN_RR, sl)
    if tp1 is None:
        # H1 levellardan topish
        tp1 = find_tp_level(swings_h1, action, entry_price, MIN_RR, sl)
    if tp1 is None:
        # Hech narsa topilmasa ATR * 2 ishlatish
        tp1 = round(entry_price + atr_val * 2, digits) if action == "BUY" else \
              round(entry_price - atr_val * 2, digits)

    # TP2: TP1 dan keyingi level
    sl_dist = abs(entry_price - sl)
    tp2_candidates = [
        x["price"] for x in (swings_h4["highs"] if action == "BUY" else swings_h4["lows"])
        if (x["price"] > tp1 if action == "BUY" else x["price"] < tp1)
    ]
    if tp2_candidates:
        tp2 = round(
            min(tp2_candidates) if action == "BUY" else max(tp2_candidates),
            digits
        )
    else:
        tp2 = round(entry_price + atr_val * 3, digits) if action == "BUY" else \
              round(entry_price - atr_val * 3, digits)

    rr = round(abs(tp1 - entry_price) / sl_dist, 2) if sl_dist > 0 else 0

    if rr < MIN_RR:
        print(f"  [{symbol}] R/R yetarli emas: {rr}")
        return None

    return {
        **item,
        "action":           action,
        "price":            round(price, digits),
        "entry_zone_top":   entry_zone_top,
        "entry_zone_bottom":entry_zone_bottom,
        "entry_price":      entry_price,
        "tp1":              tp1,
        "tp2":              tp2,
        "sl":               sl,
        "rr":               rr,
        "score":            score,
        "confirmations":    act_confs,
        "rsi_val":          round(rsi_val, 1),
        "atr":              round(atr_val, digits),
        "has_ob":           ob_zone is not None,
        "has_fvg":          fvg_zone is not None,
        "has_sweep":        sweep is not None,
    }

# ── Telegram ──────────────────────────────────────────────────────────────────

def score_bar(s: int) -> str:
    if s >= 12: return "●●●●●"
    if s >= 9:  return "●●●●○"
    if s >= 7:  return "●●●○○"
    if s >= 5:  return "●●○○○"
    return "●○○○○"

EMOJI = {"forex": "💱", "crypto": "🪙"}

def format_msg(s: dict) -> str:
    e      = "🟢" if s["action"] == "BUY"  else "🔴"
    act    = "SOTIB OL" if s["action"] == "BUY" else "SOT"
    now    = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    d      = s["digits"]
    bar    = score_bar(s["score"])
    te     = EMOJI.get(s["type"], "📊")

    # Smart Money belgilari
    sm_tags = []
    if s.get("has_ob"):     sm_tags.append("OB✓")
    if s.get("has_fvg"):    sm_tags.append("FVG✓")
    if s.get("has_sweep"):  sm_tags.append("Sweep✓")
    sm_str = "  ".join(sm_tags) if sm_tags else "—"

    conf_lines = "\n".join(f"  ✅ {c[0]}: {c[2]}" for c in s["confirmations"])

    # Entry zone bir xil bo'lsa faqat narxni ko'rsatish
    if s["entry_zone_top"] == s["entry_zone_bottom"]:
        entry_str = f"💰 Entry: <b>{s['entry_price']:.{d}f}</b>"
    else:
        entry_str = (
            f"📍 Entry Zone: <b>{s['entry_zone_bottom']:.{d}f} – {s['entry_zone_top']:.{d}f}</b>\n"
            f"💰 Optimal Entry: <b>{s['entry_price']:.{d}f}</b>"
        )

    return (
        f"{e} <b>{s['symbol']} — {act}</b> {te}\n"
        f"<i>{s['name']}</i> | {bar} ({s['score']} ball)\n"
        f"Smart Money: {sm_str}\n\n"
        f"🔴 Real narx: {s['price']:.{d}f}\n"
        f"{entry_str}\n"
        f"🎯 TP1: <b>{s['tp1']:.{d}f}</b>  ← keyingi S/R\n"
        f"🎯 TP2: <b>{s['tp2']:.{d}f}</b>  ← kuchli S/R\n"
        f"🛑 SL:  <b>{s['sl']:.{d}f}</b>   ← zona tashqarisi\n"
        f"⚖️ R/R: <b>1:{s['rr']}</b>\n\n"
        f"Tasdiqlashlar ({len(s['confirmations'])} ta):\n"
        f"{conf_lines}\n\n"
        f"📊 RSI: {s['rsi_val']} | ATR: {s['atr']:.{d}f}\n"
        f"⏰ {now}\n\n"
        f"⚠️ Faqat tahlil. Risk: 1-2% dan oshirmang."
    )

def send(msg: str):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        res = r.json()
        if res.get("ok"):
            print("  ✓ Yuborildi")
        else:
            print(f"  ✗ Xato: {res}")
    except Exception as e:
        print(f"  ✗ {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_str = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    print(f"\n{'='*55}")
    print(f"Sniper Bot: {now_str}")
    print(f"{'='*55}")

    send(
        f"🔍 <b>Sniper tahlil — {now_str}</b>\n"
        f"Smart Money + Klassik S/R\n"
        f"XAU/USD va BTC/USD..."
    )

    signals = []
    for item in SYMBOLS:
        print(f"\n[{item['symbol']}]...")
        try:
            res = analyze(item)
            if res:
                signals.append(res)
                print(
                    f"  ✓ {res['action']} | "
                    f"Entry: {res['entry_price']} | "
                    f"TP1: {res['tp1']} | "
                    f"SL: {res['sl']} | "
                    f"R/R: {res['rr']} | "
                    f"Score: {res['score']}"
                )
            else:
                print(f"  — Signal yo'q")
        except Exception as e:
            print(f"  XATO: {e}")
            import traceback; traceback.print_exc()

    now_str = datetime.utcnow().strftime("%d.%m.%Y %H:%M UTC")
    print(f"\n{'='*55}")
    print(f"Jami: {len(signals)} signal")

    if signals:
        signals.sort(key=lambda x: x["score"], reverse=True)
        send(
            f"📡 <b>SNIPER SIGNAL — {now_str}</b>\n"
            f"<b>{len(signals)} ta signal</b>\n{'─'*30}"
        )
        time.sleep(0.5)
        for s in signals:
            send(format_msg(s))
            time.sleep(0.5)
    else:
        send(
            f"📊 <b>Tahlil — {now_str}</b>\n\n"
            f"Signal yo'q — bozor hozir kutish holatida.\n"
            f"Sniper entry sharti bajarilmadi.\n"
            f"⏰ Keyingi tekshiruv 4 soatdan so'ng."
        )

    print("Tugadi.")

if __name__ == "__main__":
    main()
