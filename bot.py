import os, json, time, traceback
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ---------------- Settings ----------------
SESSION_START_UTC = 13.5   # NY session (after 1 Nov change to 14.5)
SESSION_END_UTC = 20.0     # (after 1 Nov change to 21.0)
MIN_RR = 1.2
MAX_HOURS = 24             # close paper trade after this many hours
VA_ROWS = 77               # S4 volume profile rows
VA_PCT = 0.94              # S4 value area %
FRESH = {"5m": 20, "15m": 35}
MINS = {"5m": 5, "15m": 15, "1h": 60}
SRC = {
    "GOLD": ("yahoo", "GC=F"),
    "BTC": ("yahoo", "BTC-USD"),
    "PAXG": ("binance", "PAXGUSDT"),
}
RES_TF = {"GOLD": "15m", "BTC": "15m", "PAXG": "5m"}
VOLIDX = {"GOLD": ["^GVZ", "^VIX"], "BTC": ["^VIX"]}
BINANCE_HOSTS = ["data-api.binance.vision", "api.binance.com", "api1.binance.com"]


def tg(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg}, timeout=20)
    except Exception as e:
        print("Telegram error:", e)


# ---------------- Data ----------------
def yahoo_df(sym, tf):
    period = {"5m": "5d", "15m": "30d", "1h": "60d"}[tf]
    df = yf.Ticker(sym).history(period=period, interval=tf)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = df.index.tz_convert("UTC")
    return df


def binance_df(sym, tf):
    last = None
    for host in BINANCE_HOSTS:
        try:
            r = requests.get(f"https://{host}/api/v3/klines",
                             params={"symbol": sym, "interval": tf, "limit": 500},
                             timeout=20)
            r.raise_for_status()
            rows = r.json()
            df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close",
                                             "Volume", "ct", "qv", "n", "TakerBuy",
                                             "tq", "x"])
            for c in ["Open", "High", "Low", "Close", "Volume", "TakerBuy"]:
                df[c] = df[c].astype(float)
            df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
            df["Delta"] = 2 * df.TakerBuy - df.Volume
            return df[["Open", "High", "Low", "Close", "Volume", "Delta"]]
        except Exception as e:
            last = e
            print("Binance host failed:", host, e)
    raise last


_cache = {}


def get_df(asset, tf):
    key = (asset, tf)
    if key in _cache:
        return _cache[key]
    kind, sym = SRC[asset]
    df = None
    for _ in range(3):
        try:
            df = yahoo_df(sym, tf) if kind == "yahoo" else binance_df(sym, tf)
            if df is not None and len(df) > 0:
                break
        except Exception as e:
            print("Data error", asset, tf, e)
        time.sleep(2)
    _cache[key] = df
    return df


def done_bars(df, tf):
    """keep only finished candles; None if market closed / data stale"""
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(minutes=MINS[tf]) <= now]
    if len(df) < 30:
        return None
    age = (now - (df.index[-1] + pd.Timedelta(minutes=MINS[tf]))).total_seconds() / 60
    if tf in FRESH and age > FRESH[tf]:
        return None
    return df


def stamp(sig, df, tf):
    sig["close_time"] = (df.index[-1] + pd.Timedelta(minutes=MINS[tf])).isoformat()
    return sig


# ---------------- Helpers ----------------
def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(),
                    (df.Low - pc).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def swings(df, n=2):
    hi, lo = df.High.values, df.Low.values
    H, L = [], []
    for i in range(n, len(df) - n):
        if hi[i] == max(hi[i - n:i + n + 1]):
            H.append((i, float(hi[i])))
        if lo[i] == min(lo[i - n:i + n + 1]):
            L.append((i, float(lo[i])))
    return H, L


def flip(df):
    f = df.copy()
    f["Open"], f["High"], f["Low"], f["Close"] = -df.Open, -df.Low, -df.High, -df.Close
    if "Delta" in f:
        f["Delta"] = -df.Delta
    return f


def mk(side, entry, sl, tp):
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    if side == "BUY" and tp <= entry:
        return None
    if side == "SELL" and tp >= entry:
        return None
    if abs(tp - entry) / risk < MIN_RR:
        return None
    return {"side": side, "entry": float(entry), "sl": float(sl), "tp": float(tp)}


def both(fn, *dfs):
    """run long logic; then run it on mirrored data to get the short signal"""
    r = fn(*dfs)
    if r:
        return r
    r = fn(*[flip(x) for x in dfs])
    if r:
        return dict(r, side="SELL", entry=-r["entry"], sl=-r["sl"], tp=-r["tp"])
    return None


def profile(df, rows):
    lo, hi = df.Low.min(), df.High.max()
    if hi <= lo:
        return None
    p = (df.High + df.Low + df.Close) / 3
    h, e = np.histogram(p, bins=rows, range=(lo, hi), weights=df.Volume)
    if h.sum() <= 0:
        return None
    return h, e


def value_area(h, e, pct):
    i = int(h.argmax())
    lo = hi = i
    tot, cur = h.sum(), h[i]
    while cur < pct * tot and (lo > 0 or hi < len(h) - 1):
        up = h[hi + 1] if hi < len(h) - 1 else -1
        dn = h[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1
            cur += h[hi]
        else:
            lo -= 1
            cur += h[lo]
    return (e[i] + e[i + 1]) / 2, e[hi + 1], e[lo]   # POC, VAH, VAL


def bull_engulf(p, c):
    return p.Close < p.Open and c.Close > c.Open and c.Close >= p.Open and c.Open <= p.Close


def bear_engulf(p, c):
    return p.Close > p.Open and c.Close < c.Open and c.Close <= p.Open and c.Open >= p.Close


def rejection(levels, allowed, d):
    """levels = [(key, price)] sorted by price (low to high)"""
    c, pv, a = d.iloc[-1], d.iloc[-2], atr(d)
    for i, (k, px) in enumerate(levels):
        if k not in allowed:
            continue
        sig = None
        if i > 0 and c.High >= px > c.Close and bear_engulf(pv, c):
            sig = mk("SELL", c.Close, c.High + 0.05 * a, levels[i - 1][1])
        elif i < len(levels) - 1 and c.Low <= px < c.Close and bull_engulf(pv, c):
            sig = mk("BUY", c.Close, c.Low - 0.05 * a, levels[i + 1][1])
        if sig:
            return sig
    return None


# ---------------- S1: Gold liquidity sweep ----------------
def trend_up(df):
    H, L = swings(df, 2)
    return len(H) >= 2 and len(L) >= 2 and H[-1][1] > H[-2][1] and L[-1][1] > L[-2][1]


def s1_long(d15, d1h):
    if not (trend_up(d1h) and trend_up(d15)):
        return None
    H, L = swings(d15, 2)
    sw_i, sw_p = L[-1]
    c = d15.iloc[-1]
    if not (c.Low < sw_p < c.Close):          # sweep candle: breaks low, closes back above
        return None
    if len(d15) - sw_i > 40:
        return None
    win = d15.tail(48)
    lo, hi = win.Low.min(), win.High.max()
    if hi <= lo or (c.Close - lo) / (hi - lo) > 0.4:   # must be in demand area, not middle
        return None
    ups = [p for i, p in H[-6:] if p > c.Close]
    if not ups:
        return None
    sig = mk("BUY", c.Close, c.Low - 0.05 * atr(d15), min(ups))
    if sig:
        sig["note"] = "1H+15M trend aligned, liquidity sweep in demand area"
    return sig


def s1(asset):
    d15, d1h = get_df(asset, "15m"), get_df(asset, "1h")
    if d15 is None or d1h is None:
        return None
    d15, d1h = done_bars(d15, "15m"), done_bars(d1h, "1h")
    if d15 is None or d1h is None:
        return None
    sig = both(s1_long, d15, d1h)
    return stamp(sig, d15, "15m") if sig else None


# ---------------- S2: Order flow (PAXG) ----------------
def s2_long(d):
    a = atr(d)
    leg = d.tail(36).reset_index(drop=True)
    ilo, ihi = int(leg.Low.idxmin()), int(leg.High.idxmax())
    if not (ilo < ihi) or ihi > len(leg) - 3:
        return None
    seg = leg.iloc[ilo:ihi + 1]
    if len(seg) < 5 or (seg.High.max() - seg.Low.min()) < 3 * a:   # imbalance
        return None
    if seg.Delta.sum() <= 0:
        return None
    pr = profile(seg, 12)
    if pr is None:
        return None
    h, e = pr
    j = min(range(2, len(h) - 2), key=lambda x: h[x])          # low volume node
    lvn = (e[j] + e[j + 1]) / 2
    c = leg.iloc[-1]
    if c.Low <= seg.Low.min():
        return None
    if not (c.Low <= lvn + 0.5 * a and c.Close > lvn):         # pullback to LVN
        return None
    vavg = d.Volume.tail(21).iloc[:-1].mean()
    body, rng = c.Close - c.Open, c.High - c.Low
    if rng <= 0 or body <= 0.5 * rng or c.Volume < 1.5 * vavg or c.Delta / c.Volume < 0.25:
        return None                                            # no aggression
    sig = mk("BUY", c.Close, c.Low - 0.05 * a, seg.High.max())
    if sig:
        sig["note"] = "Impulse + LVN pullback + buyer aggression (delta)"
    return sig


def s2(asset):
    d = get_df(asset, "5m")
    if d is None:
        return None
    d = done_bars(d, "5m")
    if d is None:
        return None
    t = d.index[-1]
    hr = t.hour + t.minute / 60
    if not (SESSION_START_UTC <= hr < SESSION_END_UTC):
        return None
    sig = both(s2_long, d)
    return stamp(sig, d, "5m") if sig else None


# ---------------- S3: VIX / GVZ range ----------------
def vol_open(asset):
    for sym in VOLIDX[asset]:
        try:
            df = yf.Ticker(sym).history(period="5d", interval="1d").dropna()
            if len(df) > 0:
                v = float(df.Open.iloc[-1])
                if 5 < v < 100:
                    return v, sym
        except Exception as e:
            print("Vol index error", sym, e)
    return None, None


def s3(asset):
    d, h1 = get_df(asset, "15m"), get_df(asset, "1h")
    if d is None or h1 is None:
        return None
    d, h1 = done_bars(d, "15m"), done_bars(h1, "1h")
    if d is None or h1 is None:
        return None
    v, sym = vol_open(asset)
    if v is None:
        return None
    H, L = swings(h1, 3)
    if not H or not L:
        return None
    p, sgn = (H[-1][1], -1) if H[-1][0] > L[-1][0] else (L[-1][1], 1)
    daily_pct = v / 15.87
    R = p * daily_pct / 100
    levels = sorted([(f, p + sgn * f * R) for f in (0, 0.5, 0.75, 1.0)], key=lambda x: x[1])
    sig = rejection(levels, {0.5, 0.75, 1.0}, d)
    if not sig:
        return None
    sig["note"] = f"{sym} open {v:.2f} -> daily range {daily_pct:.2f}%, engulfing rejection"
    return stamp(sig, d, "15m")


# ---------------- S4: Fixed Range Volume Profile ----------------
def s4(asset):
    d = get_df(asset, "15m")
    if d is None:
        return None
    d = done_bars(d, "15m")
    if d is None:
        return None
    tz, off = ("UTC", 0) if asset == "BTC" else ("America/New_York", 6)
    d = d.assign(day=d.index.tz_convert(tz).shift(0).to_series().add(pd.Timedelta(hours=off)).dt.date.values
                 if False else (d.index.tz_convert(tz) + pd.Timedelta(hours=off)).date)
    days = sorted(d["day"].unique())
    if len(days) < 2:
        return None
    prev, today = d[d.day == days[-2]], d[d.day == days[-1]]
    pv = prev[prev.Volume > 0]
    if len(pv) < 20 or len(today) < 3:
        return None
    pr = profile(pv, VA_ROWS)
    if pr is None:
        return None
    poc, vah, val = value_area(pr[0], pr[1], VA_PCT)
    R = poc - val
    if R < 0.1 * (prev.High.max() - prev.Low.min()):     # tiny range -> double it
        R = 2 * R
    if R <= 0:
        return None
    op = float(today.Open.iloc[0])
    ks = (-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1)
    levels = [(k, op + k * R) for k in ks]
    sig = rejection(levels, {-1, -0.75, -0.5, 0.5, 0.75, 1}, today)
    if not sig:
        return None
    sig["note"] = "Prev-day POC-VAL range projected from today's open, engulfing rejection"
    return stamp(sig, d, "15m")


CONF = {
    "S1": {"name": "Gold liquidity sweep", "assets": ["GOLD"], "fn": s1},
    "S2": {"name": "Order flow (PAXG)", "assets": ["PAXG"], "fn": s2},
    "S3": {"name": "VIX range", "assets": ["GOLD", "BTC"], "fn": s3},
    "S4": {"name": "Fixed range volume profile", "assets": ["BTC", "GOLD"], "fn": s4},
}


# ---------------- Paper trades ----------------
def load(sid):
    try:
        with open(f"trades_{sid.lower()}.json") as f:
            d = json.load(f)
    except Exception:
        d = {}
    d.setdefault("open", [])
    d.setdefault("closed", [])
    d.setdefault("last", {})
    return d


def save(sid, d):
    with open(f"trades_{sid.lower()}.json", "w") as f:
        json.dump(d, f, indent=2)


def resolve(t, df):
    risk = abs(t["entry"] - t["sl0"])
    sl = t["sl"]
    buy = t["side"] == "BUY"
    for _, c in df[df.index >= pd.Timestamp(t["time"])].iterrows():
        if buy:
            if c.Low <= sl:
                return (sl - t["entry"]) / risk
            if c.High >= t["tp"]:
                return (t["tp"] - t["entry"]) / risk
            if t["be"] and c.High >= t["entry"] + risk:
                sl = t["entry"]
        else:
            if c.High >= sl:
                return (t["entry"] - sl) / risk
            if c.Low <= t["tp"]:
                return (t["entry"] - t["tp"]) / risk
            if t["be"] and c.Low <= t["entry"] - risk:
                sl = t["entry"]
    t["sl"] = sl
    age_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(t["time"])).total_seconds() / 3600
    if age_h > MAX_HOURS:
        last = float(df.Close.iloc[-1])
        return ((last - t["entry"]) if buy else (t["entry"] - last)) / risk
    return None


def update_open(sid, d):
    keep = []
    for t in d["open"]:
        df = get_df(t["asset"], RES_TF[t["asset"]])
        r = resolve(t, df) if df is not None else None
        if r is None:
            keep.append(t)
            continue
        t["r"] = round(float(r), 2)
        d["closed"].append(t)
        wins = sum(1 for x in d["closed"] if x["r"] > 0)
        tot = len(d["closed"])
        tr = sum(x["r"] for x in d["closed"])
        tg(f"[{sid}] {'WIN' if r > 0 else 'LOSS' if r < 0 else 'BREAK-EVEN'} "
           f"{t['asset']} {t['side']}  {r:+.2f}R\n"
           f"{sid} total: {wins}/{tot} wins, {tr:+.1f}R")
    d["open"] = keep


for sid, cfg in CONF.items():
    d = load(sid)
    try:
        update_open(sid, d)
    except Exception:
        traceback.print_exc()
    for asset in cfg["assets"]:
        try:
            if any(t["asset"] == asset for t in d["open"]):
                continue
            sig = cfg["fn"](asset)
            if not sig:
                print(sid, asset, "no signal")
                continue
            if d["last"].get(asset) == sig["close_time"]:
                continue
            d["last"][asset] = sig["close_time"]
            risk = abs(sig["entry"] - sig["sl"])
            d["open"].append({
                "asset": asset, "side": sig["side"], "entry": sig["entry"],
                "sl": sig["sl"], "sl0": sig["sl"], "tp": sig["tp"],
                "be": sid == "S2", "time": sig["close_time"]})
            tg(f"[{sid}] {cfg['name']} (paper)\n"
               f"{sig['side']} {asset}\n"
               f"Entry: {sig['entry']:.2f}\nStop-loss: {sig['sl']:.2f}\n"
               f"Target: {sig['tp']:.2f}\n"
               f"Reward:Risk = {abs(sig['tp'] - sig['entry']) / risk:.1f}\n"
               f"{sig.get('note', '')}")
            print(sid, asset, "signal", sig["side"])
        except Exception:
            traceback.print_exc()
    save(sid, d)
