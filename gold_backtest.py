# -*- coding: utf-8 -*-
"""بک‌تست walk-forward: ایجنت را روی داده‌ی تاریخی کندل‌به‌کندل اجرا می‌کند (بدون نگاه به آینده)
و می‌سنجد سیگنال‌ها و هر سبک در گذشته چقدر درست بوده‌اند.
اجرا:  python gold_backtest.py            (داده‌ی واقعی GC=F روزانه، نیاز به اینترنت/VPN)
       python gold_backtest.py --demo     (داده‌ی ساختگی؛ فقط برای سنجش خود ماشین)"""
import argparse
import math

import numpy as np
import pandas as pd

import gold_agent as ga
import gold_pro as gp

COST_R = 0.02  # هزینه‌ی اسپرد و لغزش به واحد R (تقریبی)


def simulate(df, i, side, atr, max_bars, rr=2.0, sl_mult=1.5):
    """ورود در کلوز کندل i؛ SL=1.5ATR، TP=2R. اگر در یک کندل هر دو لمس شود، SL فرض می‌شود (محافظه‌کارانه)."""
    entry = df["Close"].iloc[i]
    risk = sl_mult * atr
    sl, tp = entry - side * risk, entry + side * rr * risk
    for j in range(i + 1, min(i + 1 + max_bars, len(df))):
        hi, lo = df["High"].iloc[j], df["Low"].iloc[j]
        hit_sl = lo <= sl if side == 1 else hi >= sl
        hit_tp = hi >= tp if side == 1 else lo <= tp
        if hit_sl:
            return -1.0 - COST_R
        if hit_tp:
            return rr - COST_R
    last = df["Close"].iloc[min(i + max_bars, len(df) - 1)]
    return side * (last - entry) / risk - COST_R


def run(df, horizon=5, step=3, warmup=260, thr=25, verbose=True):
    rows = []
    for i in range(warmup, len(df) - horizon - 1, step):
        sub = df.iloc[: i + 1]
        rep = ga.analyze(sub, "1d")
        reg = gp.regime(sub)
        adj = gp.adjusted_score(rep, reg["kind"])
        fwd = df["Close"].iloc[i + horizon] / df["Close"].iloc[i] - 1
        row = dict(i=i, score=rep["score"], adj=adj, fwd=fwd, regime=reg["kind"], votes=rep["votes"])
        for name, key in (("raw", rep["score"]), ("adj", adj)):
            side = 1 if key >= thr else -1 if key <= -thr else 0
            row[f"r_{name}"] = simulate(df, i, side, rep["atr"], horizon * 2) if side else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(r):
    r = r.dropna()
    if r.empty:
        return dict(n=0, win=np.nan, exp=np.nan, pf=np.nan, dd=np.nan, t=np.nan)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = r.cumsum()
    dd = (eq.cummax() - eq).max()
    t = r.mean() / (r.std() / math.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else np.nan
    return dict(n=len(r), win=(r > 0).mean(), exp=r.mean(), pf=wins / losses if losses > 0 else np.inf, dd=dd, t=t)


def report(res, label=""):
    out = [f"\n===== نتیجه‌ی بک‌تست {label} ({len(res)} نقطه‌ی تحلیل) ====="]
    base_up = (res["fwd"] > 0).mean()
    out.append(f"نرخ پایه: در {base_up:.0%} نقاط قیمت ۵ کندل بعد بالاتر بوده (مرجع مقایسه‌ی دقت جهت).")
    out.append("\nمعامله‌ی شبیه‌سازی‌شده (SL=1.5ATR، TP=2R، هزینه‌ی ۰٫۰۲R)؛ R=ریسک هر معامله")
    out.append(f"{'روش':<22}{'تعداد':>6}{'برد':>7}{'میانگین R':>11}{'ضریب سود':>10}{'افت R':>8}{'آمار t':>8}")
    for name, col in (("امتیاز خام", "r_raw"), ("امتیاز تعدیل‌شده", "r_adj")):
        for part, sub in (("کل", res), ("نیمه‌ی اول", res.iloc[: len(res) // 2]), ("نیمه‌ی دوم", res.iloc[len(res) // 2:])):
            s = _stats(sub[col])
            out.append(f"{name+' · '+part:<22}{s['n']:>6}{s['win']:>7.0%}{s['exp']:>11.3f}{s['pf']:>10.2f}{s['dd']:>8.1f}{s['t']:>8.2f}")
    out.append("\nدقت جهت هر سبک (رأی ≠ خنثی در برابر علامت بازده ۵ کندل بعد)")
    out.append(f"{'سبک':<16}{'تعداد':>6}{'دقت':>7}{'z':>7}   (z>2 یعنی احتمالاً بیش از شانس)")
    names = sorted({n for v in res["votes"] for n in v})
    rows = []
    for n in names:
        v = np.array([x.get(n, 0) for x in res["votes"]])
        m = v != 0
        if m.sum() < 20:
            continue
        hit = (np.sign(res["fwd"].values[m]) == v[m]).mean()
        # آزمون نسبت: مقایسه با ۰٫۵ (بدون نرخ پایه) تا ساده بماند
        z = (hit - 0.5) / math.sqrt(0.25 / m.sum())
        rows.append((n, int(m.sum()), hit, z))
    for n, c, h, z in sorted(rows, key=lambda x: -x[3]):
        out.append(f"{n:<16}{c:>6}{h:>7.0%}{z:>7.1f}")
    bucket = res.copy()
    bucket["b"] = pd.cut(bucket["score"], [-101, -40, -15, 15, 40, 101], labels=["نزولی قوی", "متمایل نزول", "خنثی", "متمایل صعود", "صعودی قوی"])
    out.append("\nمیانگین بازده ۵ کندل بعد بر حسب برچسب امتیاز خام")
    for b, g in bucket.groupby("b", observed=True):
        out.append(f"{b:<14}{len(g):>5} نقطه   میانگین {g['fwd'].mean():+.2%}   درصد صعود {(g['fwd'] > 0).mean():.0%}")
    out.append("\nتوجه: نمونه‌ی کوچک یا |t|<2 یعنی تفاوت از شانس قابل‌تشخیص نیست. این گذشته است و آینده را تضمین نمی‌کند.")
    return "\n".join(out)


def synthetic_series(kind, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    if kind == "random":
        ret = rng.normal(0.0002, 0.009, n)
    else:  # رژیم‌های روندی با دوام (لبه‌ی واقعی برای آزمون ماشین)
        ret = np.zeros(n)
        drift = 0.0
        for t in range(n):
            if t % 60 == 0:
                drift = rng.choice([-0.0025, 0.0025, 0.0])
            ret[t] = drift + rng.normal(0, 0.008)
    close = 1500 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    vol = rng.integers(80_000, 200_000, n).astype(float)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="داده‌ی ساختگی")
    ap.add_argument("--symbol", default="GC=F")
    ap.add_argument("--step", type=int, default=3)
    a = ap.parse_args()
    if a.demo:
        for kind in ("random", "trending"):
            print(report(run(synthetic_series(kind), step=max(a.step, 5)), f"(ساختگی: {kind})"))
    else:
        ga.PERIOD["1d"] = "max"
        df = ga.fetch(a.symbol, "1d")
        print(f"{len(df)} کندل روزانه از {df.index[0]:%Y-%m-%d} تا {df.index[-1]:%Y-%m-%d}")
        print(report(run(df, step=a.step), a.symbol))
