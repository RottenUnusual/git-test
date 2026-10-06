"""Trend-following study on daily data: tune on 2006-2017, test on 2018-today.

Run:  python trend.py            (writes results/ and REPORT.md)

Conventions
- Signal f_t in [-1, 1] (or [-2, 2] for continuous forecasts) uses closes up to t.
- Position over (t, t+LAG] is f_t / sigma_t, sigma_t = EWMA(36) stdev of daily
  price changes up to t. LAG=1 trades at the close the signal is computed on;
  LAG=2 adds a full day of execution delay.
- Each market sleeve targets 10% annual vol at |f| = 1; the portfolio is the equal-
  weighted mean of available sleeves. Sharpe ratios are scale-free, costs scale
  with the same leverage so they are consistent.
- P&L uses price differences (not % returns) so WTI's negative print in Apr-2020
  does not blow up the arithmetic. Position changes are buffered (10% of a full
  position) to avoid paying for tiny vol-rescaling trades.
- Costs: one-way cost as a fraction of traded notional, plus an optional annual
  financing charge on gross notional (retail CFD/rollover markup).
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
RNG = np.random.default_rng(20261006)

WARMUP_START = "2004-01-01"
TRAIN = ("2006-01-01", "2017-12-31")
TEST = ("2018-01-01", "2100-01-01")
VOL_TARGET = 0.10
K = VOL_TARGET / 16.0          # daily target per unit of forecast
VOL_SPAN = 36
BUFFER = 0.10
MAX_NOTIONAL = 4.0             # per-sleeve gross notional cap (x sleeve capital)
FIRST_GOOD = {"NATGAS": "2007-07-01"}   # mirror has sparse natgas data before this

FX = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
ENERGY = ["WTI", "BRENT", "NATGAS"]

# one-way cost (fraction of notional) and annual financing on gross notional
COSTS = {
    "zero": dict(fx=0.0, energy=0.0, natgas=0.0, fin=0.0),
    "low": dict(fx=0.5e-4, energy=2e-4, natgas=5e-4, fin=0.0),       # futures / ECN
    "base": dict(fx=1.5e-4, energy=5e-4, natgas=15e-4, fin=0.0),     # decent retail
    "retail": dict(fx=4e-4, energy=10e-4, natgas=30e-4, fin=0.02),   # CFD/market-maker
}


# ----------------------------------------------------------------- data
def load_prices():
    px = pd.read_csv(HERE / "data" / "prices.csv", index_col=0, parse_dates=True)
    px = px.loc[WARMUP_START:]
    for m, d in FIRST_GOOD.items():
        if m in px:
            px.loc[:d, m] = np.nan
    # common calendar: union of dates, carry last price over holidays (zero P&L)
    first = px.apply(pd.Series.first_valid_index)
    px = px.ffill()
    for m, d in first.items():
        px.loc[: d - pd.Timedelta(days=1), m] = np.nan
    return px


# ----------------------------------------------------------------- signals
def ema(p, span):
    return p.ewm(span=span, adjust=False, min_periods=span).mean()


def sig_tsmom(p, n, **_):
    return np.sign(p - p.shift(n))


def sig_emacross(p, fast, **_):
    return np.sign(ema(p, fast) - ema(p, 4 * fast))


def sig_ewmac(p, fast, sigma, scalar, **_):
    raw = (ema(p, fast) - ema(p, 4 * fast)) / sigma
    return (raw * scalar).clip(-2, 2)


def sig_donchian(p, n, **_):
    hi = p.rolling(n).max().shift(1)
    lo = p.rolling(n).min().shift(1)
    xhi = p.rolling(max(n // 2, 2)).max().shift(1)
    xlo = p.rolling(max(n // 2, 2)).min().shift(1)
    out = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for c in p.columns:
        P, H, L, XH, XL = (a[c].values for a in (p, hi, lo, xhi, xlo))
        s, res = 0.0, np.full(len(P), np.nan)
        for i in range(len(P)):
            if np.isnan(H[i]) or np.isnan(P[i]):
                continue
            if P[i] > H[i]:
                s = 1.0
            elif P[i] < L[i]:
                s = -1.0
            elif s > 0 and P[i] < XL[i]:
                s = 0.0
            elif s < 0 and P[i] > XH[i]:
                s = 0.0
            res[i] = s
        out[c] = res
    return out


def sig_long(p, **_):
    return pd.DataFrame(1.0, index=p.index, columns=p.columns).where(p.notna())


CONFIGS = (
    [("TSMOM", f"TSMOM({n})", sig_tsmom, dict(n=n)) for n in (10, 20, 40, 60, 120, 250)]
    + [("EMAX", f"EMAX({f}/{4*f})", sig_emacross, dict(fast=f)) for f in (4, 8, 16, 32, 64)]
    + [("EWMAC", f"EWMAC({f}/{4*f})", sig_ewmac, dict(fast=f)) for f in (4, 8, 16, 32, 64)]
    + [("DONCH", f"DONCH({n})", sig_donchian, dict(n=n)) for n in (20, 55, 100, 200)]
)
# Pre-registered control: fixed blend, nothing chosen on data.
ENSEMBLE = [f"EWMAC({f}/{4*f})" for f in (8, 16, 32, 64)]


# ----------------------------------------------------------------- engine
def vol(p):
    dp = p.diff()
    return np.sqrt((dp ** 2).ewm(span=VOL_SPAN, adjust=False, min_periods=VOL_SPAN).mean())


def ewmac_scalars(p, sigma):
    """Scale continuous EWMAC so mean |forecast| = 1 on TRAIN data only."""
    out = {}
    for f in (4, 8, 16, 32, 64):
        raw = ((ema(p, f) - ema(p, 4 * f)) / sigma).loc[TRAIN[0]:TRAIN[1]]
        out[f] = 1.0 / np.nanmean(np.abs(raw.values))
    return out


def positions(f, sigma, price):
    """Buffered target position in price units (per K of sleeve capital)."""
    tgt = (f / sigma).values
    width = (BUFFER / sigma).values
    cap = (MAX_NOTIONAL / K / price.abs()).values
    pos = np.full(tgt.shape, np.nan)
    for j in range(tgt.shape[1]):
        cur = 0.0
        for i in range(tgt.shape[0]):
            t = tgt[i, j]
            if np.isnan(t):
                continue
            t = max(-cap[i, j], min(cap[i, j], t))
            if t > cur + width[i, j]:
                cur = t - width[i, j]
            elif t < cur - width[i, j]:
                cur = t + width[i, j]
            if abs(t) < 1e-12:      # flat signal -> go flat
                cur = 0.0
            pos[i, j] = cur
    return pd.DataFrame(pos, index=f.index, columns=f.columns)


def sleeve_returns(pos, price, cost, lag=1):
    """Daily sleeve returns (fraction of sleeve capital) net of costs."""
    held = pos.shift(lag)
    gross = K * held * price.diff()
    traded = K * (held - held.shift(1).fillna(0)).abs() * price.abs()
    c_rate = pd.Series({m: cost["natgas"] if m == "NATGAS" else cost["energy"] if m in ENERGY
                        else cost["fx"] for m in price.columns})
    tc = traded * c_rate
    fin = K * held.abs() * price.abs() * cost["fin"] / 252.0
    net = gross - tc - fin
    return net.where(held.notna()), gross.where(held.notna())


def portfolio(r, cols=None):
    r = r if cols is None else r[cols]
    return r.mean(axis=1, skipna=True).dropna()


def stats(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return dict(sharpe=np.nan, ann_ret=np.nan, ann_vol=np.nan, maxdd=np.nan, t=np.nan, n=len(r))
    sr = r.mean() / r.std() * math.sqrt(252)
    eq = r.cumsum()
    return dict(sharpe=sr, ann_ret=r.mean() * 252, ann_vol=r.std() * math.sqrt(252),
                maxdd=(eq - eq.cummax()).min(), t=sr * math.sqrt(len(r) / 252), n=len(r))


def per(r, period):
    return r.loc[period[0]:period[1]]


def stationary_bootstrap_sharpe(r, reps=2000, block=20):
    x = r.dropna().values
    n = len(x)
    p = 1.0 / block
    out = np.empty(reps)
    for b in range(reps):
        idx = np.empty(n, dtype=int)
        i = RNG.integers(n)
        for k in range(n):
            if k == 0 or RNG.random() < p:
                i = RNG.integers(n)
            else:
                i = (i + 1) % n
            idx[k] = i
        y = x[idx]
        out[b] = y.mean() / y.std() * math.sqrt(252)
    return out


def deflated_sharpe(r, n_trials, trial_srs):
    """Bailey & Lopez de Prado DSR: P(true SR > expected max of N null trials)."""
    from scipy.stats import norm, skew, kurtosis
    x = r.dropna().values
    T = len(x)
    sr = x.mean() / x.std()                      # per-period
    v = np.var(np.asarray(trial_srs) / math.sqrt(252), ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e)))
    sk, ku = skew(x), kurtosis(x, fisher=False)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(norm.cdf(z)), float(sr0 * math.sqrt(252))


# ----------------------------------------------------------------- run
def build(px, sigma, scal, name, fn, kw):
    if fn is sig_ewmac:
        return fn(px, sigma=sigma, scalar=scal[kw["fast"]], **kw)
    return fn(px, **kw)


def run_all(px, lag=1):
    sigma = vol(px)
    scal = ewmac_scalars(px, sigma)
    sig, pos = {}, {}
    for fam, name, fn, kw in CONFIGS + [("LONG", "LONG", sig_long, {})]:
        sig[name] = build(px, sigma, scal, name, fn, kw)
    ens = sum(sig[n] for n in ENSEMBLE) / len(ENSEMBLE)
    sig["ENSEMBLE"] = ens.clip(-2, 2)
    for name, f in sig.items():
        pos[name] = positions(f, sigma, px)
    return sig, pos, sigma, scal


def null_test(px, name, n_reps=300):
    """Re-run a fixed strategy on test-period price paths whose daily changes are
    shuffled in time (jointly across markets). Kills serial dependence, keeps
    the return distribution and cross-correlation. Returns null Sharpes."""
    sub = px.loc["2017-01-01":]
    dp = sub.diff().iloc[1:]
    start = sub.iloc[0]
    fam, _, fn, kw = next(c for c in CONFIGS if c[1] == name) if name != "ENSEMBLE" else (None, None, None, None)
    out = []
    for _ in range(n_reps):
        perm = dp.values[RNG.permutation(len(dp))]
        fake = pd.DataFrame(np.vstack([start.values, start.values + np.nancumsum(perm, axis=0)]),
                            index=sub.index, columns=sub.columns).where(sub.notna())
        sigma = vol(fake)
        scal = ewmac_scalars_full(fake, sigma)
        if name == "ENSEMBLE":
            f = sum(sig_ewmac(fake, fast=int(n.split("(")[1].split("/")[0]), sigma=sigma,
                              scalar=scal[int(n.split("(")[1].split("/")[0])]) for n in ENSEMBLE) / len(ENSEMBLE)
        elif fn is sig_ewmac:
            f = fn(fake, sigma=sigma, scalar=scal[kw["fast"]], **kw)
        else:
            f = fn(fake, **kw)
        r, _ = sleeve_returns(positions(f, sigma, fake), fake, COSTS["zero"])
        out.append(stats(per(portfolio(r), TEST))["sharpe"])
    return np.array(out)


def ewmac_scalars_full(p, sigma):
    return {f: 1.0 / np.nanmean(np.abs(((ema(p, f) - ema(p, 4 * f)) / sigma).values))
            for f in (4, 8, 16, 32, 64)}


def main():
    OUT.mkdir(exist_ok=True)
    px = load_prices()
    markets = list(px.columns)
    groups = {"ALL": markets, "FX": [m for m in markets if m in FX], "ENERGY": [m for m in markets if m in ENERGY]}
    sig, pos, sigma, scal = run_all(px)
    names = [c[1] for c in CONFIGS]
    fam_of = {c[1]: c[0] for c in CONFIGS}

    # ---- every config x cost x group x period
    rows, port = [], {}
    sleeve = {}
    for cname, cost in COSTS.items():
        for lag in (1, 2):
            if lag == 2 and cname not in ("base",):
                continue
            for name in names + ["ENSEMBLE", "LONG"]:
                net, gross = sleeve_returns(pos[name], px, cost, lag)
                sleeve[(name, cname, lag)] = net
                for g, cols in groups.items():
                    pr = portfolio(net, cols)
                    port[(name, cname, lag, g)] = pr
                    for pname, period in (("train", TRAIN), ("test", TEST)):
                        s = stats(per(pr, period))
                        rows.append(dict(config=name, family=fam_of.get(name, name), cost=cname, lag=lag,
                                         group=g, period=pname, **s))
    allres = pd.DataFrame(rows)
    allres.to_csv(OUT / "all_configs.csv", index=False, float_format="%.4f")

    def sr(name, cost="base", lag=1, g="ALL", period="test"):
        q = allres.query("config==@name and cost==@cost and lag==@lag and group==@g and period==@period")
        return q.iloc[0]

    # ---- turnover & cost break-even per config (ALL, lag1)
    to_rows = []
    for name in names + ["ENSEMBLE"]:
        held = pos[name].shift(1)
        traded = (K * (held - held.shift(1).fillna(0)).abs() * px.abs())
        gross_notional = (K * held.abs() * px.abs())
        trn = per(traded.mean(axis=1), TEST).mean() * 252
        gn = per(gross_notional.mean(axis=1), TEST).mean()
        z = per(port[(name, "zero", 1, "ALL")], TEST)
        # one-way cost (bp of notional, applied to all markets) at which test Sharpe -> 0
        be = z.mean() * 252 / trn * 1e4 if trn > 0 else np.nan
        to_rows.append(dict(config=name, annual_turnover_x=trn, avg_gross_notional=gn,
                            breakeven_cost_bp_oneway=be))
    turnover = pd.DataFrame(to_rows)
    turnover.to_csv(OUT / "turnover.csv", index=False, float_format="%.3f")

    # ---- selection on TRAIN (base costs, lag 1, ALL markets)
    tr = allres.query("cost=='base' and lag==1 and group=='ALL' and period=='train' and config in @names")
    te = allres.query("cost=='base' and lag==1 and group=='ALL' and period=='test' and config in @names")
    tt = tr[["config", "family", "sharpe"]].merge(te[["config", "sharpe"]], on="config", suffixes=("_train", "_test"))
    tt.to_csv(OUT / "train_vs_test.csv", index=False, float_format="%.3f")
    best_fam = tt.loc[tt.groupby("family")["sharpe_train"].idxmax()].sort_values("sharpe_train", ascending=False)
    best_overall = best_fam.iloc[0]["config"]
    from scipy.stats import spearmanr
    rho = spearmanr(tt.sharpe_train, tt.sharpe_test)

    # ---- per-market selection (overfitting check): each market picks its own best config on train
    pm_choice, pm_rows = {}, []
    for m in markets:
        best, bsr = None, -np.inf
        for name in names:
            s = stats(per(sleeve[(name, "base", 1)][m], TRAIN))["sharpe"]
            if s > bsr:
                best, bsr = name, s
        pm_choice[m] = best
        ts = stats(per(sleeve[(best, "base", 1)][m], TEST))["sharpe"]
        pm_rows.append(dict(market=m, chosen=best, train_sharpe=bsr, test_sharpe=ts))
    pm = pd.DataFrame(pm_rows)
    pm_port = pd.concat({m: sleeve[(pm_choice[m], "base", 1)][m] for m in markets}, axis=1).mean(axis=1).dropna()

    # ---- walk-forward: each year pick best config on expanding window (base cost), trade next year
    years = range(2010, px.index[-1].year + 1)
    wf_parts, wf_log = [], []
    base_all = {n: port[(n, "base", 1, "ALL")] for n in names}
    for y in years:
        hist = {n: stats(r.loc[TRAIN[0]:f"{y-1}-12-31"])["sharpe"] for n, r in base_all.items()}
        pick = max(hist, key=hist.get)
        wf_log.append(dict(year=y, pick=pick, insample_sharpe=hist[pick],
                           oos_sharpe=stats(base_all[pick].loc[str(y)])["sharpe"]))
        wf_parts.append(base_all[pick].loc[str(y)])
    wf = pd.concat(wf_parts)
    wf_log = pd.DataFrame(wf_log)
    wf_log.to_csv(OUT / "walk_forward.csv", index=False, float_format="%.3f")

    # ---- per-market test results for the ensemble and best config
    mkt_rows = []
    for name in ["ENSEMBLE", best_overall, "LONG"]:
        for m in markets:
            for pname, period in (("train", TRAIN), ("test", TEST)):
                s = stats(per(sleeve[(name, "base", 1)][m], period))
                mkt_rows.append(dict(config=name, market=m, period=pname, **s))
    mkt = pd.DataFrame(mkt_rows)
    mkt.to_csv(OUT / "per_market.csv", index=False, float_format="%.3f")

    # ---- yearly
    yearly = pd.DataFrame({n: port[(n, "base", 1, "ALL")].groupby(port[(n, "base", 1, "ALL")].index.year).sum()
                           for n in ["ENSEMBLE", best_overall, "LONG"]})
    yearly.to_csv(OUT / "yearly_returns_base.csv", float_format="%.4f")

    # ---- significance
    ens_test = per(port[("ENSEMBLE", "base", 1, "ALL")], TEST)
    best_test = per(port[(best_overall, "base", 1, "ALL")], TEST)
    boot_ens = stationary_bootstrap_sharpe(ens_test)
    boot_best = stationary_bootstrap_sharpe(best_test)
    trial_srs = tr.sharpe.values
    dsr_best, sr0 = deflated_sharpe(per(port[(best_overall, "base", 1, "ALL")], TRAIN), len(names), trial_srs)
    null_ens = null_test(px, "ENSEMBLE")
    null_best = null_test(px, best_overall)
    ens_zero_test = stats(per(port[("ENSEMBLE", "zero", 1, "ALL")], TEST))["sharpe"]
    best_zero_test = stats(per(port[(best_overall, "zero", 1, "ALL")], TEST))["sharpe"]

    sigdict = dict(
        best_overall=best_overall,
        spearman_train_test=float(rho.statistic), spearman_p=float(rho.pvalue),
        ensemble_test_base_sharpe=float(stats(ens_test)["sharpe"]),
        ensemble_boot_ci95=[float(np.percentile(boot_ens, 2.5)), float(np.percentile(boot_ens, 97.5))],
        ensemble_boot_p_le0=float((boot_ens <= 0).mean()),
        best_test_base_sharpe=float(stats(best_test)["sharpe"]),
        best_boot_ci95=[float(np.percentile(boot_best, 2.5)), float(np.percentile(boot_best, 97.5))],
        best_boot_p_le0=float((boot_best <= 0).mean()),
        best_train_deflated_sharpe_prob=dsr_best, expected_max_null_sharpe_train=sr0,
        null_shuffle_ensemble_p=float((null_ens >= ens_zero_test).mean()),
        null_shuffle_ensemble_mean_sd=[float(null_ens.mean()), float(null_ens.std())],
        null_shuffle_best_p=float((null_best >= best_zero_test).mean()),
        null_shuffle_best_mean_sd=[float(null_best.mean()), float(null_best.std())],
        ensemble_zero_cost_test_sharpe=float(ens_zero_test), best_zero_cost_test_sharpe=float(best_zero_test),
        walk_forward_2018on=stats(per(wf, TEST)), walk_forward_2010on=stats(wf),
        per_market_selection_test=stats(per(pm_port, TEST)),
        per_market_selection_train=stats(per(pm_port, TRAIN)),
        data_end=str(px.index[-1].date()),
    )
    (OUT / "significance.json").write_text(json.dumps(sigdict, indent=2, default=float))

    plots(px, port, tt, best_overall, mkt, wf, markets)
    report(allres, tt, best_fam, pm, wf_log, mkt, turnover, sigdict, yearly, markets, scal)
    print(json.dumps(sigdict, indent=2, default=float))


# ----------------------------------------------------------------- output
def plots(px, port, tt, best, mkt, wf, markets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    blue, orange, gray, green = "#2a6fdb", "#e07b39", "#888888", "#3a9d5d"

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, col, lab in (("ENSEMBLE", blue, "EWMAC ensemble (pre-registered)"),
                           (best, orange, f"{best} (best on train)"), ("LONG", gray, "Always long (control)")):
        for cost, ls in (("base", "-"), ("retail", ":")):
            r = port[(name, cost, 1, "ALL")].loc[TRAIN[0]:]
            ax.plot(r.cumsum() * 100, color=col, ls=ls, lw=1.3, label=f"{lab}, {cost} cost")
    ax.plot(wf.cumsum() * 100, color=green, lw=1, label="Walk-forward reselection (starts 2010 at 0), base cost")
    ax.axvspan(pd.Timestamp(TEST[0]), px.index[-1], color="#f3e9c6", alpha=0.6, lw=0)
    ax.text(pd.Timestamp("2018-03-01"), ax.get_ylim()[1] * 0.92, "out-of-sample", fontsize=8)
    ax.set_ylabel("cumulative return, % (10% vol per sleeve)")
    ax.set_title("Equal-weight portfolio, 10 markets")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.set_size_inches(9, 5.2)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fams = sorted(tt.family.unique())
    cols = dict(zip(fams, [blue, orange, green, "#9b59b6"]))
    for f in fams:
        q = tt[tt.family == f]
        ax.scatter(q.sharpe_train, q.sharpe_test, color=cols[f], label=f, s=28)
        for _, r in q.iterrows():
            ax.annotate(r.config.split("(")[1].rstrip(")"), (r.sharpe_train, r.sharpe_test), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
    lim = [min(tt.sharpe_train.min(), tt.sharpe_test.min()) - 0.1, max(tt.sharpe_train.max(), tt.sharpe_test.max()) + 0.1]
    ax.plot(lim, lim, color=gray, lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Sharpe 2006-2017 (train)")
    ax.set_ylabel("Sharpe 2018-today (test)")
    ax.set_title("All 20 configs, base costs, all markets")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "train_vs_test.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    q = mkt[(mkt.config == "ENSEMBLE")].pivot(index="market", columns="period", values="sharpe").loc[markets]
    x = np.arange(len(markets))
    ax.bar(x - 0.2, q["train"], 0.4, color=gray, label="train 2006-17")
    ax.bar(x + 0.2, q["test"], 0.4, color=blue, label="test 2018-")
    ax.set_xticks(x, markets, rotation=0, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Sharpe, base cost")
    ax.set_title("EWMAC ensemble by market")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "per_market.png")
    plt.close(fig)


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_table(df, floats=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v, floats) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def report(allres, tt, best_fam, pm, wf_log, mkt, turnover, s, yearly, markets, scal):
    def get(cfg, cost, g, period, lag=1, col="sharpe"):
        q = allres.query("config==@cfg and cost==@cost and group==@g and period==@period and lag==@lag")
        return q.iloc[0][col]

    best = s["best_overall"]
    head = ["ENSEMBLE", best, "LONG"]
    cfgs = [c[1] for c in CONFIGS]
    n_ret_test = sum(get(c, "retail", "ALL", "test") > 0 for c in cfgs)
    n_ret_train = sum(get(c, "retail", "ALL", "train") > 0 for c in cfgs)
    rows = []
    for cfg in head:
        for g in ("ALL", "FX", "ENERGY"):
            rows.append(dict(strategy=cfg, markets=g,
                             **{f"{p}/{c}": get(cfg, c, g, p) for p in ("train", "test") for c in ("zero", "base", "retail")},
                             **{"test/base lag2": get(cfg, "base", g, "test", lag=2)}))
    headline = pd.DataFrame(rows)

    full = allres.query("lag==1 and group=='ALL'").pivot_table(index="config", columns=["period", "cost"], values="sharpe")
    full = full[[("train", "zero"), ("train", "base"), ("train", "retail"), ("test", "zero"), ("test", "base"), ("test", "retail")]]
    full.columns = [f"{a}/{b}" for a, b in full.columns]
    order = [c[1] for c in CONFIGS] + ["ENSEMBLE", "LONG"]
    full = full.loc[order].reset_index()
    full = full.merge(turnover[["config", "annual_turnover_x", "breakeven_cost_bp_oneway"]], on="config", how="left")

    dd = allres.query("lag==1 and group=='ALL' and cost=='base' and config in @head").pivot_table(
        index="config", columns="period", values=["ann_ret", "ann_vol", "maxdd", "t"])
    dd.columns = [f"{b} {a}" for a, b in dd.columns]
    dd = dd.loc[head].reset_index()

    pmk = mkt.pivot_table(index="market", columns=["config", "period"], values="sharpe")
    pmk = pmk[[(c, p) for c in head for p in ("train", "test")]].loc[markets]
    pmk.columns = [f"{c} {p}" for c, p in pmk.columns]
    pmk = pmk.reset_index()

    yr = yearly.copy() * 100
    yr.index.name = "year"
    yr = yr.loc[2006:].reset_index()
    yr["year"] = yr["year"].astype(int).astype(str)
    yr.columns = ["year"] + [f"{c} %" for c in yearly.columns]

    wfs, wfa, pmt, pmtr = s["walk_forward_2018on"], s["walk_forward_2010on"], s["per_market_selection_test"], s["per_market_selection_train"]
    txt = f"""# Daily trend following: 10 markets, tuned 2006-2017, tested 2018-{s['data_end'][:4]}

Generated by `trend.py` (data through {s['data_end']}). Every number below is
produced by the script; nothing was hand-edited. Raw tables are in `results/`.

## Verdict

**No robust edge out of sample, before or after costs.** In 2018-{s['data_end'][:4]} the
pre-registered EWMAC ensemble made a Sharpe of {s['ensemble_test_base_sharpe']:.2f} at base cost
(95% CI {s['ensemble_boot_ci95'][0]:.2f} to {s['ensemble_boot_ci95'][1]:.2f}) and
{get('ENSEMBLE', 'retail', 'ALL', 'test'):.2f} at retail CFD cost. The configuration that tuning
picked, `{best}`, did worse ({s['best_test_base_sharpe']:.2f}), and so did walk-forward
reselection ({fmt(wfs['sharpe'])}). The zero-cost test Sharpe ({s['ensemble_zero_cost_test_sharpe']:.2f}) is
no better than randomly reshuffled price paths (p = {s['null_shuffle_ensemble_p']:.2f}), so this is
not a cost problem: the trend signal itself did not pay on these markets in this period.

* **In sample, the edge was small.** The best of 20 configs reached a train Sharpe of
  {get(best, 'base', 'ALL', 'train'):.2f}. The expected best of 20 zero-skill configs is about
  {s['expected_max_null_sharpe_train']:.2f}, so even the train result is barely
  distinguishable from selection luck (deflated-Sharpe probability {s['best_train_deflated_sharpe_prob']:.2f}).
  Most of it came from two years, 2008 and 2014.
* **Tuning hurt.** The train-test rank correlation is negative, and per-market tuning
  went from a train Sharpe of {fmt(pmtr['sharpe'])} to a test Sharpe of {fmt(pmt['sharpe'])}.
* **The only positive pockets are weak.** Energy was slightly positive (ensemble
  {get('ENSEMBLE', 'base', 'ENERGY', 'test'):.2f}), but energy spot data omit roll costs, which is what
  you would actually pay. Fast rules such as DONCH(20) and TSMOM(20) were mildly positive
  in test, but they lost money in train, so picking them now would be hindsight.
* **Retail costs are decisive.** At retail spreads plus financing, {n_ret_test} of the 20
  configs are positive in test and {n_ret_train} are positive in train.

(The prose in this section was written for the 10-market run committed here; the
numbers are computed. Re-read it if you add markets via `data/extra/`.)

This matches the prior 5-minute finding on Deriv synthetics, USD/CAD, EUR/USD and gold:
no edge after costs. The caveat is the universe. These are FX and energy markets only;
gold, silver and the equity indices were unavailable (see below), and those are where
trend following has historically done best.

## Data, and what is missing

Stooq and Yahoo were **blocked by the sandbox network policy** (HTTP 403 on
`stooq.com` and `query1.finance.yahoo.com`), so the study uses public mirrors:

| Market | Source | Notes |
|---|---|---|
| EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCHF, USDCAD | FRED H.10 noon NY rates (github.com/datasets/exchange-rates) | spot only: **no carry/rollover** in returns |
| WTI, Brent | EIA spot (github.com/datasets/oil-prices) | spot, not futures: **no roll yield**; WTI spot printed -$36.98 in Apr-2020 |
| Henry Hub natural gas | EIA spot (github.com/datasets/natural-gas) | spot is not directly tradable; sparse before mid-2007, so it enters in 2008 |

**Gold, silver and the US indices are not in this run.** No free daily source
reaching 2026 was reachable (the only index data available ended in 2018).
`fetch_data.py` takes Stooq/Yahoo CSVs from `data/extra/` if you have them;
re-running `trend.py` then includes them automatically.

So this is 7 FX + 3 energy markets, not the 12 you asked for. It is light on the
asset classes (metals, equity indices) where trend following has historically
done best, so read the results as a test on FX and energy only.

## Method

* **Splits**: indicators warm up from 2004; **train 2006-01-01 → 2017-12-31**
  (12 years); **test 2018-01-01 → {s['data_end']}** (~{(pd.Timestamp(s['data_end']) - pd.Timestamp('2018-01-01')).days / 365.25:.1f} years). Nothing in the
  test window was used for any choice, including the EWMAC forecast scaling.
* **20 configurations, 4 families**: TSMOM(N) = sign of N-day return,
  N ∈ {{10,20,40,60,120,250}}; EMAX(f/4f) = sign of EMA crossover, f ∈ {{4,8,16,32,64}};
  EWMAC(f/4f) = the same crossover as a continuous, vol-normalised forecast, capped at ±2;
  DONCH(N) = N-day breakout, exit on an N/2-day opposite breakout, N ∈ {{20,55,100,200}}.
* **Pre-registered control**: `ENSEMBLE` = equal blend of EWMAC 8/32, 16/64,
  32/128, 64/256, fixed before looking at any result. It involves no selection, so it is
  the most honest out-of-sample estimate here. `LONG` (always long as quoted) is a
  drift control.
* **Sizing**: per-market vol targeting (EWMA-36 of daily price changes), 10% annual vol
  per sleeve at full forecast, equal-weight portfolio, 10% position buffer, 4x notional cap.
* **Execution**: signal at close t, P&L from t to t+1 (`lag 1`). `lag 2` adds a full
  extra day of delay.
* **Costs** (one-way, % of traded notional; financing is per year on gross notional):

| scenario | FX | oil | natgas | financing |
|---|---|---|---|---|
| zero | 0 | 0 | 0 | 0 |
| low (futures/ECN) | 0.5 bp | 2 bp | 5 bp | 0 |
| base (good retail) | 1.5 bp | 5 bp | 15 bp | 0 |
| retail (CFD / market maker, like Deriv) | 4 bp | 10 bp | 30 bp | 2%/yr |

## Headline: Sharpe ratios

{md_table(headline)}

`{best}` is the configuration with the best train Sharpe (base cost, all markets) out of
the 20. Choosing it after seeing all 20 is what makes its train number optimistic.

Return, vol and drawdown (base cost, all markets; returns in fraction of capital):

{md_table(dd, 3)}

## Was the best train config any good out of sample?

Best configuration per family, chosen on train:

{md_table(best_fam.rename(columns={'sharpe_train': 'train Sharpe', 'sharpe_test': 'test Sharpe'}))}

* Rank correlation (Spearman) of train vs test Sharpe across all 20 configs:
  **{s['spearman_train_test']:.2f}** (p = {s['spearman_p']:.2f}). Positive would mean tuning
  on the past helps; zero means it picks nothing useful; negative means the configs that
  looked best in 2006-2017 tended to do *worst* afterwards.
* Deflated Sharpe probability for `{best}` on train, adjusting for 20 trials:
  **{s['best_train_deflated_sharpe_prob']:.2f}** (expected max Sharpe of 20 zero-skill trials ≈
  {s['expected_max_null_sharpe_train']:.2f}).
* **Walk-forward** (each January, pick the best config on 2006 → last year, trade it for the
  year, base cost): Sharpe {fmt(wfa['sharpe'])} over 2010 → today, **{fmt(wfs['sharpe'])} over 2018 → today**.
* **Per-market tuning** (each market picks its own best config on train): train Sharpe
  {fmt(pmtr['sharpe'])} → test Sharpe **{fmt(pmt['sharpe'])}**. This shows how much
  per-market tuning overfits:

{md_table(pm)}

Walk-forward picks by year:

{md_table(wf_log)}

![train vs test](results/train_vs_test.png)

## Significance of the out-of-sample results

| test | ENSEMBLE | {best} |
|---|---|---|
| Test Sharpe, base cost | {s['ensemble_test_base_sharpe']:.2f} | {s['best_test_base_sharpe']:.2f} |
| 95% CI (stationary block bootstrap, 20-day blocks) | [{s['ensemble_boot_ci95'][0]:.2f}, {s['ensemble_boot_ci95'][1]:.2f}] | [{s['best_boot_ci95'][0]:.2f}, {s['best_boot_ci95'][1]:.2f}] |
| Bootstrap P(Sharpe ≤ 0) | {s['ensemble_boot_p_le0']:.2f} | {s['best_boot_p_le0']:.2f} |
| Zero-cost test Sharpe | {s['ensemble_zero_cost_test_sharpe']:.2f} | {s['best_zero_cost_test_sharpe']:.2f} |
| Shuffled-path null: mean ± sd of Sharpe | {s['null_shuffle_ensemble_mean_sd'][0]:.2f} ± {s['null_shuffle_ensemble_mean_sd'][1]:.2f} | {s['null_shuffle_best_mean_sd'][0]:.2f} ± {s['null_shuffle_best_mean_sd'][1]:.2f} |
| Shuffled-path null: p-value (zero cost) | {s['null_shuffle_ensemble_p']:.3f} | {s['null_shuffle_best_p']:.3f} |

The shuffled-path null re-runs the same rule on 300 test-period price paths whose daily
changes are randomly reordered (jointly across markets). That keeps the volatility, fat
tails and cross-correlation but removes any trend persistence. A small p-value means the
real paths trended more than chance would explain.

## Every configuration (all markets, lag 1)

Turnover is annual traded notional as a multiple of capital (test period). Break-even
is the one-way cost, in bp of notional across all markets, at which the test Sharpe
reaches zero.

{md_table(full)}

## By market

{md_table(pmk)}

![per market](results/per_market.png)

## Year by year (base cost, all markets, % of capital)

{md_table(yr, 1)}

![equity curves](results/equity_curves.png)

## Caveats

* Spot FX excludes interest-rate carry. A real FX position earns or pays the rate
  differential, and from 2022 that is several % a year on USDJPY and the AUD/NZD pairs.
  Trend positions often line up with carry, so spot P&L understates or overstates them
  by period. The `retail` scenario's 2% financing is a generic broker markup on top of carry,
  not the carry itself.
* Energy spot prices are not what you trade. Futures and CFDs on them pay roll (contango)
  or earn it (backwardation), and natural gas spot jumps cannot be captured by front-month
  futures. Read the energy results as indicative only.
* FRED rates are noon New York fixes, not closes, so `lag 1` is realistic. `lag 2` tests
  slower execution.
* 10 markets is a small universe. Trend following's historical Sharpe (~0.5-1 before
  fees) comes largely from diversifying across 50-100+ futures. With 10 correlated markets
  (the 7 FX pairs share USD), one expects a lower and noisier Sharpe.
* EWMAC forecast scalars (fitted on train only): {', '.join(f'{k}: {v:.3f}' for k, v in scal.items())}.
"""
    (HERE / "REPORT.md").write_text(txt)


if __name__ == "__main__":
    main()
