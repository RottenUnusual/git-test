"""Independent sanity check of trend.py: plain vectorised % returns, no buffer,
no costs, no WTI (negative price). Should broadly agree with trend.py's zero-cost rows."""
import pathlib

import numpy as np
import pandas as pd

p = pd.read_csv(pathlib.Path(__file__).parent / "data" / "prices.csv", index_col=0, parse_dates=True)
p = p.loc["2004":].drop(columns=["WTI"])
p.loc[:"2007-06-30", "NATGAS"] = np.nan
r = p.pct_change(fill_method=None)
vol = r.ewm(span=36, min_periods=36).std() * np.sqrt(252)


def sharpe(sig):
    pnl = ((sig * 0.10 / vol).shift(1) * r).mean(axis=1).dropna()
    return {k: round(x.mean() / x.std() * np.sqrt(252), 2)
            for k, x in (("train", pnl.loc["2006":"2017"]), ("test", pnl.loc["2018":]))}


for n in (20, 60, 120, 250):
    print(f"TSMOM({n})", sharpe(np.sign(p / p.shift(n) - 1)))
for f in (8, 16, 32, 64):
    print(f"EMAX({f}/{4*f})", sharpe(np.sign(p.ewm(span=f).mean() - p.ewm(span=4 * f).mean())))
