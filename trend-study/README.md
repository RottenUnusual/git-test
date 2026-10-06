# Trend-following study (daily, FX + energy)

Results and verdict: [REPORT.md](REPORT.md).

```
pip install pandas numpy scipy matplotlib
python fetch_data.py   # optional: rebuilds data/prices.csv (committed snapshot is to 2026-10-02)
python trend.py        # ~80 s; writes results/ and REPORT.md
python crosscheck.py   # independent simple re-implementation, for sanity
```

To add gold, silver or indices, save Stooq/Yahoo daily CSVs as `data/extra/XAUUSD.csv`
etc. (they need `Date` and `Close` columns), then run `fetch_data.py` and `trend.py`.
New markets are costed at the "energy" rate; edit `COSTS` in `trend.py` to change that.
