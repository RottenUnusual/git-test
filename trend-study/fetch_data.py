"""Download daily closes and build data/prices.csv.

Stooq and Yahoo were blocked by the network policy of the environment this study
was run in, so the default sources are public mirrors on GitHub (datasets/*):
  - FX majors: FRED H.10 noon buying rates (datasets/exchange-rates)
  - WTI, Brent: EIA spot prices (datasets/oil-prices)
  - Henry Hub natural gas: EIA spot (datasets/natural-gas)

To add markets (gold, silver, US indices, ...) drop Stooq/Yahoo daily CSVs into
data/extra/<NAME>.csv (needs a Date column and a Close column) and re-run.
"""
import io
import pathlib
import urllib.request

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = "https://raw.githubusercontent.com/datasets"

# FRED series in that mirror are quoted as foreign currency per USD.
FX = {
    "EURUSD": ("Euro", True),
    "GBPUSD": ("United Kingdom", True),
    "AUDUSD": ("Australia", True),
    "NZDUSD": ("New Zealand", True),
    "USDJPY": ("Japan", False),
    "USDCHF": ("Switzerland", False),
    "USDCAD": ("Canada", False),
}
COMMODITIES = {
    "WTI": f"{RAW}/oil-prices/main/data/wti-daily.csv",
    "BRENT": f"{RAW}/oil-prices/main/data/brent-daily.csv",
    "NATGAS": f"{RAW}/natural-gas/main/data/daily.csv",
}


def get_csv(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def main():
    series = {}
    fx = get_csv(f"{RAW}/exchange-rates/main/data/daily.csv")
    fx = fx.pivot_table(index="Date", columns="Country", values="Exchange rate", aggfunc="last")
    fx.index = pd.to_datetime(fx.index)
    for name, (country, invert) in FX.items():
        s = fx[country].dropna()
        series[name] = 1.0 / s if invert else s

    for name, url in COMMODITIES.items():
        df = get_csv(url)
        series[name] = pd.Series(df["Price"].values, index=pd.to_datetime(df["Date"])).dropna()

    extra = DATA / "extra"
    if extra.exists():
        for f in sorted(extra.glob("*.csv")):
            df = pd.read_csv(f)
            cols = {c.lower(): c for c in df.columns}
            series[f.stem.upper()] = pd.Series(
                df[cols["close"]].values, index=pd.to_datetime(df[cols["date"]])
            ).dropna()

    px = pd.DataFrame(series).sort_index()
    px.index.name = "Date"
    DATA.mkdir(exist_ok=True)
    px.loc["2000":].to_csv(DATA / "prices.csv", float_format="%.6g")
    print(px.loc["2000":].describe().T[["count", "min", "max"]])
    print("first/last:", px.index.min().date(), px.index.max().date())


if __name__ == "__main__":
    main()
