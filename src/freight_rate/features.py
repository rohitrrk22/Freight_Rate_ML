"""Feature engineering.

Design rules learned from validation (see report):
  * quote_signal is used only through a REGIME-AWARE signed signal. Its meaning flips by month (positive /
    negative / pure noise) and Nov-Dec are in the 'pure noise' regime like August. qs_signed = (qs - 2.05) * regime
    sign, so it is 0 in noise months and the model learns "no information" there. The raw value is never used.
  * No month / day-of-year / per-date identifiers: Nov-Dec never appear in training, and per-date values let trees
    memorise date-specific noise (this is what makes random K-fold look far better than reality).
  * Geography enters only through coordinates + distance, never city IDs, because 8 validation cities
    (12% of rows) are unseen in training.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
FEATURES = ["ldist", "dist", "hav", "eq", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "w"]  # base set
FEATURES_FINAL = FEATURES + ["qs_signed"]
QS_CENTER = 2.05  # centre of quote_signal in the symmetric (noise) regime


def haversine(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, [lat1, lon1, lat2, lon2])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))


def build_day_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Daily market level = mean of the row-level market_index over every load posted that day.

    Uses only *features* (never labels) from train + validation: legitimate denoising, since the row value is a
    ~0.025-noise reading of a daily level shared by ~150 loads. dm_dev = daily level minus its calendar-month mean:
    it isolates the day-to-day / day-of-week movement (within-month slope ~0.13 in log-rate) and carries no
    month-level information that would not extrapolate.
    """
    allrows = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    day = allrows.groupby("date").market_index.mean().rename("dm").to_frame()
    day["month"] = day.index.month
    day["dm_dev"] = day["dm"] - day.groupby("month")["dm"].transform("mean")
    # quote_signal regime per month, from the feature's own skew (no labels): +1 / -1 / 0 (noise)
    rt = regime_table(*frames)
    day["qs_sign"] = day["month"].map(rt["regime"].map({"positive": 1, "negative": -1, "noise": 0})).astype(float)
    return day


def make_features(df: pd.DataFrame, day: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["dist"] = df["distance"]
    X["ldist"] = np.log(df["distance"])
    X["hav"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["eq"] = df["equipment"].map(EQUIP)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    X["w"] = df["weight_abs"]
    X["dm_dev"] = df["date"].map(day["dm_dev"]).fillna(0.0).values  # a date with no usable market_index (or absent from the day table) gets no daily adjustment
    sign = df["date"].map(day["qs_sign"]).values
    qs = df["quote_signal"].values if "quote_signal" in df else np.full(len(df), np.nan)  # December chart inputs have none
    X["qs_signed"] = np.where(sign == 0, 0.0, (qs - QS_CENTER) * sign)
    return X


def regime_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Monthly skew of quote_signal (features only). >0.4 'positive', < -0.3 'negative', else 'noise'."""
    q = pd.concat([f[["date", "quote_signal"]] for f in frames], ignore_index=True)
    q["month"] = q.date.dt.month
    t = q.groupby("month").quote_signal.agg(["mean", "std", lambda s: stats.skew(s)])
    t.columns = ["qs_mean", "qs_std", "qs_skew"]
    t["regime"] = np.where(t.qs_skew > 0.4, "positive", np.where(t.qs_skew < -0.3, "negative", "noise"))
    return t


def build_day_table_causal(*frames: pd.DataFrame, history: int = 28, min_hist: int = 7, min_regime_days: int = 5) -> pd.DataFrame:
    """Strictly causal variant of build_day_table, for deployments that score loads before the rest of the month exists.

    * dm_dev = today's daily market level minus the mean of the previous `history` days (0 if fewer than `min_hist` days).
    * qs_sign for date t = regime of quote_signal using only rows from the start of t's month up to and including t;
      0 (no signal) until `min_regime_days` days of the month have been seen.
    Only information dated on or before t is used. Not the shipped configuration; see scripts/causal_variant_test.py."""
    allrows = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames], ignore_index=True)
    day = allrows.groupby("date").market_index.mean().rename("dm").to_frame().sort_index()
    day["month"] = day.index.month
    trailing = day["dm"].shift(1).rolling(history, min_periods=min_hist).mean()
    day["dm_dev"] = (day["dm"] - trailing).fillna(0.0)
    signs = {}
    for d in day.index:
        mrows = allrows[(allrows.date.dt.to_period("M") == d.to_period("M")) & (allrows.date <= d)]
        if mrows.date.nunique() < min_regime_days:
            signs[d] = 0.0
        else:
            s = stats.skew(mrows.quote_signal.values)
            signs[d] = 1.0 if s > 0.4 else (-1.0 if s < -0.3 else 0.0)
    day["qs_sign"] = pd.Series(signs)
    return day
