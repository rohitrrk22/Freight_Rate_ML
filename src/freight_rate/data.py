"""Loading and cleaning. Every cleaning rule here is motivated by an EDA finding (see eda.py)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0  # the weight column is clipped at +/-47,500 lb


def load(data_dir):
    d = Path(data_dir)
    tr = pd.read_csv(d / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(d / "validation.csv", parse_dates=["date"])
    tmpl = pd.read_csv(d / "validation_predictions_template.csv")
    dec = pd.read_csv(d / "december_chart_inputs.csv", parse_dates=["date"])
    return tr, va, tmpl, dec


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning (no statistics are learned here, so it is safe to apply to any split).

    * weight: ~0.6% of rows carry a minus sign (sign-flip error; |weight| distribution and rate behaviour match
      the positives) -> abs(). Missing weight stays NaN (LightGBM handles NaN natively). Values at exactly
      47,500 lb are a clipping artefact -> flagged.
    * market_index: ~0.8% missing at row level. The row value is a noisy reading of a *daily* level, so the model
      uses the daily aggregate (features.build_day_table) and never the raw row value.
    """
    out = df.copy()
    out["weight_abs"] = out["weight"].abs()
    out["weight_capped"] = (out["weight_abs"] >= WEIGHT_CAP).astype(int)
    return out
