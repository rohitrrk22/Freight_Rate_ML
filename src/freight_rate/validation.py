"""Time-aware validation. The test period (Nov-Dec) is the *future* of the development data (Jan-Oct), so each
fold trains on the past and scores the next two months - the same 2-month horizon as the real task."""
from __future__ import annotations
import numpy as np
import pandas as pd

FOLDS = [("2025-05-01", "2025-07-01"), ("2025-06-01", "2025-08-01"), ("2025-07-01", "2025-09-01"),
         ("2025-08-01", "2025-10-01"), ("2025-09-01", "2025-11-01")]  # (val_start, val_end); train = all earlier rows


def metrics(pred, actual) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    err = pred - actual
    ape = np.abs(err) / actual
    core = ape < 0.5  # 'core' = loads whose label is not a corrupted outlier; reported next to, never instead of, raw error
    return dict(MAE=np.mean(np.abs(err)), RMSE=np.sqrt(np.mean(err ** 2)), MAPE=ape.mean() * 100,
                MedAPE=np.median(ape) * 100, MAPE_core=ape[core].mean() * 100, MAE_core=np.mean(np.abs(err)[core]),
                R2=1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2), pct_core=core.mean() * 100)


def forward_cv(make_model, X, rate, dates, folds=FOLDS):
    rows, preds = [], []
    for s, e in folds:
        t = (dates < s).values
        v = ((dates >= s) & (dates < e)).values
        m = make_model().fit(X[t], rate[t], dates[t])
        p = m.predict(X[v])
        label = f"{s[:7]} to {e[:7]}"
        rows.append(dict(fold=label, n_train=int(t.sum()), n_val=int(v.sum()), market_slope=getattr(m, "slope_", np.nan), **metrics(p, rate[v].values)))
        preds.append(pd.DataFrame({"idx": np.where(v)[0], "fold": label, "pred": p}))
    return pd.DataFrame(rows), pd.concat(preds)
