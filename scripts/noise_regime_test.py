"""Fairer test of the quote_signal decision for the Nov-Dec (noise-regime) test period.

The Aug-only fold in diagnostics.py trains on Jan-Jul, i.e. on NO noise-regime month, which unfairly penalises the
regime-aware model (the final model does train on August). Here the model trains on Jan-Jul plus Aug 1-15 and is scored
on Aug 16-31 (strictly later), so it has seen the noise regime before being tested on it. Also a second split with a
one-week gap. Everything is causal (train earlier, test later)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from freight_rate.data import load, clean
from freight_rate.features import make_features, build_day_table, regime_table, FEATURES
from freight_rate.model import FreightRateModel
from freight_rate.validation import metrics
tr_raw, va_raw, _, _ = load(ROOT / "data"); tr = clean(tr_raw); day = build_day_table(tr_raw, va_raw)
X = make_features(tr, day); rate, dates = tr.posted_rate, tr.date
rt = regime_table(tr_raw, va_raw); sign = rt.regime.map({"positive": 1, "negative": -1, "noise": 0}); reg = tr.date.dt.month.map(sign).values
X["qs"] = tr.quote_signal.values; X["regime"] = reg  # qs_signed already built by make_features
fast = dict(params=dict(n_estimators=600, learning_rate=0.04), seeds=(0, 1))
rows = []
for name, feats in [("Exclude quote_signal", FEATURES), ("Include raw quote_signal", FEATURES + ["qs"]), ("Regime-aware: signed signal only (FINAL)", FEATURES + ["qs_signed"]), ("Regime-aware: signed signal + regime indicator", FEATURES + ["regime", "qs_signed"])]:
    for split, (s, e) in {"train < Aug 16, test Aug 16-31": ("2025-08-16", "2025-09-01"), "train < Aug 9, test Aug 9-31 (more noise-regime test rows)": ("2025-08-09", "2025-09-01")}.items():
        t = (dates < s).values; v = ((dates >= s) & (dates < e)).values
        m = FreightRateModel(features=feats, **fast).fit(X[t], rate[t], dates[t]); r = metrics(m.predict(X[v]), rate[v].values)
        rows.append(dict(config=name, split=split, n_noise_train=int(((dates >= "2025-08-01") & t).sum()), n_test=int(v.sum()), MAE=r["MAE"], MedAPE=r["MedAPE"], MAPE=r["MAPE"], MAPE_core=r["MAPE_core"]))
        print(rows[-1], flush=True)
out = pd.DataFrame(rows).round(2); out.to_csv(ROOT / "outputs/diag_noise_regime.csv", index=False); print(out.to_string())
