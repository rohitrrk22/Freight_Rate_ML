"""Sensitivity test: what does it cost to make the daily-market and regime features strictly causal?
All rows use the SAME folds, cleaning, metrics, 1 seed, 600 trees (the ablation protocol) so they are comparable with
Table 6 of the report. 'Batch' = shipped configuration (monthly mean / full-month regime)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from freight_rate.data import load, clean
from freight_rate.features import make_features, build_day_table, build_day_table_causal
from freight_rate.model import FreightRateModel
from freight_rate.validation import forward_cv, metrics
tr_raw, va_raw, _, _ = load(ROOT / "data"); tr = clean(tr_raw); rate, dates = tr.posted_rate, tr.date
batch = build_day_table(tr_raw, va_raw); causal = build_day_table_causal(tr_raw, va_raw)
mixed_m = batch.copy(); mixed_m["dm_dev"] = causal["dm_dev"]          # causal market only
mixed_r = batch.copy(); mixed_r["qs_sign"] = causal["qs_sign"]        # causal regime only
fast = dict(seeds=(0,), params=dict(n_estimators=600, learning_rate=0.04))
rows = []
for name, day in [("Batch (shipped)", batch), ("Causal market feature only", mixed_m), ("Causal regime only", mixed_r), ("Fully causal (both)", causal)]:
    X = make_features(tr, day); five, _ = forward_cv(lambda: FreightRateModel(**fast), X, rate, dates)
    m = five.drop(columns=["fold", "n_train", "n_val", "market_slope"]).mean()
    nz = []
    for s, e in [("2025-08-16", "2025-09-01"), ("2025-08-09", "2025-09-01")]:
        t = (dates < s).values; v = ((dates >= s) & (dates < e)).values
        nz.append(metrics(FreightRateModel(**fast).fit(X[t], rate[t], dates[t]).predict(X[v]), rate[v].values))
    rows.append(dict(variant=name, MAE_5fold=m.MAE, MedAPE_5fold=m.MedAPE, MAPE_5fold=m.MAPE, MAPEcore_5fold=m.MAPE_core, MedAPE_noise_Aug16=nz[0]["MedAPE"], MedAPE_noise_Aug9=nz[1]["MedAPE"], MAE_noise_Aug16=nz[0]["MAE"]))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows).round(3); out.to_csv(ROOT / "outputs/diag_causal_variant.csv", index=False); print(out.to_string(index=False))
