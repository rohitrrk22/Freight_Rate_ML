"""Pre-submission checks. Run from the repo root:  python tests/check_outputs.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from freight_rate.data import load, clean
from freight_rate.features import make_features, build_day_table
from freight_rate.model import FreightRateModel
ok = lambda m: print("PASS", m)
tr, va, tmpl, dec = load(ROOT / "data")
p = pd.read_csv(ROOT / "validation_predictions.csv")
assert list(p.columns) == ["load_id", "predicted_rate"]; assert len(p) == 12000 and p.load_id.is_unique and set(p.load_id) == set(tmpl.load_id) and (p.load_id.values == tmpl.load_id.values).all(); ok("validation_predictions: columns, 12,000 rows, ids match template and order")
assert np.isfinite(p.predicted_rate).all() and (p.predicted_rate > 0).all(); assert tr.posted_rate.min() * .5 <= p.predicted_rate.min() and p.predicted_rate.max() <= tr.posted_rate.max() * 1.5; ok(f"predictions finite, positive, in plausible range [{p.predicted_rate.min():.0f}, {p.predicted_rate.max():.0f}]")
d = pd.read_csv(ROOT / "december_chart_inputs.csv"); assert list(d.columns) == list(dec.columns) and len(d) == 31 and d.predicted_rate.notna().all() and (d.predicted_rate > 0).all(); ok("december_chart_inputs.csv completed: 31 rows, no missing predictions")
assert (d.drop(columns="predicted_rate").astype(str).values == dec.drop(columns="predicted_rate").assign(date=dec.date.dt.strftime("%Y-%m-%d")).astype(str).values).all(); ok("December fixed inputs unchanged")
# missing-data behaviour: a date whose market_index is entirely missing, missing weight, negative weight
tc = clean(tr); day = build_day_table(tr, va); Xt = make_features(tc, day)
m = FreightRateModel(seeds=(0,), params=dict(n_estimators=30)).fit(Xt.iloc[:3000], tc.posted_rate.iloc[:3000], tc.date.iloc[:3000])
t = va.head(40).copy(); t["market_index"] = np.nan; t["weight"] = [np.nan, -30000.0] * 20
day2 = build_day_table(t); Xb = make_features(clean(t), day2)
assert np.isfinite(Xb.dm_dev).all() and Xb.w.dropna().ge(0).all(); pr = m.predict(Xb); assert np.isfinite(pr).all() and (pr > 0).all(); ok("all-missing market_index date, missing and negative weights -> finite predictions")
unseen = (set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery)); u = va.pickup.isin(unseen) | va.delivery.isin(unseen)
pu = p.predicted_rate[u.values]; ps = p.predicted_rate[~u.values]; assert np.isfinite(pu).all(); ok(f"unseen cities ({len(unseen)}): {int(u.sum())} rows, mean prediction {pu.mean():.0f} vs {ps.mean():.0f} for seen cities")
print("ALL CHECKS PASSED")
