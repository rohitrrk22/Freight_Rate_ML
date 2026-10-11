"""Reviewer-facing diagnostics, all on the SAME forward-chaining folds as the main CV.

  1. quote_signal: exclude vs include-as-is vs regime-aware, on the 5 folds AND on a causal August-only fold
  2. recency half-life sweep
  3. 1 vs 3 seeds
  4. subgroup error / bias from the saved fold-level predictions (outputs/cv_predictions.csv)
  5. outputs/summary.json with the headline numbers used in the report and slides

Run after run_pipeline.py:   python scripts/diagnostics.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from freight_rate.data import load, clean
from freight_rate.features import make_features, build_day_table, regime_table, FEATURES, FEATURES_FINAL
from freight_rate.model import FreightRateModel
from freight_rate.validation import forward_cv, metrics, FOLDS

out = ROOT / "outputs"; (out / "figures").mkdir(exist_ok=True, parents=True)
tr_raw, va_raw, _, _ = load(ROOT / "data")
tr, va = clean(tr_raw), clean(va_raw)
day = build_day_table(tr_raw, va_raw)
X = make_features(tr, day); rate, dates = tr.posted_rate, tr.date
AUG = [("2025-08-01", "2025-09-01")]          # causal: train Jan-Jul, score August (the noise-regime month)
fast = dict(params=dict(n_estimators=600, learning_rate=0.04))
t0 = time.time()

# ---- regime-aware quote_signal (regime from monthly skew of the FEATURE only; 0 in noise months) ----
rt = regime_table(tr_raw, va_raw); sign = rt.regime.map({"positive": 1, "negative": -1, "noise": 0})
reg = tr.date.dt.month.map(sign).values
Xq = X.copy(); Xq["qs"] = tr.quote_signal.values; Xq["regime"] = reg  # Xq["qs_signed"] already in X (same definition)

def run(mk, XX):
    five, _ = forward_cv(mk, XX, rate, dates)
    aug, _ = forward_cv(mk, XX, rate, dates, folds=AUG)
    g = lambda t: t.drop(columns=["fold", "n_train", "n_val"]).mean()
    a, b = g(five), g(aug)
    return dict(MAE_5fold=a.MAE, MedAPE_5fold=a.MedAPE, MAPE_5fold=a.MAPE, MAPEcore_5fold=a.MAPE_core, MAE_Aug=b.MAE, MedAPE_Aug=b.MedAPE, MAPE_Aug=b.MAPE, MAPEcore_Aug=b.MAPE_core)

# 1. quote_signal
rows = {}
for name, feats, XX in [("Exclude quote_signal", FEATURES, Xq),
                        ("Include raw quote_signal", FEATURES + ["qs"], Xq),
                        ("Regime-aware: signed signal only (FINAL)", FEATURES + ["qs_signed"], Xq),
                        ("Regime-aware: signed signal + regime indicator", FEATURES + ["regime", "qs_signed"], Xq)]:
    rows[name] = run(lambda f=feats: FreightRateModel(features=f, seeds=(0,), **fast), XX); print(name, {k: round(v, 2) for k, v in rows[name].items()}, f"{time.time()-t0:.0f}s", flush=True)
q = pd.DataFrame(rows).T.round(3); q.to_csv(out / "diag_quote_signal.csv"); 

# 2. half-life sweep
rows = {}
for hl in [None, 21, 45, 90, 180]:
    t, _ = forward_cv(lambda h=hl: FreightRateModel(half_life=h, seeds=(0,), **fast), X, rate, dates)
    rows["none" if hl is None else f"{hl} days"] = t.drop(columns=["fold", "n_train", "n_val", "market_slope"]).mean(); print("half-life", hl, round(rows["none" if hl is None else f"{hl} days"].MedAPE, 3), f"{time.time()-t0:.0f}s", flush=True)
hlt = pd.DataFrame(rows).T.round(3); hlt.to_csv(out / "diag_half_life.csv")

# 3. seeds
rows = {}
for k in [1, 3]:
    t, _ = forward_cv(lambda k=k: FreightRateModel(seeds=tuple(range(k)), **fast), X, rate, dates)
    rows[f"{k} seed(s)"] = t.drop(columns=["fold", "n_train", "n_val", "market_slope"]).mean()
sd = pd.DataFrame(rows).T.round(3); sd.to_csv(out / "diag_seeds.csv"); print(sd[["MAE", "MedAPE", "MAPE_core"]], flush=True)

# 4. subgroup diagnostics from saved fold-level predictions of the final configuration
cvp = pd.read_csv(out / "cv_predictions.csv").merge(tr[["load_id", "date", "pickup", "delivery", "equipment", "distance", "weight"]], on="load_id")
fold_start = cvp.fold.str[:7].map(lambda s: pd.Timestamp(s + "-01"))
seen = {f: set(tr[tr.date < pd.Timestamp(f[:7] + "-01")].pickup) | set(tr[tr.date < pd.Timestamp(f[:7] + "-01")].delivery) for f in cvp.fold.unique()}
cvp["unseen_city"] = [(p not in seen[f]) or (d not in seen[f]) for p, d, f in zip(cvp.pickup, cvp.delivery, cvp.fold)]
cvp["ape"] = (cvp.pred - cvp.actual).abs() / cvp.actual; cvp["signed"] = (cvp.pred - cvp.actual) / cvp.actual; cvp["lr"] = np.log(cvp.pred / cvp.actual)
cvp["month"] = cvp.date.dt.strftime("%Y-%m"); cvp["dist_bucket"] = pd.cut(cvp.distance, [0, 300, 600, 1000, 1500, 2500, 5000]).astype(str)
cvp["rate_bucket"] = pd.qcut(cvp.actual, 8).astype(str); cvp["weight_missing"] = cvp.weight.isna()
cvp["abs_lr_gt_.5"] = cvp.lr.abs() > 0.5
def grp(col):
    g = cvp.groupby(col, observed=True)
    return pd.DataFrame({"n": g.size(), "MAE": g.apply(lambda d: (d.pred - d.actual).abs().mean()), "MedAPE%": g.ape.median() * 100, "MAPE%": g.ape.mean() * 100, "median_bias%": g.signed.median() * 100, "mean_log_ratio": g.lr.mean()}).round(2)
sub = {c: grp(c) for c in ["dist_bucket", "equipment", "month", "unseen_city", "weight_missing", "rate_bucket"]}
for c, t in sub.items(): t.to_csv(out / f"diag_by_{c}.csv"); print("\n", c, "\n", t.to_string(), flush=True)
overall = dict(median_bias_pct=float(cvp.signed.median() * 100), mean_log_ratio=float(cvp.lr.mean()), share_abs_log_ratio_gt_half=float(cvp["abs_lr_gt_.5"].mean() * 100))
print("\noverall bias", overall)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), dpi=150)
d = sub["dist_bucket"].reindex(["(0, 300]", "(300, 600]", "(600, 1000]", "(1000, 1500]", "(1500, 2500]", "(2500, 5000]"])
axs[0].bar(range(len(d)), d["MedAPE%"], color="#064A56"); axs[0].set_xticks(range(len(d))); axs[0].set_xticklabels(["<300", "300-600", "600-1k", "1-1.5k", "1.5-2.5k", ">2.5k"], fontsize=8); axs[0].set_title("Median % error by distance (mi)", loc="left", fontsize=10, fontweight="bold")
e = sub["equipment"]; axs[1].bar(e.index, e["MedAPE%"], color="#064A56"); axs[1].set_title("Median % error by equipment", loc="left", fontsize=10, fontweight="bold")
r = sub["rate_bucket"]; axs[2].bar(range(len(r)), r["median_bias%"], color=["#E07A3F" if v > 0 else "#064A56" for v in r["median_bias%"]]); axs[2].axhline(0, color="#999", lw=.8)
axs[2].set_title("Median bias by actual-rate octile (%)", loc="left", fontsize=10, fontweight="bold"); axs[2].set_xlabel("low actual rate  →  high actual rate", fontsize=8)
for ax in axs: ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(out / "figures/diag_subgroups.png"); plt.close(fig)

# 5. summary.json
cv = pd.read_csv(out / "cv_folds.csv"); m = cv.drop(columns=["fold"]).mean()
leak = pd.read_csv(out / "leakage_demo.csv", index_col=0); city = pd.read_csv(out / "unseen_city_test.csv").set_index("model")
summary = dict(cv_mean={k: float(v) for k, v in m.items()}, cv_last_fold=cv.iloc[-1].drop("fold").astype(float).to_dict(), slope_by_fold=cv.market_slope.round(3).tolist(),
               leak_random_medape=float(leak.MedAPE.iloc[0]), leak_forward_medape=float(leak.MedAPE.iloc[1]),
               city_with=float(city.loc["with coordinates", "MedAPE"]), city_without=float(city.loc["no coordinates", "MedAPE"]), overall_bias=overall)
json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
print(f"\ndiagnostics done in {time.time()-t0:.0f}s", flush=True)
