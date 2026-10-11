"""End-to-end: clean -> features -> time-aware CV + ablations -> final fit -> predictions -> December chart inputs.

    python run_pipeline.py --data-dir data --out-dir outputs
    python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight_rate.data import load, clean
from freight_rate.features import make_features, build_day_table, FEATURES, FEATURES_FINAL
from freight_rate.model import FreightRateModel
from freight_rate.validation import forward_cv, metrics, FOLDS

TEAL, GREY, ORANGE = "#064A56", "#9DAFB3", "#E07A3F"


class NaiveBaseline:
    """Median rate-per-mile by equipment x distance bin (what an analyst would do in a spreadsheet)."""
    bins = [0, 150, 300, 500, 800, 1200, 1800, 2500, 5000]
    def fit(self, X, rate, dates):
        d = pd.DataFrame({"eq": X["eq"].values, "b": pd.cut(X.dist, self.bins).values, "rpm": rate.values / X.dist.values})
        self.t_ = d.groupby(["eq", "b"], observed=True).rpm.median(); return self
    def predict(self, X):
        idx = pd.MultiIndex.from_arrays([X["eq"].values, pd.cut(X.dist, self.bins).values])
        return self.t_.reindex(idx).values * X.dist.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--out-dir", default="outputs"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--reuse-cv", action="store_true", help="reuse saved CV/ablation tables, rerun only final fit")
    a = ap.parse_args()
    if a.quick:
        print("*** QUICK MODE: CV/ablation use 1 seed and write to %s_quick; they are NOT the metrics of the final model. The final fit and predictions are unchanged (5 seeds). ***" % a.out_dir)
    out = Path(a.out_dir + ("_quick" if a.quick else "")); (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tr_raw, va_raw, tmpl, dec = load(a.data_dir)
    tr, va = clean(tr_raw), clean(va_raw)
    day = build_day_table(tr_raw, va_raw)
    Xtr, Xva = make_features(tr, day), make_features(va, day)
    rate, dates = tr.posted_rate, tr.date
    print(f"train {tr.shape}, validation {va.shape}; features: {FEATURES_FINAL} + dm_dev (linear term)")

    seeds = (0,) if a.quick else (0, 1, 2)
    fast = dict(seeds=(0,), params=dict(n_estimators=600, learning_rate=0.04))
    if a.reuse_cv:
        cv_tab = pd.read_csv(out / "cv_folds.csv"); leak = pd.read_csv(out / "leakage_demo.csv", index_col=0)
        s_, e_ = FOLDS[-1]; t_ = (dates < s_).values
        _, cv_pred = forward_cv(lambda: FreightRateModel(**fast), Xtr, rate, dates, folds=[FOLDS[-1]])
        print("reused saved CV tables")
    else:
        # ---------------- 1. time-aware cross-validation of the final model ----------------
        final = lambda: FreightRateModel(seeds=seeds)
        cv_tab, cv_pred = forward_cv(final, Xtr, rate, dates)
        cv_tab.to_csv(out / "cv_folds.csv", index=False)
        cvp = cv_pred.copy(); cvp["load_id"] = tr.load_id.values[cvp.idx.values]; cvp["actual"] = rate.values[cvp.idx.values]
        cvp[["load_id", "fold", "actual", "pred"]].to_csv(out / "cv_predictions.csv", index=False)
        print("\nForward-chaining CV (final model)\n", cv_tab.round(2).to_string(index=False))
        print("mean:", cv_tab.drop(columns="fold").mean().round(2).to_dict())

        # ---------------- 2. ablations (same folds, 1 seed for speed) ----------------
        Xab = Xtr.copy(); Xab["qs"] = tr.quote_signal.values
        variants = {
            "Naive: median $/mile by equipment x distance bin": (lambda: NaiveBaseline(), Xtr),
            "LightGBM, no recency, no market term, no quote_signal": (lambda: FreightRateModel(features=FEATURES, half_life=None, use_market=False, **fast), Xtr),
            "  + daily market term (dm_dev)": (lambda: FreightRateModel(features=FEATURES, half_life=None, use_market=True, **fast), Xtr),
            "  + recency weights (excludes quote_signal)": (lambda: FreightRateModel(features=FEATURES, use_market=True, **fast), Xtr),
            "  + regime-aware quote_signal (FINAL model)": (lambda: FreightRateModel(**fast), Xtr),
            "Final but L2 objective (mean, not median)": (lambda: FreightRateModel(params=dict(objective="l2", n_estimators=600, learning_rate=0.04), seeds=(0,)), Xtr),
            "Base + raw quote_signal (not regime-aware)": (lambda: FreightRateModel(features=FEATURES + ["qs"], **fast), Xab),
            "Final + dm_dev as a tree feature": (lambda: FreightRateModel(features=FEATURES_FINAL + ["dm_dev"], use_market=False, **fast), Xtr),
            "Final without weight": (lambda: FreightRateModel(features=[f for f in FEATURES_FINAL if f != "w"], **fast), Xtr),
        }
        rows = []
        for name, (mk, X) in variants.items():
            t, _ = forward_cv(mk, X, rate, dates)
            rows.append(dict(variant=name, **t.drop(columns=["fold", "n_train", "n_val"]).mean().to_dict()))
            print(f"  {name:55s} MAPE_core={rows[-1]['MAPE_core']:.2f}  MedAPE={rows[-1]['MedAPE']:.2f}  MAE={rows[-1]['MAE']:.1f}")
        ab = pd.DataFrame(rows); ab.to_csv(out / "ablation.csv", index=False)

        # ---------------- 3. leakage demonstration: random K-fold vs forward ----------------
        Xleak = Xtr.copy(); Xleak["dm"] = tr.date.map(day.dm).values; Xleak["doy"] = tr.date.dt.dayofyear.values
        leak_feats = FEATURES + ["dm", "doy"]
        rnd = []
        for tri, tei in KFold(5, shuffle=True, random_state=0).split(Xleak):
            m = FreightRateModel(features=leak_feats, use_market=False, half_life=None, **fast).fit(Xleak.iloc[tri], rate.iloc[tri], dates.iloc[tri])
            rnd.append(metrics(m.predict(Xleak.iloc[tei]), rate.iloc[tei].values))
        tf, _ = forward_cv(lambda: FreightRateModel(features=leak_feats, use_market=False, half_life=None, **fast), Xleak, rate, dates)
        leak = pd.DataFrame({"Random 5-fold (date features visible)": pd.DataFrame(rnd).mean(), "Forward-chaining (same model)": tf.drop(columns=["fold", "n_train", "n_val"]).mean()}).T
        leak.to_csv(out / "leakage_demo.csv"); print("\nLeakage demo\n", leak[["MAE", "MAPE_core", "MedAPE"]].round(2))

        # ---------------- 4. unseen-city generalisation ----------------
        rng = np.random.RandomState(0); cities = sorted(set(tr.pickup) | set(tr.delivery)); cr = []
        for rep in range(3):
            ho = set(rng.choice(cities, 8, replace=False)); v = (tr.pickup.isin(ho) | tr.delivery.isin(ho)).values
            m = FreightRateModel(use_market=False, half_life=None, **fast).fit(Xtr[~v], rate[~v], dates[~v]); cr.append(metrics(m.predict(Xtr[v]), rate[v].values))
            m2 = FreightRateModel(features=[f for f in FEATURES_FINAL if "lat" not in f and "lon" not in f], use_market=False, half_life=None, **fast)  # like-for-like: same features as the model it is compared with, minus coordinates.fit(Xtr[~v], rate[~v], dates[~v]); cr.append({**metrics(m2.predict(Xtr[v]), rate[v].values), "nolatlon": 1})
        city = pd.DataFrame(cr); city["model"] = np.where(city.get("nolatlon").fillna(0) == 1, "no coordinates", "with coordinates")
        cityt = city.groupby("model")[["MAE", "MAPE_core", "MedAPE"]].mean(); cityt.to_csv(out / "unseen_city_test.csv"); print("\nUnseen-city holdout\n", cityt.round(2))


    # ---------------- 5. final fit on ALL development data ----------------
    final_model = FreightRateModel(seeds=(0, 1, 2, 3, 4)).fit(Xtr, rate, dates)
    imp = final_model.importance(); imp.to_csv(out / "feature_importance.csv")
    print(f"\nfinal model: market slope = {final_model.slope_:.3f}; importance\n{imp.round(3).to_string()}")
    pred = final_model.predict(Xva)
    assert np.isfinite(pred).all() and (pred > 0).all(), "non-finite or non-positive predictions"
    assert rate.min() * 0.5 <= pred.min() and pred.max() <= rate.max() * 1.5, "predictions outside the plausible range of training rates"
    unseen_rows = ~(va.pickup.isin(set(tr.pickup) | set(tr.delivery)) & va.delivery.isin(set(tr.pickup) | set(tr.delivery))).values
    print(f"sanity: {len(pred)} finite positive predictions in [{pred.min():.0f}, {pred.max():.0f}]; rows touching unseen cities: {unseen_rows.sum()} (mean prediction {pred[unseen_rows].mean():.0f} vs {pred[~unseen_rows].mean():.0f} for seen)")
    sub = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": va.load_id.values, "predicted_rate": np.round(pred, 2)}), on="load_id", how="left")
    assert sub.predicted_rate.notna().all() and len(sub) == 12000
    sub.to_csv("validation_predictions.csv", index=False)
    print("validation predictions:", sub.predicted_rate.describe().round(1).to_dict())

    # ---------------- 6. fixed December chart inputs ----------------
    pc = pd.concat([tr_raw[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["c", "lat", "lon"], axis=1), va_raw[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["c", "lat", "lon"], axis=1)]).drop_duplicates().set_index("c")
    dcity = pd.concat([tr_raw[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["c", "lat", "lon"], axis=1), va_raw[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["c", "lat", "lon"], axis=1)]).drop_duplicates().set_index("c")
    d = dec.copy()
    d["pickup_lat"], d["pickup_lon"] = d.pickup.map(pc.lat), d.pickup.map(pc.lon)
    d["delivery_lat"], d["delivery_lon"] = d.delivery.map(dcity.lat), d.delivery.map(dcity.lon)
    d["weight_abs"] = d.weight.abs()
    Xd = make_features(d.assign(weight_abs=d.weight_abs), day)
    assert Xd.notna().all().all()
    dec_out = dec.copy(); dec_out["predicted_rate"] = np.round(final_model.predict(Xd), 2)
    dec_out["date"] = dec_out.date.dt.strftime("%Y-%m-%d")
    dec_out.to_csv(out / "december_predictions.csv", index=False)
    dec_out.to_csv("december_chart_inputs.csv", index=False)  # the completed assignment file (same name, predicted_rate filled)
    print("\nDecember predictions:", dec_out.predicted_rate.agg(["min", "mean", "max"]).round(1).to_dict())

    # ---------------- figures that need model output ----------------
    # (a) CV by fold
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=170)
    ax.bar(cv_tab.fold, cv_tab.MAPE, color=GREY, label="MAPE (all rows)"); ax.bar(cv_tab.fold, cv_tab.MAPE_core, color=TEAL, label="MAPE (core rows)")
    ax.plot(cv_tab.fold, cv_tab.MedAPE, "o-", color=ORANGE, label="Median APE")
    ax.set_ylabel("%"); ax.set_title("Forward-chaining CV: train on the past, score the next 2 months", loc="left", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(out / "figures/cv_by_fold.png"); plt.close(fig)
    # (b) leakage
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=170)
    ax.bar(["Random 5-fold", "Forward-chaining"], leak.MedAPE.values, color=[GREY, TEAL]); ax.set_ylabel("Median APE (%)")
    for i, v in enumerate(leak.MedAPE.values): ax.text(i, v + .03, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_title("Same model, same features: validation scheme\nchanges the answer ~1.7x", loc="left", fontweight="bold", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "figures/leakage_demo.png"); plt.close(fig)
    # (c) importance
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=170); im = imp.sort_values()
    ax.barh(im.index, im.values * 100, color=TEAL); ax.set_xlabel("Share of total gain (%)"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Feature importance (final model)", loc="left", fontweight="bold", fontsize=10); fig.tight_layout(); fig.savefig(out / "figures/importance.png"); plt.close(fig)
    # (d) predicted vs actual on last fold
    s, e = FOLDS[-1]; v = ((dates >= s) & (dates < e)).values; pp = cv_pred[cv_pred.idx.isin(np.where(v)[0])].sort_values("idx")
    act = rate.values[pp.idx.values]; fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=170)
    ax.scatter(act, pp.pred, s=3, alpha=.25, color=TEAL); lim = [0, 14000]; ax.plot(lim, lim, color=ORANGE, lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual posted rate ($)"); ax.set_ylabel("Predicted ($)"); ax.set_title("Sep-Oct holdout (trained Jan-Aug)", loc="left", fontweight="bold", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "figures/pred_vs_actual.png"); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
