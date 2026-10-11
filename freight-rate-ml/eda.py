"""Exploratory analysis: prints the key numbers quoted in the report and writes figures to outputs/figures."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight_rate.data import load
from freight_rate.features import haversine, regime_table

TEAL, GREY, ORANGE, RED = "#064A56", "#9DAFB3", "#E07A3F", "#C0392B"
out = Path("outputs/figures"); out.mkdir(parents=True, exist_ok=True)
tr, va, _, _ = load("data")
tr["rpm"] = tr.posted_rate / tr.distance; tr["month"] = tr.date.dt.month
va["month"] = va.date.dt.month
sty = lambda ax: ax.spines[["top", "right"]].set_visible(False)

# ---------- numbers ----------
print("rows", len(tr), len(va), "| train dates", tr.date.min().date(), tr.date.max().date(), "| val dates", va.date.min().date(), va.date.max().date())
print("missing weight/market_index train:", tr.weight.isna().sum(), tr.market_index.isna().sum(), "val:", va.weight.isna().sum(), va.market_index.isna().sum())
print("negative weights train/val:", (tr.weight < 0).sum(), (va.weight < 0).sum(), "| at cap:", (tr.weight.abs() == 47500).sum())
print("distance == 70 floor:", (tr.distance == 70).sum(), (va.distance == 70).sum())
newc = (set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery)); nv = va.pickup.isin(newc) | va.delivery.isin(newc)
print("unseen cities:", sorted(newc), "rows touching them:", nv.sum(), f"{nv.mean():.1%}")
print("market_index mean train/val:", tr.market_index.mean().round(3), va.market_index.mean().round(3))
tr["hav"] = haversine(tr.pickup_lat, tr.pickup_lon, tr.delivery_lat, tr.delivery_lon)

# ---------- label outliers: ratio of posted rate to the equipment x distance-bin typical rate, after removing lane effect ----------
tr["lane"] = tr.pickup + ">" + tr.delivery
tr["eqbin"] = tr.equipment + "|" + pd.cut(tr.distance, [0, 150, 300, 500, 800, 1200, 1800, 2500, 5000]).astype(str)
tr["expected"] = tr.groupby("eqbin").posted_rate.transform("median")
tr["lane_ratio"] = tr.posted_rate / tr.groupby("lane").posted_rate.transform("median")
lr = np.log(tr.lane_ratio); out_mask = (tr.lane_ratio > 2) | (tr.lane_ratio < 0.5)
print(f"label outliers (rate >2x or <0.5x lane median): {out_mask.sum()} ({out_mask.mean():.2%}); high {(tr.lane_ratio>2).sum()}, low {(tr.lane_ratio<0.5).sum()}")

# ---------- Fig 1: data-quality panel ----------
fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.4), dpi=150)
ax = axs[0, 0]; ax.hist(tr.weight.dropna(), bins=80, color=TEAL); ax.set_title("Weight: sign flips + clipping at ±47,500", loc="left", fontsize=10, fontweight="bold"); ax.set_xlabel("weight (lb)"); sty(ax)
ax.annotate(f"{(tr.weight<0).sum()} negative\n{(tr.weight.abs()==47500).sum()} clipped", xy=(-40000, 400), color=RED, fontsize=9)
ax = axs[0, 1]; ax.hist(np.clip(tr.distance / tr.hav, 1, 2.2), bins=80, color=TEAL); ax.set_title("Distance ÷ straight-line: floor at 70 mi, noisy", loc="left", fontsize=10, fontweight="bold"); ax.set_xlabel("ratio (clipped at 2.2)"); sty(ax)
ax.text(0.98, 0.9, f"{(tr.distance==70).sum()} loads at exactly 70 mi\n(censored short hauls)", transform=ax.transAxes, ha="right", fontsize=9, color=RED)
ax = axs[0, 2]; ax.hist(np.clip(lr, -2, 2), bins=120, color=TEAL); ax.set_yscale("log"); ax.set_title("Posted rate ÷ lane median (log scale)", loc="left", fontsize=10, fontweight="bold"); ax.set_xlabel("ln(rate / lane median)"); sty(ax)
ax.text(0.5, 0.88, f"{out_mask.mean():.1%} of labels are 0.2-0.4x or 2.3-5x\n→ label corruption; use robust (L1) loss", transform=ax.transAxes, ha="center", fontsize=8.5, color=RED)
ax = axs[1, 0]
for eq, c in zip(["Dry Van", "Flatbed", "Reefer"], [TEAL, ORANGE, GREY]):
    g = tr[(tr.equipment == eq) & (tr.rpm < 6)]; b = pd.cut(g.distance, np.arange(0, 3600, 150)); m = g.groupby(b, observed=True).rpm.median(); ax.plot([i.mid for i in m.index], m.values, "o-", ms=3, color=c, label=eq)
ax.set_title("Median $/mile falls with distance; Reefer > Flatbed > Van", loc="left", fontsize=10, fontweight="bold"); ax.set_xlabel("distance (mi)"); ax.set_ylabel("$/mile"); ax.legend(frameon=False, fontsize=8); sty(ax)
ax = axs[1, 1]; miss = pd.DataFrame({"train": [tr.weight.isna().mean(), tr.market_index.isna().mean()], "validation": [va.weight.isna().mean(), va.market_index.isna().mean()]}, index=["weight", "market_index"]) * 100
miss.plot.bar(ax=ax, color=[TEAL, ORANGE], rot=0); ax.set_ylabel("% missing"); ax.set_title("Missing values (all features otherwise complete)", loc="left", fontsize=10, fontweight="bold"); ax.legend(frameon=False, fontsize=8); sty(ax)
ax = axs[1, 2]; ax.bar(["seen cities", "8 unseen cities"], [100 - nv.mean() * 100, nv.mean() * 100], color=[TEAL, RED]); ax.set_ylabel("% of validation rows"); ax.set_title("Validation rows touching unseen cities", loc="left", fontsize=10, fontweight="bold"); sty(ax)
ax.text(1, nv.mean() * 100 + 2, f"{nv.mean():.1%}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(out / "eda_quality.png"); plt.close(fig)

# ---------- Fig 2: quote_signal regime ----------
allx = pd.concat([tr.assign(src="train"), va.assign(src="val")], ignore_index=True); allx["week"] = allx.date.dt.to_period("W-SUN").dt.start_time
wk = allx.groupby("week").quote_signal.apply(stats.skew)
tr["res"] = np.log(tr.posted_rate) - np.log(tr.expected)
tr = tr.assign(week=tr.date.dt.to_period("W-SUN").dt.start_time)
cw = tr[(tr.distance > 1200) & (tr.lane_ratio.between(.5, 2))].groupby("week").apply(lambda g: np.corrcoef(g.quote_signal, g.res)[0, 1])
fig, axs = plt.subplots(2, 1, figsize=(10.5, 5.6), dpi=150, sharex=True)
axs[0].plot(wk.index, wk.values, "o-", color=TEAL, ms=3); axs[0].axhline(0, color=GREY, lw=.8); axs[0].set_ylabel("skew of quote_signal"); sty(axs[0])
axs[0].set_title("quote_signal switches regime: right-skewed (+), left-skewed (−), or symmetric noise (0)", loc="left", fontsize=10.5, fontweight="bold")
axs[1].bar(cw.index, cw.values, width=5, color=np.where(cw.values > 0.05, TEAL, np.where(cw.values < -0.05, ORANGE, GREY))); axs[1].axhline(0, color=GREY, lw=.8)
axs[1].set_ylabel("corr(quote_signal, rate/mile)\nlong hauls"); sty(axs[1])
for ax in axs: ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=RED, alpha=.10)
axs[0].text(pd.Timestamp("2025-11-03"), wk.max() * .8, "Validation\n(Nov-Dec):\npure noise\nregime", fontsize=8, color=RED)
axs[1].text(pd.Timestamp("2025-08-04"), .27, "Aug: noise regime\n(corr ≈ 0)", fontsize=8, color="#555")
fig.tight_layout(); fig.savefig(out / "eda_regime.png"); plt.close(fig)
print(regime_table(tr, va).round(2).to_string())

# ---------- Fig 3: market structure ----------
dm = allx.groupby("date").market_index.mean()
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), dpi=150, gridspec_kw={"width_ratios": [1.7, 1]})
axs[0].plot(dm.index, dm.values, color=TEAL, lw=1.2); axs[0].axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=RED, alpha=.10)
axs[0].set_title("Daily market level = mean market_index over ~150 loads/day", loc="left", fontsize=10, fontweight="bold"); sty(axs[0])
dd = dm.groupby(dm.index.dayofweek).mean(); axs[1].bar(["M", "T", "W", "T", "F", "S", "S"], dd.values, color=TEAL); axs[1].set_ylim(0.9, 1.2); axs[1].set_title("Strong day-of-week cycle", loc="left", fontsize=10, fontweight="bold"); sty(axs[1])
fig.tight_layout(); fig.savefig(out / "eda_market.png"); plt.close(fig)

# ---------- Fig 4: split schematic ----------
fig, ax = plt.subplots(figsize=(10.5, 3.3), dpi=150)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
folds = [(0, 4, 4, 6), (0, 5, 5, 7), (0, 6, 6, 8), (0, 7, 7, 9), (0, 8, 8, 10)]
for i, (a, b, c, d) in enumerate(folds):
    y = 4 - i; ax.barh(y, b - a, left=a, color=TEAL, height=.6); ax.barh(y, d - c, left=c, color=ORANGE, height=.6); ax.text(-.15, y, f"Fold {i+1}", ha="right", va="center", fontsize=8.5)
ax.barh(-1.2, 10, left=0, color=TEAL, height=.6); ax.barh(-1.2, 2, left=10, color=RED, height=.6); ax.text(-.15, -1.2, "FINAL", ha="right", va="center", fontsize=8.5, fontweight="bold")
ax.text(5, -1.2, "fit on all of Jan-Oct", color="white", ha="center", va="center", fontsize=8.5); ax.text(11, -1.2, "predict Nov-Dec", color="white", ha="center", va="center", fontsize=8.5)
ax.set_xticks(np.arange(12) + .5); ax.set_xticklabels(months); ax.set_yticks([]); ax.set_xlim(0, 12); sty(ax); ax.spines["left"].set_visible(False)
ax.set_title("Forward-chaining validation: train on the past (teal), score the next 2 months (orange)", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(out / "split_schematic.png"); plt.close(fig)
dq = pd.DataFrame({
    "train": [len(tr), tr.weight.isna().sum(), tr.market_index.isna().sum(), (tr.weight < 0).sum(), (tr.weight.abs() == 47500).sum(), (tr.distance == 70).sum(), tr.duplicated("load_id").sum(), 0],
    "validation": [len(va), va.weight.isna().sum(), va.market_index.isna().sum(), (va.weight < 0).sum(), (va.weight.abs() == 47500).sum(), (va.distance == 70).sum(), va.duplicated("load_id").sum(), int(nv.sum())]},
    index=["rows", "missing weight", "missing market_index", "negative weight", "weight at +/-47,500", "distance == 70", "duplicate load_id", "rows with unseen cities"])
dq.to_csv("outputs/data_quality.csv"); print(dq)
print("figures written")
