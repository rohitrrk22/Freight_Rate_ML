"""Is the quote_signal regime knowable EARLY in a month, i.e. without seeing the whole month?
Compares the regime from the full month (used by the model) with the regime from only the first 7 / 14 days."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from freight_rate.data import load
tr, va, _, _ = load(ROOT / "data"); q = pd.concat([tr[["date", "quote_signal"]], va[["date", "quote_signal"]]])
q["month"] = q.date.dt.month; q["dom"] = q.date.dt.day
cls = lambda s: "positive" if s > 0.4 else ("negative" if s < -0.3 else "noise")
rows = []
for m, g in q.groupby("month"):
    d = {"month": m, "n": len(g), "skew_full": stats.skew(g.quote_signal), "regime_full": cls(stats.skew(g.quote_signal))}
    for k in (7, 14):
        s = stats.skew(g[g.dom <= k].quote_signal); d[f"skew_first{k}d"] = s; d[f"regime_first{k}d"] = cls(s)
    rows.append(d)
t = pd.DataFrame(rows); t["agree_7d"] = t.regime_full == t.regime_first7d; t["agree_14d"] = t.regime_full == t.regime_first14d
t.round(2).to_csv(ROOT / "outputs/diag_regime_causality.csv", index=False); print(t.round(2).to_string(index=False))
print("agreement first-7-days:", t.agree_7d.mean(), " first-14-days:", t.agree_14d.mean())
