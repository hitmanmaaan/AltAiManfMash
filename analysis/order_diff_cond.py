"""Differential order spectra within each operating condition.

Uses the order-normalised levels from order_diff.py. A trial's excess is measured against the
median of the other trials run in the same operating condition, so condition-dependent supply
and stand components cancel out and only trial-specific (state-specific) components remain.
"""
import numpy as np
import pandas as pd

from conditions import conditions
from mcsa import ROOT

d = np.load(ROOT / "analysis" / "order_levels.npz")
grid, L = d["grid"], d["L"]
cond = conditions()

# Ignore the fundamental's skirt and the sub-8 Hz region dominated by 1/f noise.
valid = (np.abs(grid) > 0.15) & (grid * 24.2 + 48.5 > 8)

excess = np.full_like(L, np.nan)
for c, grp in cond.groupby("condition"):
    idx = grp.index.values - 1
    for i in idx:
        others = [j for j in idx if j != i]
        excess[i] = L[i] - np.median(L[others], axis=0)

np.save(ROOT / "analysis" / "order_excess.npy", excess)

rows = []
for i in range(45):
    hot = np.where(valid & (excess[i] >= 10) & (L[i] > -72))[0]
    groups = np.split(hot, np.where(np.diff(hot) > 3)[0] + 1) if len(hot) else []
    for g in groups:
        k = g[np.argmax(excess[i, g])]
        rows.append(dict(trial=i + 1, condition=cond.loc[i + 1, "condition"], order=round(grid[k], 3),
                         level_dB=L[i, k], excess_dB=excess[i, k]))

ev = pd.DataFrame(rows)
ev.to_csv(ROOT / "analysis" / "order_events_cond.csv", index=False)

# Cluster events by order (within 0.01) and list which trials share each component.
ev = ev.sort_values("order")
ev["cluster"] = (ev.order.diff() > 0.01).cumsum()
summary = ev.groupby("cluster").agg(order=("order", "median"), n_trials=("trial", "nunique"),
                                   trials=("trial", lambda s: sorted(set(s))),
                                   max_excess=("excess_dB", "max"), mean_level=("level_dB", "mean"))
summary = summary[summary.n_trials >= 2].sort_values("n_trials", ascending=False)
pd.set_option("display.width", 250, "display.max_colwidth", 120)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nevents per trial:")
print(ev.groupby("trial").size().reindex(range(1, 46), fill_value=0).to_string())
