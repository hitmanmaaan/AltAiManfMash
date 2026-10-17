"""Within-group deviations for trials sharing exactly the same supply frequency (48.48 Hz) and load.

For each band of the fingerprint, a trial's deviation is compared with the spread of the group
(robust z-score: (x - median) / (1.4826 * MAD)). Bands where a trial deviates strongly and
consistently are listed.
"""
import numpy as np
import pandas as pd

from conditions import conditions
from fingerprint import band_edges
from mcsa import ROOT

F = np.load(ROOT / "analysis" / "fingerprint.npy")
e = band_edges()
centers = (e[:-1] + e[1:]) / 2
cond = conditions()
nominal = cond[cond.condition.str.endswith("nominal")]

rows = []
for c, grp in nominal.groupby("condition"):
    idx = grp.index.values - 1
    X = F[idx]
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0) + 0.3  # 0.3 dB floor against tiny MADs
    Zs = (X - med) / mad
    print(f"\n=== {c}: trials {list(grp.index)}")
    for j, t in enumerate(grp.index):
        z = Zs[j]
        dev_db = X[j] - med
        strong = np.where((np.abs(z) >= 4) & (np.abs(dev_db) >= 3))[0]
        lo = (centers < 1000)
        summary = dict(trial=t, condition=c,
                       n_bands_up=int(((z >= 4) & (dev_db >= 3)).sum()),
                       n_bands_down=int(((z <= -4) & (dev_db <= -3)).sum()),
                       mean_dev_LF_dB=float(dev_db[lo].mean()),
                       mean_dev_HF_dB=float(dev_db[~lo].mean()))
        top = strong[np.argsort(-np.abs(dev_db[strong]))][:6]
        summary["top_bands"] = ", ".join(f"{centers[k]:.0f}Hz:{dev_db[k]:+.1f}" for k in top)
        rows.append(summary)
        print(f"  trial {t:2d}: up={summary['n_bands_up']:3d} down={summary['n_bands_down']:3d} "
              f"LF {summary['mean_dev_LF_dB']:+.2f} dB  HF {summary['mean_dev_HF_dB']:+.2f} dB  | {summary['top_bands']}")

pd.DataFrame(rows).to_csv(ROOT / "analysis" / "nominal_group_deviations.csv", index=False)
