"""Narrowband differences between trials with identical supply frequency and load.

Within each nominal group f_s is identical (48.4816 Hz) and shaft speed agrees to ~0.005 Hz,
so 60 s spectra (df = 0.0167 Hz) can be compared bin by bin. A component is reported when a
trial exceeds the group median by >= 12 dB and stands >= 15 dB above its own local noise.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter

from conditions import conditions
from mcsa import ROOT, load, spectrum

FMAX = 2000.0
cond = conditions()
nominal = cond[cond.condition.str.endswith("nominal")]

rows = []
for c, grp in nominal.groupby("condition"):
    trials = list(grp.index)
    specs = {}
    for t in trials:
        f, amp, db = spectrum(load(t), 60)
        m = f <= FMAX
        specs[t] = maximum_filter1d(db[m], 3)  # tolerate ±1 bin
        f = f[m]
    S = np.vstack([specs[t] for t in trials])
    for j, t in enumerate(trials):
        others = np.delete(S, j, axis=0)
        excess = S[j] - np.median(others, axis=0)
        floor = median_filter(S[j], size=301)  # ~5 Hz running median = local noise
        hot = np.where((excess >= 12) & (S[j] - floor >= 15) & (f > 3))[0]
        groups = np.split(hot, np.where(np.diff(hot) > 5)[0] + 1) if len(hot) else []
        for g in groups:
            k = g[np.argmax(excess[g])]
            fs, fr1 = cond.loc[t, ["f_s", "f_r1"]]
            rows.append(dict(condition=c, trial=t, freq=f[k], level_dB=S[j, k], excess_dB=excess[k],
                             snr_dB=S[j, k] - floor[k], off_fs=f[k] - fs, off_over_fr1=(f[k] - fs) / fr1,
                             f_over_fs=f[k] / fs))

ev = pd.DataFrame(rows)
ev.to_csv(ROOT / "analysis" / "narrowband_events.csv", index=False)
pd.set_option("display.width", 220)
if len(ev):
    print(ev.sort_values(["condition", "trial", "freq"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nevents per trial:\n", ev.groupby(["condition", "trial"]).size().to_string())
else:
    print("no narrowband trial-specific components found")
