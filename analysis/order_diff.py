"""Order-normalised differential spectra across all trials.

Each spectrum is re-indexed by order q = (f - f_s) / f_r1, so shaft-locked components line up
across trials regardless of load-dependent speed. The per-order median over all trials is the
"typical stand" signature; each trial's excess over it highlights trial-specific components.
"""
import numpy as np
import pandas as pd

from mcsa import ROOT, load, spectrum

speeds = pd.read_csv(ROOT / "analysis" / "speeds.csv").set_index("trial")
Q_MIN, Q_MAX, DQ = -1.95, 25.0, 0.002
grid = np.arange(Q_MIN, Q_MAX, DQ)
edges = np.concatenate([grid - DQ / 2, [grid[-1] + DQ / 2]])

L = np.empty((45, len(grid)), dtype=np.float32)
for t in range(1, 46):
    f, amp, _ = spectrum(load(t), 60)
    fs, fr = speeds.loc[t, ["f_s", "f_r1"]]
    a0 = amp[np.argmin(np.abs(f - fs))]
    q = (f - fs) / fr
    m = (q >= edges[0]) & (q < edges[-1])
    idx = np.digitize(q[m], edges) - 1
    out = np.full(len(grid), 1e-12)
    np.maximum.at(out, idx, amp[m])
    L[t - 1] = 20 * np.log10(out / a0)
    print("trial", t, flush=True)

np.savez_compressed(ROOT / "analysis" / "order_levels.npz", grid=grid, L=L)

median = np.median(L, axis=0)
excess = L - median

# Report orders where at least one trial is >= 12 dB above the stand median and that
# component itself is well above the noise floor (> -75 dB re fundamental).
rows = []
for t in range(45):
    hot = np.where((excess[t] >= 12) & (L[t] > -75))[0]
    # merge neighbouring bins into single events
    groups = np.split(hot, np.where(np.diff(hot) > 3)[0] + 1) if len(hot) else []
    for g in groups:
        k = g[np.argmax(excess[t, g])]
        rows.append(dict(trial=t + 1, order=grid[k], freq_offset_Hz=grid[k] * speeds.loc[t + 1, "f_r1"],
                         level_dB=L[t, k], excess_dB=excess[t, k]))

ev = pd.DataFrame(rows)
ev.to_csv(ROOT / "analysis" / "order_events.csv", index=False)
pd.set_option("display.width", 200)
print(ev.sort_values(["order", "trial"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# How many distinct trials light up at each (rounded) order.
ev["order_r"] = ev.order.round(2)
print(ev.groupby("order_r").trial.apply(lambda s: sorted(set(s))).to_string())
