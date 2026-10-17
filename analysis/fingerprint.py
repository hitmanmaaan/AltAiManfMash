"""Broadband spectral fingerprint per trial and test of the "block = machine state" hypothesis.

1. Welch PSD (2 s Hann windows, 0.5 Hz resolution) of phase R, normalised to the fundamental.
2. Supply harmonics (k·f_s) and fixed stand lines are notched out.
3. Log energy in bands (fine 5 Hz bands up to 1 kHz, 100 Hz bands up to 12.8 kHz).
4. Within each operating condition the condition mean is subtracted, leaving only
   trial-specific deviations. Trials are then clustered on those deviations.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import welch

from conditions import conditions
from mcsa import FS, ROOT, load

FIXED_LINES = [43.07, 86.13, 28.72]  # stand interference found in classify_peaks.py
NOTCH = 0.75


def band_edges():
    fine = np.arange(10, 1000, 5)
    coarse = np.arange(1000, 12_801, 100)
    return np.concatenate([fine, coarse])


def fingerprint(t, fs):
    x = load(t)
    f, p = welch(x - x.mean(), fs=FS, window="hann", nperseg=2 * FS, noverlap=FS)
    p = p / p[np.argmin(np.abs(f - fs))]
    keep = np.ones_like(f, bool)
    for k in range(1, int(f[-1] / fs) + 1):
        keep &= np.abs(f - k * fs) > NOTCH
    for fl in FIXED_LINES:
        for k in (1, 2, 3):
            keep &= np.abs(f - k * fl) > NOTCH
            keep &= np.abs(f - (fs + k * fl)) > NOTCH
            keep &= np.abs(f - abs(fs - k * fl)) > NOTCH
    e = band_edges()
    out = np.empty(len(e) - 1)
    for i in range(len(e) - 1):
        m = keep & (f >= e[i]) & (f < e[i + 1])
        out[i] = 10 * np.log10(p[m].mean())
    return out


if __name__ == "__main__":
    cond = conditions()
    F = np.vstack([fingerprint(t, cond.loc[t, "f_s"]) for t in range(1, 46)])
    np.save(ROOT / "analysis" / "fingerprint.npy", F)

    D = F.copy()
    for c, grp in cond.groupby("condition"):
        idx = grp.index.values - 1
        D[idx] -= F[idx].mean(axis=0)

    Z = linkage(D, method="ward")
    res = pd.DataFrame({"trial": range(1, 46), "condition": cond.condition.values,
                        "block": [(t - 1) // 6 + 1 if t <= 30 else 0 for t in range(1, 46)]})
    for k in (2, 3, 4, 5, 6):
        res[f"k{k}"] = fcluster(Z, k, criterion="maxclust")
    res.to_csv(ROOT / "analysis" / "clusters.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))

    # Does clustering follow blocks? Contingency block x cluster for trials 1-30.
    sub = res[res.block > 0]
    for k in (5, 6):
        print(f"\nblock vs cluster (k={k}):")
        print(pd.crosstab(sub.block, sub[f"k{k}"]))
    # Adjusted Rand index vs blocks (1-30), and vs condition as a sanity check.
    from sklearn.metrics import adjusted_rand_score
    print("\nARI(cluster k=5, block) =", round(adjusted_rand_score(sub.block, sub.k5), 3))
    print("ARI(cluster k=6, condition) =", round(adjusted_rand_score(res.condition, res.k6), 3))
