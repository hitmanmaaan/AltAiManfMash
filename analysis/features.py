"""Physics-targeted MCSA features for every trial (phase R).

Spectrum: Welch average of 30 s Hann windows (df = 0.033 Hz). For each diagnostic frequency
f_s ± F, the level is the strongest bin within the search window, expressed as SNR in dB over
the local noise floor (median level in a ±3 Hz ring, excluding the window itself). The two
sidebands are averaged in power, so a component must be present on both sides to score high.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

from conditions import conditions
from mcsa import BEARING_FACTORS, FS, ROOT, load

WIN_S = 30
OUTPUT_RATIO = 1.5  # f_r2 / f_r1, measured from the ±1.5·f_r1 sideband family


def psd(x):
    f, p = welch(x - x.mean(), fs=FS, window="hann", nperseg=WIN_S * FS, noverlap=0,
                 scaling="spectrum")
    return f, p


def snr_at(f, p, x, tol):
    if x <= 1:
        return np.nan, np.nan
    win = np.abs(f - x) <= tol
    ring = (np.abs(f - x) <= 3) & ~(np.abs(f - x) <= max(tol, 0.2))
    k = np.where(win)[0][np.argmax(p[win])]
    floor = np.median(p[ring])
    return 10 * np.log10(p[k] / floor), f[k]


def sideband_snr(f, p, fs, F, tol):
    lo, flo = snr_at(f, p, abs(fs - F), tol)
    hi, fhi = snr_at(f, p, fs + F, tol)
    pair = 10 * np.log10((10 ** (lo / 10) + 10 ** (hi / 10)) / 2) if np.isfinite(lo) else hi
    return pair, lo, hi


def trial_features(t, cond):
    x = load(t)
    f, p = psd(x)
    fs, fr1 = cond.loc[t, ["f_s", "f_r1"]]
    fr2 = OUTPUT_RATIO * fr1
    feats = {}
    tight = 0.08  # speed is measured to ~0.005 Hz, so shaft sidebands are tightly located
    for name, F in [("U1", fr1), ("M1", 2 * fr1), ("U2", fr2), ("M2", 2 * fr2)]:
        feats[name], feats[name + "_lo"], feats[name + "_hi"] = sideband_snr(f, p, fs, F, tight)
    for name, k in BEARING_FACTORS.items():
        F = k * fr2
        tol = max(0.015 * F, 0.15)  # bearing geometry / slip uncertainty ~1.5 %
        feats[name], feats[name + "_lo"], feats[name + "_hi"] = sideband_snr(f, p, fs, F, tol)
    fund = p[np.argmin(np.abs(f - fs))]
    harm = sum(p[np.argmin(np.abs(f - h * fs))] for h in range(2, 11))
    feats["THD_dB"] = 10 * np.log10(harm / fund)
    return feats


if __name__ == "__main__":
    cond = conditions()
    rows = []
    for t in range(1, 46):
        r = dict(trial=t, condition=cond.loc[t, "condition"], block=(t - 1) // 6 + 1 if t <= 30 else None)
        r.update(trial_features(t, cond))
        rows.append(r)
        print(t, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "analysis" / "features.csv", index=False)
    cols = ["trial", "condition", "block", "U1", "M1", "U2", "M2", "FTF", "BSF", "BPFO", "BPFI", "THD_dB"]
    pd.set_option("display.width", 200)
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:6.1f}"))
