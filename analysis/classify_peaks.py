"""Classify spectral peaks as mechanical (scale with shaft speed), supply (scale with f_s) or fixed.

Peaks are picked in a light-load trial and looked up in heavy-load trials, where shaft speed
differs by ~0.6%. Each hypothesis predicts a different location; the one with the strongest,
consistent peak wins.
"""
import numpy as np
import pandas as pd

from mcsa import ROOT, load, peak_near, spectrum, top_peaks

speeds = pd.read_csv(ROOT / "analysis" / "speeds.csv").set_index("trial")
REF = 1
TESTS = [6, 12, 18, 24, 30, 45]  # heaviest-load group: largest speed change vs REF

f1, a1, d1 = spectrum(load(REF), 60)
fs1, fr1 = speeds.loc[REF, ["f_s", "f_r1"]]
peaks = [f1[k] for k in top_peaks(f1, d1, 2, 700, n=80, min_sep=0.4) if d1[k] > -70]

spectra = {t: spectrum(load(t), 60) for t in TESTS}


def level(f, amp, a0, x, tol=0.04):
    idx = np.where(np.abs(f - x) <= tol)[0]
    return 20 * np.log10(amp[idx].max() / a0)


rows = []
for F in peaks:
    o = F - fs1
    score = {"mechanical": [], "supply": [], "fixed": []}
    for t, (f, amp, _) in spectra.items():
        fs, fr = speeds.loc[t, ["f_s", "f_r1"]]
        a0 = amp[np.argmin(np.abs(f - fs))]
        # A mechanical component sits at |k*f_s + m*f_r|; express F that way with fr scaling.
        k = round(F / fs1) if abs(F / fs1 - round(F / fs1)) < 0.02 else None
        preds = {
            "mechanical": fs + o * fr / fr1,
            "supply": F * fs / fs1,
            "fixed": fs + o,
        }
        for h, x in preds.items():
            score[h].append(level(f, amp, a0, abs(x)))
    med = {h: float(np.median(v)) for h, v in score.items()}
    best = max(med, key=med.get)
    spread = sorted(med.values())[-1] - sorted(med.values())[-2]
    rows.append(dict(freq=F, offset=o, level_ref=d1[np.argmin(np.abs(f1 - F))], kind=best,
                     margin_dB=spread, off_over_fr1=o / fr1, **{f"L_{h}": v for h, v in med.items()}))

df = pd.DataFrame(rows)
df.loc[df.margin_dB < 3, "kind"] = "ambiguous"
df.to_csv(ROOT / "analysis" / "peak_classes.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(df.kind.value_counts())
