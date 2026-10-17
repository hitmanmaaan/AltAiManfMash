"""Impulsiveness (bearing-impact) indicators per trial.

The current is high-passed above 1 kHz to remove the supply and its low harmonics. On that
residual: kurtosis and crest factor (impacts raise both), and the envelope spectrum (Hilbert),
where bearing defect frequencies BPFO/BPFI/BSF/FTF of the output shaft are checked for peaks.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis

from conditions import conditions
from mcsa import BEARING_FACTORS, FS, ROOT, load

SOS = butter(6, [1000, 10_000], btype="band", fs=FS, output="sos")
SECONDS = 60


def envelope_peak_snr(env_spec, f, F):
    tol = max(0.015 * F, 0.3)
    win = np.abs(f - F) <= tol
    ring = (np.abs(f - F) <= 5) & ~win
    return 10 * np.log10(env_spec[win].max() / np.median(env_spec[ring]))


cond = conditions()
rows = []
for t in range(1, 46):
    x = load(t)[: SECONDS * FS]
    y = sosfiltfilt(SOS, x - x.mean())
    env = np.abs(hilbert(y))
    env -= env.mean()
    E = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    fe = np.fft.rfftfreq(len(env), 1 / FS)
    fr2 = 1.5 * cond.loc[t, "f_r1"]
    r = dict(trial=t, condition=cond.loc[t, "condition"],
             kurtosis=kurtosis(y, fisher=False), crest=np.abs(y).max() / y.std(), hf_rms=y.std())
    for name, k in BEARING_FACTORS.items():
        r["env_" + name] = envelope_peak_snr(E, fe, k * fr2)
    r["env_fr1"] = envelope_peak_snr(E, fe, cond.loc[t, "f_r1"])
    r["env_fr2"] = envelope_peak_snr(E, fe, fr2)
    rows.append(r)
    print(t, flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "analysis" / "impulsiveness.csv", index=False)
pd.set_option("display.width", 220)
print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
