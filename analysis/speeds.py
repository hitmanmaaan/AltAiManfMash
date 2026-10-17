"""Measure supply frequency and both shaft speeds for every trial from current sidebands."""
import pandas as pd

from mcsa import ROOT, load, peak_near, spectrum

rows = []
for t in range(1, 46):
    f, amp, db = spectrum(load(t), 60)
    fs, a0 = peak_near(f, amp, 48.6, 0.6)
    lo1, _ = peak_near(f, amp, fs - 24.2, 0.4)
    hi1, _ = peak_near(f, amp, fs + 24.2, 0.4)
    hi2, _ = peak_near(f, amp, fs + 43.07, 0.4)
    lo2, _ = peak_near(f, amp, fs - 43.07, 0.4)
    fr1 = ((hi1 - fs) + (fs - lo1)) / 2
    fr2 = ((hi2 - fs) + (fs - lo2)) / 2
    rows.append(dict(trial=t, f_s=fs, f_r1=fr1, rpm1=fr1 * 60, f_r2=fr2, rpm2=fr2 * 60,
                     ratio=fr2 / fr1, slip=1 - fr1 / (fs / 2), I_fund=a0))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "analysis" / "speeds.csv", index=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nratio f_r2/f_r1: mean %.4f  std %.5f" % (df.ratio.mean(), df.ratio.std()))
