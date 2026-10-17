"""Where does the impulsive 4-5 kHz component come from? Both kurtograms pick 4.0-5.2 kHz in all 45 trials.

Its strongest envelope line near 147 Hz obeys line = C - f_s with one constant C for all trials
(a fixed-frequency inverter component mixed with the supply). A mechanical line would scale with the
shaft speed instead. Same f_s -> same line, whatever the load. Writes pwm_lines.csv.
"""
import pandas as pd

import envelope as E
from conditions import conditions
from mcsa import FS, ROOT, load, peak_near

cond = conditions()
rows = []
for t in range(1, 46):
    fs, fr1 = cond.loc[t, ["f_s", "f_r1"]]
    f, a = E.amp_spectrum(E.band_envelope(load(t)[: 60 * FS], (4000, 4800)), E.FS_LOW)
    line, amp = peak_near(f, a, 3 * fs + 1.6, 1.2)
    rows.append(dict(trial=t, condition=cond.loc[t, "condition"], f_s=fs, f_r1=fr1, line=line,
                     delta=line - 3 * fs, amp=amp))
    print(t, flush=True)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "analysis" / "pwm_lines.csv", index=False)
g = df.groupby(df.f_s.round(3)).agg(n=("trial", "size"), delta_mean=("delta", "mean"), delta_std=("delta", "std"),
                                     loads=("condition", lambda c: ", ".join(sorted({x.split("/")[0] for x in c}))))
print(g.round(3).to_string())
C = df.line + df.f_s
print("line + f_s = C: mean %.3f Hz, std %.4f Hz, range %.3f..%.3f" % (C.mean(), C.std(), C.min(), C.max()))
print("corr(delta, f_s) = %.2f, corr(delta, f_r1) = %.2f" % (df.delta.corr(df.f_s), df.delta.corr(df.f_r1)))
