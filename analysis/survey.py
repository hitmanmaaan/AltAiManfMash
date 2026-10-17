"""Convert raw CSV trials to float32 .npy and print a per-trial overview."""
from pathlib import Path

import numpy as np
import pandas as pd

FS = 25_600
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data" / "Датасеты" / "csv"
NPY_DIR = ROOT / "data" / "npy"
NPY_DIR.mkdir(parents=True, exist_ok=True)


def supply_freq(x: np.ndarray) -> float:
    n = min(len(x), FS * 20)
    seg = x[:n] - x[:n].mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / FS)
    band = (f > 40) & (f < 60)
    k = np.argmax(spec * band)
    # parabolic interpolation for sub-bin accuracy
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-12)
    return f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])


rows = []
for i in range(1, 46):
    npy = NPY_DIR / f"current_{i}.npy"
    if npy.exists():
        x = np.load(npy)
    else:
        x = pd.read_csv(CSV_DIR / f"current_{i}.csv", dtype=np.float32, engine="c").to_numpy()
        np.save(npy, x)
    rms = np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))
    rows.append(
        dict(trial=i, samples=len(x), seconds=round(len(x) / FS, 1),
             rms_R=rms[0], rms_S=rms[1], rms_T=rms[2],
             unbalance_pct=100 * (rms.max() - rms.min()) / rms.mean(),
             f_supply=supply_freq(x[:, 0]))
    )
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "analysis" / "survey.csv", index=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
