"""Core MCSA helpers: loading trials and high-resolution spectra of stator current."""
from pathlib import Path

import numpy as np

FS = 25_600
ROOT = Path(__file__).resolve().parent.parent
NPY_DIR = ROOT / "data" / "npy"
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# NSK 6205: 9 balls, ball d=7.938 mm, pitch D=38.5 mm, contact angle 0.
BEARING_FACTORS = {"FTF": 0.397, "BSF": 2.322, "BPFO": 3.572, "BPFI": 5.428}


def load(trial: int, phase: int = 0) -> np.ndarray:
    x = np.load(NPY_DIR / f"current_{trial}.npy", mmap_mode="r")[:, phase]
    return np.asarray(x, dtype=np.float64)


def spectrum(x: np.ndarray, seconds: float = 60.0):
    """Hann-windowed amplitude spectrum in dB relative to the strongest peak."""
    n = min(len(x), int(seconds * FS))
    seg = x[:n] - x[:n].mean()
    w = np.hanning(n)
    amp = np.abs(np.fft.rfft(seg * w)) * 2 / w.sum()
    f = np.fft.rfftfreq(n, 1 / FS)
    db = 20 * np.log10(amp / amp.max() + 1e-15)
    return f, amp, db


def peak_near(f, amp, f0, tol):
    """Refined (parabolic) frequency and amplitude of the largest peak within f0±tol."""
    idx = np.where(np.abs(f - f0) <= tol)[0]
    k = idx[np.argmax(amp[idx])]
    a, b, c = np.log(amp[k - 1 : k + 2] + 1e-15)
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return f[k] + d * (f[1] - f[0]), amp[k]


def top_peaks(f, db, fmin, fmax, n=25, min_sep=0.5):
    """Strongest local maxima in [fmin, fmax], separated by at least min_sep Hz."""
    sel = np.where((f >= fmin) & (f <= fmax))[0]
    order = sel[np.argsort(db[sel])[::-1]]
    picked = []
    for k in order:
        if db[k] < db[k - 1] or db[k] < db[k + 1]:
            continue
        if all(abs(f[k] - f[p]) >= min_sep for p in picked):
            picked.append(k)
        if len(picked) == n:
            break
    return sorted(picked, key=lambda k: f[k])
