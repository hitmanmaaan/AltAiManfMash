"""Envelope analysis of stator current: two flavours, because a mechanical fault shows up in two ways.

1. Low-band (carrier) envelope. A fault modulates the load torque, so the current amplitude is
   i(t) = A(1 + m g(t)) cos(2 pi f_s t). The direct spectrum shows this as a pair f_s +- F; the
   envelope |hilbert(i)| / mean - 1 gives m g(t) itself, so the fault sits right at F and its
   amplitude reads directly as the modulation depth m. Single phase: for F > f_s the lower sideband
   folds through 0 Hz and the envelope line splits into F and |F - 2 f_s| (m/2 each); both are
   used as targets. With all three phases the Park-vector modulus has no such split.
2. High-band envelope. Impacts (spalls, cracks) are short bursts that ring at some high frequency;
   their direct spectrum is smeared (random slip of the rolling elements), but the squared envelope
   of the right band has clean lines at the impact rate. The band is picked by the spectral
   kurtosis (STFT kurtogram, Antoni 2006): the band where the signal is most impulsive.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, stft

from mcsa import FS

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))
from common import FIXED_LINES, fault_freqs  # noqa: E402

DS = 25  # 25.6 kHz -> 1024 Hz for everything that lives below 400 Hz
FS_LOW = FS // DS
LOW_BAND = (5.0, 400.0)  # f_s with sidebands up to f_s + BPFI ~ 245 Hz
LOW_SOS = butter(4, LOW_BAND, btype="band", fs=FS_LOW, output="sos")
HF_FIXED = (1000.0, 10_000.0)  # fixed high band, the "no kurtogram" baseline
HF_MIN = 500.0  # kurtogram never picks bands below this: supply harmonics live there
TRIM = 0.02  # fraction cut from both ends of every envelope (Hilbert/filter edge effects)


def _trim(e):
    k = int(len(e) * TRIM)
    return e[k : len(e) - k]


def amp_spectrum(x, fs):
    """Hann amplitude spectrum, amplitude-calibrated (a tone of amplitude a reads a)."""
    x = x - x.mean()
    w = np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1 / fs), np.abs(np.fft.rfft(x * w)) * 2 / w.sum()


def low_envelope(x):
    """Relative envelope m g(t) of the supply carrier, sampled at FS_LOW."""
    y = sosfiltfilt(LOW_SOS, resample_poly(x - x.mean(), 1, DS))
    a = _trim(np.abs(hilbert(y)))
    return a / a.mean() - 1


def park_envelope(X):
    """Relative modulus of the Park vector from three phases (n, 3); None if a phase is missing."""
    if np.isnan(X).any():
        return None
    Y = np.stack([sosfiltfilt(LOW_SOS, resample_poly(c - c.mean(), 1, DS)) for c in X.T], 1)
    Y /= Y.std(0)  # phase T is measured ~23 % low: equalise before combining
    ia, ib, ic = Y.T
    a = _trim(np.hypot((2 * ia - ib - ic) / 3, (ib - ic) / np.sqrt(3)))
    return a / a.mean() - 1


def band_envelope(x, band, squared=True):
    """(Squared) envelope of x filtered to band, decimated to FS_LOW, relative to its mean."""
    lo, hi = band
    sos = butter(4, [lo, min(hi, 0.98 * FS / 2)], btype="band", fs=FS, output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, x - x.mean())))
    e = e**2 if squared else e
    e = _trim(resample_poly(e - e.mean(), 1, DS)) + e.mean()
    return e / e.mean() - 1


def kurtogram(x, nfft=(16, 32, 64, 128, 256, 512, 1024), fmin=HF_MIN):
    """STFT spectral kurtosis for several window lengths.

    Returns [(nfft, f, SK)], and the band (lo, hi) with the largest SK. A stationary sinusoid has
    SK = -1, Gaussian noise 0, a band dominated by repetitive impacts SK >> 0.
    """
    grid, best = [], (-np.inf, None)
    for n in nfft:
        f, _, Z = stft(x - x.mean(), FS, window="hann", nperseg=n, noverlap=3 * n // 4, boundary=None)
        P = np.abs(Z) ** 2
        sk = (P**2).mean(1) / P.mean(1) ** 2 - 2
        grid.append((n, f, sk))
        bw = 2 * FS / n  # Hann main lobe
        ok = (f - bw / 2 >= fmin) & (f + bw / 2 <= 0.98 * FS / 2)
        if ok.any():
            k = np.flatnonzero(ok)[np.argmax(sk[ok])]
            if sk[k] > best[0]:
                best = (sk[k], (f[k] - bw / 2, f[k] + bw / 2))
    return grid, best[1], best[0]


def stft_band_envelopes(x, nperseg=128, hop=16, fmin=1000.0):
    """Squared envelopes of narrow bands at once: |STFT|^2 of each bin is the squared envelope of that
    200 Hz band, sampled at FS/hop = 1600 Hz. Adjacent bins are summed into 400/800/1600 Hz bands.
    Returns [((lo, hi), envelope)]."""
    f, _, Z = stft(x - x.mean(), FS, window="hann", nperseg=nperseg, noverlap=nperseg - hop,
                   boundary=None, padded=False)
    P = np.abs(Z) ** 2
    df = f[1] - f[0]
    first = int(np.ceil(fmin / df))
    out = []
    for w in (1, 2, 4, 8):
        for i in range(first, len(f) - w, max(1, w // 2)):
            out.append(((f[i] - df / 2, f[i + w - 1] + df / 2), P[i : i + w].sum(0)))
    return out, FS / hop


# ---------------------------------------------------------------- line detection


def mask(f, fs, fmax=None):
    """Exclude known non-fault lines: supply harmonics and stand interference (and its sidebands)."""
    keep = np.ones_like(f, bool) if fmax is None else f <= fmax
    for k in range(1, int(f[-1] / fs) + 2):
        keep &= np.abs(f - k * fs) > 0.3
    for fl in FIXED_LINES:
        for x in (fl, fs + fl, abs(fs - fl), 2 * fl):
            keep &= np.abs(f - x) > 0.3
    return keep


def line_snr(f, amp, targets, keep, ring=3.0):
    """10 log10 of the mean (peak power / local median power) over target lines [(freq, tol), ...]."""
    p = amp**2
    vals = []
    for x, tol in targets:
        if x < 2 or x + ring > f[-1]:
            continue
        win = (np.abs(f - x) <= tol) & keep
        ref = (np.abs(f - x) <= ring) & (np.abs(f - x) > max(tol, 0.5)) & keep
        if win.any() and ref.sum() >= 10:
            vals.append(p[win].max() / np.median(p[ref]))
    return 10 * np.log10(np.mean(vals)) if vals else 0.0


# Which modulation frequencies each fault produces (same physics as synth/common.modulation).
def defect_lines(defect, F):
    return {
        "outer_race": [F["BPFO"], 2 * F["BPFO"]],
        "inner_race": [F["BPFI"], F["BPFI"] - F["fr2"], F["BPFI"] + F["fr2"]],
        "ball": [2 * F["BSF"], 2 * F["BSF"] - F["FTF"], 2 * F["BSF"] + F["FTF"]],
        "cage": [F["FTF"]],
        "unbalance": [F["fr2"]],
        "misalignment": [2 * F["fr2"], F["fr2"]],
    }[defect]


def _tol(d):
    return max(0.006 * d, 0.15)  # +-0.3 % geometry/speed uncertainty, x2 margin


def envelope_targets(lines, fs, fold=True):
    """Targets in an envelope spectrum; with fold=True also the single-phase image |d - 2 f_s|."""
    t = [(d, _tol(d)) for d in lines]
    if fold:
        t += [(abs(d - 2 * fs), _tol(d)) for d in lines if d > fs]
    return t


def direct_targets(lines, fs):
    """Targets in the direct current spectrum: sidebands f_s +- d and the line d itself."""
    return [(x, _tol(d)) for d in lines for x in (abs(fs - d), fs + d, d)]


ENV_FMAX = 400.0
DIRECT_FMAX = 1000.0


def scores(x, fs, f_r1, X3=None, kurt=True):
    """Fault scores (dB) of one segment for every defect type, by every method in this module.

    x: phase R at FS; X3: optional (n, 3) for the Park-vector variant.
    Returns {method: {defect: dB}} plus 'band' (the kurtogram band).
    """
    F = fault_freqs(f_r1)
    out = {}
    spectra = {}
    f, a = amp_spectrum(x, FS)
    spectra["direct"] = (f, a, mask(f, fs, DIRECT_FMAX), "direct")
    f, a = amp_spectrum(low_envelope(x), FS_LOW)
    spectra["env_low"] = (f, a, mask(f, fs, ENV_FMAX), "fold")
    if X3 is not None:
        e = park_envelope(X3)
        if e is not None:
            f, a = amp_spectrum(e, FS_LOW)
            spectra["env_park"] = (f, a, mask(f, fs, ENV_FMAX), "nofold")
    f, a = amp_spectrum(band_envelope(x, HF_FIXED), FS_LOW)
    spectra["env_hf"] = (f, a, mask(f, fs, ENV_FMAX), "nofold")
    band = None
    envs, fe = stft_band_envelopes(x)
    out["env_sel"], out["band_sel"] = best_band_scores([(b, *amp_spectrum(_trim(e) / e.mean() - 1, fe))
                                                        for b, e in envs], fs, F)
    if kurt:
        _, band, _ = kurtogram(x)
        f, a = amp_spectrum(band_envelope(x, band), FS_LOW)
        spectra["env_kurt"] = (f, a, mask(f, fs, ENV_FMAX), "nofold")
    for name, (f, a, keep, kind) in spectra.items():
        out[name] = {}
        for d in ["outer_race", "inner_race", "ball", "cage", "unbalance", "misalignment"]:
            lines = defect_lines(d, F)
            if kind == "direct":
                t = direct_targets(lines, fs)
            else:
                t = envelope_targets(lines, fs, fold=kind == "fold")
            out[name][d] = line_snr(f, a, t, keep)
    out["band"] = band
    return out


DEFECT_NAMES = ["outer_race", "inner_race", "ball", "cage", "unbalance", "misalignment"]


def best_band_scores(spectra, fs, F):
    """Targeted band selection ("protrugram"): for every fault hypothesis take the band whose envelope
    spectrum has the strongest lines at that fault's frequencies. spectra: [(band, f, amp)].
    The max over bands inflates the score of clean records too, which the clean-data threshold absorbs."""
    best = {d: -np.inf for d in DEFECT_NAMES}
    where = {}
    keep = None
    for band, f, a in spectra:
        if keep is None or len(keep) != len(f):
            keep = mask(f, fs, ENV_FMAX)
        for d in DEFECT_NAMES:
            v = line_snr(f, a, envelope_targets(defect_lines(d, F), fs, fold=False), keep)
            if v > best[d]:
                best[d], where[d] = v, band
    return best, where


if __name__ == "__main__":
    # Real data: envelope spectra of all 45 trials (phase R, first 60 s), SNR at every fault line.
    import pandas as pd

    from conditions import conditions
    from mcsa import ROOT, load

    cond = conditions()
    rows = []
    for t in range(1, 46):
        X = np.asarray(np.load(ROOT / "data" / "npy" / f"current_{t}.npy", mmap_mode="r")[: 60 * FS], np.float64)
        fs, fr1 = cond.loc[t, ["f_s", "f_r1"]]
        s = scores(X[:, 0], fs, fr1, X3=X)
        r = dict(trial=t, condition=cond.loc[t, "condition"], kurt_lo=s["band"][0], kurt_hi=s["band"][1])
        for m, v in s.items():
            if not m.startswith("band"):
                r.update({f"{m}:{d}": v[d] for d in v})
        rows.append(r)
        print(t, r["condition"], f"band {s['band'][0]:.0f}-{s['band'][1]:.0f} Hz", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "analysis" / "envelope.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(1).to_string(index=False))
