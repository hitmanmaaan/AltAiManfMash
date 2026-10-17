"""Wavelet analysis of stator current.

1. CWT scalogram (complex Morlet): time-frequency picture. A 60 s FFT assumes the signal does not
   change; the scalogram shows whether lines drift, appear or vanish in time.
2. Wavelet-packet kurtogram: the WPT (db8) splits 0-12.8 kHz into 2, 4, ... 64 equal bands; the
   node with the most impulsive coefficients (largest kurtosis) is reconstructed alone and its
   envelope spectrum is taken. Wavelet counterpart of the STFT kurtogram in envelope.py.
3. Wavelet denoising: soft universal threshold (Donoho) on the DWT details of the high band.
   Impacts are sparse in the wavelet basis and survive; broadband noise is removed before the
   envelope is taken.
"""
import numpy as np
import pywt
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt
from scipy.stats import kurtosis

from envelope import (DS, ENV_FMAX, FS_LOW, HF_FIXED, HF_MIN, TRIM, amp_spectrum, best_band_scores,
                      defect_lines, envelope_targets, line_snr, low_envelope, mask)
from mcsa import FS

WAVELET = "db8"
WPT_LEVEL = 6  # 64 bands of 200 Hz
CMOR = "cmor1.5-1.0"


def _rel_env(y, squared=True):
    e = np.abs(hilbert(y))
    e = e**2 if squared else e
    e = resample_poly(e - e.mean(), 1, DS) + e.mean()
    k = int(len(e) * TRIM)
    e = e[k : len(e) - k]
    return e / e.mean() - 1


def scalogram(x, fs, freqs, wavelet=CMOR):
    """|CWT| of x at the given frequencies (rows) — amplitude-like, comparable across rows."""
    fc = pywt.central_frequency(wavelet)
    scales = fc * fs / np.asarray(freqs)
    W, _ = pywt.cwt(x - x.mean(), scales, wavelet, sampling_period=1 / fs, method="fft")
    return np.abs(W) / np.sqrt(scales)[:, None]


def wpt_kurtogram(x, wavelet=WAVELET, maxlevel=WPT_LEVEL, fmin=HF_MIN):
    """Kurtosis of every WPT node. Returns {level: (bands (n,2), kurt (n,))}, best (level, path, band)."""
    wp = pywt.WaveletPacket(x - x.mean(), wavelet, mode="periodization", maxlevel=maxlevel)
    grid, best = {}, (-np.inf, None)
    for lev in range(1, maxlevel + 1):
        nodes = wp.get_level(lev, order="freq")
        w = FS / 2 / 2**lev
        bands = np.array([(i * w, (i + 1) * w) for i in range(len(nodes))])
        # periodization wraps the record end onto its start: drop edge coefficients
        k = np.array([kurtosis(n.data[int(TRIM * len(n.data)) : len(n.data) - int(TRIM * len(n.data))])
                      for n in nodes])
        grid[lev] = (bands, k)
        for i, n in enumerate(nodes):
            if bands[i, 0] >= fmin and k[i] > best[0]:
                best = (k[i], (lev, n.path, tuple(bands[i])))
    return wp, grid, best[1], best[0]


def wpt_node_signal(wp, path, n):
    """Time signal of one WPT node alone (all other nodes zeroed)."""
    new = pywt.WaveletPacket(None, wp.wavelet, mode=wp.mode, maxlevel=len(path))
    new[path] = wp[path].data
    return new.reconstruct(update=False)[:n]


def wavelet_denoise(x, wavelet=WAVELET, level=5):
    """Soft universal-threshold DWT denoising (noise sigma from the finest details)."""
    c = pywt.wavedec(x, wavelet, mode="periodization", level=level)
    sigma = np.median(np.abs(c[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    c = [c[0]] + [pywt.threshold(d, thr, mode="soft") for d in c[1:]]
    return pywt.waverec(c, wavelet, mode="periodization")[: len(x)]


def wpt_band_envelopes(wp, n, fine=WPT_LEVEL, widths=(1, 2, 4, 8), fmin=1000.0):
    """Band envelopes from the wavelet-packet tree, built like the STFT version in envelope.py:
    every finest node (level 6, 200 Hz) is reconstructed alone, its squared Hilbert envelope taken
    and decimated to FS_LOW; wider bands (400/800/1600 Hz = parent nodes) are the SUM of their
    children's envelopes. The incoherent sum keeps impact energy but drops beats between unrelated
    tones in different 200 Hz nodes (PWM sidebands 4 f_s apart would otherwise beat at ~4 f_s, right
    next to BPFI). Returns [((lo, hi), relative envelope)]."""
    w = FS / 2 / 2**fine
    nodes = wp.get_level(fine, order="freq")
    first = int(np.ceil(fmin / w))
    env = {}
    for i in range(first, len(nodes)):
        e = np.abs(hilbert(wpt_node_signal(wp, nodes[i].path, n))) ** 2
        e = resample_poly(e - e.mean(), 1, DS) + e.mean()
        k = int(len(e) * TRIM)
        env[i] = e[k : len(e) - k]
    out = []
    for m in widths:  # m children = one node m levels up (aligned groups)
        for i in range(first + (-first) % m, len(nodes) - m + 1, m):
            e = sum(env[j] for j in range(i, i + m))
            out.append(((i * w, (i + m) * w), e / e.mean() - 1))
    return out


HF_SOS = butter(4, HF_FIXED, btype="band", fs=FS, output="sos")


def scores(x, fs, f_r1):
    """Fault scores (dB) by the wavelet methods, same targets/masks as envelope.scores."""
    from common import fault_freqs

    F = fault_freqs(f_r1)
    wp, _, (lev, path, band), kmax = wpt_kurtogram(x)
    spectra = {
        "wpt_kurt": amp_spectrum(_rel_env(wpt_node_signal(wp, path, len(x))), FS_LOW),
        "wden_hf": amp_spectrum(_rel_env(wavelet_denoise(sosfiltfilt(HF_SOS, x - x.mean()))), FS_LOW),
    }
    out = {}
    out["wpt_sel"], out["band_sel"] = best_band_scores(
        [(b, *amp_spectrum(e, FS_LOW)) for b, e in wpt_band_envelopes(wp, len(x))], fs, F)
    for name, (f, a) in spectra.items():
        keep = mask(f, fs, ENV_FMAX)
        out[name] = {d: line_snr(f, a, envelope_targets(defect_lines(d, F), fs, fold=False), keep)
                     for d in ["outer_race", "inner_race", "ball", "cage", "unbalance", "misalignment"]}
    out["band"] = band
    out["kurt"] = kmax
    return out


def stationarity(x, fs, f_r1, seconds=60):
    """How constant in time the main low-band lines are: CV of the CWT magnitude of the envelope at
    each line, plus the instantaneous supply frequency spread. Low CV = a 60 s FFT is legitimate."""
    e = low_envelope(x[: seconds * FS])
    lines = {"f_r1": f_r1, "f_r2": 1.5 * f_r1, "43.07": 43.07, "2f_s": 2 * fs}
    S = scalogram(e, FS_LOW, list(lines.values()), wavelet="cmor20-1.0")  # ~ 0.1 Hz-ish bandwidth
    k = S.shape[1] // 20  # drop CWT edge zones
    out = {f"cv_{n}": S[i, k:-k].std() / S[i, k:-k].mean() for i, n in enumerate(lines)}
    # instantaneous supply frequency from the analytic low-band current
    y = sosfiltfilt(butter(4, (fs - 4, fs + 4), btype="band", fs=FS_LOW, output="sos"),
                    resample_poly(x[: seconds * FS] - x.mean(), 1, DS))
    ph = np.unwrap(np.angle(hilbert(y)))
    fi = np.diff(ph) * FS_LOW / (2 * np.pi)
    fi = np.convolve(fi[k:-k], np.ones(FS_LOW) / FS_LOW, "valid")  # 1 s moving average
    out["fs_std_mHz"] = 1000 * fi.std()
    return out


if __name__ == "__main__":
    import pandas as pd

    from conditions import conditions
    from mcsa import ROOT, load

    cond = conditions()
    rows = []
    for t in range(1, 46):
        x = load(t)[: 60 * FS]
        fs, fr1 = cond.loc[t, ["f_s", "f_r1"]]
        s = scores(x, fs, fr1)
        r = dict(trial=t, condition=cond.loc[t, "condition"], wpt_lo=s["band"][0], wpt_hi=s["band"][1],
                 wpt_kurt=s["kurt"])
        for m in ("wpt_kurt", "wpt_sel", "wden_hf"):
            r.update({f"{m}:{d}": v for d, v in s[m].items()})
        r.update(stationarity(x, fs, fr1))
        rows.append(r)
        print(t, r["condition"], f"WPT band {r['wpt_lo']:.0f}-{r['wpt_hi']:.0f} Hz, kurt {r['wpt_kurt']:.2f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "analysis" / "wavelet.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
