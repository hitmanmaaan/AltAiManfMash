"""Synthetic fault symptoms in stator current and the spike-ready features used to detect them.

Physics: a mechanical fault modulates the load torque and therefore the stator current
amplitude: i'(t) = i(t) * (1 + m * g(t)). Each fault has its own g(t) built from the real
shaft speed of the base recording (measured from the f_s ± f_r1 sidebands), so the symptom
lands exactly where it would on this machine. A pure AM term m*cos(2*pi*F*t) produces the
sidebands f_s ± F at 20*log10(m/2) dB relative to the fundamental.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcsa import FS, ROOT  # noqa: E402

OUTPUT_RATIO = 1.5  # f_r2 / f_r1, measured from the ±1.5·f_r1 sideband family
BEARING = {"FTF": 0.397, "BSF": 2.322, "BPFO": 3.572, "BPFI": 5.428}  # NSK 6205, × f_r2

DEFECTS = ["outer_race", "inner_race", "ball", "cage", "unbalance", "misalignment"]
DEFECT_RU = {"normal": "норма", "outer_race": "наружное кольцо", "inner_race": "внутреннее кольцо",
             "ball": "тела качения", "cage": "сепаратор", "unbalance": "дисбаланс",
             "misalignment": "расцентровка"}
SEVERITY_DB = {1: -66.0, 2: -58.0, 3: -50.0}  # main sideband level re fundamental
# The stand already has its own 1x line of the output shaft at about -54 dB, so unbalance severity is
# set relative to it (+6/+12/+18 dB); otherwise a weak unbalance is physically invisible.
SEVERITY_DB_BY_DEFECT = {"unbalance": {1: -48.0, 2: -42.0, 3: -36.0}}
SEVERITY_RU = {0: "—", 1: "слабая", 2: "средняя", 3: "сильная"}

SEG_S = 10  # feature window: 10 s -> 0.1 Hz resolution
SYNTH_DIR = ROOT / "analysis" / "synth"
DATA_DIR = ROOT / "data" / "synthetic"


def fault_freqs(f_r1: float) -> dict:
    fr2 = OUTPUT_RATIO * f_r1
    f = {k: v * fr2 for k, v in BEARING.items()}
    f["fr1"], f["fr2"] = f_r1, fr2
    return f


def modulation(defect: str, f_r1: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """g(t) for one defect; the main term has unit amplitude."""
    t = np.arange(n) / FS
    jitter = 1 + rng.uniform(-0.003, 0.003)  # bearing geometry / speed uncertainty
    F = {k: v * jitter for k, v in fault_freqs(f_r1).items()}
    ph = rng.uniform(0, 2 * np.pi, 4)

    def c(freq, p):
        return np.cos(2 * np.pi * freq * t + p)

    if defect == "outer_race":
        return c(F["BPFO"], ph[0]) + 0.5 * c(2 * F["BPFO"], ph[1])
    if defect == "inner_race":
        return c(F["BPFI"], ph[0]) * (1 + 0.5 * c(F["fr2"], ph[1]))
    if defect == "ball":
        return c(2 * F["BSF"], ph[0]) * (1 + 0.5 * c(F["FTF"], ph[1]))
    if defect == "cage":
        return c(F["FTF"], ph[0])
    if defect == "unbalance":
        return c(F["fr2"], ph[0])
    if defect == "misalignment":
        return 0.6 * c(F["fr2"], ph[0]) + c(2 * F["fr2"], ph[1])
    raise ValueError(defect)


def inject(x: np.ndarray, defect: str, severity: int, f_r1: float, rng: np.random.Generator):
    """Return (x_with_symptom, level_dB). x is (n,) or (n, phases); NaN phases stay NaN."""
    if defect == "normal":
        return x.copy(), np.nan
    level = SEVERITY_DB_BY_DEFECT.get(defect, SEVERITY_DB)[severity] + rng.uniform(-2, 2)
    m = 2 * 10 ** (level / 20)
    g = modulation(defect, f_r1, len(x), rng)
    mod = 1 + m * g
    return (x * (mod[:, None] if x.ndim == 2 else mod)), level


IMPACT_DEFECTS = ["outer_race", "inner_race", "ball"]
IMPACT_DB = {1: -12.0, 2: -6.0, 3: 0.0}  # impact peak re RMS of the 1-10 kHz residual of the record
IMPACT_TAU = 0.4e-3  # ringing decay time, s
IMPACT_SLIP = 0.01  # random spread of the impact interval (rolling elements slip), 1 sigma


def inject_impacts(x: np.ndarray, defect: str, severity: int, f_r1: float, rng: np.random.Generator):
    """Impulsive (spall) model: a train of decaying bursts at a random resonance 2-6 kHz, added to x.

    Unlike inject(), the symptom is not a clean torque modulation: impact times jitter by
    IMPACT_SLIP, so the direct spectrum has no stable lines and only the envelope reveals the rate.
    Inner race impacts are amplitude-modulated by the load zone (f_r2), ball impacts by the cage (FTF).
    Returns (x_with_impacts, level_dB re HF RMS). x is (n,) or (n, phases); all phases get the burst.
    """
    from scipy.signal import butter, sosfiltfilt

    n = len(x)
    F = fault_freqs(f_r1)
    rate, am = {"outer_race": (F["BPFO"], None), "inner_race": (F["BPFI"], F["fr2"]),
                "ball": (2 * F["BSF"], F["FTF"])}[defect]
    T = 1 / (rate * (1 + rng.uniform(-0.003, 0.003)))
    tk = np.cumsum(T * (1 + IMPACT_SLIP * rng.standard_normal(int(n / FS / T) + 10)))
    tk = tk[tk < n / FS - 0.01] + rng.uniform(0, T)
    a = np.ones_like(tk) if am is None else 1 + 0.5 * np.cos(2 * np.pi * am * tk + rng.uniform(0, 2 * np.pi))
    f_res = rng.uniform(2000, 6000)
    th = np.arange(int(8 * IMPACT_TAU * FS)) / FS
    h = np.exp(-th / IMPACT_TAU) * np.sin(2 * np.pi * f_res * th)
    s = np.zeros(n)
    idx = (tk * FS).astype(int)
    np.add.at(s, idx, a)
    s = np.convolve(s, h)[:n]
    x0 = x[:, 0] if x.ndim == 2 else x
    sos = butter(4, [1000, 10_000], btype="band", fs=FS, output="sos")
    sigma = sosfiltfilt(sos, x0 - x0.mean()).std()
    level = IMPACT_DB[severity] + rng.uniform(-2, 2)
    s *= 10 ** (level / 20) * sigma / np.abs(h).max()
    return (x + (s[:, None] if x.ndim == 2 else s)), level


# ---------------------------------------------------------------- features


FMAX = 150.0  # analysis band: everything above 150 Hz is ignored
DECIM = 50  # 25.6 kHz -> 512 Hz after anti-alias filtering (Nyquist 256 Hz > FMAX)
FS_DS = FS // DECIM


def downsample(seg):
    """Anti-alias low-pass + decimation to 512 Hz: the device only needs a 512 S/s current channel."""
    return resample_poly(seg - seg.mean(), 1, DECIM)


def _spectrum(seg):
    y = downsample(seg)
    w = np.hanning(len(y))
    amp = np.abs(np.fft.rfft(y * w)) * 2 / w.sum()
    f = np.fft.rfftfreq(len(y), 1 / FS_DS)
    return f, amp


def _peak(f, amp, f0, tol):
    idx = np.where(np.abs(f - f0) <= tol)[0]
    k = idx[np.argmax(amp[idx])]
    a, b, c = np.log(amp[k - 1 : k + 2] + 1e-15)
    return f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]), amp[k]


def measure_speed(f, amp):
    fs, a0 = _peak(f, amp, 48.6, 0.6)
    hi, _ = _peak(f, amp, fs + 24.2, 0.4)
    lo, _ = _peak(f, amp, fs - 24.2, 0.4)
    return fs, ((hi - fs) + (fs - lo)) / 2, a0


FIXED_LINES = [43.07, 86.13, 28.72]  # stand interference found in analysis/classify_peaks.py

# BPFO2 (f_s ± 2·BPFO ≈ 210/308 Hz) is outside the 0-150 Hz band, so it is not a feature any more.
FEATURES = ["U1x", "M2x", "BPFO", "BPFI", "BPFIsb", "BSF2", "FTF"]


def feature_targets(fs, f_r1):
    """{feature: [(sideband frequency, defect frequency it comes from), ...]}"""
    F = fault_freqs(f_r1)
    sb = lambda d: [(abs(fs - d), d), (fs + d, d)]  # noqa: E731
    return {
        "U1x": sb(F["fr2"]),
        "M2x": [(fs + 2 * F["fr2"], 2 * F["fr2"])],  # lower sideband ≈ f_s - 2f_r2 collides with the f_r1 line
        "BPFO": sb(F["BPFO"]),
        "BPFI": sb(F["BPFI"]),
        "BPFIsb": sb(F["BPFI"] - F["fr2"]) + sb(F["BPFI"] + F["fr2"]),
        "BSF2": sb(2 * F["BSF"]),
        "FTF": sb(F["FTF"]),
    }


def _mask(f, fs):
    keep = f <= FMAX
    for k in range(1, 30):
        keep &= np.abs(f - k * fs) > 0.3
    for fl in FIXED_LINES:
        for x in (fl, fs + fl, abs(fs - fl)):
            keep &= np.abs(f - x) > 0.3
    return keep


def segment_features(seg: np.ndarray):
    """7 SNR features (dB over local noise), 0-150 Hz only, for one 10 s segment of phase R."""
    f, amp = _spectrum(seg)
    fs, fr1, a0 = measure_speed(f, amp)
    keep = _mask(f, fs)
    p = amp ** 2
    out = []
    for name, targets in feature_targets(fs, fr1).items():
        vals = []
        for x, fdef in targets:
            # The defect frequency itself is uncertain by ±0.3 % (geometry/speed), ×2 margin. Using the
            # distance from f_s instead would make the window far too narrow for folded sidebands |f_s - F|.
            tol = max(0.006 * fdef, 0.15)
            if x < 2 or x + tol > FMAX:
                continue
            win = (np.abs(f - x) <= tol) & keep
            ring = (np.abs(f - x) <= 3) & (np.abs(f - x) > max(tol, 0.5)) & keep
            if not win.any() or ring.sum() < 10:
                continue
            vals.append(p[win].max() / np.median(p[ring]))
        out.append(10 * np.log10(np.mean(vals)) if vals else 0.0)
    return np.array(out), fs, fr1


def operating_mode(fs, fr1):
    slip = 1 - fr1 / (fs / 2)
    load = 0 if slip < 0.0027 else (1 if slip < 0.0055 else 2)
    return load * 2 + int(fs > 48.5)  # 0..5


def operating_point(fs, fr1):
    """Finer operating point: load level and supply frequency on a 0.05 Hz grid (the drive's setpoint)."""
    return operating_mode(fs, fr1) // 2 * 1000 + int(round(fs / 0.05))


THERMO_DB = [6, 9, 12, 16, 20, 24]


def thermometer(excess: np.ndarray) -> np.ndarray:
    """(n, F) excess dB over normal -> (n, F*L) binary spikes: bit (f, l) = excess_f > THERMO_DB[l]."""
    t = np.array(THERMO_DB)
    return (excess[:, :, None] > t[None, None, :]).reshape(len(excess), -1).astype(np.uint8)
