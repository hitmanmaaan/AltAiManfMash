"""Sanity tests of the envelope and wavelet methods on pure synthetic signals with a known answer.

Run: ../.venv/bin/python test_methods.py   (prints each check, exits non-zero on failure)
"""
import numpy as np

import envelope as E
import wavelet as W
from mcsa import FS

rng = np.random.default_rng(0)
N = 10 * FS
t = np.arange(N) / FS
FS_SUP = 48.5
A = 3.3
failed = 0


def check(name, ok, detail):
    global failed
    failed += not ok
    print(f"[{'OK ' if ok else 'FAIL'}] {name} => {detail}")


def peak(f, a, f0, tol=0.3):
    w = np.abs(f - f0) <= tol
    k = np.flatnonzero(w)[np.argmax(a[w])]
    return f[k], a[k]


m = 2 * 10 ** (-50 / 20)  # sidebands at -50 dB re fundamental

# 1. AM below f_s: single-phase envelope gives one line at F with amplitude m
for F in (14.4, 36.3):
    x = A * np.cos(2 * np.pi * FS_SUP * t) * (1 + m * np.cos(2 * np.pi * F * t)) + 0.01 * rng.standard_normal(N)
    f, a = E.amp_spectrum(E.low_envelope(x), E.FS_LOW)
    fp, ap = peak(f, a, F)
    # tolerance 15 %: Hann scalloping loss is up to 1.4 dB when F falls between FFT bins
    check(f"огибающая несущей, F={F} Гц < f_s", abs(fp - F) < 0.1 and abs(ap / m - 1) < 0.15,
          f"линия {fp:.2f} Гц, амплитуда {ap / m:.2f}·m")

# 2. AM above f_s: single phase splits into F and |F - 2 f_s|, m/2 each; Park vector gives m at F
F = 129.8
g = 1 + m * np.cos(2 * np.pi * F * t)
X = np.stack([A * np.cos(2 * np.pi * FS_SUP * t - k * 2 * np.pi / 3) * g for k in range(3)], 1)
X += 0.01 * rng.standard_normal(X.shape)
f, a = E.amp_spectrum(E.low_envelope(X[:, 0]), E.FS_LOW)
_, a1 = peak(f, a, F)
_, a2 = peak(f, a, abs(F - 2 * FS_SUP))
check("огибающая 1 фазы, F > f_s: расщепление", abs(a1 / m - 0.5) < 0.1 and abs(a2 / m - 0.5) < 0.1,
      f"на F {a1 / m:.2f}·m, на |F−2f_s| {a2 / m:.2f}·m (теория 0,5 и 0,5)")
f, a = E.amp_spectrum(E.park_envelope(X), E.FS_LOW)
_, a1 = peak(f, a, F)
_, a2 = peak(f, a, abs(F - 2 * FS_SUP))
check("огибающая вектора Парка, F > f_s", abs(a1 / m - 1) < 0.1 and a2 / m < 0.05,
      f"на F {a1 / m:.2f}·m, на |F−2f_s| {a2 / m:.3f}·m (теория 1 и 0)")

# 3. Impacts at 4 kHz, rate 129.8 Hz with 1 % slip: kurtogram band, envelope line, WPT node
rate, f_res = 129.8, 4000.0
tk = np.cumsum(1 / rate * (1 + 0.01 * rng.standard_normal(int(10 * rate) + 10)))
tk = tk[tk < 9.99]
th = np.arange(64) / FS
h = np.exp(-th / 0.4e-3) * np.sin(2 * np.pi * f_res * th)
s = np.zeros(N)
s[(tk * FS).astype(int)] = 1
x = A * np.cos(2 * np.pi * FS_SUP * t) + 0.05 * rng.standard_normal(N) + 0.1 * np.convolve(s, h)[:N]
_, band, sk = E.kurtogram(x)
check("кюртограмма STFT находит полосу удара", band[0] <= f_res <= band[1], f"{band[0]:.0f}–{band[1]:.0f} Гц, SK={sk:.2f}")
f, a = E.amp_spectrum(E.band_envelope(x, band), E.FS_LOW)
k = np.argmax(a * (f > 5))
check("спектр огибающей: частота ударов", abs(f[k] - rate) < 0.3, f"пик {f[k]:.2f} Гц (задано {rate})")
f, a = E.amp_spectrum(x, FS)
_, ad = peak(f, a, rate, 0.5)
_, ae = peak(*E.amp_spectrum(E.band_envelope(x, band), E.FS_LOW), rate, 0.5)
check("прямой спектр на частоте ударов почти пуст", ad / a.max() < 1e-3, f"{20 * np.log10(ad / a.max()):.0f} дБ отн. основной")
wp, _, best, kmax = W.wpt_kurtogram(x)
check("вейвлет-пакетная кюртограмма находит полосу удара", best[2][0] <= f_res <= best[2][1],
      f"узел {best[1]} = {best[2][0]:.0f}–{best[2][1]:.0f} Гц, эксцесс {kmax:.1f}")
F_dummy = {"BPFO": rate, "BPFI": 197.0, "BSF": 84.0, "FTF": 14.4, "fr1": 24.2, "fr2": 36.3}
sc, where = E.best_band_scores([(b, *E.amp_spectrum(e, E.FS_LOW)) for b, e in W.wpt_band_envelopes(wp, N)],
                               FS_SUP, F_dummy)
lo, hi = where["outer_race"]
check("вейвлет-пакеты: выбор узла по BPFO", lo - 400 <= f_res <= hi + 400 and sc["outer_race"] > 15,
      f"{lo:.0f}–{hi:.0f} Гц, {sc['outer_race']:.1f} дБ")

# 4. Scalogram: a tone that switches on half-way is seen in time (the FFT would only average it)
tt =np.arange(4 * 1024) / 1024
y = np.where(tt > 2, np.cos(2 * np.pi * 80 * tt), 0.0) + 0.01 * rng.standard_normal(len(tt))
S = W.scalogram(y, 1024, [80.0])[0]
check("скалограмма видит включение тона на 2 с", S[512:1536].mean() < 0.1 * S[2560:3584].mean(),
      f"до {S[512:1536].mean():.3f}, после {S[2560:3584].mean():.3f}")

# 5. Wavelet denoising: strong impacts in Gaussian noise are recovered with a smaller error
imp = 0.5 * np.convolve(s, h)[:N]
zi = imp + 0.05 * rng.standard_normal(N)
err0, err1 = (zi - imp).std(), (W.wavelet_denoise(zi) - imp).std()
check("вейвлет-шумоподавление сильных ударов (выигрыш мал: мягкий порог срезает и удары)", err1 < err0,
      f"ошибка до {err0:.4f}, после {err1:.4f}")

print("\nвсе проверки пройдены" if not failed else f"\nпровалено: {failed}")
raise SystemExit(failed)
