"""Figures for the envelope / wavelet report (analysis/figures/m_*.png)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))
import envelope as E  # noqa: E402
import wavelet as W  # noqa: E402
from common import fault_freqs, inject, inject_impacts  # noqa: E402
from compare_methods import METHOD_RU, METHODS  # noqa: E402
from conditions import conditions  # noqa: E402
from mcsa import FIG_DIR, FS, ROOT, load  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
TRIAL = 1
cond = conditions()
fs, fr1 = cond.loc[TRIAL, ["f_s", "f_r1"]]
F = fault_freqs(fr1)
x10 = load(TRIAL)[20 * FS : 30 * FS]
rng = np.random.default_rng(7)


def db(a, ref):
    return 20 * np.log10(a / ref + 1e-12)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"m_{name}.png", dpi=150)
    plt.close(fig)
    print(name)


# 1. AM fault: direct spectrum (pair f_s +- F) vs carrier envelope (line at F)
xo, _ = inject(x10, "outer_race", 2, fr1, rng)
fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
for col, (sig, lab) in enumerate([(x10, "норма"), (xo, "наружное кольцо, средняя (−58 дБ)")]):
    f, a = E.amp_spectrum(sig, FS)
    a0 = a.max()
    ax[0, col].plot(f, db(a, a0), lw=0.5, color="0.3")
    for x in (abs(fs - F["BPFO"]), fs + F["BPFO"]):
        ax[0, col].axvline(x, color="tab:red", ls=":", lw=1)
    ax[0, col].set(xlim=(0, 300), ylim=(-110, 5), title=f"Прямой спектр, {lab}", xlabel="Гц", ylabel="дБ отн. f_s")
    f, a = E.amp_spectrum(E.low_envelope(sig), E.FS_LOW)
    ax[1, col].plot(f, db(a, 1), lw=0.6, color="tab:blue")
    for x in (F["BPFO"], abs(F["BPFO"] - 2 * fs), 2 * F["BPFO"]):
        ax[1, col].axvline(x, color="tab:red", ls=":", lw=1)
    ax[1, col].set(xlim=(0, 300), ylim=(-110, -20), title=f"Спектр огибающей несущей, {lab}", xlabel="Гц",
                   ylabel="дБ (0 дБ = 100 % модуляции)")
ax[0, 0].text(fs + F["BPFO"], -15, " f_s ± BPFO", color="tab:red")
ax[1, 0].text(F["BPFO"], -30, " BPFO, |BPFO−2f_s|, 2·BPFO", color="tab:red")
save(fig, "am_direct_vs_envelope")

# 2. Impact fault: direct spectrum has nothing; kurtogram picks the PWM band; the band chosen by the
#    defect frequencies shows BPFO
def targeted_band(sig, defect="outer_race"):
    envs, fe = E.stft_band_envelopes(sig)
    _, where = E.best_band_scores([(b, *E.amp_spectrum(E._trim(e) / e.mean() - 1, fe)) for b, e in envs], fs, F)
    return where[defect]


xi, lvl = inject_impacts(x10, "outer_race", 3, fr1, rng)
fig, ax = plt.subplots(2, 4, figsize=(18, 7.5))
for row, (sig, lab) in enumerate([(x10, "норма"), (xi, f"удары ({lvl:+.0f} дБ)")]):
    f, a = E.amp_spectrum(sig, FS)
    ax[row, 0].plot(f, db(a, a.max()), lw=0.3, color="0.3")
    ax[row, 0].set(xlim=(0, 8000), ylim=(-140, 5), title=f"{lab}: прямой спектр", xlabel="Гц", ylabel="дБ")
    grid, band, sk = E.kurtogram(sig)
    img = np.array([np.interp(np.linspace(0, FS / 2, 400), f_, s_) for n, f_, s_ in grid])
    im = ax[row, 1].imshow(img, aspect="auto", origin="lower", extent=(0, FS / 2000, -0.5, len(grid) - 0.5),
                           cmap="magma", vmin=-1, vmax=max(1, img.max()))
    ax[row, 1].set_yticks(range(len(grid)), [str(n) for n, _, _ in grid])
    ax[row, 1].set(xlabel="кГц", ylabel="окно STFT", title=f"{lab}: кюртограмма, max {band[0]:.0f}–{band[1]:.0f} Гц")
    plt.colorbar(im, ax=ax[row, 1], label="SK")
    tb = targeted_band(sig)
    for col, bd, what in [(2, band, "полоса кюртограммы"), (3, tb, "полоса по BPFO")]:
        f, a = E.amp_spectrum(E.band_envelope(sig, bd), E.FS_LOW)
        ax[row, col].plot(f, a, lw=0.6, color="tab:blue")
        for k in (1, 2, 3):
            ax[row, col].axvline(k * F["BPFO"], color="tab:red", ls=":", lw=1)
        ax[row, col].set(xlim=(0, 400), xlabel="Гц", ylabel="амплитуда (отн.)",
                         title=f"{lab}: огибающая, {what}\n{bd[0]:.0f}–{bd[1]:.0f} Гц")
save(fig, "impact_kurtogram_envelope")

# 3. Real data: HF envelope is all electrical (PWM): multiples of f_s, not bearing lines
x60 = load(TRIAL)[: 60 * FS]
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5))
for a_, band, title in [(ax[0], E.HF_FIXED, "широкая полоса 1–10 кГц"), (ax[1], (4000, 4800), "полоса кюртограммы 4,0–4,8 кГц")]:
    f, a = E.amp_spectrum(E.band_envelope(x60, band), E.FS_LOW)
    a_.plot(f, a, lw=0.5, color="0.3")
    for k in range(1, 9):
        a_.axvline(k * fs, color="tab:green", ls="--", lw=0.6)
    for name in ("FTF", "BSF", "BPFO", "BPFI"):
        a_.axvline(F[name], color="tab:red", ls=":", lw=1)
        a_.text(F[name], a.max() * 0.95, " " + name, color="tab:red", fontsize=8)
    a_.set(xlim=(0, 400), xlabel="Гц", ylabel="амплитуда (отн.)",
           title=f"Испытание {TRIAL}, спектр огибающей, {title}. Зелёные — k·f_s, красные — частоты подшипника")
save(fig, "real_hf_envelope")

# 4. Scalogram: low-band current and carrier envelope, normal vs AM inner race vs impacts
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
freqs = np.linspace(5, 300, 160)
for a_, (sig, lab) in zip(ax, [(x10, "норма"), (inject(x10, "inner_race", 3, fr1, rng)[0], "внутр. кольцо, модуляция −50 дБ"),
                                (inject_impacts(x10, "outer_race", 3, fr1, rng)[0], "удары наружного кольца")]):
    if "удары" in lab:
        bd = targeted_band(sig)
        S = W.scalogram(E.band_envelope(sig, bd)[: 4 * E.FS_LOW], E.FS_LOW, freqs)
        lab += f"\nогибающая полосы {bd[0]:.0f}–{bd[1]:.0f} Гц"
    else:
        S = W.scalogram(E.low_envelope(sig)[: 4 * E.FS_LOW], E.FS_LOW, freqs)
        lab += "\nогибающая несущей"
    im = a_.imshow(20 * np.log10(S + 1e-12), aspect="auto", origin="lower", cmap="viridis",
                   extent=(0, 4, freqs[0], freqs[-1]))
    im.set_clim(im.get_clim()[1] - 50, im.get_clim()[1])
    a_.set(title=lab, xlabel="время, с", ylabel="Гц")
save(fig, "scalogram")

# 5. WPT kurtogram on real data and on synthetic impacts
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for a_, (sig, lab) in zip(ax, [(x60[:10 * FS], f"испытание {TRIAL}"), (xi, "синтетические удары")]):
    _, grid, best, k = W.wpt_kurtogram(sig)
    img = np.full((len(grid), 64), np.nan)
    for lev, (bands, kk) in grid.items():
        img[lev - 1] = np.repeat(kk, 64 // len(kk))
    im = a_.imshow(img, aspect="auto", origin="lower", cmap="magma", extent=(0, FS / 2000, 0.5, len(grid) + 0.5))
    plt.colorbar(im, ax=a_, label="эксцесс коэфф.")
    a_.set(xlabel="кГц", ylabel="уровень WPT", title=f"{lab}: max эксцесс {k:.1f}\nв {best[2][0]:.0f}–{best[2][1]:.0f} Гц")
save(fig, "wpt_kurtogram")

# 6. Detection vs severity
res = pd.read_csv(ROOT / "analysis" / "compare_methods.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
for a_, fam, title in [(ax[0], "am", "Модуляция тока (6 дефектов)"), (ax[1], "impact", "Ударные дефекты (3 подшипниковых)")]:
    r = res[res.family == fam].groupby(["method", "severity"]).detected.mean().unstack()
    for m in METHODS:
        if m in r.index:
            a_.plot(r.columns, 100 * r.loc[m], "o-", label=METHOD_RU[m])
    a_.set(title=title, xlabel="степень (1 — слабая, 3 — сильная)", xticks=[1, 2, 3], ylim=(-3, 103))
ax[0].set_ylabel("обнаружено, % (ложные тревоги ≤ ~1 %)")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(FIG_DIR / "m_detection.png", dpi=150)
plt.close(fig)
print("detection")
