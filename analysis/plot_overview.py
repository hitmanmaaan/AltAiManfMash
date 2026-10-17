"""Annotated overview spectrum of one trial (phase R)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcsa import FIG_DIR, load, peak_near, spectrum

trial = int(sys.argv[1]) if len(sys.argv) > 1 else 1
f, amp, db = spectrum(load(trial), 60)
fs, _ = peak_near(f, amp, 48.5, 1.0)

FR1, FR2 = 24.22, 43.07
marks = [
    (fs, "f_s (сеть)", "k"),
    (fs - FR1, "f_s−f_r1", "tab:blue"), (fs + FR1, "f_s+f_r1", "tab:blue"),
    (fs - FR2, "f_s−f_r2", "tab:green"), (fs + FR2, "f_s+f_r2", "tab:green"),
    (fs + 86.13, "f_s+3.56·f_r1\n(≈BPFO вала ЭД)", "tab:red"),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for ax, (lo, hi) in zip(axes, [(0, 700), (0, 150)]):
    m = (f >= lo) & (f <= hi)
    ax.plot(f[m], db[m], lw=0.6, color="0.3")
    for x, label, c in marks:
        if lo <= x <= hi:
            ax.axvline(x, color=c, ls=":", lw=1)
            ax.text(x, -8, label, color=c, fontsize=8, rotation=90, va="top", ha="right")
    for h in range(2, 15):
        if lo <= h * fs <= hi:
            ax.axvline(h * fs, color="0.7", ls="--", lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-110, 5)
    ax.set_ylabel("дБ отн. основной")
    ax.grid(alpha=0.3)
axes[0].set_title(f"Испытание {trial}, фаза R: спектр тока статора (Δf = {f[1]:.4f} Гц), f_s = {fs:.3f} Гц")
axes[1].set_xlabel("Частота, Гц (серые пунктиры — гармоники сети)")
fig.tight_layout()
out = FIG_DIR / f"overview_trial{trial}.png"
fig.savefig(out, dpi=130)
print(out)
