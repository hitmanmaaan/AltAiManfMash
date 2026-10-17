"""Evidence figure: spectra overlay within exact-condition groups.

Top: medium/nominal group, 0-400 Hz — trials are practically indistinguishable.
Bottom: heavy/nominal group, zoom f_s ± 8 Hz — the f_s ± ~4.67 Hz sideband pair is present in
every trial; trials 6 and 45 only differ by a small frequency shift of that pair.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcsa import FIG_DIR, load, spectrum

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
for t in [4, 10, 16, 22, 28, 34]:
    f, amp, db = spectrum(load(t), 60)
    m = f <= 400
    ax.plot(f[m], db[m], lw=0.5, label=f"№{t}", alpha=0.7)
ax.set_xlim(0, 400)
ax.set_ylim(-110, 5)
ax.set_title("Средняя нагрузка, f_s = 48,48 Гц: 6 испытаний наложены — спектры практически совпадают")
ax.set_ylabel("дБ отн. основной")
ax.legend(ncol=6, fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for t, style in [(6, dict(color="tab:red", lw=1.2)), (45, dict(color="tab:orange", lw=1.2)),
                 (12, dict(color="0.5", lw=0.6)), (18, dict(color="0.6", lw=0.6)),
                 (24, dict(color="0.7", lw=0.6)), (30, dict(color="0.8", lw=0.6))]:
    f, amp, db = spectrum(load(t), 60)
    m = (f >= 40.5) & (f <= 56.5)
    ax.plot(f[m], db[m], label=f"№{t}", **style)
ax.axvline(48.48 - 4.7, color="tab:red", ls=":", lw=1)
ax.axvline(48.48 + 4.7, color="tab:red", ls=":", lw=1)
ax.set_xlim(40.5, 56.5)
ax.set_ylim(-100, 5)
ax.set_title("Тяжёлая нагрузка, f_s = 48,48 Гц: боковые f_s ± ~4,67 Гц есть у всех; у №6/№45 они лишь сдвинуты (4,72 / 4,60 Гц)")
ax.set_xlabel("Частота, Гц")
ax.set_ylabel("дБ отн. основной")
ax.legend(ncol=6, fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
out = FIG_DIR / "within_group_evidence.png"
fig.savefig(out, dpi=130)
print(out)
