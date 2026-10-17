"""Run the trained ternary network on the neuromorphic backend for the whole test split and compare
every output spike with the numpy reference (sum w x >= theta)."""
import sys

import numpy as np

import chip
from common import DEFECTS, SYNTH_DIR, thermometer
from train_ternary import baselines, decide, load, mode_medians, severity_from


def test_inputs():
    d, tr, va, te = load()
    med = mode_medians(d, tr | va)
    ref = baselines(d, np.arange(1, 46), med)
    X = thermometer(d["feat"] - ref)
    return d, te, X


def main(backend):
    d, te, X = test_inputs()
    m = np.load(SYNTH_DIR / "model.npz")
    W, TH = m["W"], m["TH"]
    Xt = X[te]
    ref_fires = (Xt @ W.T >= TH).astype(int)

    raw, _, log = chip.run(Xt, SYNTH_DIR / "weights.csv", backend=backend, tag=f"test_{backend}")
    print("\n".join(l for l in log.splitlines() if "info" not in l.lower() or "Device" in l)[:1500])
    best = max(range(0, 3), key=lambda lag: (chip.align(raw, len(Xt), W.shape[0], lag) == ref_fires).all(1).mean())
    fires = chip.align(raw, len(Xt), W.shape[0], best)
    mism = (fires != ref_fires).any(1).sum()
    print(f"\nbackend={backend}: samples {len(Xt)}, output spikes {len(raw)}, lag {best} step(s)")
    print(f"bit-exact vs numpy reference: {len(Xt) - mism}/{len(Xt)} samples identical, mismatches {mism}")

    yt, ys = d["y_type"][te], d["y_sev"][te]
    pt, ps = decide(fires), severity_from(fires)
    print(f"accuracy from chip spikes: type {(pt == yt).mean():.3f}; "
          f"severity where type correct {(ps[(yt > 0) & (pt == yt)] == ys[(yt > 0) & (pt == yt)]).mean():.3f}")
    np.save(SYNTH_DIR / f"chip_fires_test_{backend}.npy", fires)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu-v2")
