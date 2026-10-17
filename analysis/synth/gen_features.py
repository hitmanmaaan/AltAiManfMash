"""Generate labelled feature sets: every 10 s segment of every base recording × {normal, 6 defects × 3 severities}.

Base recordings 1-30 -> training/validation, 31-45 -> test (split by recording: no leakage).
Only phase R is used (present in all trials). Output: synth/features.npz
"""
import sys
from pathlib import Path

import numpy as np

from common import DEFECTS, FS, SEG_S, SYNTH_DIR, inject, operating_mode, operating_point, segment_features

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcsa import load  # noqa: E402

VARIANTS = [("normal", 0)] + [(d, s) for d in DEFECTS for s in (1, 2, 3)]


def main():
    rng = np.random.default_rng(42)
    rows = dict(feat=[], y_type=[], y_sev=[], level=[], base=[], seg=[], mode=[], point=[])
    seg_n = SEG_S * FS
    for base in range(1, 46):
        x = load(base)
        for k in range(len(x) // seg_n):
            seg = x[k * seg_n:(k + 1) * seg_n]
            _, fs, fr1 = segment_features(seg)  # speed of the clean segment drives the fault frequencies
            mode = operating_mode(fs, fr1)
            point = operating_point(fs, fr1)
            for d, s in VARIANTS:
                y, level = inject(seg, d, s, fr1, rng)
                feat, _, _ = segment_features(y)
                rows["feat"].append(feat)
                rows["y_type"].append(0 if d == "normal" else DEFECTS.index(d) + 1)
                rows["y_sev"].append(s)
                rows["level"].append(level)
                rows["base"].append(base)
                rows["seg"].append(k)
                rows["mode"].append(mode)
                rows["point"].append(point)
        print(f"base {base}: {len(x) // seg_n} segments", flush=True)
    np.savez_compressed(SYNTH_DIR / "features.npz", **{k: np.array(v) for k, v in rows.items()})
    print("samples:", len(rows["feat"]))


if __name__ == "__main__":
    main()
