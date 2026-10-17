"""Blind test set in the teacher's format: 45 CSV files (current_R, current_S, current_T, 60 s at 25.6 kHz)
built from base recordings 31-45 (never used for training) with injected symptoms, plus the answer key.

9 normal + 6 defects x 6 files (each severity twice). Symptoms are applied to all three phases.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DATA_DIR, DEFECT_RU, DEFECTS, FS, SEVERITY_RU, inject, operating_point,
                    segment_features)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcsa import NPY_DIR  # noqa: E402

WIN = 60 * FS


def main():
    rng = np.random.default_rng(2026)
    out_dir = DATA_DIR / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = [("normal", 0)] * 9 + [(d, s) for d in DEFECTS for s in (1, 2, 3) for _ in range(2)]
    rng.shuffle(plan)

    windows = []  # (base, offset_s) — 3 per base; 120 s bases give offsets 0/30/60, 60 s bases reuse 0
    for base in range(31, 46):
        n = np.load(NPY_DIR / f"current_{base}.npy", mmap_mode="r").shape[0]
        offs = [0, 30, 60] if n >= 120 * FS else [0, 0, 0]
        windows += [(base, o) for o in offs]

    rows = []
    for i, ((defect, sev), (base, off)) in enumerate(zip(plan, windows), start=1):
        x = np.array(np.load(NPY_DIR / f"current_{base}.npy", mmap_mode="r")[off * FS: off * FS + WIN],
                     dtype=np.float64)
        _, fs, fr1 = segment_features(x[: 10 * FS, 0])
        y, level = inject(x, defect, sev, fr1, rng)
        name = f"syn_{i:02d}.csv"
        pd.DataFrame(y, columns=["current_R", "current_S", "current_T"]).to_csv(
            out_dir / name, index=False, float_format="%.6f", na_rep="")
        rows.append(dict(file=name, base_trial=base, offset_s=off, point=operating_point(fs, fr1),
                         defect=defect, defect_ru=DEFECT_RU[defect], severity=sev,
                         severity_ru=SEVERITY_RU[sev], level_dB=round(level, 1) if sev else "",
                         f_s=round(fs, 3), f_r1=round(fr1, 3)))
        print(name, base, off, defect, sev, flush=True)

    pd.DataFrame(rows).to_csv(DATA_DIR / "labels.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
