"""Diagnose one CSV file on the neuromorphic backend: python diagnose_file.py syn_07.csv [cpu-v2|altai-v1]"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

import chip
from common import (DATA_DIR, DEFECT_RU, DEFECTS, FEATURES, FS, SEG_S, SEVERITY_RU, SYNTH_DIR, operating_mode,
                    operating_point, segment_features, thermometer)
from infer_blind import commissioning_baseline
from train_ternary import decide, severity_from

TYPES = ["normal"] + DEFECTS


def main(name, backend):
    labels = pd.read_csv(DATA_DIR / "labels.csv").set_index("file")
    row = labels.loc[name]
    d = dict(np.load(SYNTH_DIR / "features.npz"))
    m = np.load(SYNTH_DIR / "model.npz")
    x = pd.read_csv(DATA_DIR / "csv" / name, usecols=["current_R"]).to_numpy()[:, 0]
    bits, excess = [], []
    for k in range(len(x) // (SEG_S * FS)):
        feat, fs, fr1 = segment_features(x[k * SEG_S * FS:(k + 1) * SEG_S * FS])
        e = feat - commissioning_baseline(d, row.base_trial, operating_point(fs, fr1), operating_mode(fs, fr1),
                                          m["medians"])
        excess.append(e)
        bits.append(thermometer(e[None])[0])
    bits = np.array(bits)
    raw, _, log = chip.run(bits, SYNTH_DIR / "weights.csv", backend=backend, tag="one")
    fires = chip.align(raw, len(bits), m["W"].shape[0], lag=chip.BACKENDS[backend]["lag"])
    st, ss = decide(fires), severity_from(fires)

    print(f"Файл {name}: бэкенд {backend}; входных спайков {int(bits.sum())}, выходных {len(raw)}")
    print("Превышение признаков над «нормой», дБ (по 10-секундным кускам):")
    print("        " + " ".join(f"{f:>7s}" for f in FEATURES))
    for k, e in enumerate(excess):
        print(f"  {10*k:2d}-{10*k+10:2d}с " + " ".join(f"{v:7.1f}" for v in e)
              + f"   -> {DEFECT_RU[TYPES[st[k]]] if st[k] >= 0 else 'неоднозначно'}")
    votes = Counter(t for t in st if t >= 0)
    t = votes.most_common(1)[0][0] if votes else 0
    sv = ss[st == t]
    s = int(np.round(np.median(sv))) if t > 0 and len(sv) else 0
    print(f"ДИАГНОЗ «Алтая»: {DEFECT_RU[TYPES[t]]}, степень: {SEVERITY_RU[s]}")
    print(f"ПРАВИЛЬНЫЙ ОТВЕТ: {row.defect_ru}, степень: {row.severity_ru}   -> "
          + ("ВЕРНО" if TYPES[t] == row.defect else "ОШИБКА"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cpu-v2")
