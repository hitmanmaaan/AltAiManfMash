"""Diagnose the 45 blind CSV files on the neuromorphic backend and score against the answer key.

Per file: 6 × 10 s segments of phase R -> 8 features -> excess over the commissioning baseline of the
same operating point (other recordings only) -> 48 thermometer spikes -> ternary AltAILIF network ->
per-segment decision. File decision = majority vote of the 6 segments (one decision per minute).
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

import chip
from common import (DATA_DIR, DEFECT_RU, DEFECTS, FS, SEG_S, SEVERITY_RU, SYNTH_DIR, operating_mode,
                    operating_point, segment_features, thermometer)
from train_ternary import decide, severity_from

TYPES = ["normal"] + DEFECTS


def commissioning_baseline(d, base, point, mode, fallback):
    ref = (d["y_type"] == 0) & (d["point"] == point) & (d["base"] != base)
    return np.median(d["feat"][ref], axis=0) if ref.sum() >= 6 else fallback[mode]


def main(backend):
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    d = dict(np.load(SYNTH_DIR / "features.npz"))
    m = np.load(SYNTH_DIR / "model.npz")
    W, TH, med = m["W"], m["TH"], m["medians"]

    bits, owner = [], []
    for i, row in labels.iterrows():
        x = pd.read_csv(DATA_DIR / "csv" / row.file, usecols=["current_R"]).to_numpy()[:, 0]
        for k in range(len(x) // (SEG_S * FS)):
            seg = x[k * SEG_S * FS:(k + 1) * SEG_S * FS]
            feat, fs, fr1 = segment_features(seg)
            base = commissioning_baseline(d, row.base_trial, operating_point(fs, fr1), operating_mode(fs, fr1), med)
            bits.append(thermometer((feat - base)[None])[0])
            owner.append(i)
        print(row.file, flush=True)
    bits, owner = np.array(bits), np.array(owner)

    raw, _, log = chip.run(bits, SYNTH_DIR / "weights.csv", backend=backend, tag=f"blind_{backend}")
    fires = chip.align(raw, len(bits), W.shape[0], lag=chip.BACKENDS[backend]["lag"])
    ref = (bits @ W.T >= TH).astype(int)
    print(f"backend {backend}: {len(bits)} segments, bit-exact vs numpy: {(fires == ref).all(1).sum()}/{len(bits)}")

    seg_type, seg_sev = decide(fires), severity_from(fires)
    out = []
    for i, row in labels.iterrows():
        st = seg_type[owner == i]
        votes = Counter(t for t in st if t >= 0)
        t = votes.most_common(1)[0][0] if votes else 0
        sv = seg_sev[owner == i][st == t]
        s = int(np.round(np.median(sv))) if t > 0 and len(sv) else 0
        out.append(dict(file=row.file, truth=row.defect_ru, truth_sev=SEVERITY_RU[row.severity],
                        pred=DEFECT_RU[TYPES[t]], pred_sev=SEVERITY_RU[s],
                        type_ok=TYPES[t] == row.defect, sev_ok=(s == row.severity),
                        segments=" ".join(TYPES[x][:4] if x >= 0 else "??" for x in st)))
    res = pd.DataFrame(out)
    res.to_csv(DATA_DIR / f"blind_results_{backend}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(res.to_string(index=False))
    print(f"\nтип верно: {res.type_ok.sum()}/45 ({res.type_ok.mean():.0%});  "
          f"степень верно (где тип верен): {res[res.type_ok].sev_ok.mean():.0%}")
    print(pd.crosstab(res.truth, res.pred))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu-v2")
