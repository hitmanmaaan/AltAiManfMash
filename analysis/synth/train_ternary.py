"""Train a one-layer ternary spiking classifier that satisfies AltAI constraints.

Inputs: thermometer-coded binary spikes (48 = 8 features × 6 levels).
Outputs (9 AltAILIF neurons, fire iff sum_i w_i x_i >= theta, w in {-1,0,1}, theta integer >= 1):
  0..5 -> defect type (outer, inner, ball, cage, unbalance, misalignment); none firing -> "normal"
  6..8 -> severity levels 1..3 (OR over the features' level bits, weight +1, theta 1).
Split by base recording: 1-24 train, 25-30 validation (thresholds), 31-45 test.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from common import DEFECTS, FEATURES, SYNTH_DIR, THERMO_DB, thermometer

N_MODES = 6
LEVELS = len(THERMO_DB)


def load():
    d = dict(np.load(SYNTH_DIR / "features.npz"))
    train = d["base"] <= 24
    val = (d["base"] >= 25) & (d["base"] <= 30)
    test = d["base"] >= 31
    return d, train, val, test


def mode_medians(d, sel):
    med = np.zeros((N_MODES, len(FEATURES)))
    for m in range(N_MODES):
        s = sel & (d["y_type"] == 0) & (d["mode"] == m)
        med[m] = np.median(d["feat"][s], axis=0)
    return med


def baselines(d, pool_bases, fallback):
    """Commissioning baseline for every sample: median of *normal* features of the other recordings
    taken at the same operating point (load + drive frequency). The recording itself is excluded so
    its own noise is never subtracted from itself. Falls back to the coarse mode median."""
    base_of = np.zeros((len(d["base"]), len(FEATURES)))
    normal = d["y_type"] == 0
    for b in np.unique(d["base"]):
        for p in np.unique(d["point"][d["base"] == b]):
            ref = normal & (d["point"] == p) & (d["base"] != b) & np.isin(d["base"], pool_bases)
            sel = (d["base"] == b) & (d["point"] == p)
            if ref.sum() >= 6:
                base_of[sel] = np.median(d["feat"][ref], axis=0)
            else:
                base_of[sel] = fallback[d["mode"][sel]]
    return base_of


def decide(fires):
    """Host-side reading of the type neurons: exactly one -> that defect, none -> normal (0), several -> -1."""
    n = fires[:, :6].sum(1)
    out = np.where(n == 0, 0, -1)
    single = n == 1
    out[single] = fires[single, :6].argmax(1) + 1
    return out


def severity_from(fires):
    sev = np.zeros(len(fires), int)
    for k in (1, 2, 3):
        sev[fires[:, 5 + k] == 1] = k
    return sev


def f1(y, p):
    tp = (y & p).sum()
    return 2 * tp / max(y.sum() + p.sum(), 1)


def train_type_neuron(Xtr, ytr, Xva, yva):
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5).fit(Xtr, ytr)
    w = lr.coef_[0]
    order = np.argsort(-np.abs(w))
    best = (-1, None, None)
    for k in range(1, len(w) + 1):  # keep the k strongest weights as ±1
        wt = np.zeros(len(w), int)
        wt[order[:k]] = np.sign(w[order[:k]]).astype(int)
        s_va = Xva @ wt
        for theta in range(1, int(np.abs(wt).sum()) + 1):
            score = f1(yva, s_va >= theta)
            if score > best[0]:
                best = (score, wt.copy(), theta)
    return best


def main():
    d, tr, va, te = load()
    med = mode_medians(d, tr | va)
    ref_train = baselines(d, np.arange(1, 31), med)  # train/val samples: baseline from train/val only
    ref_all = baselines(d, np.arange(1, 46), med)  # test: commissioning data of any other recording
    ref = np.where(te[:, None], ref_all, ref_train)
    excess = d["feat"] - ref
    X = thermometer(excess)
    yt, ys = d["y_type"], d["y_sev"]

    W, TH = [], []
    for j in range(6):
        score, wt, theta = train_type_neuron(X[tr], yt[tr] == j + 1, X[va], yt[va] == j + 1)
        W.append(wt)
        TH.append(theta)
        used = [f"{FEATURES[i // LEVELS]}>{THERMO_DB[i % LEVELS]}дБ:{wt[i]:+d}" for i in np.nonzero(wt)[0]]
        print(f"{DEFECTS[j]:13s} F1(val)={score:.3f} theta={theta}  {', '.join(used)}")

    # Severity neurons: OR of level bit L_k across all features. Pick (L1<L2<L3) on train+val.
    tv = (tr | va) & (yt > 0)
    best = (-1, None)
    for L1 in range(LEVELS):
        for L2 in range(L1 + 1, LEVELS):
            for L3 in range(L2 + 1, LEVELS):
                sev = np.zeros(tv.sum(), int)
                for k, L in ((1, L1), (2, L2), (3, L3)):
                    sev[X[tv][:, L::LEVELS].any(1)] = k
                acc = (sev == ys[tv]).mean()
                if acc > best[0]:
                    best = (acc, (L1, L2, L3))
    sev_levels = best[1]
    print(f"severity levels: >{[THERMO_DB[L] for L in sev_levels]} dB, train+val acc {best[0]:.3f}")
    for L in sev_levels:
        w = np.zeros(X.shape[1], int)
        w[L::LEVELS] = 1
        W.append(w)
        TH.append(1)

    W, TH = np.array(W), np.array(TH)
    fires = (X @ W.T >= TH).astype(int)
    pred_type = decide(fires)
    pred_sev = severity_from(fires)

    metrics = {}
    for name, sel in (("validation", va), ("test", te)):
        acc = (pred_type[sel] == yt[sel]).mean()
        amb = (pred_type[sel] == -1).mean()
        dsel = sel & (yt > 0) & (pred_type == yt)
        sev_acc = (pred_sev[dsel] == ys[dsel]).mean()
        metrics[name] = dict(type_accuracy=acc, ambiguous=amb, severity_accuracy_on_correct=sev_acc)
        print(f"{name}: тип {acc:.3f}, неоднозначно {amb:.3f}, степень (где тип верен) {sev_acc:.3f}")

    # Per-type, per-severity detection on test.
    print("\ntest: доля верно определённого типа по степени")
    per = {}
    for j in range(7):
        row = []
        for s in ((0,) if j == 0 else (1, 2, 3)):
            sel = te & (yt == j) & (ys == s)
            row.append(f"s{s}:{(pred_type[sel] == j).mean():.2f}")
            per[f"{j}_{s}"] = float((pred_type[sel] == j).mean())
        print(f"  {(['normal'] + DEFECTS)[j]:13s} " + "  ".join(row))
    metrics["per_class_test"] = per

    labels = [-1, 0, 1, 2, 3, 4, 5, 6]
    cm = confusion_matrix(yt[te], pred_type[te], labels=labels)
    np.savetxt(SYNTH_DIR / "confusion_test.csv", cm, fmt="%d", delimiter=",")

    # Float baselines (no chip constraints): continuous features and same binary inputs.
    for tag, Z in (("float, непрерывные признаки", excess), ("float, те же бинарные входы", X)):
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(Z[tr | va], yt[tr | va])
        acc = (lr.predict(Z[te]) == yt[te]).mean()
        metrics[f"baseline {tag}"] = acc
        print(f"база без ограничений чипа ({tag}): тип {acc:.3f}")

    with open(SYNTH_DIR / "weights.csv", "w") as fh:
        fh.write(f"{W.shape[1]},{W.shape[0]}\n")
        for w, t in zip(W, TH):
            fh.write(",".join([str(int(t))] + [str(int(v)) for v in w]) + "\n")
    np.savez(SYNTH_DIR / "model.npz", medians=med, W=W, TH=TH, sev_levels=np.array(sev_levels))
    with open(SYNTH_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2, default=float)
    print(f"\nвеса: {W.shape[0]} нейронов × {W.shape[1]} входов, ненулевых синапсов {int(np.abs(W).sum())}")


if __name__ == "__main__":
    main()
