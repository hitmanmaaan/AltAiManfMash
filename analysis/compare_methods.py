"""Verification: direct spectrum vs envelope vs wavelet methods on synthetic faults with a known answer.

Background = real recordings (all 45 trials, phase R, plus all phases for the Park variant).
- Threshold: every method/defect score on 90 clean 10 s segments (offsets 0 and 20 s); threshold =
  their maximum (false-alarm rate <~ 1 %).
- Test: a third segment (offset 40 s) of every trial, clean or with one injected fault:
  * "am"     — torque modulation of the current (synth/common.inject), 6 faults x 3 severities;
  * "impact" — impulsive bursts with slip jitter (synth/common.inject_impacts), 3 bearing faults x 3.
- Detected = score of the true fault's detector above its threshold. Type correct = among detectors
  above threshold, the true one has the largest margin.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "synth"))
import envelope  # noqa: E402
import wavelet  # noqa: E402
from common import DEFECTS, IMPACT_DEFECTS, inject, inject_impacts  # noqa: E402
from conditions import conditions  # noqa: E402
from mcsa import FS, ROOT  # noqa: E402

SEG = 10 * FS
# Чувствительность: порог обнаружения = max признака на чистых кусках − SENS_DB.
# SENS_DB > 0 — порог ниже, ловим более слабые дефекты, но растут ложные тревоги; < 0 — строже.
# Можно задать из командной строки: compare_methods.py --sens 2
SENS_DB = 0.0
CLEAN_OFFSETS = (0, 20)
TEST_OFFSET = 40
METHODS = ["direct", "env_low", "env_park", "env_hf", "env_kurt", "env_sel", "wpt_kurt", "wpt_sel", "wden_hf"]
METHOD_RU = {
    "direct": "Прямой спектр (f_s ± F)",
    "env_low": "Огибающая несущей, 1 фаза",
    "env_park": "Огибающая вектора Парка, 3 фазы",
    "env_hf": "Огибающая ВЧ 1–10 кГц",
    "env_kurt": "Огибающая ВЧ + кюртограмма",
    "env_sel": "Огибающая ВЧ + выбор полосы по частотам дефекта",
    "wpt_kurt": "Вейвлет-пакеты + кюртограмма",
    "wpt_sel": "Вейвлет-пакеты + выбор узла по частотам дефекта",
    "wden_hf": "Вейвлет-шумоподавление + огибающая ВЧ",
}


def all_scores(X, fs, fr1):
    s = envelope.scores(X[:, 0], fs, fr1, X3=X)
    w = wavelet.scores(X[:, 0], fs, fr1)
    out = {}
    for m in METHODS:
        src = s if m in s else w if m in w else None
        for d in DEFECTS:
            out[f"{m}:{d}"] = np.nan if src is None else src[m][d]
    return out


def run_trial(t):
    cond = conditions()
    fs, fr1 = cond.loc[t, ["f_s", "f_r1"]]
    raw = np.load(ROOT / "data" / "npy" / f"current_{t}.npy", mmap_mode="r")
    seg = lambda off: np.asarray(raw[off * FS : off * FS + SEG], np.float64)  # noqa: E731
    rng = np.random.default_rng(1000 + t)
    rows = []
    for off in CLEAN_OFFSETS:
        rows.append(dict(trial=t, set="clean", family="normal", defect="normal", severity=0, level_dB=np.nan,
                         **all_scores(seg(off), fs, fr1)))
    base = seg(TEST_OFFSET)
    cases = [("normal", "normal", 0)]
    cases += [("am", d, s) for d in DEFECTS for s in (1, 2, 3)]
    cases += [("impact", d, s) for d in IMPACT_DEFECTS for s in (1, 2, 3)]
    for fam, d, s in cases:
        if fam == "normal":
            X, lvl = base, np.nan
        elif fam == "am":
            X, lvl = inject(base, d, s, fr1, rng)
        else:
            X, lvl = inject_impacts(base, d, s, fr1, rng)
        rows.append(dict(trial=t, set="test", family=fam, defect=d, severity=s, level_dB=lvl,
                         **all_scores(X, fs, fr1)))
    print(t, flush=True)
    return rows


def evaluate(df, sens_db=SENS_DB):
    clean = df[df.set == "clean"]
    test = df[df.set == "test"]
    thr = {c: clean[c].max() - sens_db for c in df.columns if ":" in c}
    rows = []
    for m in METHODS:
        cols = [f"{m}:{d}" for d in DEFECTS]
        if test[cols[0]].isna().all():
            continue
        margin = pd.DataFrame({d: test[f"{m}:{d}"] - thr[f"{m}:{d}"] for d in DEFECTS})
        normal = test.family == "normal"
        ok = margin[normal].notna().all(axis=1)
        fa = (margin[normal][ok] > 0).any(axis=1).mean()
        rows.append(dict(method=m, family="normal", defect="normal", severity=0,
                         n=int(ok.sum()), detected=fa, type_ok=np.nan))
        for (fam, d, s), g in test[~normal].groupby(["family", "defect", "severity"]):
            mg = margin.loc[g.index].dropna()
            det = mg[d] > 0
            best = mg.idxmax(axis=1)
            rows.append(dict(method=m, family=fam, defect=d, severity=s, n=len(mg),
                             detected=det.mean(), type_ok=((best == d) & det).mean()))
    return pd.DataFrame(rows), thr


if __name__ == "__main__":
    out_csv = ROOT / "analysis" / "compare_scores.csv"
    if out_csv.exists() and "--rerun" not in sys.argv:
        df = pd.read_csv(out_csv)
    else:
        with ProcessPoolExecutor(max_workers=6) as ex:
            df = pd.DataFrame([r for rows in ex.map(run_trial, range(1, 46)) for r in rows])
        df.to_csv(out_csv, index=False)
    sens = float(sys.argv[sys.argv.index("--sens") + 1]) if "--sens" in sys.argv else SENS_DB
    print(f"чувствительность: порог = max(чистые) − {sens:g} дБ")
    res, thr = evaluate(df, sens)
    res.to_csv(ROOT / "analysis" / "compare_methods.csv", index=False)
    pd.set_option("display.width", 250)
    for what in ("detected", "type_ok"):
        print(f"\n=== {what} (share of segments) ===")
        p = res.pivot_table(index=["family", "defect", "severity"], columns="method", values=what)
        print((100 * p[[m for m in METHODS if m in p]]).round(0).to_string())
    print("\n=== mean over faults, by family and severity: detected % ===")
    p = res[res.family != "normal"].pivot_table(index=["family", "severity"], columns="method", values="detected")
    print((100 * p[[m for m in METHODS if m in p]]).round(0).to_string())
