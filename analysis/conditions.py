"""Operating condition of each trial: load level (from slip) x supply variant (nominal 48.48 Hz or raised)."""
import pandas as pd

from mcsa import ROOT


def conditions() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "analysis" / "speeds.csv")
    load = pd.cut(s.slip, [0, 0.0027, 0.0055, 1], labels=["light", "medium", "heavy"]).astype(str)
    supply = (s.f_s > 48.50).map({True: "raised", False: "nominal"})
    s["condition"] = load + "/" + supply
    return s.set_index("trial")
