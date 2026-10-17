"""Export trials from data/npy to MATLAB files data/mat/current_N.mat (variables R, S, T as float32, fs).

Run after survey.py. The MATLAB demo (matlab_demo/) reads only these files.
"""
import numpy as np
from scipy.io import savemat

from mcsa import FS, NPY_DIR, ROOT

MAT_DIR = ROOT / "data" / "mat"
MAT_DIR.mkdir(parents=True, exist_ok=True)

for t in range(1, 46):
    x = np.load(NPY_DIR / f"current_{t}.npy")
    savemat(MAT_DIR / f"current_{t}.mat", dict(R=x[:, 0], S=x[:, 1], T=x[:, 2], fs=float(FS)))
    print(t, flush=True)
