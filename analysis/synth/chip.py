"""Run the ternary classifier on the neuromorphic backend inside Docker and read the spikes back.

Backends:
  "altai-v1": AltAI golden-model backend from KNP v1.0.0 (container knp-lab, AltAI device selected)
  "cpu-v2":   AltAILIF neurons on the KNP v2 CPU backend (container knp-run) — functional emulation
Exchange directory: knp-lab/chip_io (mounted as /work/chip_io in both containers).
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np

from common import ROOT

IO = ROOT / "knp-lab" / "chip_io"
DOCKER = shutil.which("docker") or "/usr/local/bin/docker"
# lag: output spike step minus input step (AltAI golden model answers within the same step).
BACKENDS = {
    "altai-v1": dict(container="knp-lab", bindir="/work/repo-v1/build/bin",
                     backend="/opt/altai-v1/usr/lib/knp-altai-backend", extra="--select-device", lag=0),
    "cpu-v2": dict(container="knp-run", bindir="/work/repo/build-gcc14/bin",
                   backend="./knp-cpu-single-threaded-backend", extra="", lag=1),
}


def run(bits: np.ndarray, weights_csv: Path, backend: str = "cpu-v2", tag: str = "run"):
    """bits: (n_samples, n_inputs) 0/1. Returns (fires (n_samples, n_outputs), log text)."""
    IO.mkdir(parents=True, exist_ok=True)
    inp, out, w = IO / f"{tag}_in.csv", IO / f"{tag}_out.csv", IO / f"{tag}_weights.csv"
    w.write_text(Path(weights_csv).read_text())
    W = np.loadtxt(weights_csv, delimiter=",", skiprows=1, dtype=int, ndmin=2)[:, 1:]
    connected = W.any(axis=0)  # AltAI creates input axons only for inputs that have synapses
    with open(inp, "w") as fh:
        for row in bits:
            fh.write(",".join(str(i) for i in np.nonzero(row & connected)[0]) + "\n")
    if out.exists():
        out.unlink()
    b = BACKENDS[backend]
    cmd = (f'cd {b["bindir"]} && ./altai-fault-classifier {b["backend"]} /work/chip_io/{w.name} '
           f'/work/chip_io/{inp.name} /work/chip_io/{out.name} {b["extra"]} 2>&1')
    res = subprocess.run([DOCKER, "exec", b["container"], "bash", "-c", cmd], capture_output=True, text=True)
    log = res.stdout + res.stderr
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f"backend {backend} failed (exit {res.returncode}):\n{log}")
    n_out = int(Path(weights_csv).read_text().splitlines()[0].split(",")[1])
    fires = np.zeros((len(bits), n_out), dtype=int)
    raw = np.loadtxt(out, delimiter=",", dtype=int, ndmin=2) if out.stat().st_size else np.zeros((0, 2), int)
    return raw, fires, log


def align(raw: np.ndarray, n_samples: int, n_out: int, lag: int):
    """Output spike at step (k + lag) belongs to sample k."""
    fires = np.zeros((n_samples, n_out), dtype=int)
    for step, neuron in raw:
        k = step - lag
        if 0 <= k < n_samples:
            fires[k, neuron] = 1
    return fires
