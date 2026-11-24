"""Noise profile of one benchmark from its stored versions (step 05, equations 1-3 and 8).

  c_w  within-run noise of one block, from differences between consecutive blocks   (eq. 1)
  c_t  between-run noise, from differences between run means of the same version   (eq. 2)
  n_b  blocks per confirmation run that make delta_min confirmable in K_max runs     (eq. 8)
All are relative to the version's level (0.01 = 1 %).
"""

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

MAD = 1.4826


@dataclass
class Noise:
    c_w: float
    c_t: float
    n_b: int                  # blocks per confirmation run (n_max if infeasible)
    feasible: bool            # delta_min confirmable within k_max runs of at most n_max blocks
    delta_feasible: float     # the smallest slowdown it CAN confirm with k_max runs of n_max blocks
    versions: int             # how many stored versions the estimate uses
    estimated: bool           # False = not enough history yet, the defaults from the settings were used

    def to_dict(self):
        return asdict(self)


def plan(c_w, c_t, settings, versions=0, estimated=True):
    """Equation (8): blocks per run, feasibility and the smallest confirmable slowdown."""
    a, b, k, d = settings["alpha"], settings["beta"], settings["k_max"], settings["delta_min"]
    n_min, n_max = 1, settings["n_max"]
    z = NormalDist().inv_cdf(1 - a) + NormalDist().inv_cdf(1 - b)
    room = k * d ** 2 / z ** 2 - 2 * c_t ** 2
    n_needed = math.ceil(2 * c_w ** 2 / room) if room > 0 else math.inf
    feasible = n_needed <= n_max
    n_b = int(min(max(n_needed, n_min), n_max)) if feasible else n_max
    delta_feasible = z * math.sqrt(2 * (c_t ** 2 + c_w ** 2 / n_max) / k)
    return Noise(c_w, c_t, n_b, bool(feasible), delta_feasible, versions, estimated)


def estimate(versions, settings):
    """versions: list of arrays (runs, blocks), one per stored version. Needs at least
    settings['noise_min_versions'] versions with two or more runs; otherwise uses the defaults."""
    usable = [v for v in versions if v.shape[0] >= 2 and v.shape[1] >= 2]
    if len(usable) < settings["noise_min_versions"]:
        return plan(settings["default_c_w"], settings["default_c_t"], settings, len(usable), False)
    steps, gaps, blocks_per_run = [], [], []
    for v in usable:
        rel = v / np.median(v) - 1                         # relative to the version's level
        steps.append(np.abs(np.diff(rel, axis=1)).ravel())
        means = rel.mean(axis=1)
        i, j = np.triu_indices(len(means), k=1)            # each pair of runs once
        gaps.append(np.abs(means[i] - means[j]))
        blocks_per_run.append(v.shape[1])
    c_w = MAD / math.sqrt(2) * float(np.median(np.concatenate(steps)))
    total = (MAD / math.sqrt(2) * float(np.median(np.concatenate(gaps)))) ** 2
    c_t = math.sqrt(max(total - c_w ** 2 / float(np.median(blocks_per_run)), 0.0))
    return plan(c_w, c_t, settings, len(usable), True)


def load_profile(path, settings):
    """Reads step 05's <project>_noise_profile.csv (index = benchmark, c_w_pct, c_t_pct, ...)
    into {benchmark: Noise}. n_b and feasible are taken from the file as step 05 computed them
    (the % columns are rounded, so recomputing could differ at the edge); missing -> equation (8)."""
    import csv
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("bench_id") or row.get("bench_key") or row[next(iter(row))]
            c_w, c_t = float(row["c_w_pct"]) / 100, float(row["c_t_pct"]) / 100
            n = plan(c_w, c_t, settings, int(float(row.get("versions") or 0)), True)
            if row.get("feasible") not in (None, ""):
                n.feasible = row["feasible"].strip().lower() == "true"
                n.n_b = int(float(row["n_b"])) if n.feasible and row.get("n_b") else settings["n_max"]
            out[name] = n
    return out
