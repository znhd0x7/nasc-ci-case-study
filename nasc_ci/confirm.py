"""NASC, the sequential confirmation (III-E; nasc_sim.nasc_confirm with terminal="level" and the cap).

A batch is one fresh run of the candidate and one fresh run of the reference (the parent version),
n_b blocks each, run back to back so both see the same machine state. After every batch the
log-likelihood ratio is compared with Wald's boundaries: alert at A, dismiss at B. If neither is
reached after k_max batches, the level rule decides (one-sided test at level alpha on all batches).
"""

from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import numpy as np


@dataclass
class Confirmation:
    confirmed: bool
    batches: int                 # K: confirmation batches used (each = 1 run per side)
    extra_runs: int              # 2 x K
    extra_blocks: int            # 2 x K x n_b
    change: float                # slowdown seen in the confirmation runs, mean(cand) / mean(ref) - 1
    ended_by: str                # "alert boundary", "dismiss boundary" or "level rule after K_max"
    llr: list = field(default_factory=list)       # log-likelihood ratio after each batch
    pairs: list = field(default_factory=list)     # (candidate run mean, reference run mean) per batch

    def to_dict(self):
        return asdict(self)


def wald_bounds(alpha, beta):
    """Wald's boundaries (7): alert when the log-likelihood ratio reaches A, dismiss at B."""
    return float(np.log((1 - beta) / alpha)), float(np.log(beta / (1 - alpha)))


def nasc_confirm(measure, noise, ref_level, settings):
    """measure(side, n_blocks) -> mean of one fresh run, side = "candidate" or "reference".
    noise: the benchmark's Noise; ref_level: the stored mean of the reference version (m_b)."""
    alpha, beta, k_max = settings["alpha"], settings["beta"], settings["k_max"]
    cap, n_b = settings["cap"], noise.n_b
    s_b = np.sqrt(noise.c_t ** 2 + noise.c_w ** 2 / n_b)                    # equation (3)
    s_b = max(s_b, 1e-6)
    theta = settings["delta_min"] / s_b                                     # equation (5)
    var = 2.0                                                               # y_j = one run minus one run
    upper, lower = wald_bounds(alpha, beta)
    y_low = -cap * np.sqrt(var / 2)                                         # the A4 cap (Huber-style)
    y_high = theta / 2 + min(cap * np.sqrt(var / 2), 2 / 3 * var * upper / theta)
    total, llrs, pairs = 0.0, [], []
    for k in range(1, k_max + 1):
        cand = measure("candidate", n_b)
        ref = measure("reference", n_b)
        pairs.append((cand, ref))
        y = (cand - ref) / (ref_level * s_b)                                # equation (4)
        total += float(np.clip(y, y_low, y_high))
        llr = theta / (2 * var) * (2 * total - k * theta)                   # equation (6)
        llrs.append(float(llr))
        if llr >= upper or llr <= lower:
            return _result(llr >= upper, k, n_b, pairs, llrs,
                           "alert boundary" if llr >= upper else "dismiss boundary")
    passed = total >= NormalDist().inv_cdf(1 - alpha) * np.sqrt(var * k_max)
    return _result(bool(passed), k_max, n_b, pairs, llrs, "level rule after K_max")


def _result(confirmed, k, n_b, pairs, llrs, ended_by):
    c = np.array(pairs)
    change = float(c[:, 0].mean() / c[:, 1].mean() - 1)
    return Confirmation(bool(confirmed), k, 2 * k, 2 * k * n_b, change, ended_by, llrs,
                        [tuple(map(float, p)) for p in pairs])
