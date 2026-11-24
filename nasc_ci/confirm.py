"""NASC, the sequential confirmation (III-E; nasc_sim.nasc_confirm with terminal="level" and the cap).

A batch is one fresh run of the candidate and one fresh run of the reference (the parent version),
n_b blocks each, run back to back so both see the same machine state. After every batch the
log-likelihood ratio is compared with Wald's boundaries: alert at A, dismiss at B. If neither is
reached after k_max batches, the level rule decides (one-sided test at level alpha on all batches).

Two settings (v0.3, 24 Nov 2026):
  scale  "fresh"  (default) y_k is divided by the mean of the reference runs of THIS confirmation
                  so far (batches 1..k), i.e. the parent's level on the machine doing the re-runs;
         "stored" y_k is divided by ref_level, the value the caller passes: the simulation's true
                  parent level (as nasc_sim), or the parent's stored mean. On cloud runners the
                  stored mean can come from another machine (the pilot saw a ~60 % offset), so
                  the case study uses "fresh". The equivalence test with nasc_sim uses "stored".
  order  "abba"   (default) the order inside a batch alternates: candidate then reference in odd
                  batches, reference then candidate in even batches, so a "first run is slower"
                  effect cancels over every two batches; the order of each batch is recorded;
         "fixed"  candidate then reference in every batch (as nasc_sim draws them).
"""

from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import numpy as np

CAND_FIRST, REF_FIRST = "candidate first", "reference first"


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
    orders: list = field(default_factory=list)    # "candidate first" / "reference first" per batch
    scales: list = field(default_factory=list)    # the level y_k was divided by, per batch
    scale: str = "stored"                          # "fresh" or "stored"

    def to_dict(self):
        return asdict(self)


def wald_bounds(alpha, beta):
    """Wald's boundaries (7): alert when the log-likelihood ratio reaches A, dismiss at B."""
    return float(np.log((1 - beta) / alpha)), float(np.log(beta / (1 - alpha)))


def batch_order(k, order):
    """Order of batch k (1-based): ABBA alternates, "fixed" is always candidate first."""
    if order not in ("abba", "fixed"):
        raise ValueError(f"unknown order {order!r} (use abba or fixed)")
    if order == "fixed" or k % 2 == 1:
        return CAND_FIRST
    return REF_FIRST


def nasc_confirm(measure, noise, ref_level, settings):
    """measure(side, n_blocks) -> mean of one fresh run, side = "candidate" or "reference".
    noise: the benchmark's Noise; ref_level: the reference level used when settings["scale"] is
    "stored" (the stored mean of the reference version, m_b, or the simulation's true level)."""
    alpha, beta, k_max = settings["alpha"], settings["beta"], settings["k_max"]
    cap, n_b = settings["cap"], noise.n_b
    scale_mode = settings.get("scale", "fresh")
    order = settings.get("order", "abba")
    if scale_mode not in ("fresh", "stored"):
        raise ValueError(f"unknown scale {scale_mode!r} (use fresh or stored)")
    s_b = np.sqrt(noise.c_t ** 2 + noise.c_w ** 2 / n_b)                    # equation (3)
    s_b = max(s_b, 1e-6)
    theta = settings["delta_min"] / s_b                                     # equation (5)
    var = 2.0                                                               # y_j = one run minus one run
    upper, lower = wald_bounds(alpha, beta)
    y_low = -cap * np.sqrt(var / 2)                                         # the A4 cap (Huber-style)
    y_high = theta / 2 + min(cap * np.sqrt(var / 2), 2 / 3 * var * upper / theta)
    total, llrs, pairs, orders, scales = 0.0, [], [], [], []
    for k in range(1, k_max + 1):
        o = batch_order(k, order)
        if o == CAND_FIRST:
            cand = measure("candidate", n_b)
            ref = measure("reference", n_b)
        else:
            ref = measure("reference", n_b)
            cand = measure("candidate", n_b)
        pairs.append((cand, ref))
        orders.append(o)
        level = float(np.mean([p[1] for p in pairs])) if scale_mode == "fresh" else float(ref_level)
        scales.append(level)
        y = (cand - ref) / (level * s_b)                                    # equation (4)
        total += float(np.clip(y, y_low, y_high))
        llr = theta / (2 * var) * (2 * total - k * theta)                   # equation (6)
        llrs.append(float(llr))
        if llr >= upper or llr <= lower:
            return _result(llr >= upper, k, n_b, pairs, llrs,
                           "alert boundary" if llr >= upper else "dismiss boundary", orders, scales, scale_mode)
    passed = total >= NormalDist().inv_cdf(1 - alpha) * np.sqrt(var * k_max)
    return _result(bool(passed), k_max, n_b, pairs, llrs, "level rule after K_max", orders, scales, scale_mode)


def _result(confirmed, k, n_b, pairs, llrs, ended_by, orders, scales, scale_mode):
    c = np.array(pairs)
    change = float(c[:, 0].mean() / c[:, 1].mean() - 1)
    return Confirmation(bool(confirmed), k, 2 * k, 2 * k * n_b, change, ended_by, llrs,
                        [tuple(map(float, p)) for p in pairs], orders, scales, scale_mode)
