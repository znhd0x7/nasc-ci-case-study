"""Base detectors, the same rules as nasc_sim.py (step 14 / A8a).

Every detector gets
  window: the stored versions it compares with, oldest first; each one an array of shape
          (runs, blocks) with the block means of that version
  cand:   the new version, also (runs, blocks)
and returns a Flag. Only slowdowns are flagged (a larger time per operation).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Flag:
    flagged: bool
    change: float        # relative change the detector saw, e.g. 0.031 = 3.1 % slower
    statistic: float     # the detector's own number (ratio, t value or lower interval end)
    rule: str            # short text of the rule, for the report


def threshold_detector(window, cand, p=0.02):
    """Percentage threshold: the version's mean against the median of the window's version means."""
    history = np.median([v.mean(axis=1).mean() for v in window])     # mean of run means, as nasc_sim
    change = cand.mean(axis=1).mean() / history - 1
    return Flag(change > p, change, change, f"change > {p:.0%}")


def ttest_detector(window, cand, tau=2.0, min_change=0.02):
    """Perfherder-style t-test: the version's runs (fore window) against all runs of the window
    (back window). Flags when t > tau AND the slowdown is at least min_change."""
    fore = cand.mean(axis=1)
    back = np.concatenate([v.mean(axis=1) for v in window])
    diff = fore.mean() - back.mean()
    se = np.sqrt(fore.var(ddof=1) / len(fore) + back.var(ddof=1) / len(back)) if len(fore) > 1 else 0.0
    t = diff / se if se > 0 else 0.0
    change = diff / back.mean()
    return Flag(bool(se > 0 and t > tau and change > min_change), change, t,
                f"t > {tau:g} and change > {min_change:.0%}")


def bootstrap_ratio(cand, ref, rng, n_boot=1000, level=0.99):
    """Hierarchical bootstrap (Kalibera & Jones) of the ratio of means cand / ref: runs drawn with
    replacement, then blocks inside each drawn run. Returns (median, lower, upper)."""
    def resampled_means(x):
        n_runs, n_blocks = x.shape
        r = rng.integers(n_runs, size=(n_boot, n_runs, 1))
        b = rng.integers(n_blocks, size=(n_boot, n_runs, n_blocks))
        return x[r, b].mean(axis=(1, 2))
    ratio = resampled_means(cand) / resampled_means(ref)
    lo, mid, hi = np.quantile(ratio, [(1 - level) / 2, 0.5, (1 + level) / 2])
    return mid, lo, hi


def bootstrap_detector(window, cand, rng, level=0.99, min_change=0.0, n_boot=1000):
    """Laaber's rule: the version against its PARENT (the last stored version); flags when the
    whole bootstrap interval of the ratio lies above 1 + min_change."""
    parent = window[-1]
    blocks = min(cand.shape[1], parent.shape[1])
    mid, lo, _ = bootstrap_ratio(cand[:, :blocks], parent[:, :blocks], rng, n_boot, level)
    return Flag(lo > 1 + min_change, mid - 1, lo, f"{level:.0%} interval above {1 + min_change:g}")


def detect(name, window, cand, rng, settings):
    """Runs the detector chosen in the settings."""
    if name == "threshold":
        return threshold_detector(window, cand, settings["threshold_p"])
    if name == "ttest":
        return ttest_detector(window, cand, settings["ttest_tau"], settings["delta_min"])
    if name == "bootstrap":
        return bootstrap_detector(window, cand, rng, settings["bootstrap_level"])
    raise ValueError(f"unknown detector {name!r} (use threshold, ttest or bootstrap)")
