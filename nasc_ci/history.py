"""The history store: one JSON file with every checked version and every decision.

{
  "tool": "nasc-ci 0.1 ...",
  "settings": {...},
  "versions": [{"index": 1, "commit": "a1b2c3", "label": "v01", "time": "...",
                "results": {benchmark: {"runs": [[blocks]], "unit": "ns/op"}}}, ...],
  "checks":   [{"version": 6, "benchmark": "...", "decision": "confirmed", ...}, ...],
  "baselines": {benchmark: first version index of the current baseline}
}
When a slowdown is confirmed, the confirmed version becomes the new baseline (triage accepts it,
as in nasc_sim.check_series), so later versions are compared with the new level.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

from . import VERSION

DEFAULT_SETTINGS = {
    "detector": "bootstrap",      # threshold, ttest or bootstrap
    "window": 5,                  # W: stored versions the detector compares with
    "threshold_p": 0.02,
    "ttest_tau": 2.0,            # Perfherder t-test threshold; 2 as in the paper (main grid sweeps 2, 3, 5, 7)
    "bootstrap_level": 0.99,
    "delta_min": 0.02,            # smallest slowdown worth an alert
    "alpha": 0.05,
    "beta": 0.05,
    "k_max": 5,
    "n_max": 4,                   # longest confirmation run, in blocks (a recorded run has 4)
    "cap": 3.0,
    "noise_min_versions": 3,      # versions needed before c_w and c_t are estimated
    "default_c_w": 0.01,          # used until then
    "default_c_t": 0.01,
}


def new(settings=None):
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    return {"tool": VERSION, "settings": s, "versions": [], "checks": [], "baselines": {}}


def load(path, settings=None):
    """Loads the history, or starts an empty one. Settings given here override the stored ones."""
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as f:
            h = json.load(f)
        h["settings"] = {**DEFAULT_SETTINGS, **h.get("settings", {}), **(settings or {})}
        return h
    return new(settings)


def save(h, path):
    h["tool"] = VERSION
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(h, f, indent=1)
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def versions_of(h, benchmark):
    """All stored versions that have this benchmark: list of (version entry, runs array)."""
    return [(v, np.array(v["results"][benchmark]["runs"])) for v in h["versions"]
            if benchmark in v["results"]]


def window_of(h, benchmark):
    """The versions the detector compares with: the last W since the current baseline started.
    Returns (list of (entry, runs), warming_up). Warming up = fewer than W versions stored in total
    (the first versions of a new benchmark are never checked, as versions 1..W in the simulation)."""
    w = h["settings"]["window"]
    stored = versions_of(h, benchmark)
    start = h["baselines"].get(benchmark, 0)
    since = [(v, r) for v, r in stored if v["index"] >= start]
    return since[-w:], len(stored) < w


def overhead(h):
    """NASC's measurement overhead as the paper defines it, O = (T_policy - T_none) / T_none
    = confirmation blocks / recorded blocks, counted over the CHECKED benchmark-versions only
    (warm-up versions are recorded but never checked), as nasc_sim.summarize_scores computes T.
    Returns (extra blocks, recorded blocks, O)."""
    size = {(v["index"], b): len(r["runs"]) * len(r["runs"][0])
            for v in h["versions"] for b, r in v["results"].items()}
    checked = [c for c in h["checks"] if c["decision"] != "warm-up"]
    recorded = sum(size[c["version"], c["benchmark"]] for c in checked)
    extra = sum(c["extra_blocks"] for c in checked)
    return extra, recorded, (extra / recorded if recorded else 0.0)


def overhead_text(h):
    extra, recorded, o = overhead(h)
    return f"overhead O = {o:+.1%} ({extra} extra blocks / {recorded} recorded blocks of checked versions)"

