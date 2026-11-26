"""Tests of case-study/study.py, step 2 (26 Nov 2026): confirm. No JMH is run: the recordings are
synthetic (as in test_study.py) and the confirmation runs come from a fake runner that returns
block means at a chosen level per commit, so every decision can be checked by hand.

    python -m pytest -q tests/test_study_confirm.py
"""

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-study"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import study  # noqa: E402
from nasc_ci import confirm as nasc  # noqa: E402
from nasc_ci import noise  # noqa: E402
from test_study import record  # noqa: E402

S = study.settings()
H80 = "io.jenetics.internal.util.IndexSorterPerf.heapSort80"
LONG = "io.jenetics.util.RandomEnginePerf.RandomPerf.nextLong"


def history_with(tmp_path, levels):
    """Records commits 1-15 with levels {pos: {short: level}} and runs the profile step."""
    for pos in range(1, 16):
        record(tmp_path, pos, levels.get(pos, {}))
    study.main(["profile", "--records", str(tmp_path / "records"), "--out", str(tmp_path / "p")])
    return json.loads((tmp_path / "p" / "history.json").read_text())


class FakeRunner:
    """run(pos, n_blocks, tag) -> (blocks, file): level[pos] x (1 + small noise); logs every call."""

    def __init__(self, level, sd=0.002, seed=0):
        self.level, self.sd, self.rng, self.calls = level, sd, np.random.default_rng(seed), []

    def __call__(self, pos, n_blocks, tag):
        self.calls.append((pos, n_blocks, tag))
        x = self.level(pos) * (1 + self.rng.normal(0, self.sd, n_blocks))
        return [float(v) for v in x], f"runs/{tag}.json"


def step_levels(v_star, size, bench="heapSort80", base=1350.0):
    return {p: {bench: base * (1 + size if p >= v_star else 1)} for p in range(1, 16)}


def test_ground_truth_helper(capsys):
    for b, want in [("heapSort80", "9 heapsort"), ("heapSort320", "9 heapsort"),
                    ("serialIndexOf250", "13 indexof"), ("nextLong", "")]:
        study.main(["ground-truth", "--bench", b])
        assert capsys.readouterr().out.strip() == want


def test_flat_benchmark_no_confirmations(tmp_path, capsys):
    h = history_with(tmp_path, {})
    run = FakeRunner(lambda p: 100.0)
    R = study.run_confirm(h, LONG, run)
    capsys.readouterr()
    assert [c["pos"] for c in R["checks"]] == list(range(6, 16))
    for d in S["detectors"]:
        for a in S["arms"]:
            assert [r[0] for r in R["series"][d][a]] == list(range(6, 16))
    n_conf = sum(c["confirmation"] is not None for c in R["checks"])
    assert len(run.calls) == n_conf * 2 * S["nasc"]["k_max"]
    assert R["v_star"] is None and R["complete"]


def test_step_detected_then_baseline_restarts(tmp_path, capsys):
    h = history_with(tmp_path, step_levels(9, 0.12))
    run = FakeRunner(lambda p: 1350.0 * (1.12 if p >= 9 else 1.0))
    R = study.run_confirm(h, H80, run)
    capsys.readouterr()
    c9 = next(c for c in R["checks"] if c["pos"] == 9)
    assert c9["confirmation"] is not None
    conf = c9["confirmation"]
    assert len(conf["batches"]) == 5                                # always K_max batches
    assert conf["nasc"]["confirmed"] and conf["nasc"]["batches"] < 5  # NASC stops early on +12 %
    assert conf["static"]["confirmed"] and conf["static-mid"]["confirmed"]
    assert conf["nasc"]["extra_blocks"] == 2 * conf["nasc"]["batches"] * conf["n_b"]
    for d in ("threshold", "ttest"):
        for a in S["arms"]:
            f9 = c9["flags"][d][a]
            assert f9["flagged"] and f9["alerted"] and f9["window"] == [4, 8]
            f10 = next(c for c in R["checks"] if c["pos"] == 10)["flags"][d][a]
            assert f10["start"] == 9 and f10["window"] == [9, 9] and not f10["flagged"]
    # confirmation of commit 9 against commit 8, ABBA order, n_b blocks per run
    calls = [c for c in run.calls if c[2].startswith("c09/")]
    assert [c[2] for c in calls[:4]] == ["c09/batch1-candidate", "c09/batch1-reference",
                                         "c09/batch2-reference", "c09/batch2-candidate"]
    assert [c[0] for c in calls[:2]] == [9, 8] and {c[1] for c in calls} == {conf["n_b"]}
    assert [b["order"] for b in conf["batches"]] == [nasc.batch_order(j, "abba") for j in range(1, 6)]
    # cost columns
    row = next(r for r in R["series"]["threshold"]["static"] if r[0] == 9)
    assert row == [9, True, True, 2 * 5 * conf["n_b"]]
    assert next(r for r in R["series"]["threshold"]["none"] if r[0] == 9) == [9, True, True, 0]


def test_dismissed_arm_keeps_old_window(tmp_path, capsys):
    """The recordings show a step at 9, but the re-runs on the confirm VM show none: every
    confirming arm dismisses, so only arm none accepts the step and restarts its history."""
    h = history_with(tmp_path, step_levels(9, 0.12))
    R = study.run_confirm(h, H80, FakeRunner(lambda p: 1350.0))
    capsys.readouterr()
    c9, c10 = (next(c for c in R["checks"] if c["pos"] == p) for p in (9, 10))
    assert not c9["confirmation"]["nasc"]["confirmed"] and not c9["confirmation"]["static"]["confirmed"]
    assert c9["flags"]["threshold"]["none"]["alerted"] and not c9["flags"]["threshold"]["nasc"]["alerted"]
    assert c10["flags"]["threshold"]["none"]["window"] == [9, 9]
    assert c10["flags"]["threshold"]["nasc"]["window"] == [5, 9]       # old level still in its window
    assert c10["flags"]["threshold"]["nasc"]["flagged"]                # so it flags (and confirms) again
    assert c10["confirmation"] is not None


def test_static_and_static_mid_by_hand():
    prof = noise.Noise(c_w=0.004, c_t=0.001, n_b=2, feasible=True, delta_feasible=0.01, versions=5, estimated=True)
    s = {**S["nasc"]}
    cand, ref = [100.6, 100.5, 100.7, 100.6, 100.5], [100.0, 100.0, 100.1, 99.9, 100.0]
    batches = [{"candidate": c, "reference": r} for c, r in zip(cand, ref)]
    D = study.static_decisions(batches, prof, s)
    level = np.mean(ref)
    s_b = math.sqrt(0.001 ** 2 + 0.004 ** 2 / 2)
    total = sum(np.clip((c - r) / (level * s_b), -3, 3) for c, r in zip(cand, ref))
    assert D["static"]["total"] == pytest.approx(total)
    assert D["static"]["bound"] == pytest.approx(NormalDist().inv_cdf(0.95) * math.sqrt(10))
    # +0.58 % on average: significant (static alerts) but below delta_min / 2 = 1 % (static-mid dismisses)
    assert D["static"]["confirmed"] and not D["static-mid"]["confirmed"]
    assert D["static-mid"]["slowdown"] == pytest.approx(np.mean([(c - r) / level for c, r in zip(cand, ref)]))


def test_nasc_ignores_batches_after_its_stop():
    prof = noise.Noise(c_w=0.005, c_t=0.002, n_b=1, feasible=True, delta_feasible=0.01, versions=5, estimated=True)
    s = {**S["nasc"]}
    big = [{"candidate": 110.0, "reference": 100.0}] * 2
    late = [{"candidate": 90.0, "reference": 100.0}] * 3
    D = study.nasc_decision(big + late, prof, s)
    D2 = study.nasc_decision(big + big + [big[0]], prof, s)
    assert D["confirmed"] and D["batches"] <= 2 and D == D2


def test_runner_classpath_from_packed_build(tmp_path):
    d = tmp_path / "builds" / "c09"
    d.mkdir(parents=True)
    (d / "000-x.jar").write_bytes(b"j")
    (d / "classpath.txt").write_text("000-x.jar\n")
    r = study.JmhRunner(H80, tmp_path / "builds", tmp_path / "out", 10, "1s")
    assert r.classpath(9) == str((d / "000-x.jar").resolve())
