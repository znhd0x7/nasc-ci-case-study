"""Checks NASC-CI's reader for JMH sample mode (-bm sample) against the iteration-mean recipe of
the noise-profile pipeline (src/04_iteration_means.py), which turned the bencher CSV of Laaber et
al. into one mean per iteration:

    KEY = ["commit", "benchmark", "params", "instance", "trial", "fork", "iteration"]
    chunk["weighted"] = chunk["value"] * chunk["value_count"]
    totals = chunk.groupby(KEY)[["weighted", "value_count"]].sum()
    totals["mean"] = totals["weighted"] / totals["value_count"]

The JMH JSON is flattened into rows like that CSV (one row per [value, count] pair) and the same
lines are applied; NASC-CI's iteration scores must be the same numbers.

The fixture tests/data/jmh_sample_serialIndexOf250.json is a real JMH 1.17.4 result of jenetics
(ProbabilitySelectorIndexOfPerf.serialIndexOf250, Java 8, -bm sample -tu ns -f 2 -wi 0 -i 6
-r 50ms), kept short so the file stays small.

    python -m pytest -q tests/test_jmh_sample.py
v1 (24 Nov 2026)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # the folder that holds nasc_ci/

from nasc_ci import jmh  # noqa: E402

FIXTURE = HERE / "data" / "jmh_sample_serialIndexOf250.json"
KEY = ["commit", "benchmark", "params", "instance", "trial", "fork", "iteration"]


def as_bencher_rows(entries, commit="c1", trial=1):
    """JMH sample-mode JSON -> one row per [value, count] pair, like the bencher CSV."""
    rows = []
    for e in entries:
        for f, fork in enumerate(e["primaryMetric"]["rawDataHistogram"], start=1):
            for i, iteration in enumerate(fork, start=1):
                for value, count in iteration:
                    rows.append({"commit": commit, "benchmark": e["benchmark"], "params": "",
                                 "instance": "", "trial": trial, "fork": f, "iteration": i,
                                 "value": value, "value_count": count})
    return pd.DataFrame(rows)


def recipe_04(chunk):
    """The lines of src/04_iteration_means.py (Part 2 and 3) on one chunk."""
    chunk = chunk.copy()
    chunk["weighted"] = chunk["value"] * chunk["value_count"]
    totals = chunk.groupby(KEY, as_index=False)[["weighted", "value_count"]].sum()
    totals["mean"] = totals["weighted"] / totals["value_count"]
    return totals


def compare(entries):
    """Max relative difference between NASC-CI's iteration scores and recipe_04, and the count."""
    ref = recipe_04(as_bencher_rows(entries)).set_index(["benchmark", "fork", "iteration"])["mean"]
    worst, n = 0.0, 0
    for e in entries:
        for f, fork in enumerate(jmh.iteration_scores(e["primaryMetric"]), start=1):
            for i, score in enumerate(fork, start=1):
                expected = ref[(e["benchmark"], f, i)]
                worst = max(worst, abs(score / expected - 1))
                n += 1
    assert n == len(ref), f"NASC-CI gave {n} iterations, the recipe {len(ref)}"
    return worst, n


def test_hand_example():
    # 1 x 10 ns and 3 x 20 ns -> (10 + 60) / 4 = 17.5 ns (not the plain mean of the values, 15)
    assert jmh.histogram_mean([[10.0, 1], [20.0, 3]]) == 17.5


def test_real_fixture_same_as_recipe_04():
    entries = json.loads(FIXTURE.read_text(encoding="utf-8"))
    worst, n = compare(entries)
    assert n == 12                       # 2 forks x 6 iterations
    assert worst < 1e-12, f"largest relative difference {worst:.3g}"


def test_real_fixture_pooled_mean_is_jmh_score():
    # independent check that the [value, count] pairs are read right: JMH's own score in sample
    # mode is the pooled mean of all samples of all forks
    m = json.loads(FIXTURE.read_text(encoding="utf-8"))[0]["primaryMetric"]
    pairs = np.array([p for fork in m["rawDataHistogram"] for it in fork for p in it], dtype=float)
    pooled = (pairs[:, 0] * pairs[:, 1]).sum() / pairs[:, 1].sum()
    assert abs(pooled / m["score"] - 1) < 1e-12


def test_random_histograms_same_as_recipe_04():
    # 3 forks x 20 iterations, like the case study, with repeated values inside an iteration
    rng = np.random.default_rng(4)
    hist = [[[[float(v), int(c)] for v, c in zip(rng.choice(np.arange(100, 400, 2), 50),
                                                  rng.integers(1, 30, 50))]
             for _ in range(20)] for _ in range(3)]
    entries = [{"benchmark": "x.Bench.m", "primaryMetric": {"rawDataHistogram": hist}}]
    worst, n = compare(entries)
    assert n == 60 and worst < 1e-12


def test_blocks_and_reader(tmp_path):
    # read(): 20 iterations -> 4 blocks, each the plain mean of 5 iteration means
    rng = np.random.default_rng(7)
    hist = [[[[float(v), int(c)] for v, c in zip(rng.integers(50, 90, 8), rng.integers(1, 9, 8))]
             for _ in range(20)] for _ in range(3)]
    entry = {"benchmark": "x.Bench.m", "mode": "sample",
             "primaryMetric": {"scoreUnit": "ns/op", "rawDataHistogram": hist}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    r = jmh.read(path)["x.Bench.m"]
    its = recipe_04(as_bencher_rows([entry])).sort_values(["fork", "iteration"])["mean"].to_numpy()
    expected = its.reshape(3, 4, 5).mean(axis=2)
    assert r["mode"] == "sample" and r["unit"] == "ns/op"
    assert np.allclose(r["runs"], expected, rtol=1e-12, atol=0)


def test_average_time_files_read_as_before(tmp_path):
    # rawData (avgt / thrpt) is unchanged: blocks of 5 plain iteration scores
    entry = {"benchmark": "x.Bench.m", "mode": "avgt",
             "primaryMetric": {"scoreUnit": "ns/op", "rawData": [list(range(1, 11)), list(range(11, 21))]}}
    path = tmp_path / "a.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    assert jmh.read(path)["x.Bench.m"]["runs"] == [[3.0, 8.0], [13.0, 18.0]]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
