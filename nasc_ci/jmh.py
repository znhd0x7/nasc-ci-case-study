"""Reads JMH JSON results (java -jar benchmarks.jar -rf json -rff result.json).

A run is one JMH fork. Every measurement iteration of a run gives one score:
  - average-time / throughput modes: primaryMetric.rawData, one list per fork, one score per iteration;
  - sample mode (-bm sample): primaryMetric.rawDataHistogram, one list per fork, one list per
    iteration of [value, count] pairs. The iteration's score is the weighted mean
    sum(value * count) / sum(count), exactly as the noise-profile step computes it from the
    bencher CSV of Laaber et al. (weighted = value * value_count, summed per iteration, divided
    by the summed value_count).
As in the thesis, iterations are grouped into blocks of BLOCK_SIZE (5) and a run becomes the list
of its block means: 20 iterations -> 4 blocks.

v2 (24 Nov 2026): sample mode (rawDataHistogram) added; rawData files are read as before.
"""

import json

import numpy as np

BLOCK_SIZE = 5


def bench_name(entry):
    """Benchmark method plus its @Param values, e.g. demo.SortBench.quickSort [size=1000]."""
    params = entry.get("params") or {}
    if not params:
        return entry["benchmark"]
    return entry["benchmark"] + " [" + ", ".join(f"{k}={v}" for k, v in sorted(params.items())) + "]"


def histogram_mean(pairs):
    """One sample-mode iteration: [[value, count], ...] -> sum(value * count) / sum(count)."""
    x = np.asarray(pairs, dtype=float).reshape(-1, 2)
    return float((x[:, 0] * x[:, 1]).sum() / x[:, 1].sum())


def iteration_scores(metric):
    """-> one list per fork with the score of every measurement iteration."""
    if "rawDataHistogram" in metric:
        return [[histogram_mean(it) for it in fork] for fork in metric["rawDataHistogram"]]
    return [[float(s) for s in fork] for fork in metric["rawData"]]


def to_blocks(iterations, block_size=BLOCK_SIZE):
    x = np.asarray(iterations, dtype=float)
    n = len(x) // block_size
    if n == 0:                                 # shorter than one block: the run is one block
        return np.array([x.mean()])
    return x[: n * block_size].reshape(n, block_size).mean(axis=1)


def read(path):
    """-> {benchmark: {"runs": [[block means], ...], "unit": "ns/op", "mode": "avgt"}}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for entry in data:
        metric = entry["primaryMetric"]
        runs = [to_blocks(fork).tolist() for fork in iteration_scores(metric)]
        out[bench_name(entry)] = {"runs": runs, "unit": metric.get("scoreUnit", ""),
                                  "mode": entry.get("mode", "")}
    return out


def mean_of(path, benchmark):
    """Mean of all blocks of one benchmark in a JMH file (used for a confirmation run)."""
    results = read(path)
    if benchmark not in results:
        raise KeyError(f"{benchmark} not found in {path}; found: {', '.join(results)}")
    return float(np.mean([np.mean(r) for r in results[benchmark]["runs"]]))
