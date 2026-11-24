"""NASC-CI: Noise-Adaptive Sequential Confirmation for performance alerts in CI.

The statistics are the ones of the thesis simulator (nasc_sim.py), rewritten so they work on
real benchmark results instead of simulated series:
  detectors.py  base detectors: threshold, Perfherder t-test, hierarchical bootstrap
  noise.py      noise profile c_w, c_t and blocks per run n_b (equations 1-3 and 8)
  confirm.py    NASC, the sequential confirmation (equations 3-7, level rule, cap)
  jmh.py        reads JMH JSON results into runs of blocks
  history.py    the JSON history store
  runner.py     re-runs a benchmark for a confirmation batch (any shell command)
  check.py      one CI check: detect, confirm, store, report
"""

VERSION = "nasc-ci 0.2 (24 Nov 2026)"
