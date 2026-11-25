# NASC-CI case study on jenetics

This branch holds a performance case study: the jenetics benchmarks (JMH) are measured on GitHub
Actions for 15 consecutive commits of the jenetics v4.4.0 release line, and a confirmation layer
for performance-regression alerts (NASC-CI) decides which base-detector alerts to keep.

The jenetics source itself is untouched: it lives on the other branches of this fork (`master`,
tags `v4.2.0` .. `v4.4.0`). Jenetics is licensed under the Apache License 2.0 (see `master`).

| Path | What |
|---|---|
| `case-study/commits.txt` | the 15 studied commits, oldest first |
| `case-study/benchmarks.txt` | the 16 studied benchmarks |
| `case-study/reference_noise_profile.csv` | reference noise (c_w, c_t, n_b) of the 16 benchmarks, from the traces of Laaber et al. (FSE 2020) |
| `case-study/pilot.py` | the two-commit pilot: checks a recording, base detectors, forced NASC confirmation, report (pilot v1 ran with NASC-CI 0.2) |
| `case-study/ab_check.py` | same-VM A/B check: back-to-back runs of two commits in alternating (ABBA) order, the difference per order, the order effect and the same-VM noise; with an injection patch on B, the injected slowdown and the CPU it was measured on |
| `case-study/injections/` | the two injected slowdowns (`heapsort.patch`: extra `sink()` passes in `HeapSorter.sort(double[], int[])`; `indexof.patch`: a partial re-scan in `ProbabilitySelector.indexOfSerial`); they only add lines, keep every result unchanged, and `INJECTED_EXTRA_PERCENT` sets their size |
| `nasc_ci/` | NASC-CI, the confirmation layer; `nasc_ci/VERSION` is the commit of the source it was copied from |
| `tests/` | unit tests (sample-mode iteration means; the A/B check with a simulated machine) |
| `.github/workflows/build-check.yml` | builds each commit's JMH jar on Java 8 and checks the 16 benchmarks are in it (no measurement) |
| `.github/workflows/pilot.yml` | the pilot: records commits 1 and 2, then one forced NASC confirmation of two benchmarks |
| `.github/workflows/ab-check.yml` | the same-VM A/B check: one job builds two commits (optionally with an injection patch on the second) and runs `case-study/ab_check.py` (started by hand) |

All runs use Eclipse Temurin 1.8.0_514-b01 and JMH in sample mode
(`-bm sample -tu ns -f 3 -wi 10 -i 20 -w 1s -r 1s`; one fork per run in the A/B check) on GitHub's
standard `ubuntu-latest` runner. Every measuring job logs its CPU model (`machine.json`, `lscpu.txt`).
