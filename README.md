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
| `case-study/ab_check.py` | same-VM A/B check: back-to-back runs of two commits in alternating (ABBA) order, the difference per order, the order effect and the same-VM noise; with an injection patch on B, the injected slowdown and the CPU it was measured on; `pool` combines copies run on different VMs into the difference per CPU type (`--expected` names copies that are missing) |
| `case-study/study.py`, `case-study/study_settings.json` | the full study (15 commits, 10 benchmarks): checks each recording, the completeness gate, the noise profile frozen from commits 1-5, the confirmations of commits 6-15 per benchmark (base detectors threshold, t-test, bootstrap; arms none, static, static-mid, NASC), the scores and the report; all settings in the JSON file |
| `case-study/calibration_background.json` | the injected slowdowns measured per CPU type before the full study (background in the report) |
| `case-study/injections/` | the two injected slowdowns (`heapsort.patch`: extra `sink()` passes in `HeapSorter.sort(double[], int[])`; `indexof.patch`: the search loop of `ProbabilitySelector.indexOfSerial` run again over part of the prefix); they only add lines, keep every result unchanged, and `INJECTED_EXTRA_PERCENT` sets their size; `targets.txt` gives each patch's first commit and the benchmarks it slows down |
| `nasc_ci/` | NASC-CI, the confirmation layer; `nasc_ci/VERSION` is the commit of the source it was copied from |
| `tests/` | unit tests (sample-mode iteration means; the A/B check with a simulated machine) |
| `.github/workflows/build-check.yml` | builds each commit's JMH jar on Java 8 and checks the 16 benchmarks are in it (no measurement) |
| `.github/workflows/pilot.yml` | the pilot: records commits 1 and 2, then one forced NASC confirmation of two benchmarks |
| `.github/workflows/ab-check.yml` | the same-VM A/B check: one job builds two commits (optionally with an injection patch on the second) and runs `case-study/ab_check.py` (started by hand); with `copies` 2-4, that many jobs run in parallel on separate VMs and a summary job reports the result per CPU type |
| `.github/workflows/full-study.yml` | the full study (started by hand): 15 record jobs (one VM per commit), the frozen profile, 10 confirm jobs (one per benchmark, with a same-VM A/B of each injection), and the report, which is marked INCOMPLETE and fails if any job or artifact is missing; every build is retried twice and every failed job leaves `FAILED.txt` in its artifact |

All runs use Eclipse Temurin 1.8.0_514-b01 and JMH in sample mode
(`-bm sample -tu ns -f 3 -wi 10 -i 20 -w 1s -r 1s`; one fork per run in the A/B check) on GitHub's
standard `ubuntu-latest` runner. Every measuring job logs its CPU model (`machine.json`, `lscpu.txt`).
