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
| `.github/workflows/build-check.yml` | builds each commit's JMH jar on Java 8 and checks the 16 benchmarks are in it (no measurement) |
