"""pilot v1 (24 Nov 2026): the two-commit pilot of the NASC-CI case study.

  python3 case-study/pilot.py regex
      prints the JMH regular expression that selects the benchmarks of case-study/benchmarks.txt
  python3 case-study/pilot.py record --json raw.json --commit SHA --pos 1 --out DIR
      checks one full recording (every benchmark x 3 forks x 20 iterations, sample mode, ns/op) and
      writes iteration_means.csv (the columns of the noise-profile pipeline: one weighted mean per
      iteration) and results.json (NASC-CI's format: the block means of every run)
  python3 case-study/pilot.py confirm --parent DIR --candidate DIR --parent-cp FILE --candidate-cp FILE
      --force heapSort80,serialIndexOf250 --out DIR
      1. base detectors on the candidate against the parent: threshold 2 %, Perfherder t-test
         (tau = 2, change > 2 %), bootstrap 99 %. Pilot only: the window is the parent alone (W = 1),
         because there are only two commits; the study uses W = 5;
      2. a forced NASC confirmation of the --force benchmarks, flagged or not: the candidate and the
         parent are re-measured here, back to back in this job, n_b blocks per run from the
         reference noise profile, scaled by the parent's stored mean (as NASC-CI does in CI);
      3. the runner's c_w and c_t from the recorded runs, next to the reference noise profile;
      4. report.md (also the job summary), pilot.json, history.json (NASC-CI's history format) and
         the raw JMH output of every re-run.
Needs numpy. Run from the repository root.
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nasc_ci import VERSION, history, jmh, noise  # noqa: E402
from nasc_ci import confirm as nasc  # noqa: E402
from nasc_ci import detectors  # noqa: E402

PILOT = "pilot v1 (24 Nov 2026)"
HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE / "benchmarks.txt"
PROFILE = HERE / "reference_noise_profile.csv"
DETECTORS = ["threshold", "ttest", "bootstrap"]
BOOTSTRAP_SEED = 1


def benchmarks():
    return [l.strip() for l in BENCHMARKS.read_text().splitlines() if l.strip() and not l.startswith("#")]


def short(name):
    return ".".join(name.split(".")[-2:])


def source_hash():
    f = ROOT / "nasc_ci" / "VERSION"
    return f.read_text().strip() if f.exists() else "unknown"


def minutes(t0, t1):
    return (float(t1) - float(t0)) / 60 if t0 and t1 else None


def read_timing(path):
    """KEY=epoch-seconds lines written by the workflow."""
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = float(v)
    return out


# ---------------------------------------------------------------- regex
def cmd_regex(a):
    print("^(" + "|".join(re.escape(b) for b in benchmarks()) + ")$")


# ---------------------------------------------------------------- record
def cmd_record(a):
    data = json.loads(Path(a.json).read_text(encoding="utf-8"))
    by_name = {e["benchmark"]: e for e in data}
    problems = []
    for b in benchmarks():
        e = by_name.get(b)
        if e is None:
            problems.append(f"{b}: missing")
            continue
        m = e["primaryMetric"]
        want = {"mode": "sample", "forks": a.forks, "warmupIterations": a.warmup,
                "measurementIterations": a.iterations, "warmupTime": a.time, "measurementTime": a.time}
        for k, v in want.items():
            if e.get(k) != v:
                problems.append(f"{b}: {k} = {e.get(k)!r}, expected {v!r}")
        if m.get("scoreUnit") != "ns/op":
            problems.append(f"{b}: unit {m.get('scoreUnit')!r}, expected 'ns/op'")
        hist = m.get("rawDataHistogram")
        if not hist or len(hist) != a.forks or any(len(f) != a.iterations for f in hist):
            problems.append(f"{b}: expected {a.forks} forks x {a.iterations} iterations of histograms")
    extra = sorted(set(by_name) - set(benchmarks()))
    if extra:
        problems.append("unexpected benchmarks: " + ", ".join(extra))
    if problems:
        print("::error title=Recording incomplete::" + "; ".join(problems[:5]))
        for p in problems:
            print("  -", p)
        sys.exit(1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "iteration_means.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["commit", "benchmark", "params", "instance", "trial", "fork", "iteration",
                    "n_invocations", "mean"])
        for b in benchmarks():
            m = by_name[b]["primaryMetric"]
            for fi, fork in enumerate(m["rawDataHistogram"], start=1):
                for ii, it in enumerate(fork, start=1):
                    count = int(sum(c for _, c in it))
                    w.writerow([a.commit, b, "", "", 1, fi, ii, count, repr(jmh.histogram_mean(it))])
    results = jmh.read(a.json)
    record = {"pilot": PILOT, "tool": VERSION, "nasc_ci_source": source_hash(), "pos": a.pos,
              "commit": a.commit, "results": {b: results[b] for b in benchmarks()}}
    (out / "results.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{PILOT}; {VERSION}; source {source_hash()}")
    print(f"commit {a.pos} {a.commit[:10]}: {len(benchmarks())} benchmarks x {a.forks} forks x "
          f"{a.iterations} iterations recorded; iteration means and results.json in {out}")
    print(f"{'benchmark':<52} {'mean ns/op':>12}  run means")
    for b in benchmarks():
        runs = np.array(results[b]["runs"])
        print(f"{short(b):<52} {runs.mean():>12.4g}  " + " ".join(f"{x:.4g}" for x in runs.mean(axis=1)))


# ---------------------------------------------------------------- confirm
class Measurer:
    """Re-runs one benchmark with JMH for a confirmation batch and keeps the raw output."""

    def __init__(self, cps, out, wi, time_):
        self.cps, self.out, self.wi, self.time = cps, Path(out), wi, time_
        self.count = {}
        self.runs = []

    def for_benchmark(self, bench):
        def measure(side, n_blocks):
            k = self.count[(bench, side)] = self.count.get((bench, side), 0) + 1
            stem = self.out / "confirm" / short(bench) / f"batch{k}-{side}"
            stem.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["java", "-cp", self.cps[side], "org.openjdk.jmh.Main", "^" + re.escape(bench) + "$",
                   "-bm", "sample", "-tu", "ns", "-f", "1", "-wi", str(self.wi),
                   "-i", str(n_blocks * jmh.BLOCK_SIZE), "-w", self.time, "-r", self.time, "-foe", "true",
                   "-rf", "json", "-rff", str(stem) + ".json"]
            t0 = time.time()
            with open(str(stem) + ".log", "w", encoding="utf-8") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            value = jmh.mean_of(str(stem) + ".json", bench)
            self.runs.append({"benchmark": bench, "side": side, "batch": k, "blocks": n_blocks,
                              "mean": value, "seconds": round(time.time() - t0, 1),
                              "file": str(stem.relative_to(self.out)) + ".json"})
            print(f"   {short(bench)} batch {k} {side:<9} {n_blocks} blocks: {value:.5g} ns/op "
                  f"({time.time() - t0:.0f} s)", flush=True)
            return value
        return measure


def runner_noise(versions, s, min_versions):
    return noise.estimate(versions, {**s, "noise_min_versions": min_versions})


def pct(x, d=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def spct(x, d=2):
    return "–" if x is None else f"{x * 100:+.{d}f}%"


def cmd_confirm(a):
    P = json.loads((Path(a.parent) / "results.json").read_text(encoding="utf-8"))
    C = json.loads((Path(a.candidate) / "results.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h = history.new({"detector": "ttest", "ttest_tau": 2.0, "window": 1})
    s = h["settings"]
    h["pilot"] = (f"{PILOT}: two commits only, so the detector window is the parent alone (W = 1; "
                  "the study uses W = 5); NASC forced on " + ", ".join(a.force.split(",")))
    h["nasc_ci_source"] = source_hash()
    profile = noise.load_profile(PROFILE, s)
    names = benchmarks()
    forced = [b for b in names if any(b.endswith("." + f.strip()) for f in a.force.split(","))]
    if len(forced) != len(a.force.split(",")):
        sys.exit(f"--force {a.force}: found only {forced}")
    labels = {P["pos"]: f"c{P['pos']:02d}", C["pos"]: f"c{C['pos']:02d}"}
    vP = {"index": 1, "commit": P["commit"], "label": labels[P["pos"]], "time": history.now(),
          "results": P["results"]}
    vC = {"index": 2, "commit": C["commit"], "label": labels[C["pos"]], "time": history.now(),
          "results": C["results"]}
    h["versions"] = [vP, vC]

    cps = {"candidate": Path(a.candidate_cp).read_text().strip(),
           "reference": Path(a.parent_cp).read_text().strip()}
    meas = Measurer(cps, out, a.wi, a.time)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows, detail = [], []
    for b in names:
        pr = np.array(P["results"][b]["runs"], dtype=float)
        h["checks"].append({"version": 1, "label": vP["label"], "commit": P["commit"], "benchmark": b,
                            "mean": float(pr.mean()), "unit": "ns/op", "detector": "ttest",
                            "flagged": False, "detector_change": None, "decision": "warm-up",
                            "noise": profile[b].to_dict(), "confirmation": None, "extra_blocks": 0})
    for b in names:
        pr = np.array(P["results"][b]["runs"], dtype=float)
        cr = np.array(C["results"][b]["runs"], dtype=float)
        flags = {d: detectors.detect(d, [pr], cr, rng, s) for d in DETECTORS}
        row = {"version": 2, "label": vC["label"], "commit": C["commit"], "benchmark": b,
               "mean": float(cr.mean()), "parent_mean": float(pr.mean()), "unit": "ns/op", "detector": "ttest",
               "flagged": bool(flags["ttest"].flagged), "detector_change": float(flags["ttest"].change),
               "detector_statistic": float(flags["ttest"].statistic), "rule": flags["ttest"].rule,
               "parent": vP["label"], "flags": {d: {"flagged": bool(f.flagged), "change": float(f.change),
                                                     "statistic": float(f.statistic), "rule": f.rule}
                                                 for d, f in flags.items()},
               "decision": "no flag" if not flags["ttest"].flagged else "flagged, NASC not run (pilot)",
               "noise": profile[b].to_dict(), "confirmation": None, "extra_blocks": 0, "forced": b in forced}
        rows.append(row)
    h["checks"].extend(rows)

    print(f"{PILOT}; {VERSION}; source {source_hash()}")
    print(f"forced NASC on {', '.join(short(b) for b in forced)}: candidate {C['commit'][:10]}, "
          f"parent {P['commit'][:10]}, re-measured in this job", flush=True)
    t_nasc0 = time.time()
    nasc_rows = []
    for b in forced:
        row = next(r for r in rows if r["benchmark"] == b)
        pr = np.array(P["results"][b]["runs"], dtype=float)
        cr = np.array(C["results"][b]["runs"], dtype=float)
        t0 = time.time()
        c = nasc.nasc_confirm(meas.for_benchmark(b), profile[b], float(pr.mean()), s)
        pairs = np.array(c.pairs)
        row.update(decision="confirmed" if c.confirmed else "dismissed", confirmation=c.to_dict(),
                   extra_blocks=c.extra_blocks)
        nasc_rows.append({"benchmark": b, "n_b": profile[b].n_b, "batches": c.batches,
                          "ended_by": c.ended_by, "decision": row["decision"], "change": c.change,
                          "extra_blocks": c.extra_blocks, "llr": c.llr,
                          "fresh_parent_vs_stored": float(pairs[:, 1].mean() / pr.mean() - 1),
                          "fresh_candidate_vs_stored": float(pairs[:, 0].mean() / cr.mean() - 1),
                          "minutes": (time.time() - t0) / 60})
    nasc_minutes = (time.time() - t_nasc0) / 60
    history.save(h, str(out / "history.json"))

    # runner noise next to the reference profile
    noise_rows = []
    for b in names:
        pr = np.array(P["results"][b]["runs"], dtype=float)
        cr = np.array(C["results"][b]["runs"], dtype=float)
        nP, nC = runner_noise([pr], s, 1), runner_noise([cr], s, 1)
        both = runner_noise([pr, cr], s, 2)
        noise_rows.append({"benchmark": b, "reference": profile[b].to_dict(),
                           "runner_parent": nP.to_dict(), "runner_candidate": nC.to_dict(),
                           "runner_both": both.to_dict()})

    tP, tC, tX = read_timing(a.parent_timing), read_timing(a.candidate_timing), read_timing(a.timing)
    now = time.time()
    jobs = [
        {"job": f"record commit {P['pos']}", "setup_build": minutes(tP.get("job_start"), tP.get("measure_start")),
         "measure": minutes(tP.get("measure_start"), tP.get("measure_end")),
         "total": minutes(tP.get("job_start"), tP.get("measure_end"))},
        {"job": f"record commit {C['pos']}", "setup_build": minutes(tC.get("job_start"), tC.get("measure_start")),
         "measure": minutes(tC.get("measure_start"), tC.get("measure_end")),
         "total": minutes(tC.get("job_start"), tC.get("measure_end"))},
        {"job": "forced NASC", "setup_build": minutes(tX.get("job_start"), tX.get("measure_start")),
         "measure": nasc_minutes, "total": minutes(tX.get("job_start"), now)},
    ]
    summary = {"pilot": PILOT, "tool": VERSION, "nasc_ci_source": source_hash(),
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "parent": {"pos": P["pos"], "commit": P["commit"]},
               "candidate": {"pos": C["pos"], "commit": C["commit"]},
               "settings": s, "bootstrap_seed": BOOTSTRAP_SEED, "rerun": {"wi": a.wi, "time": a.time},
               "detector_rows": rows, "nasc": nasc_rows, "confirmation_runs": meas.runs,
               "noise": noise_rows, "jobs": jobs}
    (out / "pilot.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    text = report(summary)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


def report(S):
    P, C = S["parent"], S["candidate"]
    rows, nrows, jobs = S["detector_rows"], S["nasc"], S["jobs"]
    count = {d: sum(r["flags"][d]["flagged"] for r in rows) for d in DETECTORS}
    L = [f"## NASC-CI pilot: commits {P['pos']} and {C['pos']}", "",
         f"`{S['tool']}`, source `{S['nasc_ci_source']}`; {S['pilot']}. "
         "Pilot only: the detectors compare commit "
         f"{C['pos']} with commit {P['pos']} alone (W = 1); NASC was forced on "
         f"{', '.join('`' + short(n['benchmark']) + '`' for n in nrows)}.", "",
         "### Per commit", "",
         "| Commit | Base flags: threshold 2% / t-test τ=2 / bootstrap 99% | NASC decisions | Extra blocks | Runner minutes |",
         "|---|---|---|---|---|"]
    m = lambda x: "–" if x is None else f"{x:.1f}"  # noqa: E731
    L.append(f"| {P['pos']} `{P['commit'][:10]}` | warm-up (first commit, nothing to compare) | – | 0 | "
             f"{m(jobs[0]['total'])} (record) |")
    dec = "; ".join(f"{short(n['benchmark'])}: **{n['decision']}** (K={n['batches']}, {n['ended_by']})"
                    for n in nrows)
    L.append(f"| {C['pos']} `{C['commit'][:10]}` | {count['threshold']} / {count['ttest']} / {count['bootstrap']} "
             f"of {len(rows)} | {dec} | {sum(n['extra_blocks'] for n in nrows)} | "
             f"{m(jobs[1]['total'])} (record) + {m(jobs[2]['total'])} (forced NASC) |")
    L += ["", "### Runner minutes per job (from the job's first step)", "",
          "| Job | Set-up + build | Measuring | Total |", "|---|---|---|---|"]
    for j in jobs:
        L.append(f"| {j['job']} | {m(j['setup_build'])} | {m(j['measure'])} | {m(j['total'])} |")
    L.append(f"| **all jobs** | | | **{m(sum(j['total'] or 0 for j in jobs))}** |")
    L += ["", f"### Base detectors, commit {C['pos']} against commit {P['pos']}", "",
          f"| Benchmark | Commit {P['pos']} ns/op | Commit {C['pos']} ns/op | Change | Threshold | t-test (t) | "
          "Bootstrap (lower end) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = r["flags"]
        flag = lambda d, v: ("🚩 " if f[d]["flagged"] else "") + v  # noqa: E731
        t_value = "{:.2f}".format(f["ttest"]["statistic"])
        lower = "{:.4f}".format(f["bootstrap"]["statistic"])
        L.append(f"| `{short(r['benchmark'])}` | {r['parent_mean']:.5g} | {r['mean']:.5g} | "
                 f"{spct(f['threshold']['change'])} | {flag('threshold', 'flag' if f['threshold']['flagged'] else 'no')} | "
                 f"{flag('ttest', t_value)} | {flag('bootstrap', lower)} |")
    L += ["", "### Forced NASC (candidate and parent re-measured in the same job)", "",
          "| Benchmark | n_b | Batches K | Ended by | Decision | Change in re-runs | Extra blocks | "
          "Fresh parent vs stored | Fresh candidate vs stored | Minutes |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in nrows:
        L.append(f"| `{short(n['benchmark'])}` | {n['n_b']} | {n['batches']} | {n['ended_by']} | {n['decision']} | "
                 f"{spct(n['change'])} | {n['extra_blocks']} | {spct(n['fresh_parent_vs_stored'])} | "
                 f"{spct(n['fresh_candidate_vs_stored'])} | {n['minutes']:.1f} |")
    L += ["", "LLR after each batch: " + "; ".join(
        f"{short(n['benchmark'])}: " + ", ".join(f"{x:+.2f}" for x in n["llr"]) for n in nrows)
        + " (alert at +2.94, dismiss at −2.94)."]
    L += ["", "### Noise: this runner next to the reference profile (Laaber et al. traces)", "",
          "c_w = within-run noise of one block, c_t = between-run noise. Runner values come from 3 forks "
          "per commit (only 3 run pairs for c_t, so c_t is rough); n_b from equation 8 with these values.", "",
          f"| Benchmark | Reference c_w / c_t (n_b) | Runner commit {P['pos']} c_w / c_t | "
          f"Runner commit {C['pos']} c_w / c_t | Runner both c_w / c_t (n_b) |", "|---|---|---|---|---|"]
    for x in S["noise"]:
        r, a, b, both = x["reference"], x["runner_parent"], x["runner_candidate"], x["runner_both"]
        nb = lambda n: f"{n['n_b']}" if n["feasible"] else "infeasible"  # noqa: E731
        cw = lambda n: f"{pct(n['c_w'])} / {pct(n['c_t'])}" if n["estimated"] else "n/a (too few blocks)"  # noqa: E731
        L.append(f"| `{short(x['benchmark'])}` | {cw(r)} ({nb(r)}) | {cw(a)} | {cw(b)} | "
                 f"{cw(both)}{f' ({nb(both)})' if both['estimated'] else ''} |")
    L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=PILOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regex")
    r = sub.add_parser("record")
    r.add_argument("--json", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--pos", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--forks", type=int, default=3)
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--iterations", type=int, default=20)
    r.add_argument("--time", default="1 s", help="JMH's iteration time as the JSON writes it")
    c = sub.add_parser("confirm")
    c.add_argument("--parent", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--parent-cp", required=True)
    c.add_argument("--candidate-cp", required=True)
    c.add_argument("--force", default="heapSort80,serialIndexOf250")
    c.add_argument("--out", required=True)
    c.add_argument("--wi", type=int, default=10, help="warm-up iterations of a re-run")
    c.add_argument("--time", default="1s", help="iteration time of a re-run (JMH -w / -r)")
    c.add_argument("--parent-timing")
    c.add_argument("--candidate-timing")
    c.add_argument("--timing")
    a = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"regex": cmd_regex, "record": cmd_record, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
