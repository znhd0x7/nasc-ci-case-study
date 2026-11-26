"""full study v1 (26 Nov 2026): the NASC-CI case study on 15 jenetics commits.

Step 1 of 4 (26 Nov 2026): the settings, the injections per commit, recording one commit and
freezing the noise profile. Step 2 (26 Nov 2026): confirm, one benchmark over commits 6-15.
Steps 3-4 add the scorer and report, and the workflow.

  python3 case-study/study.py regex
      the JMH regular expression that selects exactly the 10 benchmarks of study_settings.json
  python3 case-study/study.py patches --pos K [--without-new]
      the injection patches applied to commit K (they persist from their first commit on, see
      injections/targets.txt), space-separated, in the order they are applied; --without-new
      leaves out a patch that STARTS at K (the clean side of the same-VM ground truth)
  python3 case-study/study.py new-patch --pos K
      the patch that starts at commit K, or nothing (commits 9 and 13 build an extra clean variant)
  python3 case-study/study.py record --json raw.json --pos K --commit SHA --patches "heapsort" \
      --machine machine.json --code code.json --timing timing.txt --out DIR
      checks one recording (every benchmark x 3 forks x 20 iterations, sample mode, ns/op) and
      writes results.json (block means per run, machine, code hash, patches with their sha256)
      and iteration_means.csv; exits 1 with the reasons if anything is missing or different
  python3 case-study/study.py pack --classpath FILE --out DIR
      copies the jars of a class path into DIR and writes DIR/classpath.txt (names relative to DIR),
      so a later job on another VM can run the same build
  python3 case-study/study.py classpath --dir DIR
      prints the class path of a packed folder, as absolute paths
  python3 case-study/study.py check-records --records DIR
      the completeness gate: every commit 1-15 recorded exactly once, complete, with the right
      commit and patches; prints the problems and exits 1 if there are any
  python3 case-study/study.py profile --records DIR --out DIR
      after the gate: the history of the 15 recordings and the runner noise profile frozen from
      commits 1-5 (NASC-CI 0.3 freeze_profile: c_w, c_t, n_b, feasibility, the CPU of each of
      those five VMs and whether they mix CPU types), next to the reference profile
  python3 case-study/study.py confirm --history DIR/history.json --bench heapSort80 \
      --builds DIR --machine machine.json --out DIR
      one benchmark, commits 6-15 in order, as nasc_sim.check_series walks a series: for every
      detector (threshold, t-test, bootstrap) and arm (none, static, static-mid, NASC) its own
      window and baseline; when any confirming arm flags commit k, ONE confirmation of k against
      k-1 on this VM: always K_max = 5 batches of n_b blocks per side in ABBA order, from the packed
      builds DIR/c<k> and DIR/c<k-1>. NASC decides at its own stopping point (fresh scale, level
      rule, cap), static and static-mid from all 5 batches (scale: the mean of the 5 reference
      runs; static-mid also needs a mean slowdown >= delta_min / 2). An accepted alert at or after
      the injection's first commit restarts that detector-arm's history there. Writes confirm.json
      (every check, every flag, every batch, the series per detector and arm) and the raw JMH files.
  python3 case-study/study.py ground-truth --bench heapSort80
      "<first commit> <patch>" if the benchmark has an injection (its confirm job then runs the
      same-VM A/B with ab_check.py), nothing otherwise
Needs numpy. Run from the repository root.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from statistics import NormalDist
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nasc_ci import VERSION, check, detectors, history, jmh, machine, noise  # noqa: E402
from nasc_ci import confirm as nasc  # noqa: E402

HERE = Path(__file__).resolve().parent
SETTINGS = HERE / "study_settings.json"
COMMITS = HERE / "commits.txt"
TARGETS = HERE / "injections" / "targets.txt"
INJECTIONS = HERE / "injections"
BENCHMARKS = HERE / "benchmarks.txt"
PROFILE = HERE / "reference_noise_profile.csv"


def settings():
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


STUDY = settings()["study"]


def short(name):
    return name.split(".")[-1]


def source_hash():
    f = ROOT / "nasc_ci" / "VERSION"
    return f.read_text().strip() if f.exists() else "unknown"


def lines(path):
    return [l.split() for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def commits():
    """{position: full sha} from commits.txt."""
    return {int(r[0]): r[1] for r in lines(COMMITS)}


def injections():
    """[(name, first position, [short benchmark names])] in the order of targets.txt."""
    return [(r[0], int(r[1]), r[2].split(",")) for r in lines(TARGETS)]


def patches_at(pos, without_new=False):
    """The patches applied to commit pos, ordered by their first commit (then by name)."""
    rows = sorted(injections(), key=lambda r: (r[1], r[0]))
    return [n for n, first, _ in rows if first <= pos and not (without_new and first == pos)]


def new_patch(pos):
    return next((n for n, first, _ in injections() if first == pos), None)


def step_of(bench):
    """(patch name, first position) of the injection that slows this benchmark down, or None."""
    for n, first, targets in injections():
        if short(bench) in targets:
            return n, first
    return None


def patch_sha(name):
    return hashlib.sha256((INJECTIONS / f"{name}.patch").read_bytes()).hexdigest()


def regex(names):
    return "^(" + "|".join(re.escape(b) for b in names) + ")$"


def read_timing(path):
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = float(v)
    return out


def error(title, problems):
    print(f"::error title={title}::" + "; ".join(problems[:5]))
    for p in problems:
        print("  -", p)
    sys.exit(1)


# ---------------------------------------------------------------- regex / patches
def cmd_regex(a):
    print(regex(settings()["benchmarks"]))


def cmd_patches(a):
    print(" ".join(patches_at(a.pos, a.without_new)))


def cmd_new_patch(a):
    print(new_patch(a.pos) or "")


# ---------------------------------------------------------------- record
def check_jmh(data, S):
    """Problems of one recording's JMH JSON (empty list = complete and as configured)."""
    r = S["jmh_record"]
    seconds = r["time"].replace("s", " s")             # JMH writes "1 s" for -w 1s
    by_name = {e["benchmark"]: e for e in data}
    problems = []
    for b in S["benchmarks"]:
        e = by_name.get(b)
        if e is None:
            problems.append(f"{short(b)}: missing")
            continue
        want = {"mode": "sample", "forks": r["forks"], "warmupIterations": r["warmup"],
                "measurementIterations": r["iterations"], "warmupTime": seconds, "measurementTime": seconds}
        for k, v in want.items():
            if e.get(k) != v:
                problems.append(f"{short(b)}: {k} = {e.get(k)!r}, expected {v!r}")
        m = e["primaryMetric"]
        if m.get("scoreUnit") != "ns/op":
            problems.append(f"{short(b)}: unit {m.get('scoreUnit')!r}, expected 'ns/op'")
        hist = m.get("rawDataHistogram")
        if not hist or len(hist) != r["forks"] or any(len(f) != r["iterations"] for f in hist):
            problems.append(f"{short(b)}: expected {r['forks']} forks x {r['iterations']} iterations of histograms")
    extra = sorted(set(by_name) - set(S["benchmarks"]))
    if extra:
        problems.append("unexpected benchmarks: " + ", ".join(short(b) for b in extra))
    return problems


def cmd_record(a):
    S = settings()
    sha = commits().get(a.pos)
    problems = []
    if sha is None:
        problems.append(f"position {a.pos} is not in commits.txt")
    elif sha != a.commit:
        problems.append(f"commit {a.commit[:10]} is not position {a.pos} ({sha[:10]})")
    applied = a.patches.split()
    if applied != patches_at(a.pos):
        problems.append(f"patches applied {applied or 'none'}, expected {patches_at(a.pos) or 'none'}")
    data = json.loads(Path(a.json).read_text(encoding="utf-8"))
    problems += check_jmh(data, S)
    if problems:
        error("Recording incomplete", problems)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by_name = {e["benchmark"]: e for e in data}
    with open(out / "iteration_means.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["commit", "benchmark", "params", "instance", "trial", "fork", "iteration",
                    "n_invocations", "mean"])
        for b in S["benchmarks"]:
            for fi, fork in enumerate(by_name[b]["primaryMetric"]["rawDataHistogram"], start=1):
                for ii, it in enumerate(fork, start=1):
                    w.writerow([a.commit, b, "", "", 1, fi, ii, int(sum(c for _, c in it)),
                                repr(jmh.histogram_mean(it))])
    results = jmh.read(a.json)
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p else None  # noqa: E731
    record = {"study": STUDY, "tool": VERSION, "nasc_ci_source": source_hash(), "pos": a.pos,
              "commit": a.commit, "label": f"c{a.pos:02d}",
              "patches": [{"name": n, "sha256": patch_sha(n)} for n in applied],
              "machine": load(a.machine), "code": load(a.code), "timing": read_timing(a.timing),
              "results": {b: results[b] for b in S["benchmarks"]}, "complete": True}
    (out / "results.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    cpu = machine.cpu_of(record["machine"]) or "unknown CPU"
    print(f"{STUDY}; {VERSION}; source {source_hash()}")
    print(f"commit {a.pos} {a.commit[:10]} ({', '.join(applied) or 'no patch'}), {cpu}: "
          f"{len(S['benchmarks'])} benchmarks x {S['jmh_record']['forks']} forks x "
          f"{S['jmh_record']['iterations']} iterations recorded")
    print(f"{'benchmark':<20} {'mean ns/op':>12}  run means")
    for b in S["benchmarks"]:
        runs = np.array(results[b]["runs"])
        print(f"{short(b):<20} {runs.mean():>12.5g}  " + " ".join(f"{x:.5g}" for x in runs.mean(axis=1)))


# ---------------------------------------------------------------- pack / classpath
def cmd_pack(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for i, entry in enumerate(p for p in Path(a.classpath).read_text().strip().split(":") if p):
        src = Path(entry)
        if not src.exists():
            error("Class path entry missing", [entry])
        name = f"{i:03d}-{src.name}"
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, out / name)
        names.append(name)
    (out / "classpath.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"packed {len(names)} class path entries into {out}")


def cmd_classpath(a):
    d = Path(a.dir).resolve()
    names = [n for n in (d / "classpath.txt").read_text(encoding="utf-8").split() if n]
    missing = [n for n in names if not (d / n).exists()]
    if missing:
        error("Packed class path incomplete", missing)
    print(":".join(str(d / n) for n in names))


# ---------------------------------------------------------------- gate and profile
def load_records(folder):
    """{position: record} of every */results.json under folder, and the problems found."""
    S = settings()
    sha = commits()
    first, last = S["positions"]
    found, problems = {}, []
    for f in sorted(Path(folder).glob("*/results.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"{f.parent.name}: unreadable results.json ({e})")
            continue
        pos = r.get("pos")
        if pos in found:
            problems.append(f"commit {pos} recorded twice ({f.parent.name})")
            continue
        found[pos] = r
        if not r.get("complete"):
            problems.append(f"commit {pos}: recording not marked complete")
        if r.get("commit") != sha.get(pos):
            problems.append(f"commit {pos}: sha {str(r.get('commit'))[:10]} is not the one in commits.txt")
        if [p["name"] for p in r.get("patches", [])] != patches_at(pos):
            problems.append(f"commit {pos}: patches {[p['name'] for p in r.get('patches', [])]}, "
                            f"expected {patches_at(pos)}")
        for p in r.get("patches", []):
            if p["sha256"] != patch_sha(p["name"]):
                problems.append(f"commit {pos}: {p['name']}.patch differs from this repository's")
        missing = [short(b) for b in S["benchmarks"] if b not in r.get("results", {})]
        if missing:
            problems.append(f"commit {pos}: benchmarks missing: {', '.join(missing)}")
        for b in S["benchmarks"]:
            runs = r.get("results", {}).get(b, {}).get("runs")
            if runs is not None and (len(runs) != S["jmh_record"]["forks"]
                                     or any(len(x) != S["jmh_record"]["iterations"] // jmh.BLOCK_SIZE for x in runs)):
                problems.append(f"commit {pos}: {short(b)} has not {S['jmh_record']['forks']} runs of "
                                f"{S['jmh_record']['iterations'] // jmh.BLOCK_SIZE} blocks")
    absent = [p for p in range(first, last + 1) if p not in found]
    if absent:
        problems.insert(0, "commits not recorded (job failed or artifact missing): "
                        + ", ".join(map(str, absent)))
    return found, problems


def cmd_check_records(a):
    found, problems = load_records(a.records)
    if problems:
        error("Recordings incomplete", problems)
    print(f"all {len(found)} recordings complete")


def build_history(found, S):
    h = history.new(S["nasc"])
    for pos in sorted(found):
        r = found[pos]
        h["versions"].append({"index": pos, "commit": r["commit"], "label": r["label"],
                              "time": history.now(), "results": r["results"], "machine": r.get("machine"),
                              "code": r.get("code"), "patches": [p["name"] for p in r.get("patches", [])]})
    return h


def pct(x, d=2):
    return "–" if x is None else f"{x * 100:.{d}f}%"


def cmd_profile(a):
    S = settings()
    found, problems = load_records(a.records)
    if problems:
        error("Recordings incomplete", problems)
    h = build_history(found, S)
    first, last = S["history_commits"]
    n = last - first + 1
    h["profile"] = check.freeze_profile(h, n)
    h["study"] = STUDY
    h["nasc_ci_source"] = source_hash()
    reference = noise.load_profile(PROFILE, h["settings"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    history.save(h, str(out / "history.json"))
    prof = h["profile"]
    summary = {"study": STUDY, "tool": VERSION, "nasc_ci_source": source_hash(),
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "profile": prof,
               "reference": {b: reference[b].to_dict() for b in S["benchmarks"] if b in reference},
               "record_cpus": {v["index"]: machine.cpu_of(v.get("machine")) for v in h["versions"]},
               "code": {v["index"]: (v.get("code") or {}).get("sha256") for v in h["versions"]}}
    (out / "profile.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    text = profile_report(summary, S)
    (out / "profile.md").write_text(text, encoding="utf-8")
    print(text)


def profile_report(P, S):
    prof = P["profile"]
    kinds = "MIXED CPU types" if prof["mixed_cpu"] else "one CPU type"
    L = [f"## Runner noise profile, frozen from commits {prof['versions'][0]}-{prof['versions'][-1]}", "",
         f"`{P['tool']}`, source `{P['nasc_ci_source']}`; {STUDY}.", "",
         f"The five history VMs: {kinds} ({', '.join(str(c) for c in prof['cpus'])}).", "",
         "| Benchmark | Runner c_w / c_t | n_b | Feasible at 2 % | Smallest confirmable | Reference c_w / c_t |",
         "|---|---|---|---|---|---|"]
    for b in S["benchmarks"]:
        n = prof["benchmarks"][b]
        r = P["reference"].get(b)
        ref = f"{pct(r['c_w'])} / {pct(r['c_t'])}" if r else "–"
        L.append(f"| `{short(b)}` | {pct(n['c_w'])} / {pct(n['c_t'])} | {n['n_b']} | "
                 f"{'yes' if n['feasible'] else 'no'} | {pct(n['delta_feasible'], 1)} | {ref} |")
    L += ["", "CPU of every recording VM: " + "; ".join(
        f"{k}: {v or 'unknown'}" for k, v in sorted(P["record_cpus"].items(), key=lambda kv: int(kv[0]))), ""]
    return "\n".join(L)


# ---------------------------------------------------------------- confirm (step 2)
CONFIRMING = ("static", "static-mid", "nasc")


def full_name(bench_short, S):
    hits = [b for b in S["benchmarks"] if short(b) == bench_short]
    if len(hits) != 1:
        error("Unknown benchmark", [f"{bench_short}: {len(hits)} matches in study_settings.json"])
    return hits[0]


def cmd_ground_truth(a):
    step = step_of(full_name(a.bench, settings()))
    print("" if step is None else f"{step[1]} {step[0]}")


class JmhRunner:
    """One fresh JMH run (one JVM fork) of one benchmark from a packed build of commit pos."""

    def __init__(self, bench, builds, out, wi, time_):
        self.bench, self.builds, self.out, self.wi, self.time = bench, Path(builds), Path(out), wi, time_

    def classpath(self, pos):
        d = (self.builds / f"c{pos:02d}").resolve()
        names = [n for n in (d / "classpath.txt").read_text(encoding="utf-8").split() if n]
        return ":".join(str(d / n) for n in names)

    def __call__(self, pos, n_blocks, tag):
        stem = self.out / "runs" / tag
        stem.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["java", "-cp", self.classpath(pos), "org.openjdk.jmh.Main", "^" + re.escape(self.bench) + "$",
               "-bm", "sample", "-tu", "ns", "-f", "1", "-wi", str(self.wi), "-i", str(n_blocks * jmh.BLOCK_SIZE),
               "-w", self.time, "-r", self.time, "-foe", "true", "-rf", "json", "-rff", str(stem) + ".json"]
        with open(str(stem) + ".log", "w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs = jmh.read(str(stem) + ".json")[self.bench]["runs"]
        if len(runs) != 1 or len(runs[0]) != n_blocks:
            raise RuntimeError(f"{stem}.json: expected one fork of {n_blocks} blocks")
        return [float(x) for x in runs[0]], str(stem.relative_to(self.out)) + ".json"


def measure_batches(run, cand_pos, ref_pos, n_b, k_max, order):
    """K_max batches of candidate (commit cand_pos) and reference (ref_pos) runs, n_b blocks each,
    in NASC-CI's batch order. -> list of {"batch", "order", "candidate", "reference", ...}."""
    out = []
    for j in range(1, k_max + 1):
        o = nasc.batch_order(j, order)
        sides = ("candidate", "reference") if o == nasc.CAND_FIRST else ("reference", "candidate")
        rec = {"batch": j, "order": o}
        for side in sides:
            pos = cand_pos if side == "candidate" else ref_pos
            t0 = time.time()
            blocks, file = run(pos, n_b, f"c{cand_pos:02d}/batch{j}-{side}")
            rec[side] = float(np.mean(blocks))
            rec[side + "_blocks"] = blocks
            rec[side + "_file"] = file
            rec[side + "_seconds"] = round(time.time() - t0, 1)
        out.append(rec)
        print(f"   commit {cand_pos} vs {ref_pos}, batch {j} ({o}): candidate {rec['candidate']:.5g}, "
              f"reference {rec['reference']:.5g} ns/op", flush=True)
    return out


def nasc_decision(batches, prof, s):
    """NASC-CI's nasc_confirm on the measured batches, read in the order they were measured; it
    stops at its own boundary, so later batches do not change its decision."""
    queue = {"candidate": [b["candidate"] for b in batches], "reference": [b["reference"] for b in batches]}
    used = {"candidate": 0, "reference": 0}

    def measure(side, n_blocks):
        used[side] += 1
        return queue[side][used[side] - 1]
    return nasc.nasc_confirm(measure, prof, float(np.mean(queue["reference"])), s).to_dict()


def static_decisions(batches, prof, s):
    """static (nasc_sim.repeat_confirm with n_blocks = n_b and the cap) and static-mid (src/34
    static_mid) on all K_max batches. Scale: the mean of the reference runs of this confirmation."""
    cand = np.array([b["candidate"] for b in batches])
    ref = np.array([b["reference"] for b in batches])
    k = len(batches)
    level = float(ref.mean())
    s_b = math.sqrt(prof.c_t ** 2 + prof.c_w ** 2 / prof.n_b)
    total = float(np.clip((cand - ref) / (level * s_b), -s["cap"], s["cap"]).sum())
    bound = NormalDist().inv_cdf(1 - s["alpha"]) * math.sqrt(2 * k)
    slowdown = float(((cand - ref) / level).mean())
    significant = total >= bound
    return {"static": {"confirmed": bool(significant), "total": total, "bound": bound, "scale": level,
                       "s_b": s_b, "batches": k},
            "static-mid": {"confirmed": bool(significant and slowdown >= s["delta_min"] / 2), "total": total,
                           "bound": bound, "slowdown": slowdown, "scale": level, "s_b": s_b, "batches": k}}


def run_confirm(h, bench, run, machine_info=None, seed=1):
    """Walks commits 6-15 for one benchmark. h: the profile job's history (versions 1-15, frozen
    profile). run(pos, n_blocks, tag) -> (block means, file) measures one fresh run."""
    S = settings()
    s = h["settings"]
    first, last = S["checked_commits"]
    w = s["window"]
    versions = {v["index"]: v for v in h["versions"]}
    runs = {p: np.array(v["results"][bench]["runs"], dtype=float) for p, v in versions.items()}
    prof = noise.Noise(**h["profile"]["benchmarks"][bench])
    step = step_of(bench)
    v_star = step[1] if step else None
    dets, arms = S["detectors"], S["arms"]
    start = {(d, a): 1 for d in dets for a in arms}             # first version each history may use
    series = {d: {a: [] for a in arms} for d in dets}
    checks = []
    bench_index = S["benchmarks"].index(bench)
    for k in range(first, last + 1):
        boot = detectors.detect("bootstrap", [runs[k - 1]], runs[k],
                                np.random.default_rng([seed, bench_index, k]), s)   # parent only: same for every arm
        flags = {}
        for d in dets:
            for a in arms:
                window = list(range(max(start[(d, a)], k - w), k))
                f = boot if d == "bootstrap" else detectors.detect(d, [runs[p] for p in window], runs[k], None, s)
                flags[(d, a)] = {"window": [window[0], window[-1]], "flagged": bool(f.flagged),
                                 "change": float(f.change), "statistic": float(f.statistic)}
        need = any(flags[(d, a)]["flagged"] for d in dets for a in CONFIRMING)
        conf = None
        if need:
            t0 = time.time()
            batches = measure_batches(run, k, k - 1, prof.n_b, s["k_max"], s.get("order", "abba"))
            n = nasc_decision(batches, prof, s)
            st = static_decisions(batches, prof, s)
            d_rel = np.array([b["candidate"] / b["reference"] - 1 for b in batches])
            conf = {"candidate": k, "reference": k - 1, "n_b": prof.n_b, "batches": batches,
                    "nasc": n, "static": st["static"], "static-mid": st["static-mid"],
                    "pair_sd": float(d_rel.std(ddof=1)), "pair_mean": float(d_rel.mean()),
                    "minutes": (time.time() - t0) / 60}
        decided = {"nasc": conf and conf["nasc"]["confirmed"], "static": conf and conf["static"]["confirmed"],
                   "static-mid": conf and conf["static-mid"]["confirmed"]}
        cost = {"none": 0} if conf is None else {
            "none": 0, "static": 2 * s["k_max"] * prof.n_b, "static-mid": 2 * s["k_max"] * prof.n_b,
            "nasc": conf["nasc"]["extra_blocks"]}
        for d in dets:
            for a in arms:
                f = flags[(d, a)]
                alerted = f["flagged"] if a == "none" else bool(f["flagged"] and decided[a])
                blocks = cost.get(a, 0) if f["flagged"] else 0
                f.update(alerted=alerted, blocks=blocks, start=start[(d, a)])
                series[d][a].append([k, f["flagged"], alerted, blocks])
                if alerted and v_star is not None and k >= v_star:
                    start[(d, a)] = k                          # accepted: new baseline from here
        checks.append({"pos": k, "parent": k - 1,
                       "record_cpus": [machine.cpu_of(versions[k - 1].get("machine")),
                                       machine.cpu_of(versions[k].get("machine"))],
                       "same_code_as_parent": (versions[k].get("code") or {}).get("sha256")
                       == (versions[k - 1].get("code") or {}).get("sha256"),
                       "flags": {d: {a: flags[(d, a)] for a in arms} for d in dets},
                       "confirmation": conf})
        n_flag = sum(flags[(d, "none")]["flagged"] for d in dets)
        print(f"commit {k}: {n_flag} of {len(dets)} detectors flag (arm none)"
              + ("" if conf is None else f"; confirmation: NASC {'alert' if conf['nasc']['confirmed'] else 'dismiss'} "
                 f"after {conf['nasc']['batches']}, static {'alert' if conf['static']['confirmed'] else 'dismiss'}, "
                 f"static-mid {'alert' if conf['static-mid']['confirmed'] else 'dismiss'}"), flush=True)
    return {"study": STUDY, "tool": VERSION, "nasc_ci_source": source_hash(),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "benchmark": bench, "v_star": v_star, "patch": step[0] if step else None,
            "settings": s, "noise": prof.to_dict(), "machine": machine_info, "bootstrap_seed": seed,
            "checks": checks, "series": series, "complete": True}


def cmd_confirm(a):
    S = settings()
    bench = full_name(a.bench, S)
    h = history.load(a.history)
    if "profile" not in h or len(h["versions"]) != S["positions"][1]:
        error("History incomplete", ["the profile job's history.json must hold all 15 versions and the frozen profile"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p else None  # noqa: E731
    run = JmhRunner(bench, a.builds, out, S["jmh_rerun"]["warmup"], S["jmh_rerun"]["time"])
    print(f"{STUDY}; {VERSION}; source {source_hash()}; confirm {short(bench)}", flush=True)
    result = run_confirm(h, bench, run, load(a.machine), S["bootstrap_seed"])
    result["timing"] = read_timing(a.timing)
    (out / "confirm.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"confirm.json written: {sum(c['confirmation'] is not None for c in result['checks'])} confirmations")


def main(argv=None):
    p = argparse.ArgumentParser(description=STUDY)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regex")
    x = sub.add_parser("patches")
    x.add_argument("--pos", type=int, required=True)
    x.add_argument("--without-new", action="store_true")
    x = sub.add_parser("new-patch")
    x.add_argument("--pos", type=int, required=True)
    r = sub.add_parser("record")
    r.add_argument("--json", required=True)
    r.add_argument("--pos", type=int, required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--patches", default="")
    r.add_argument("--machine")
    r.add_argument("--code")
    r.add_argument("--timing")
    r.add_argument("--out", required=True)
    x = sub.add_parser("pack")
    x.add_argument("--classpath", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("classpath")
    x.add_argument("--dir", required=True)
    x = sub.add_parser("check-records")
    x.add_argument("--records", required=True)
    x = sub.add_parser("profile")
    x.add_argument("--records", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("confirm")
    x.add_argument("--history", required=True)
    x.add_argument("--bench", required=True, help="short name, e.g. heapSort80")
    x.add_argument("--builds", required=True, help="folder with the packed builds c05 ... c15")
    x.add_argument("--machine")
    x.add_argument("--timing")
    x.add_argument("--out", required=True)
    x = sub.add_parser("ground-truth")
    x.add_argument("--bench", required=True)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"regex": cmd_regex, "patches": cmd_patches, "new-patch": cmd_new_patch, "record": cmd_record,
     "pack": cmd_pack, "classpath": cmd_classpath, "check-records": cmd_check_records,
     "profile": cmd_profile, "confirm": cmd_confirm, "ground-truth": cmd_ground_truth}[a.cmd](a)


if __name__ == "__main__":
    main()
