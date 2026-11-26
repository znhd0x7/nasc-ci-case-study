"""full study v1 (26 Nov 2026): the NASC-CI case study on 15 jenetics commits.

Step 1 of 4 (26 Nov 2026): the settings, the injections per commit, recording one commit and
freezing the noise profile. Step 2 (26 Nov 2026): confirm, one benchmark over commits 6-15.
Step 3 (26 Nov 2026): the scorer (nasc_sim.score_series' rules) and the report. Step 4 adds the
workflow.

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
  python3 case-study/study.py report --records DIR --profile DIR --confirms DIR --ground-truth DIR --out DIR
      the completeness gate over everything (15 recordings, the profile, 10 confirm.json, the same-VM
      A/B of the 4 injected benchmarks); if anything is missing: report.md starts with INCOMPLETE,
      names it, and the command exits 1 (no scores are written). Otherwise every detector-arm series
      is scored with score_series' rules (false alerts only before v*, the first alert at or after
      v* is the detection, later ones duplicates) and pooled as summarize_scores pools them;
      report.md and study.json give the scores, every confirmation, the same-VM ground truth per
      injected benchmark (with the calibration of that CPU type as background), the observed sd of
      the confirmation pairs against the frozen profile's sqrt(2) s_b (one-sided chi-square, 5 %),
      base-flag rates for record pairs on the same and on mixed CPU types, the CPU mix of the
      frozen profile, and the limitation (4 injected instances among 100 checks).
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


def nasc_decision(batches, prof, s, ref_level=None):
    """NASC-CI's nasc_confirm on the measured batches, read in the order they were measured; it
    stops at its own boundary, so later batches do not change its decision. ref_level is used only
    with s["scale"] == "stored" (the equivalence test); the study uses "fresh"."""
    queue = {"candidate": [b["candidate"] for b in batches], "reference": [b["reference"] for b in batches]}
    used = {"candidate": 0, "reference": 0}

    def measure(side, n_blocks):
        used[side] += 1
        return queue[side][used[side] - 1]
    level = float(np.mean(queue["reference"])) if ref_level is None else float(ref_level)
    return nasc.nasc_confirm(measure, prof, level, s).to_dict()


def static_decisions(batches, prof, s, scale=None):
    """static (nasc_sim.repeat_confirm with n_blocks = n_b and the cap) and static-mid (src/34
    static_mid) on all K_max batches. Scale: the mean of the reference runs of this confirmation
    (scale=None, the study), or the level given (the equivalence test passes nasc_sim's m_b)."""
    cand = np.array([b["candidate"] for b in batches])
    ref = np.array([b["reference"] for b in batches])
    k = len(batches)
    level = float(ref.mean()) if scale is None else float(scale)
    s_b = math.sqrt(prof.c_t ** 2 + prof.c_w ** 2 / prof.n_b)
    total = 0.0
    for c, r in zip(cand, ref):                                  # summed batch by batch, as nasc_sim
        total += float(np.clip((c - r) / (level * s_b), -s["cap"], s["cap"]))
    bound = NormalDist().inv_cdf(1 - s["alpha"]) * np.sqrt(2 * k)
    slowdown = 0.0
    for c, r in zip(cand, ref):                                  # as src/34 static_mid accumulates it
        slowdown += (c - r) / level / k
    significant = bool(total >= bound)
    return {"static": {"confirmed": significant, "total": total, "bound": float(bound), "scale": level,
                       "s_b": s_b, "batches": k},
            "static-mid": {"confirmed": bool(significant and slowdown >= s["delta_min"] / 2), "total": total,
                           "bound": float(bound), "slowdown": float(slowdown), "scale": level, "s_b": s_b,
                           "batches": k}}


def walk_series(runs, s, dets, arms, v_star, first, last, decide, seed=1, bench_index=0):
    """The walk of nasc_sim.check_series for every detector-arm at once. runs: {pos: (runs, blocks)};
    decide(k) -> the confirmation of k against k-1 (a dict with "n_b" and "nasc", "static",
    "static-mid", each with "confirmed"; "nasc" also "extra_blocks"), asked at most once per commit
    and only when a confirming arm flags k. -> ({pos: {(d, a): flag}}, {pos: confirmation}, series)."""
    w = s["window"]
    start = {(d, a): 1 for d in dets for a in arms}             # first version each history may use
    series = {d: {a: [] for a in arms} for d in dets}
    all_flags, confirmations = {}, {}
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
        need = any(flags[(d, a)]["flagged"] for d in dets for a in arms if a in CONFIRMING)
        conf = decide(k) if need else None
        cost = {} if conf is None else {"static": 2 * s["k_max"] * conf["n_b"],
                                        "static-mid": 2 * s["k_max"] * conf["n_b"],
                                        "nasc": conf["nasc"]["extra_blocks"]}
        for d in dets:
            for a in arms:
                f = flags[(d, a)]
                alerted = f["flagged"] if a == "none" else bool(f["flagged"] and conf[a]["confirmed"])
                blocks = cost.get(a, 0) if f["flagged"] else 0
                f.update(alerted=alerted, blocks=blocks, start=start[(d, a)])
                series[d][a].append([k, f["flagged"], alerted, blocks])
                if alerted and v_star is not None and k >= v_star:
                    start[(d, a)] = k                          # accepted: new baseline from here
        all_flags[k], confirmations[k] = flags, conf
    return all_flags, confirmations, series


def run_confirm(h, bench, run, machine_info=None, seed=1):
    """Walks commits 6-15 for one benchmark. h: the profile job's history (versions 1-15, frozen
    profile). run(pos, n_blocks, tag) -> (block means, file) measures one fresh run."""
    S = settings()
    s = h["settings"]
    first, last = S["checked_commits"]
    versions = {v["index"]: v for v in h["versions"]}
    runs = {p: np.array(v["results"][bench]["runs"], dtype=float) for p, v in versions.items()}
    prof = noise.Noise(**h["profile"]["benchmarks"][bench])
    step = step_of(bench)
    v_star = step[1] if step else None
    dets, arms = S["detectors"], S["arms"]

    def decide(k):
        t0 = time.time()
        batches = measure_batches(run, k, k - 1, prof.n_b, s["k_max"], s.get("order", "abba"))
        st = static_decisions(batches, prof, s)
        d_rel = np.array([b["candidate"] / b["reference"] - 1 for b in batches])
        conf = {"candidate": k, "reference": k - 1, "n_b": prof.n_b, "batches": batches,
                "nasc": nasc_decision(batches, prof, s), "static": st["static"], "static-mid": st["static-mid"],
                "pair_sd": float(d_rel.std(ddof=1)), "pair_mean": float(d_rel.mean()),
                "minutes": (time.time() - t0) / 60}
        print(f"commit {k}: confirmation: NASC {'alert' if conf['nasc']['confirmed'] else 'dismiss'} after "
              f"{conf['nasc']['batches']}, static {'alert' if conf['static']['confirmed'] else 'dismiss'}, "
              f"static-mid {'alert' if conf['static-mid']['confirmed'] else 'dismiss'}", flush=True)
        return conf

    flags, confs, series = walk_series(runs, s, dets, arms, v_star, first, last, decide, seed,
                                       S["benchmarks"].index(bench))
    checks = []
    for k in range(first, last + 1):
        checks.append({"pos": k, "parent": k - 1,
                       "record_cpus": [machine.cpu_of(versions[k - 1].get("machine")),
                                       machine.cpu_of(versions[k].get("machine"))],
                       "same_code_as_parent": (versions[k].get("code") or {}).get("sha256")
                       == (versions[k - 1].get("code") or {}).get("sha256"),
                       "flags": {d: {a: flags[k][(d, a)] for a in arms} for d in dets},
                       "confirmation": confs[k]})
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


# ---------------------------------------------------------------- scorer and report (step 3)
CALIBRATION = HERE / "calibration_background.json"
BLOCKS_PER_VERSION = 12                       # 3 runs x 4 blocks, as nasc_sim.BLOCKS_PER_VERSION
# one-sided 95 % quantiles of chi-square, df 1..30 (above 30: Wilson-Hilferty)
CHI2_95 = [3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026,
           22.362, 23.685, 24.996, 26.296, 27.587, 28.869, 30.144, 31.410, 32.671, 33.924, 35.172, 36.415,
           37.652, 38.885, 40.113, 41.337, 42.557, 43.773]


def chi2_95(df):
    if 1 <= df <= 30:
        return CHI2_95[df - 1]
    z = NormalDist().inv_cdf(0.95)
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def score(rows, v_star):
    """nasc_sim.score_series for one series. rows: [commit, flagged, alerted, blocks];
    v_star: the injection's first commit, None for a benchmark without a step (delta = 0)."""
    has_step = v_star is not None
    clean = [r for r in rows if not has_step or r[0] < v_star]
    after = [r for r in rows if has_step and r[0] >= v_star]
    hits = [r[0] for r in after if r[2]]
    return {"has_step": has_step, "clean_checks": len(clean), "clean_flags": sum(bool(r[1]) for r in clean),
            "false_alerts": sum(bool(r[2]) for r in clean),
            "flagged_at_v_star": any(r[1] for r in after if r[0] == v_star),
            "detected": len(hits) > 0, "delay": hits[0] - v_star if hits else None,
            "duplicates": max(len(hits) - 1, 0), "checked_versions": len(rows),
            "confirm_blocks": sum(r[3] for r in rows)}


def summarize(scores):
    """Pools the scores of several series as nasc_sim.summarize_scores (sums first, then ratios)."""
    steps = [x for x in scores if x["has_step"]]
    clean = sum(x["clean_checks"] for x in scores)
    fa = sum(x["false_alerts"] for x in scores)
    det = sum(x["detected"] for x in steps)
    recorded = BLOCKS_PER_VERSION * sum(x["checked_versions"] for x in scores)
    ratio = lambda a, b: a / b if b else None  # noqa: E731
    return {"series": len(scores), "clean_checks": clean, "clean_flags": sum(x["clean_flags"] for x in scores),
            "base_flag_rate": sum(x["clean_flags"] for x in scores) / max(clean, 1),   # max(.., 1) as nasc_sim
            "false_alerts": fa, "FPR": fa / max(clean, 1), "steps": len(steps), "detected": det,
            "base_hit_rate": ratio(sum(x["flagged_at_v_star"] for x in steps), len(steps)),
            "recall": ratio(det, len(steps)),
            "recall_at_v_star": ratio(sum(x["delay"] == 0 for x in steps), len(steps)),
            "recall_late": ratio(sum(x["delay"] is not None and x["delay"] > 0 for x in steps), len(steps)),
            "FDP": ratio(fa, fa + det) if steps else None,
            "duplicates": sum(x["duplicates"] for x in scores),
            "confirm_blocks": sum(x["confirm_blocks"] for x in scores),
            "T": (recorded + sum(x["confirm_blocks"] for x in scores)) / recorded if recorded else None}


def load_confirms(folder, S):
    found, problems = {}, []
    for f in sorted(Path(folder).glob("*/confirm.json")):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"{f.parent.name}: unreadable confirm.json ({e})")
            continue
        b = c.get("benchmark")
        if b in found:
            problems.append(f"{short(b)}: confirmed twice")
        elif not c.get("complete") or len(c.get("checks", [])) != S["checked_commits"][1] - S["checked_commits"][0] + 1:
            problems.append(f"{short(b)}: confirm.json incomplete")
        found[b] = c
    absent = [short(b) for b in S["benchmarks"] if b not in found]
    if absent:
        problems.insert(0, "benchmarks not confirmed (job failed or artifact missing): " + ", ".join(absent))
    return found, problems


def load_ground_truth(folder, S):
    found, problems = {}, []
    for f in sorted(Path(folder).glob("*/ab.json")):
        try:
            g = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"{f.parent.name}: unreadable ab.json ({e})")
            continue
        for r in g.get("results", []):
            found[r["benchmark"]] = {"difference": r["difference"], "ci95": r["ci95_welch"], "pairs": r["pairs"],
                                     "order_effect": r["order_effect"],
                                     "cpu": (g.get("machine") or {}).get("cpu_model"),
                                     "a": g.get("a"), "b": g.get("b"), "same_code": g.get("same_code")}
    absent = [short(b) for b in S["benchmarks"] if step_of(b) and b not in found]
    if absent:
        problems.insert(0, "same-VM ground truth missing (job failed or artifact missing): " + ", ".join(absent))
    return found, problems


def sd_check(conf_list, prof):
    """Observed sd of the relative pair differences c / r - 1 (pooled within confirmations) against
    the frozen profile's prediction sqrt(2) s_b; one-sided chi-square test at 5 %."""
    ss, df = 0.0, 0
    for c in conf_list:
        d = np.array([b["candidate"] / b["reference"] - 1 for b in c["batches"]])
        ss += float(((d - d.mean()) ** 2).sum())
        df += len(d) - 1
    if df == 0:
        return None
    s_b = math.sqrt(prof["c_t"] ** 2 + prof["c_w"] ** 2 / prof["n_b"])
    predicted = math.sqrt(2) * s_b
    observed = math.sqrt(ss / df)
    stat = ss / predicted ** 2
    return {"confirmations": len(conf_list), "df": df, "observed_sd": observed, "predicted_sd": predicted,
            "chi2": stat, "chi2_95": chi2_95(df), "higher": bool(stat > chi2_95(df))}


def cpu_kind(cpus):
    if any(c is None for c in cpus):
        return "unknown"
    return "same" if len(set(cpus)) == 1 else "mixed"


def cmd_report(a):
    S = settings()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records, problems = load_records(a.records)
    pfile = Path(a.profile) / "profile.json"
    if not pfile.exists():
        problems.append("profile.json missing (profile job failed or artifact missing)")
    confirms, p2 = load_confirms(a.confirms, S)
    truth, p3 = load_ground_truth(a.ground_truth, S)
    problems += p2 + p3
    if problems:
        text = "\n".join(["## INCOMPLETE: the full study is not complete; no scores are given", ""]
                         + [f"- {p}" for p in problems] + ["", "Re-run the failed jobs (Re-run failed jobs keeps "
                                                          "the jobs that succeeded), then the report again.", ""])
        (out / "report.md").write_text(text, encoding="utf-8")
        print(text)
        error("Full study incomplete", problems)
    P = json.loads(pfile.read_text(encoding="utf-8"))
    calib = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    dets, arms = S["detectors"], S["arms"]
    per_series = {(d, ar): {} for d in dets for ar in arms}
    for b, c in confirms.items():
        for d in dets:
            for ar in arms:
                per_series[(d, ar)][b] = score(c["series"][d][ar], c["v_star"])
    pooled = {f"{d} | {ar}": summarize(list(per_series[(d, ar)].values())) for d in dets for ar in arms}
    # NASC batches against static's K_max
    nasc_k = [c2["confirmation"]["nasc"]["batches"] for c in confirms.values() for c2 in c["checks"]
              if c2["confirmation"]]
    # confirmations, ground truth, sd check, CPU grouping
    conf_rows, sd_rows, gt_rows, cpu_groups = [], {}, [], {}
    for b in S["benchmarks"]:
        c = confirms[b]
        cpu_c = machine.cpu_of(c.get("machine"))
        cl = [x["confirmation"] for x in c["checks"] if x["confirmation"]]
        sd_rows[b] = sd_check(cl, c["noise"])
        for x in c["checks"]:
            kind = cpu_kind(x["record_cpus"])
            if c["v_star"] is None or x["pos"] < c["v_star"]:
                for d in dets:
                    g = cpu_groups.setdefault((d, kind), [0, 0])
                    g[0] += 1
                    g[1] += bool(x["flags"][d]["none"]["flagged"])
            cf = x["confirmation"]
            if cf:
                conf_rows.append({"benchmark": b, "pos": x["pos"], "record": kind, "record_cpus": x["record_cpus"],
                                  "confirm_cpu": cpu_c, "n_b": cf["n_b"], "pair_mean": cf["pair_mean"],
                                  "pair_sd": cf["pair_sd"], "nasc": cf["nasc"]["confirmed"],
                                  "nasc_k": cf["nasc"]["batches"], "nasc_end": cf["nasc"]["ended_by"],
                                  "static": cf["static"]["confirmed"], "static_mid": cf["static-mid"]["confirmed"],
                                  "same_code": x["same_code_as_parent"]})
        if c["v_star"] is not None:
            t = truth[b]
            bg = (calib.get(c["patch"]) or {}).get(t["cpu"] or "", {}).get(short(b))
            det = {f"{d} | {ar}": per_series[(d, ar)][b]["delay"] for d in dets for ar in arms}
            gt_rows.append({"benchmark": b, "v_star": c["v_star"], "patch": c["patch"], "confirm_cpu": cpu_c,
                            "truth": t, "background": bg, "delay": det,
                            "other_cpu_than_truth": cpu_c != t["cpu"]})
    checks = sum(x["checked_versions"] for v in per_series[(dets[0], "none")].values() for x in [v])
    clean = sum(x["clean_checks"] for x in per_series[(dets[0], "none")].values())
    at_v = sum(1 for c in confirms.values() if c["v_star"] is not None)
    summary = {"study": STUDY, "tool": VERSION, "nasc_ci_source": source_hash(),
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "settings": S, "pooled": pooled,
               "per_series": {f"{d} | {ar}": {short(b): v for b, v in per_series[(d, ar)].items()}
                              for d in dets for ar in arms},
               "nasc_batches": nasc_k, "confirmations": conf_rows,
               "sd_check": {short(b): v for b, v in sd_rows.items()}, "ground_truth": gt_rows,
               "base_flags_by_record_cpus": {f"{d} | {k}": v for (d, k), v in sorted(cpu_groups.items())},
               "profile_cpus": {k: P["profile"][k] for k in ("versions", "cpus", "cpu_types", "mixed_cpu")},
               "record_cpus": P["record_cpus"],
               "limitation": {"checks": checks, "clean": clean, "at_v_star": at_v, "after_v_star": checks - clean - at_v,
                              "instances": at_v}}
    (out / "study.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    text = study_report(summary)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


def study_report(R):
    S = R["settings"]
    p = lambda x, d=1: "–" if x is None else f"{x * 100:.{d}f}%"  # noqa: E731
    sp = lambda x, d=2: "–" if x is None else f"{x * 100:+.{d}f}%"  # noqa: E731
    lim = R["limitation"]
    L = ["## NASC-CI case study: full study", "",
         f"`{R['tool']}`, source `{R['nasc_ci_source']}`; {R['study']}.", "",
         f"**Limitation.** The study has {lim['instances']} injected regression instances among {lim['checks']} "
         f"checked benchmark-commits ({lim['clean']} clean checks, {lim['at_v_star']} at the injection's first "
         f"commit, {lim['after_v_star']} after it). It shows NASC-CI working in real CI; error rates and costs "
         "with statistical weight come from the simulation.", "",
         "### Scores per detector and arm (rules of nasc_sim.score_series, pooled as summarize_scores)", "",
         "| Detector | Arm | Clean checks | Base flags | False alerts (FPR) | Detected (of steps) | At v* / later | "
         "Duplicates | Extra blocks | T |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, x in R["pooled"].items():
        d, ar = name.split(" | ")
        L.append(f"| {d} | {ar} | {x['clean_checks']} | {x['clean_flags']} | {x['false_alerts']} ({p(x['FPR'])}) | "
                 f"{x['detected']} of {x['steps']} | {p(x['recall_at_v_star'], 0)} / {p(x['recall_late'], 0)} | "
                 f"{x['duplicates']} | {x['confirm_blocks']} | {x['T']:.3f} |")
    k = R["nasc_batches"]
    if k:
        L += ["", f"NASC used {np.mean(k):.2f} batches per confirmation on average (static and static-mid always "
                  f"{S['nasc']['k_max']}): {len(k)} confirmations, batches " + ", ".join(map(str, k)) + "."]
    L += ["", "### Same-VM ground truth of the injections (measured on the confirm job's VM)", "",
          "| Benchmark | Injection | Confirm VM | True size on this VM | Calibration on this CPU type | "
          "Delay per detector-arm (– = missed) |", "|---|---|---|---|---|---|"]
    for g in R["ground_truth"]:
        t, bg = g["truth"], g["background"]
        delays = "; ".join(f"{n.replace(' | ', ' ')}: {'–' if v is None else v}" for n, v in g["delay"].items())
        L.append(f"| `{short(g['benchmark'])}` | {g['patch']} from commit {g['v_star']} | {g['confirm_cpu']} | "
                 f"{sp(t['difference'])} ± {p(t['ci95'], 2)} ({t['pairs']} pairs) | "
                 f"{'not calibrated' if bg is None else f'{sp(bg[0])} ± {p(bg[1], 2)}'} | {delays} |")
    L += ["", "### Confirmations (each one: 5 batches of commit k against k-1 on the confirm VM)", "",
          "| Benchmark | Commit | Record VMs k-1 / k | Confirm VM | n_b | Pair mean (sd) | NASC | static | "
          "static-mid | Same code as k-1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda v: "alert" if v else "dismiss"  # noqa: E731
    for c in R["confirmations"]:
        L.append(f"| `{short(c['benchmark'])}` | {c['pos']} | {c['record']} | {c['confirm_cpu']} | {c['n_b']} | "
                 f"{sp(c['pair_mean'])} ({p(c['pair_sd'], 2)}) | {yn(c['nasc'])} (K={c['nasc_k']}) | "
                 f"{yn(c['static'])} | {yn(c['static_mid'])} | {'yes' if c['same_code'] else 'no'} |")
    L += ["", "### Noise of the confirmation pairs against the frozen profile", "",
          "| Benchmark | Confirmations | Observed sd of pairs | Predicted sqrt(2) s_b | Clearly higher (chi-square, 5 %) |",
          "|---|---|---|---|---|"]
    for b, v in R["sd_check"].items():
        if v:
            L.append(f"| `{b}` | {v['confirmations']} | {p(v['observed_sd'], 2)} | {p(v['predicted_sd'], 2)} | "
                     f"{'**yes**' if v['higher'] else 'no'} (chi2 {v['chi2']:.1f} vs {v['chi2_95']:.1f}, df {v['df']}) |")
    L += ["", "### Base flags on clean checks by the CPU types of the two recording VMs (arm none)", "",
          "| Detector | Record VMs | Clean checks | Base flags |", "|---|---|---|---|"]
    for name, (n, f) in R["base_flags_by_record_cpus"].items():
        d, kind = name.split(" | ")
        L.append(f"| {d} | {kind} CPU type | {n} | {f} ({p(f / n if n else None)}) |")
    pc = R["profile_cpus"]
    L += ["", f"Frozen noise profile: commits {pc['versions'][0]}-{pc['versions'][-1]}, "
              f"{'MIXED CPU types' if pc['mixed_cpu'] else 'one CPU type'}: " + ", ".join(map(str, pc["cpus"])) + ".",
          "Recording VMs: " + "; ".join(f"{k}: {v}" for k, v in sorted(R["record_cpus"].items(), key=lambda kv: int(kv[0])))
          + ".", ""]
    return "\n".join(L)


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
    x = sub.add_parser("report")
    x.add_argument("--records", required=True)
    x.add_argument("--profile", required=True)
    x.add_argument("--confirms", required=True)
    x.add_argument("--ground-truth", required=True)
    x.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"regex": cmd_regex, "patches": cmd_patches, "new-patch": cmd_new_patch, "record": cmd_record,
     "pack": cmd_pack, "classpath": cmd_classpath, "check-records": cmd_check_records,
     "profile": cmd_profile, "confirm": cmd_confirm, "ground-truth": cmd_ground_truth,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
