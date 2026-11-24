"""ab_check v1 (24 Nov 2026): same-VM A/B check of two commits for a few benchmarks.

  python3 case-study/ab_check.py run --a-cp FILE --b-cp FILE --a SHA --b SHA \
      --bench heapSort80,serialIndexOf250 --pairs 10 --out DIR
A is the earlier commit (the reference, "parent"), B the later one (the candidate).
Pair i of a benchmark is one run of B and one run of A, back to back in this job, in the order
NASC-CI 0.3 uses for batch i (nasc_ci.confirm.batch_order, "abba"): odd pairs candidate first,
even pairs reference first, so with 10 pairs 5 are measured each way. The benchmarks take turns
pair by pair, so slow drift during the job falls on all of them alike.
Every run: java -cp <jar:class path> org.openjdk.jmh.Main "^<benchmark>$" -bm sample -tu ns -f 1
-wi 10 -i 20 -w 1s -r 1s -foe true -rf json  (20 iterations = 4 blocks, a recording-sized run).

Per pair: d = mean(B run) / mean(A run) - 1. Per benchmark the report gives
  - mean d over the "candidate first" pairs and over the "reference first" pairs (95 % t-interval);
  - the order-balanced difference B vs A = the average of the two means (the real change);
  - the order effect = half their difference: how much slower a run is when it goes first;
  - same-VM noise of the runs (NASC-CI's c_w, c_t estimate per commit and both together) with the
    blocks per run (n_b) and the smallest confirmable slowdown it implies, next to the reference
    profile (Laaber et al. traces).
Writes ab.json, report.md and the raw JMH output of every run (runs/<benchmark>/pairNN-<side>.json/.log).
Needs numpy. Run from the repository root.
"""

import argparse
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

from nasc_ci import VERSION, codehash, history, jmh, noise  # noqa: E402
from nasc_ci.confirm import CAND_FIRST, REF_FIRST, batch_order  # noqa: E402

AB = "ab_check v1 (24 Nov 2026)"
HERE = Path(__file__).resolve().parent
PROFILE = HERE / "reference_noise_profile.csv"
BENCHMARKS = HERE / "benchmarks.txt"
# two-sided 95 % quantiles of Student's t, df 1..30 (above 30: 1.96 is close enough here)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042]


def t975(df):
    df = int(math.floor(df))
    return T975[df - 1] if 1 <= df <= 30 else 1.96


def short(name):
    return ".".join(name.split(".")[-2:])


def source_hash():
    f = ROOT / "nasc_ci" / "VERSION"
    return f.read_text().strip() if f.exists() else "unknown"


def full_names(wanted):
    names = [l.strip() for l in BENCHMARKS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    out = []
    for w in wanted:
        hits = [n for n in names if n.endswith("." + w.strip())]
        if len(hits) != 1:
            sys.exit(f"--bench {w}: {len(hits)} matches in {BENCHMARKS.name}")
        out.append(hits[0])
    return out


# ---------------------------------------------------------------- measuring
def schedule(benchmarks, pairs):
    """The run order: pair by pair, benchmarks taking turns, sides in NASC-CI's ABBA order."""
    out = []
    for i in range(1, pairs + 1):
        order = batch_order(i, "abba")
        for b in benchmarks:
            sides = ("B", "A") if order == CAND_FIRST else ("A", "B")
            out.append({"pair": i, "benchmark": b, "order": order, "sides": sides})
    return out


def jmh_runner(cps, out, wi, iterations, time_):
    """-> run(benchmark, side, pair): runs JMH once, keeps its JSON and log, returns the run's
    block means (NASC-CI's jmh reader: sample-mode iteration means, blocks of 5)."""
    def run(bench, side, pair):
        stem = Path(out) / "runs" / short(bench) / f"pair{pair:02d}-{side}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["java", "-cp", cps[side], "org.openjdk.jmh.Main", "^" + re.escape(bench) + "$",
               "-bm", "sample", "-tu", "ns", "-f", "1", "-wi", str(wi), "-i", str(iterations),
               "-w", time_, "-r", time_, "-foe", "true", "-rf", "json", "-rff", str(stem) + ".json"]
        with open(str(stem) + ".log", "w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        res = jmh.read(str(stem) + ".json")[bench]
        assert len(res["runs"]) == 1, f"{stem}.json: expected one fork"
        return res["runs"][0], str(stem.relative_to(out)) + ".json"
    return run


def measure_all(plan, run):
    """Runs the plan with run(benchmark, side, pair) -> (block means, file). -> list of run records."""
    records = []
    for step in plan:
        for pos, side in enumerate(step["sides"], start=1):
            t0 = time.time()
            blocks, file = run(step["benchmark"], side, step["pair"])
            records.append({"benchmark": step["benchmark"], "pair": step["pair"], "side": side,
                            "position": pos, "order": step["order"], "blocks": [float(x) for x in blocks],
                            "mean": float(np.mean(blocks)), "seconds": round(time.time() - t0, 1),
                            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "file": file})
            print(f"   pair {step['pair']:>2} {short(step['benchmark']):<48} {side} ({pos}) "
                  f"{np.mean(blocks):10.5g} ns/op  {time.time() - t0:4.0f} s", flush=True)
    return records


# ---------------------------------------------------------------- analysis
def mean_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else None, None
    half = t975(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), float(half)


def analyse(records, settings, reference=None):
    """-> one summary per benchmark (see the module docstring)."""
    out = []
    for b in dict.fromkeys(r["benchmark"] for r in records):
        rs = [r for r in records if r["benchmark"] == b]
        pairs = sorted({r["pair"] for r in rs})
        d, orders = [], []
        for p in pairs:
            a = next(r for r in rs if r["pair"] == p and r["side"] == "A")
            bb = next(r for r in rs if r["pair"] == p and r["side"] == "B")
            d.append(bb["mean"] / a["mean"] - 1)
            orders.append(a["order"])
        d, orders = np.array(d), np.array(orders)
        cf, rf = d[orders == CAND_FIRST], d[orders == REF_FIRST]
        m_cf, h_cf = mean_ci(cf)
        m_rf, h_rf = mean_ci(rf)
        both = (m_cf + m_rf) / 2 if len(cf) and len(rf) else float(d.mean())
        effect = (m_cf - m_rf) / 2 if len(cf) and len(rf) else None
        half = None
        if len(cf) > 1 and len(rf) > 1:                       # Welch interval for both quantities
            v1, v2 = cf.var(ddof=1) / len(cf), rf.var(ddof=1) / len(rf)
            df = (v1 + v2) ** 2 / (v1 ** 2 / (len(cf) - 1) + v2 ** 2 / (len(rf) - 1)) if v1 + v2 > 0 else 1
            half = t975(df) * math.sqrt(v1 + v2) / 2
        runs = {side: np.array([r["blocks"] for r in rs if r["side"] == side]) for side in ("A", "B")}
        n_a = noise.estimate([runs["A"]], {**settings, "noise_min_versions": 1})
        n_b = noise.estimate([runs["B"]], {**settings, "noise_min_versions": 1})
        n_ab = noise.estimate([runs["A"], runs["B"]], {**settings, "noise_min_versions": 2})
        out.append({"benchmark": b, "pairs": len(pairs), "d": d.tolist(), "orders": orders.tolist(),
                    "candidate_first": {"n": int(len(cf)), "mean": m_cf, "ci95": h_cf},
                    "reference_first": {"n": int(len(rf)), "mean": m_rf, "ci95": h_rf},
                    "difference": both, "order_effect": effect, "ci95_welch": half,
                    "sd_d": float(d.std(ddof=1)) if len(d) > 1 else None,
                    "noise_A": n_a.to_dict(), "noise_B": n_b.to_dict(), "noise_AB": n_ab.to_dict(),
                    "reference": (reference or {}).get(b)})
    return out


def pct(x, d=2):
    return "–" if x is None else f"{x * 100:+.{d}f}%"


def ci(m, h):
    return "–" if m is None else (f"{m * 100:+.2f}%" + ("" if h is None else f" ± {h * 100:.2f}%"))


def sd(x):
    return "–" if x is None else f"{x * 100:.2f}%"


def report(S):
    L = [f"## Same-VM A/B check: commit {S['a']['pos']} (A) vs commit {S['b']['pos']} (B)", "",
         f"`{S['tool']}`, source `{S['nasc_ci_source']}`; {S['ab']}. {S['pairs']} pairs per benchmark, "
         "order alternating as NASC-CI 0.3 (odd pairs candidate B first, even pairs reference A first).", "",
         f"- Machine: {S['machine'].get('cpu_model')} ({S['machine'].get('cpus')} CPUs, clock source "
         f"{S['machine'].get('clocksource')}).",
         f"- Compiled code: A `{(S['code_A'] or {}).get('sha256', '?')[:12]}`, B `{(S['code_B'] or {}).get('sha256', '?')[:12]}` "
         f"-> **{ {True: 'identical', False: 'different', None: 'unknown'}[S['same_code']] }**.",
         f"- Runner minutes: {S['minutes']['setup_build']:.1f} set-up + build, {S['minutes']['measure']:.1f} measuring, "
         f"{S['minutes']['total']:.1f} in total (from the job's first step).", "",
         "### B vs A per order (d = mean B run / mean A run - 1; mean ± 95 % t-interval)", "",
         "| Benchmark | Candidate (B) first | Reference (A) first | B vs A, order-balanced | Order effect (first run slower by) | sd of d |",
         "|---|---|---|---|---|---|"]
    for r in S["results"]:
        L.append(f"| `{short(r['benchmark'])}` | {ci(r['candidate_first']['mean'], r['candidate_first']['ci95'])} (n={r['candidate_first']['n']}) | "
                 f"{ci(r['reference_first']['mean'], r['reference_first']['ci95'])} (n={r['reference_first']['n']}) | "
                 f"{ci(r['difference'], r['ci95_welch'])} | {ci(r['order_effect'], r['ci95_welch'])} | "
                 f"{sd(r['sd_d'])} |")
    L += ["", "### Same-VM noise of the runs (NASC-CI estimate) next to the reference profile", "",
          "| Benchmark | Reference c_w / c_t (n_b) | A: c_w / c_t | B: c_w / c_t | A and B: c_w / c_t | n_b | Smallest confirmable slowdown |",
          "|---|---|---|---|---|---|---|"]
    for r in S["results"]:
        ref = r["reference"]
        rtxt = "–" if not ref else f"{ref['c_w'] * 100:.2f}% / {ref['c_t'] * 100:.2f}% ({ref['n_b'] if ref['feasible'] else 'infeasible'})"
        f = lambda n: f"{n['c_w'] * 100:.2f}% / {n['c_t'] * 100:.2f}%"  # noqa: E731
        ab = r["noise_AB"]
        L.append(f"| `{short(r['benchmark'])}` | {rtxt} | {f(r['noise_A'])} | {f(r['noise_B'])} | {f(ab)} | "
                 f"{ab['n_b'] if ab['feasible'] else 'infeasible'} | {ab['delta_feasible'] * 100:.1f}% |")
    L += ["", "Pairs (d per pair, in run order): " + "; ".join(
        f"{short(r['benchmark'])}: " + ", ".join(f"{x * 100:+.2f}%{'ᶜ' if o == CAND_FIRST else 'ʳ'}"
                                                   for x, o in zip(r["d"], r["orders"])) for r in S["results"])
        + " (ᶜ candidate first, ʳ reference first).", ""]
    return "\n".join(L)


def read_timing(path):
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = float(v)
    return out


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    benches = full_names(a.bench.split(","))
    cps = {"A": Path(a.a_cp).read_text().strip(), "B": Path(a.b_cp).read_text().strip()}
    print(f"{AB}; {VERSION}; source {source_hash()}")
    plan = schedule(benches, a.pairs)
    print(f"{len(plan) * 2} runs: {a.pairs} pairs x {len(benches)} benchmarks, A = {a.a[:10]}, B = {a.b[:10]}", flush=True)
    t_start = time.time()
    records = measure_all(plan, jmh_runner(cps, out, a.wi, a.iterations, a.time))
    t_end = time.time()
    s = history.new()["settings"]
    reference = {b: n.to_dict() for b, n in noise.load_profile(PROFILE, s).items()}
    load = lambda p: json.loads(Path(p).read_text()) if p and Path(p).exists() else None  # noqa: E731
    code_a, code_b = load(a.code_a), load(a.code_b)
    timing = read_timing(a.timing)
    t0 = timing.get("job_start", t_start)
    S = {"ab": AB, "tool": VERSION, "nasc_ci_source": source_hash(),
         "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "a": {"pos": a.a_pos, "commit": a.a}, "b": {"pos": a.b_pos, "commit": a.b}, "pairs": a.pairs,
         "run_settings": {"wi": a.wi, "iterations": a.iterations, "time": a.time},
         "machine": load(a.machine) or {}, "code_A": code_a, "code_B": code_b,
         "same_code": codehash.same_code(code_a, code_b),
         "minutes": {"setup_build": (t_start - t0) / 60, "measure": (t_end - t_start) / 60, "total": (t_end - t0) / 60},
         "runs": records, "results": analyse(records, s, reference)}
    (out / "ab.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    text = report(S)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description=AB)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--a", required=True, help="commit A (reference)")
    r.add_argument("--b", required=True, help="commit B (candidate)")
    r.add_argument("--a-pos", type=int, default=0)
    r.add_argument("--b-pos", type=int, default=0)
    r.add_argument("--a-cp", required=True)
    r.add_argument("--b-cp", required=True)
    r.add_argument("--bench", default="heapSort80,serialIndexOf250")
    r.add_argument("--pairs", type=int, default=10)
    r.add_argument("--out", required=True)
    r.add_argument("--wi", type=int, default=10)
    r.add_argument("--iterations", type=int, default=20)
    r.add_argument("--time", default="1s")
    r.add_argument("--machine")
    r.add_argument("--code-a")
    r.add_argument("--code-b")
    r.add_argument("--timing")
    a = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd_run(a)


if __name__ == "__main__":
    main()
