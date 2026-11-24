"""One CI check of a new version: detect, confirm, store and report.

For every benchmark in the new results:
  1. warm-up: fewer than W versions stored -> only stored, not checked;
  2. the base detector compares the version with the last W stored versions (since the baseline);
  3. no flag -> done; a flag -> NASC re-runs the version and its PARENT in batches until the
     evidence is clear: "confirmed" (a real slowdown, alert) or "dismissed" (noise, no alert);
  4. a confirmed version becomes the new baseline for that benchmark.

Noise profile (v0.3): unless a fixed profile is passed in, c_w and c_t are estimated from the
recorded runs of the first settings["profile_versions"] (5) stored versions, each version
divided by its own level (so a faster or slower machine does not count as noise), and then
FROZEN in h["profile"], together with the CPU model of each of those versions. Benchmarks whose
profile is infeasible (equation 8) are still confirmed, with n_b = n_max, but reported apart
with the smallest slowdown they can confirm.
"""

import numpy as np

from . import codehash, detectors, history, machine, noise
from . import confirm as nasc

ICON = {"confirmed": "🔴", "dismissed": "🟢", "no flag": "⚪", "warm-up": "⏳"}


def freeze_profile(h, n_versions):
    """The noise profile from the first n_versions stored versions -> h["profile"]."""
    s = h["settings"]
    first = h["versions"][:n_versions]
    names = sorted({b for v in first for b in v["results"]})
    prof = {}
    for b in names:
        runs = [np.array(v["results"][b]["runs"], dtype=float) for v in first if b in v["results"]]
        prof[b] = noise.estimate(runs, s).to_dict()
    cpus = [machine.cpu_of(v.get("machine")) for v in first]
    known = sorted({c for c in cpus if c})
    return {"versions": [v["index"] for v in first], "benchmarks": prof, "cpus": cpus,
            "cpu_types": known, "mixed_cpu": len(known) > 1,
            "cpu_unknown": sum(c is None for c in cpus)}


def profile_of(h, bench, profile):
    """Fixed profile passed in > frozen profile in the history > estimate from the history so far."""
    if profile and bench in profile:
        return profile[bench]
    frozen = h.get("profile")
    if frozen and bench in frozen["benchmarks"]:
        return noise.Noise(**frozen["benchmarks"][bench])
    return noise.estimate([r for _, r in history.versions_of(h, bench)], h["settings"])


def run_check(h, results, commit, label, runner, rng, save=True, profile=None, machine_info=None,
              code=None):
    """profile: optional fixed noise profile {benchmark: Noise}, e.g. from step 05's
    noise_profile.csv, as the thesis simulation uses. Otherwise the frozen runner profile (see above).
    machine_info: machine.describe() of the job that RECORDED these results; code: codehash.code_hash()
    of this commit's compiled classes. Both are stored with the version and shown in the report.
    With settings["scale"] == "stored", NASC scales by runner.reference_level(parent) if the runner
    has it (the simulation passes the TRUE parent level there), otherwise by the parent's stored
    mean; with "fresh" (default) it scales by the mean of its own reference re-runs."""
    s = h["settings"]
    entry = {"index": len(h["versions"]) + 1, "commit": commit, "label": label,
             "time": history.now(), "results": results}
    if machine_info is not None:
        entry["machine"] = machine_info
    if code is not None:
        entry["code"] = code
    rows = []
    for bench in sorted(results):
        cand = np.array(results[bench]["runs"], dtype=float)
        window, warming = history.window_of(h, bench)
        prof = profile_of(h, bench, profile)
        row = {"version": entry["index"], "label": label, "commit": commit, "benchmark": bench,
               "mean": float(cand.mean()), "unit": results[bench].get("unit", ""),
               "detector": s["detector"], "flagged": False, "detector_change": None,
               "decision": "warm-up", "noise": prof.to_dict(), "confirmation": None,
               "extra_blocks": 0}
        if not warming and window:
            flag = detectors.detect(s["detector"], [r for _, r in window], cand, rng, s)
            parent_entry = window[-1][0]
            row.update(flagged=bool(flag.flagged), detector_change=float(flag.change),
                       detector_statistic=float(flag.statistic), rule=flag.rule,
                       parent=parent_entry["label"], decision="no flag",
                       same_code_as_parent=codehash.same_code(code, parent_entry.get("code")))
            if flag.flagged:
                parent, parent_runs = window[-1]
                measure = runner.measurer(bench, entry, parent)
                scale = (runner.reference_level(parent) if hasattr(runner, "reference_level")
                         else float(parent_runs.mean()))
                c = nasc.nasc_confirm(measure, prof, scale, s)
                row.update(decision="confirmed" if c.confirmed else "dismissed",
                           confirmation=c.to_dict(), extra_blocks=c.extra_blocks)
                if c.confirmed and save:
                    h["baselines"][bench] = entry["index"]   # triage accepts it: new baseline
        rows.append(row)
    if save:
        h["versions"].append(entry)
        h["checks"].extend(rows)
        n = s.get("profile_versions", 0)
        if n and "profile" not in h and len(h["versions"]) >= n:
            h["profile"] = freeze_profile(h, n)
    return rows


def pct(x, digits=1):
    return "–" if x is None else f"{x * 100:+.{digits}f}%"


def decision_text(r):
    c = r["confirmation"]
    if r["decision"] == "confirmed":
        return (f"**confirmed slowdown {c['change'] * 100:.1f}%** "
                f"({c['batches']} round{'s' * (c['batches'] > 1)}, {c['extra_runs']} extra runs)")
    if r["decision"] == "dismissed":
        return (f"dismissed as noise ({c['batches']} round{'s' * (c['batches'] > 1)}, "
                f"{c['extra_runs']} extra runs; re-runs showed {c['change'] * 100:+.1f}%)")
    if r["decision"] == "warm-up":
        return "stored (warm-up, history too short to check)"
    return "no slowdown"


def _flag_table(rows, feasible):
    col_name = "n_b" if feasible else "Smallest confirmable slowdown"
    out = [f"| Benchmark | Detector saw | NASC decision | Noise (c_w / c_t) | {col_name} | Same code as parent |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        n = r["noise"]
        same = {True: "yes", False: "no", None: "–"}[r.get("same_code_as_parent")]
        col = f"{n['n_b']}" if feasible else f"{n['delta_feasible'] * 100:.1f}%"
        out.append(f"| {ICON[r['decision']]} `{r['benchmark']}` | {pct(r['detector_change'])} "
                   f"vs `{r['parent']}` | {decision_text(r)} | "
                   f"{n['c_w'] * 100:.2f}% / {n['c_t'] * 100:.2f}% | {col} | {same} |")
    return out


def report(rows, h, commit, label):
    s = h["settings"]
    flagged = [r for r in rows if r["flagged"]]
    confirmed = [r for r in rows if r["decision"] == "confirmed"]
    dismissed = [r for r in rows if r["decision"] == "dismissed"]
    checked = [r for r in rows if r["decision"] != "warm-up"]
    blocks = sum(r["extra_blocks"] for r in rows)
    lines = [f"## NASC-CI performance check: `{label}` ({commit[:7]})", ""]
    if confirmed:
        lines.append(f"🔴 **{len(confirmed)} confirmed slowdown{'s' * (len(confirmed) > 1)}** "
                     f"out of {len(checked)} checked benchmarks.")
    elif checked:
        lines.append(f"✅ No confirmed slowdown in {len(checked)} checked benchmarks.")
    else:
        lines.append(f"⏳ Warm-up: {len(rows)} benchmarks stored, checks start after {s['window']} versions.")
    if checked:
        lines.append(f"The {s['detector']} detector flagged {len(flagged)}; NASC confirmed {len(confirmed)} and "
                     f"dismissed {len(dismissed)} as noise, using {blocks} extra blocks "
                     f"({blocks * 5} iterations).")
    entry = next((v for v in h["versions"] if v["label"] == label), None)
    if entry and entry.get("machine"):
        lines.append(f"Recorded on: {entry['machine'].get('cpu_model')} ({entry['machine'].get('cpus')} CPUs).")
    prof = h.get("profile")
    if prof:
        cpu = ("MIXED CPU types: " if prof["mixed_cpu"] else "one CPU type: ") + (", ".join(prof["cpu_types"]) or "unknown")
        lines.append(f"Noise profile: frozen from versions {prof['versions'][0]}–{prof['versions'][-1]} ({cpu}).")
    lines.append("")
    feas = [r for r in flagged if r["noise"]["feasible"]]
    infeas = [r for r in flagged if not r["noise"]["feasible"]]
    if feas:
        lines += ["**Feasible benchmarks** (can confirm a slowdown of "
                  f"{s['delta_min']:.0%} within K_max = {s['k_max']} rounds)", ""] + _flag_table(feas, True) + [""]
    if infeas:
        lines += ["**Infeasible benchmarks** (too noisy for "
                  f"{s['delta_min']:.0%}; confirmed with n_b = {s['n_max']}, reported apart)", ""] \
            + _flag_table(infeas, False) + [""]
    lines += ["<details><summary>All benchmarks</summary>", "",
              "| Benchmark | Mean | Change | Decision | n_b | Feasible at "
              f"{s['delta_min']:.0%} |", "|---|---|---|---|---|---|"]
    for r in rows:
        n = r["noise"]
        lines.append(f"| {ICON[r['decision']]} `{r['benchmark']}` | {r['mean']:.4g} {r['unit']} | "
                     f"{pct(r['detector_change'])} | {r['decision']} | {n['n_b']} | "
                     f"{'yes' if n['feasible'] else 'no'}{'' if n['estimated'] else ' (default noise)'} |")
    lines += ["", "</details>", "",
              f"<sub>NASC: K_max={s['k_max']}, α=β={s['alpha']}, δ_min={s['delta_min']:.0%}, "
              f"window W={s['window']}, scale={s.get('scale')}, order={s.get('order')}. {h['tool']}.</sub>", ""]
    return "\n".join(lines)
