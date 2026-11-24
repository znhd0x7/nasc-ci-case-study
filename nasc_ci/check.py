"""One CI check of a new version: detect, confirm, store and report.

For every benchmark in the new results:
  1. warm-up: fewer than W versions stored -> only stored, not checked;
  2. the base detector compares the version with the last W stored versions (since the baseline);
  3. no flag -> done; a flag -> NASC re-runs the version and its PARENT in batches until the
     evidence is clear: "confirmed" (a real slowdown, alert) or "dismissed" (noise, no alert);
  4. a confirmed version becomes the new baseline for that benchmark.
"""

import numpy as np

from . import confirm as nasc
from . import detectors, history, noise

ICON = {"confirmed": "🔴", "dismissed": "🟢", "no flag": "⚪", "warm-up": "⏳"}


def run_check(h, results, commit, label, runner, rng, save=True, profile=None):
    """profile: optional fixed noise profile {benchmark: Noise}, e.g. from step 05's
    noise_profile.csv, as the thesis uses. Benchmarks without one estimate it from the history.
    If the runner has reference_level(parent entry), NASC scales by it (the simulation passes the
    TRUE parent level there); otherwise by the stored mean of the parent's runs, as a real CI must."""
    s = h["settings"]
    entry = {"index": len(h["versions"]) + 1, "commit": commit, "label": label,
             "time": history.now(), "results": results}
    rows = []
    for bench in sorted(results):
        cand = np.array(results[bench]["runs"], dtype=float)
        window, warming = history.window_of(h, bench)
        if profile and bench in profile:
            prof = profile[bench]
        else:
            prof = noise.estimate([r for _, r in history.versions_of(h, bench)], s)
        row = {"version": entry["index"], "label": label, "commit": commit, "benchmark": bench,
               "mean": float(cand.mean()), "unit": results[bench].get("unit", ""),
               "detector": s["detector"], "flagged": False, "detector_change": None,
               "decision": "warm-up", "noise": prof.to_dict(), "confirmation": None,
               "extra_blocks": 0}
        if not warming and window:
            flag = detectors.detect(s["detector"], [r for _, r in window], cand, rng, s)
            row.update(flagged=bool(flag.flagged), detector_change=float(flag.change),
                       detector_statistic=float(flag.statistic), rule=flag.rule,
                       parent=window[-1][0]["label"], decision="no flag")
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
    lines.append("")
    if flagged:
        lines += ["| Benchmark | Detector saw | NASC decision | Noise (c_w / c_t) |",
                  "|---|---|---|---|"]
        for r in flagged:
            n = r["noise"]
            note = "" if n["feasible"] else f" ⚠️ can only confirm ≥ {n['delta_feasible'] * 100:.1f}%"
            lines.append(f"| {ICON[r['decision']]} `{r['benchmark']}` | {pct(r['detector_change'])} "
                         f"vs `{r['parent']}` | {decision_text(r)} | "
                         f"{n['c_w'] * 100:.2f}% / {n['c_t'] * 100:.2f}%{note} |")
        lines.append("")
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
              f"window W={s['window']}. {h['tool']}.</sub>", ""]
    return "\n".join(lines)
