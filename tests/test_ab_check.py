"""Tests of case-study/ab_check.py v2.3 (26 Nov 2026) with a fake machine, no Java needed.

v2.3 (26 Nov 2026): pool --expected names missing copies (guard G5).

v2.2 (25 Nov 2026): pool combines copies run on different VMs and reports the difference per CPU type.

v2.1 (25 Nov 2026): the benchmarks measured head the report; a patch's benchmarks come from
case-study/injections/targets.txt and any not measured gives a warning; targets.txt is checked.

v2 (25 Nov 2026): the injection patch on B is recorded and its slowdown is reported with the CPU
model it was measured on; the two patch files in case-study/injections/ are checked as text.

The fake machine makes whichever run of a pair goes FIRST slower by a fixed share (an order
effect), and B slower than A by a chosen true change. The check must split the two: the
order-balanced difference recovers the true change, the order effect recovers the first-run penalty.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ab_check", ROOT / "case-study" / "ab_check.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

HEAP = "io.jenetics.internal.util.IndexSorterPerf.heapSort80"
INDEX = "io.jenetics.ProbabilitySelectorIndexOfPerf.serialIndexOf250"


def fake_machine(true_change, first_penalty, c_w=0.003, c_t=0.002, seed=1):
    """-> run(benchmark, side, pair) like ab_check.jmh_runner, plus the call log."""
    rng = np.random.default_rng(seed)
    level = {HEAP: 1350.0, INDEX: 90.0}
    calls = []

    def run(bench, side, pair):
        first = not calls or calls[-1][:2] != (bench, pair)      # first run of this pair?
        calls.append((bench, pair, side))
        mu = level[bench] * (1 + true_change * (side == "B")) * (1 + first_penalty * first)
        mu *= 1 + rng.normal(0, c_t)
        blocks = mu * (1 + rng.normal(0, c_w, size=4))
        return blocks.tolist(), f"runs/{bench}/pair{pair:02d}-{side}.json"
    return run, calls


def summarise(true_change, first_penalty, pairs=10):
    plan = ab.schedule([HEAP, INDEX], pairs)
    run, calls = fake_machine(true_change, first_penalty)
    records = ab.measure_all(plan, run)
    s = ab.history.new()["settings"]
    return {r["benchmark"]: r for r in ab.analyse(records, s)}, records, calls


def test_schedule_is_abba_and_benchmarks_take_turns():
    plan = ab.schedule([HEAP, INDEX], 10)
    assert len(plan) == 20
    assert [p["benchmark"] for p in plan[:4]] == [HEAP, INDEX, HEAP, INDEX]
    heap = [p for p in plan if p["benchmark"] == HEAP]
    assert "".join("".join(p["sides"]) for p in heap) == "BAAB" * 5          # B = candidate, A = reference
    assert sum(p["order"] == ab.CAND_FIRST for p in heap) == 5
    assert sum(p["order"] == ab.REF_FIRST for p in heap) == 5


def test_runs_follow_the_schedule():
    _, records, calls = summarise(0.0, 0.0)
    assert len(records) == 40
    assert [(c[0], c[2]) for c in calls[:4]] == [(HEAP, "B"), (HEAP, "A"), (INDEX, "B"), (INDEX, "A")]
    assert [(c[0], c[2]) for c in calls[4:8]] == [(HEAP, "A"), (HEAP, "B"), (INDEX, "A"), (INDEX, "B")]
    assert all(r["position"] in (1, 2) for r in records)


def test_order_effect_on_identical_code_is_split_from_the_difference():
    res, _, _ = summarise(true_change=0.0, first_penalty=0.03)
    for b in (HEAP, INDEX):
        r = res[b]
        assert r["candidate_first"]["mean"] == pytest.approx(0.03, abs=0.006)   # B first: B looks slower
        assert r["reference_first"]["mean"] == pytest.approx(-0.03, abs=0.006)  # A first: B looks faster
        assert r["difference"] == pytest.approx(0.0, abs=0.004)
        assert r["order_effect"] == pytest.approx(0.03, abs=0.004)
        assert r["candidate_first"]["n"] == r["reference_first"]["n"] == 5


def test_real_change_is_recovered_with_and_without_order_effect():
    for penalty in (0.0, 0.03):
        res, _, _ = summarise(true_change=0.05, first_penalty=penalty)
        for b in (HEAP, INDEX):
            assert res[b]["difference"] == pytest.approx(0.05, abs=0.006)
            assert res[b]["order_effect"] == pytest.approx(penalty, abs=0.006)
            lo = res[b]["difference"] - res[b]["ci95_welch"]
            hi = res[b]["difference"] + res[b]["ci95_welch"]
            assert lo < 0.05 < hi


def test_same_vm_noise_is_estimated_per_side():
    res, _, _ = summarise(true_change=0.0, first_penalty=0.0)
    r = res[HEAP]
    for key in ("noise_A", "noise_B", "noise_AB"):
        assert r[key]["estimated"] is True
        assert 0.0015 < r[key]["c_w"] < 0.006
    assert r["noise_AB"]["feasible"] is True


def test_t_quantiles():
    assert ab.t975(4) == pytest.approx(2.776)
    assert ab.t975(1.7) == pytest.approx(12.706)
    assert ab.t975(100) == 1.96


def test_full_names_and_report():
    assert ab.full_names(["heapSort80", "serialIndexOf250"]) == [HEAP, INDEX]
    with pytest.raises(SystemExit):
        ab.full_names(["noSuchBenchmark"])
    res, records, _ = summarise(0.0, 0.03)
    S = {"ab": ab.AB, "tool": ab.VERSION, "nasc_ci_source": "0" * 40, "a": {"pos": 1}, "b": {"pos": 2},
         "pairs": 10, "machine": {"cpu_model": "Test CPU", "cpus": 4, "clocksource": "tsc"},
         "code_A": {"sha256": "ab" * 32}, "code_B": {"sha256": "ab" * 32}, "same_code": True,
         "minutes": {"setup_build": 2.0, "measure": 20.5, "total": 22.5},
         "results": list(res.values())}
    text = ab.report(S)
    assert "Candidate (B) first" in text and "Reference (A) first" in text
    assert "**identical**" in text and "Test CPU" in text
    assert "`IndexSorterPerf.heapSort80`" in text and "`ProbabilitySelectorIndexOfPerf.serialIndexOf250`" in text
    assert text.count("ᶜ") == 10 + 1 and text.count("ʳ") == 10 + 1             # 5 + 5 per benchmark, + legend


def report_input(res, patch=None, same_code=True):
    return {"ab": ab.AB, "tool": ab.VERSION, "nasc_ci_source": "0" * 40, "a": {"pos": 9},
            "b": {"pos": 9, "patch": patch}, "pairs": 10,
            "machine": {"cpu_model": "Test CPU", "cpus": 4, "clocksource": "tsc"},
            "code_A": {"sha256": "ab" * 32}, "code_B": {"sha256": ("ab" if same_code else "cd") * 32},
            "same_code": same_code, "minutes": {"setup_build": 2.0, "measure": 30.0, "total": 32.0},
            "results": list(res.values())}


def test_injection_patch_is_recorded_and_its_size_reported_with_the_cpu():
    path = ROOT / "case-study" / "injections" / "heapsort.patch"
    info = ab.patch_info(str(path))
    assert info["file"] == "heapsort.patch" and len(info["sha256"]) == 64
    assert info["added"] > 0 and info["removed"] == 0
    assert ab.patch_info(None) is None
    res, _, _ = summarise(true_change=0.05, first_penalty=0.0)
    text = ab.report(report_input(res, info, same_code=False))
    assert "commit 9 (A) vs commit 9 + `heapsort.patch` (B)" in text
    assert "**Benchmarks measured (10 pairs each): heapSort80, serialIndexOf250**" in text
    assert "it slows down heapSort80, heapSort250, heapSort320" in text
    assert "Warning: not measured in this run: heapSort250, heapSort320" in text
    assert "### Injected slowdown, measured on Test CPU (4 CPUs)" in text
    assert "may differ on other CPU types" in text
    assert "expected: B carries the injection patch" in text
    assert "+5." in text.split("### Injected slowdown")[1]           # about +5 %
    warn = ab.report(report_input(res, info, same_code=True))
    assert "the patch did not reach the compiled code" in warn


def test_report_without_patch_has_no_injection_section():
    res, _, _ = summarise(true_change=0.0, first_penalty=0.0)
    text = ab.report(report_input(res))
    assert "Injected slowdown" not in text and "Injection patch" not in text
    assert "**identical**" in text


@pytest.mark.parametrize("name, java_file", [
    ("heapsort.patch", "jenetics/src/main/java/io/jenetics/internal/util/IndexSorter.java"),
    ("indexof.patch", "jenetics/src/main/java/io/jenetics/ProbabilitySelector.java"),
])
def test_injection_patches_only_add_lines_to_one_file(name, java_file):
    text = (ROOT / "case-study" / "injections" / name).read_text(encoding="utf-8")
    files = [l[len("+++ b/"):] for l in text.splitlines() if l.startswith("+++ ")]
    assert files == [java_file]
    body = [l for l in text.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    assert body and all(l.startswith("+") for l in body)                 # adds only, removes nothing
    added = "\n".join(l[1:] for l in body)
    assert "private static final int INJECTED_EXTRA_PERCENT = " in added
    assert added.count("INJECTED_EXTRA_PERCENT") == 3                    # constant, comment, use
    assert "NASC-CI case study, not part of jenetics" in added


def test_targets_file_matches_patches_and_benchmarks():
    rows = [l.split() for l in ab.TARGETS.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    assert [(r[0], r[1]) for r in rows] == [("heapsort", "9"), ("indexof", "13")]
    for name, _, benches in rows:
        assert (ROOT / "case-study" / "injections" / f"{name}.patch").is_file()
        assert ab.full_names(benches.split(","))                         # every name is in benchmarks.txt
    assert ab.patch_targets("x/heapsort.patch") == ["heapSort80", "heapSort250", "heapSort320"]
    assert ab.patch_targets("indexof.patch") == ["serialIndexOf250"]
    assert ab.patch_targets("other.patch") == []


def test_no_warning_when_all_patched_benchmarks_are_measured():
    res, _, _ = summarise(true_change=0.05, first_penalty=0.0)
    info = ab.patch_info(str(ROOT / "case-study" / "injections" / "indexof.patch"))
    text = ab.report(report_input(res, info, same_code=False))   # measures heapSort80 + serialIndexOf250
    assert "it slows down serialIndexOf250" in text and "Warning" not in text


def copy_result(cpu, true_change, seed, copy, patch_sha="ff" * 32):
    """ab.json of one simulated copy: serialIndexOf250 only, on a VM with the given CPU."""
    plan = ab.schedule([INDEX], 10)
    run, _ = fake_machine(true_change, 0.0, c_w=0.01, c_t=0.005, seed=seed)
    records = ab.measure_all(plan, run)
    res = ab.analyse(records, ab.history.new()["settings"])
    return {"copy": copy, "a": {"pos": 13, "commit": "a" * 40},
            "b": {"pos": 13, "commit": "a" * 40, "patch": {"file": "indexof.patch", "sha256": patch_sha,
                                                           "added": 18, "removed": 0}},
            "machine": {"cpu_model": cpu, "cpus": 4}, "results": res}


def test_pool_gives_the_size_per_cpu_type():
    runs = [copy_result("Intel X", 0.15, 1, 1), copy_result("AMD Y", 0.10, 2, 2),
            copy_result("Intel X", 0.15, 3, 3), copy_result("AMD Y", 0.10, 4, 4)]
    P = ab.pool(runs)
    assert P["cpu_types"] == ["AMD Y", "Intel X"]
    by = {g["cpu"]: g for g in P["by_cpu"]}
    assert by["Intel X"]["copies"] == 2 and by["Intel X"]["pairs"] == 20
    assert by["Intel X"]["difference"] == pytest.approx(0.15, abs=0.01)
    assert by["AMD Y"]["difference"] == pytest.approx(0.10, abs=0.01)
    assert by["AMD Y"]["ci95"] < 0.01
    assert len(P["copies"]) == 4
    text = ab.pool_report(P, runs[0])
    assert "4 copies" in text and "CPU types seen: AMD Y, Intel X." in text
    assert "Only one CPU type" not in text
    assert "| `ProbabilitySelectorIndexOfPerf.serialIndexOf250` | Intel X | 2 | 20 | +1" in text


def test_pool_warns_when_only_one_cpu_type_was_seen_and_refuses_mixed_runs():
    P = ab.pool([copy_result("Intel X", 0.15, 1, 1), copy_result("Intel X", 0.15, 2, 2)])
    assert "Only one CPU type was seen" in ab.pool_report(P, copy_result("Intel X", 0.15, 1, 1))
    with pytest.raises(SystemExit):
        ab.pool([copy_result("Intel X", 0.15, 1, 1), copy_result("AMD Y", 0.15, 2, 2, patch_sha="ee" * 32)])


def test_balanced_matches_analyse():
    res, _, _ = summarise(true_change=0.05, first_penalty=0.03)
    r = res[HEAP]
    m, h, n = ab.balanced(r["d"], r["orders"])
    assert m == pytest.approx(r["difference"]) and h == pytest.approx(r["ci95_welch"]) and n == 10


def test_pool_command_reads_copy_folders(tmp_path):
    for i, cpu in ((1, "Intel X"), (2, "AMD Y")):
        S = copy_result(cpu, 0.15, i, i)
        del S["copy"]
        d = tmp_path / f"ab-check-a13-b13-indexof-run1-copy{i}"
        d.mkdir()
        (d / "ab.json").write_text(__import__("json").dumps(S), encoding="utf-8")
    import types
    ab.cmd_pool(types.SimpleNamespace(files=sorted(str(p) for p in tmp_path.glob("*/ab.json")),
                                      out=str(tmp_path / "summary")))
    P = __import__("json").loads((tmp_path / "summary" / "pool.json").read_text(encoding="utf-8"))
    assert sorted(c["copy"] for c in P["copies"]) == [1, 2]
    assert (tmp_path / "summary" / "pool.md").exists()


def test_pool_names_missing_copies(tmp_path, capsys):
    """G5 (26 Nov 2026): 4 copies started, copy 3 failed (no ab.json): the summary says so."""
    import json
    import types
    for i in (1, 2, 4):
        d = tmp_path / f"ab-check-a13-b13-indexof-run1-copy{i}"
        d.mkdir()
        S = copy_result("AMD Y", 0.15, i, i)
        del S["copy"]
        (d / "ab.json").write_text(json.dumps(S), encoding="utf-8")
    files = sorted(str(p) for p in tmp_path.glob("*/ab.json"))
    ab.cmd_pool(types.SimpleNamespace(files=files, out=str(tmp_path / "s"), expected=4))
    out = capsys.readouterr().out
    P = json.loads((tmp_path / "s" / "pool.json").read_text(encoding="utf-8"))
    assert P["missing_copies"] == [3] and P["expected"] == 4
    text = (tmp_path / "s" / "pool.md").read_text(encoding="utf-8")
    assert "**Incomplete: 1 of 4 copies missing (job failed or artifact missing): copy 3." in text
    assert "::warning title=Copies missing::1 of 4 copies have no result: copy 3" in out
    ab.cmd_pool(types.SimpleNamespace(files=files, out=str(tmp_path / "s2"), expected=3))
    assert "Incomplete" in (tmp_path / "s2" / "pool.md").read_text(encoding="utf-8")   # copy 3 of 3 missing
    ab.cmd_pool(types.SimpleNamespace(files=files[:2], out=str(tmp_path / "s3"), expected=2))
    assert "Incomplete" not in (tmp_path / "s3" / "pool.md").read_text(encoding="utf-8")
