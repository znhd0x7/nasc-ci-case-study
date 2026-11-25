"""Tests of case-study/ab_check.py v2 (25 Nov 2026) with a fake machine, no Java needed.

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
