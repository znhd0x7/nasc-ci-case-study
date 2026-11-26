"""Tests of case-study/study.py, step 3 (26 Nov 2026): the scorer and the report, on a synthetic
full study (15 recordings, the profile, 10 confirm jobs with a fake runner, 4 same-VM A/B files).
The equality with nasc_sim.score_series / check_series is tested in the private thesis repository
(nasc_sim.py is not public); here the report's arithmetic and its completeness gate are checked.

    python -m pytest -q tests/test_study_report.py
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-study"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import study  # noqa: E402
from test_study import record  # noqa: E402
from test_study_confirm import FakeRunner  # noqa: E402

S = study.settings()
STEP = {"heapSort80": 0.12, "heapSort250": 0.12, "heapSort320": 0.10, "serialIndexOf250": 0.34}
BASE = {"heapSort80": 1350.0, "heapSort250": 4600.0, "heapSort320": 6200.0, "serialIndexOf250": 90.0}


def level(short, pos):
    if short not in STEP:
        return 100.0
    v_star = 13 if short == "serialIndexOf250" else 9
    return BASE[short] * (1 + STEP[short] if pos >= v_star else 1)


@pytest.fixture(scope="module")
def study_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("study")
    cpus = {p: ("INTEL(R) XEON(R) PLATINUM 8573C" if p in (7, 11) else "AMD EPYC 7763 64-Core Processor")
            for p in range(1, 16)}
    for pos in range(1, 16):
        record(tmp, pos, {b: level(b, pos) for b in STEP}, cpu=cpus[pos])
    study.main(["profile", "--records", str(tmp / "records"), "--out", str(tmp / "profile")])
    h = json.loads((tmp / "profile" / "history.json").read_text())
    for b in S["benchmarks"]:
        sb = study.short(b)
        R = study.run_confirm(h, b, FakeRunner(lambda p, sb=sb: level(sb, p)),
                              {"cpu_model": "AMD EPYC 9V74 80-Core Processor", "cpus": 4})
        d = tmp / "confirms" / f"study-confirm-{sb}"
        d.mkdir(parents=True)
        (d / "confirm.json").write_text(json.dumps(R))
        if study.step_of(b):
            g = tmp / "ground-truth" / f"study-gt-{sb}"
            g.mkdir(parents=True)
            (g / "ab.json").write_text(json.dumps({
                "machine": {"cpu_model": "AMD EPYC 9V74 80-Core Processor"}, "a": {"pos": 9}, "b": {"pos": 9},
                "same_code": False, "results": [{"benchmark": b, "difference": STEP[sb], "ci95_welch": 0.01,
                                                 "pairs": 10, "order_effect": 0.0}]}))
    return tmp


def args(d, out="out"):
    return ["report", "--records", str(d / "records"), "--profile", str(d / "profile"), "--confirms",
            str(d / "confirms"), "--ground-truth", str(d / "ground-truth"), "--out", str(d / out)]


def test_score_rules_by_hand():
    rows = [[6, 1, 1, 0], [7, 1, 0, 40], [8, 0, 0, 0], [9, 1, 1, 40], [10, 1, 1, 0], [11, 0, 0, 0],
            [12, 0, 0, 0], [13, 0, 0, 0], [14, 0, 0, 0], [15, 1, 1, 20]]
    x = study.score(rows, 9)
    assert (x["clean_checks"], x["clean_flags"], x["false_alerts"]) == (3, 2, 1)
    assert x["detected"] and x["delay"] == 0 and x["duplicates"] == 2 and x["flagged_at_v_star"]
    assert x["confirm_blocks"] == 100 and x["checked_versions"] == 10
    y = study.score(rows, None)
    assert (y["clean_checks"], y["false_alerts"], y["detected"], y["delay"]) == (10, 4, False, None)
    z = study.score([[6, 0, 0, 0], [13, 1, 0, 8], [14, 1, 1, 8], [15, 0, 0, 0]], 13)
    assert z["delay"] == 1 and z["duplicates"] == 0 and z["flagged_at_v_star"]


def test_summarize_pools_like_summarize_scores():
    a = study.score([[6, 1, 1, 0], [9, 1, 1, 40]] + [[k, 0, 0, 0] for k in (7, 8, 10, 11, 12, 13, 14, 15)], 9)
    b = study.score([[k, 0, 0, 0] for k in range(6, 16)], None)
    x = study.summarize([a, b])
    assert x["clean_checks"] == 13 and x["false_alerts"] == 1 and x["FPR"] == pytest.approx(1 / 13)
    assert x["recall"] == 1 and x["recall_at_v_star"] == 1 and x["FDP"] == pytest.approx(0.5)
    assert x["T"] == pytest.approx((12 * 20 + 40) / (12 * 20))


def test_report_complete_study(study_dir, capsys):
    study.main(args(study_dir))
    text = capsys.readouterr().out
    R = json.loads((study_dir / "out" / "study.json").read_text())
    assert R["limitation"] == {"checks": 100, "clean": 76, "at_v_star": 4, "after_v_star": 20, "instances": 4}
    assert len(R["pooled"]) == 12
    for name, x in R["pooled"].items():
        assert x["steps"] == 4 and x["clean_checks"] == 76
    assert R["pooled"]["threshold | static"]["detected"] == 4
    assert R["pooled"]["threshold | none"]["confirm_blocks"] == 0
    gt = {study.short(g["benchmark"]): g for g in R["ground_truth"]}
    assert set(gt) == set(STEP) and gt["heapSort80"]["truth"]["difference"] == pytest.approx(0.12)
    assert gt["heapSort80"]["background"] is None                    # 9V74 not calibrated for heapsort
    assert gt["serialIndexOf250"]["background"] == [0.3393, 0.0119]
    assert gt["heapSort80"]["delay"]["ttest | nasc"] == 0
    kinds = {k.split(" | ")[1] for k in R["base_flags_by_record_cpus"]}
    assert kinds == {"same", "mixed"}                                  # commits 7 and 11 on Intel
    assert R["profile_cpus"]["mixed_cpu"] is False
    assert "INCOMPLETE" not in text and "### Scores per detector and arm" in text
    assert "4 injected regression instances among 100" in text


def test_report_gate_refuses_incomplete(study_dir, capsys):
    shutil.rmtree(study_dir / "confirms" / "study-confirm-nextLong")
    shutil.rmtree(study_dir / "ground-truth" / "study-gt-heapSort250")
    with pytest.raises(SystemExit) as e:
        study.main(args(study_dir, "out2"))
    assert e.value.code == 1
    text = (study_dir / "out2" / "report.md").read_text()
    assert text.startswith("## INCOMPLETE")
    assert "benchmarks not confirmed (job failed or artifact missing): nextLong" in text
    assert "same-VM ground truth missing (job failed or artifact missing): heapSort250" in text
    assert not (study_dir / "out2" / "study.json").exists()


def test_sd_check():
    prof = {"c_w": 0.01, "c_t": 0.005, "n_b": 4}
    calm = [{"batches": [{"candidate": 100 * (1 + e), "reference": 100.0} for e in (0.0, 0.001, -0.001, 0.0005, 0.0)]}]
    wild = [{"batches": [{"candidate": 100 * (1 + e), "reference": 100.0} for e in (0.0, 0.05, -0.05, 0.04, -0.03)]}]
    assert not study.sd_check(calm, prof)["higher"]
    x = study.sd_check(wild, prof)
    assert x["higher"] and x["df"] == 4 and x["chi2_95"] == pytest.approx(9.488)
    assert study.chi2_95(40) == pytest.approx(55.758, abs=0.05)       # Wilson-Hilferty above df 30
