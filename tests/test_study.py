"""Tests of case-study/study.py, step 1 (26 Nov 2026): settings, injections per commit, recording,
packing a class path, the completeness gate and the frozen noise profile. No JMH is run: the
recordings are synthetic JMH sample-mode JSON with the shape the record jobs produce.

    python -m pytest -q tests/test_study.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-study"))
sys.path.insert(0, str(ROOT))

import study  # noqa: E402
from nasc_ci import noise  # noqa: E402

S = study.settings()
SHORT = [b.split(".")[-1] for b in S["benchmarks"]]


def jmh_json(levels, forks=3, iterations=20, seed=0, time_="1 s", drop=None):
    """Synthetic JMH sample-mode JSON: one entry per benchmark, each iteration a small histogram."""
    rng = np.random.default_rng(seed)
    out = []
    for b in S["benchmarks"]:
        if drop and b.endswith(drop):
            continue
        level = levels.get(b.split(".")[-1], 100.0)
        hist = []
        for _ in range(forks):
            run_level = level * (1 + rng.normal(0, 0.01))
            hist.append([[[round(run_level * (1 + rng.normal(0, 0.01)), 3), 7],
                          [round(run_level * (1 + rng.normal(0, 0.01)), 3), 3]] for _ in range(iterations)])
        out.append({"benchmark": b, "mode": "sample", "forks": forks, "warmupIterations": 10,
                    "measurementIterations": iterations, "warmupTime": time_, "measurementTime": time_,
                    "primaryMetric": {"scoreUnit": "ns/op", "rawDataHistogram": hist}})
    return out


def record(tmp, pos, levels=None, cpu="AMD EPYC 7763 64-Core Processor", **kw):
    raw = tmp / f"raw{pos}.json"
    raw.write_text(json.dumps(jmh_json(levels or {}, seed=pos, **kw)))
    m = tmp / f"m{pos}.json"
    m.write_text(json.dumps({"cpu_model": cpu, "cpus": 4}))
    c = tmp / f"c{pos}.json"
    c.write_text(json.dumps({"sha256": f"{pos:064x}", "parts": {}}))
    t = tmp / f"t{pos}.txt"
    t.write_text("job_start=100\nmeasure_start=160\nmeasure_end=1060\n")
    out = tmp / "records" / f"study-record-c{pos:02d}"
    study.main(["record", "--json", str(raw), "--pos", str(pos), "--commit", study.commits()[pos],
                "--patches", " ".join(study.patches_at(pos)), "--machine", str(m), "--code", str(c),
                "--timing", str(t), "--out", str(out)])
    return out


def test_settings_benchmarks():
    listed = [l.strip() for l in study.BENCHMARKS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(S["benchmarks"]) == 10 and len(set(S["benchmarks"])) == 10
    assert set(S["benchmarks"]) <= set(listed)
    assert not set(S["excluded_timer_floor"]) & set(S["benchmarks"])
    assert set(S["benchmarks"]) | set(S["excluded_timer_floor"]) == set(listed)
    for _, _, targets in study.injections():
        assert set(targets) <= set(SHORT)
    assert S["nasc"]["k_max"] == 5 and S["nasc"]["alpha"] == S["nasc"]["beta"] == 0.05
    assert S["nasc"]["delta_min"] == 0.02 and S["nasc"]["window"] == 5 and S["nasc"]["ttest_tau"] == 2.0


def test_patches_persist_from_their_commit():
    assert study.patches_at(8) == []
    assert study.patches_at(9) == ["heapsort"]
    assert study.patches_at(12) == ["heapsort"]
    assert study.patches_at(13) == ["heapsort", "indexof"]
    assert study.patches_at(15) == ["heapsort", "indexof"]
    assert study.patches_at(9, without_new=True) == []
    assert study.patches_at(13, without_new=True) == ["heapsort"]
    assert [study.new_patch(p) for p in range(1, 16)].count(None) == 13
    assert study.new_patch(9) == "heapsort" and study.new_patch(13) == "indexof"
    assert study.step_of("io.x.IndexSorterPerf.heapSort250") == ("heapsort", 9)
    assert study.step_of("io.x.ProbabilitySelectorIndexOfPerf.serialIndexOf250") == ("indexof", 13)
    assert study.step_of("io.x.RandomPerf.nextLong") is None


def test_regex_selects_exactly_the_ten(capsys):
    study.main(["regex"])
    rx = re.compile(capsys.readouterr().out.strip())
    listed = [l.strip() for l in study.BENCHMARKS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert [b for b in listed if rx.search(b)] == [b for b in listed if b in S["benchmarks"]]
    assert not rx.search("io.jenetics.internal.util.IndexSorterPerf.heapSort8")


def test_record_writes_blocks_and_iteration_means(tmp_path):
    out = record(tmp_path, 9, {"heapSort80": 1350.0})
    r = json.loads((out / "results.json").read_text())
    assert r["complete"] and r["pos"] == 9 and r["label"] == "c09"
    assert [p["name"] for p in r["patches"]] == ["heapsort"]
    assert r["patches"][0]["sha256"] == study.patch_sha("heapsort")
    assert r["machine"]["cpu_model"].startswith("AMD") and r["timing"]["measure_end"] == 1060
    runs = np.array(r["results"][S["benchmarks"][0]]["runs"])
    assert runs.shape == (3, 4) and abs(runs.mean() / 1350 - 1) < 0.05
    rows = (out / "iteration_means.csv").read_text().splitlines()
    assert len(rows) == 1 + 10 * 3 * 20


@pytest.mark.parametrize("change, message", [
    (dict(drop="nextLong"), "nextLong: missing"),
    (dict(forks=2), "forks = 2"),
    (dict(iterations=15), "measurementIterations = 15"),
    (dict(time_="2 s"), "warmupTime"),
])
def test_record_refuses_incomplete(tmp_path, capsys, change, message):
    with pytest.raises(SystemExit) as e:
        record(tmp_path, 3, **change)
    assert e.value.code == 1
    assert message in capsys.readouterr().out


def test_record_refuses_wrong_patches_or_commit(tmp_path, capsys):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(jmh_json({})))
    with pytest.raises(SystemExit):
        study.main(["record", "--json", str(raw), "--pos", "13", "--commit", study.commits()[13],
                    "--patches", "heapsort", "--out", str(tmp_path / "o")])
    assert "expected ['heapsort', 'indexof']" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        study.main(["record", "--json", str(raw), "--pos", "2", "--commit", study.commits()[3],
                    "--out", str(tmp_path / "o")])
    assert "is not position 2" in capsys.readouterr().out


def test_pack_and_classpath(tmp_path, capsys):
    a = tmp_path / "x" / "lib.jar"
    a.parent.mkdir()
    a.write_bytes(b"jar")
    b = tmp_path / "y" / "lib.jar"                       # same file name, other folder
    b.parent.mkdir()
    b.write_bytes(b"jar2")
    d = tmp_path / "classes"
    (d / "p").mkdir(parents=True)
    (d / "p" / "A.class").write_bytes(b"c")
    cp = tmp_path / "cp.txt"
    cp.write_text(f"{a}:{b}:{d}\n")
    study.main(["pack", "--classpath", str(cp), "--out", str(tmp_path / "packed")])
    capsys.readouterr()
    study.main(["classpath", "--dir", str(tmp_path / "packed")])
    parts = capsys.readouterr().out.strip().split(":")
    assert len(parts) == 3 and Path(parts[0]).read_bytes() == b"jar" and Path(parts[1]).read_bytes() == b"jar2"
    assert (Path(parts[2]) / "p" / "A.class").exists()


def test_gate_names_missing_and_wrong_commits(tmp_path, capsys):
    for pos in range(1, 16):
        if pos != 7:
            record(tmp_path, pos)
    capsys.readouterr()
    found, problems = study.load_records(tmp_path / "records")
    assert len(found) == 14
    assert problems == ["commits not recorded (job failed or artifact missing): 7"]
    record(tmp_path, 7)
    found, problems = study.load_records(tmp_path / "records")
    assert len(found) == 15 and problems == []
    f = tmp_path / "records" / "study-record-c12" / "results.json"
    r = json.loads(f.read_text())
    r["patches"] = []
    r["complete"] = False
    f.write_text(json.dumps(r))
    _, problems = study.load_records(tmp_path / "records")
    assert any("commit 12: recording not marked complete" in p for p in problems)
    assert any("commit 12: patches []" in p for p in problems)
    with pytest.raises(SystemExit):
        study.main(["profile", "--records", str(tmp_path / "records"), "--out", str(tmp_path / "p")])


def test_profile_frozen_from_commits_1_to_5(tmp_path, capsys):
    cpus = {3: "INTEL(R) XEON(R) PLATINUM 8573C"}
    levels = {p: {"heapSort80": 1350.0 * (1.3 if p == 9 else 1.0)} for p in range(1, 16)}
    for pos in range(1, 16):
        record(tmp_path, pos, levels[pos], cpu=cpus.get(pos, "AMD EPYC 7763 64-Core Processor"))
    capsys.readouterr()
    study.main(["profile", "--records", str(tmp_path / "records"), "--out", str(tmp_path / "p")])
    text = capsys.readouterr().out
    P = json.loads((tmp_path / "p" / "profile.json").read_text())
    prof = P["profile"]
    assert prof["versions"] == [1, 2, 3, 4, 5]
    assert prof["mixed_cpu"] and len(prof["cpu_types"]) == 2
    assert "MIXED CPU types" in text
    h = json.loads((tmp_path / "p" / "history.json").read_text())
    assert [v["index"] for v in h["versions"]] == list(range(1, 16))
    b = S["benchmarks"][0]
    runs = [np.array(v["results"][b]["runs"]) for v in h["versions"][:5]]
    expected = noise.estimate(runs, h["settings"])
    assert prof["benchmarks"][b]["c_w"] == pytest.approx(expected.c_w)
    assert prof["benchmarks"][b]["n_b"] == expected.n_b
    # commit 9's 30 % jump must not enter the frozen profile
    assert prof["benchmarks"][b]["c_t"] < 0.05
    assert set(P["reference"]) == set(S["benchmarks"])
