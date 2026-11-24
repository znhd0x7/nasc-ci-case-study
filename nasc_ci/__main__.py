"""Command line:  python -m nasc_ci check --results jmh-result.json --history nasc_history.json
                 --commit <sha> [--label v07] --rerun-cmd "<command template>" [--report report.md]

  --no-save            check only, do not store the version (pull requests: compare, don't record)
  --fail-on-confirmed  exit with code 1 when a slowdown is confirmed (to fail the CI job)
  --set key=value      change a setting, e.g. --set detector=bootstrap --set ttest_tau=3
  --machine auto|none|FILE   machine that recorded the results (default auto: this machine)
  --code FILE          compiled-code hash of the commit (JSON written by the codehash command)

                 python -m nasc_ci machine [--out machine.json]
                 python -m nasc_ci codehash DIR [DIR ...] [--out code.json]
"""

import argparse
import json
import sys

import numpy as np

from . import VERSION, codehash, history, jmh, machine
from .check import report, run_check
from .runner import CommandRunner


def parse_value(text):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def main(argv=None):
    p = argparse.ArgumentParser(prog="nasc_ci", description=VERSION)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check one new version")
    c.add_argument("--results", required=True, help="JMH JSON of the new version")
    c.add_argument("--history", required=True, help="history JSON (created if missing)")
    c.add_argument("--commit", required=True)
    c.add_argument("--label", help="short name of the version (default: first 7 characters of the commit)")
    c.add_argument("--rerun-cmd", required=True, help="command template for confirmation runs")
    c.add_argument("--report", help="write the Markdown report here")
    c.add_argument("--no-save", action="store_true")
    c.add_argument("--fail-on-confirmed", action="store_true")
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    c.add_argument("--quiet", action="store_true", help="do not print the report")
    c.add_argument("--machine", default="auto", help="auto, none, or a JSON file from the machine command")
    c.add_argument("--code", help="JSON file from the codehash command")
    m = sub.add_parser("machine", help="print this machine's CPU model etc. as JSON")
    m.add_argument("--out")
    ch = sub.add_parser("codehash", help="hash of the compiled classes under the given folders")
    ch.add_argument("folders", nargs="+")
    ch.add_argument("--out")
    a = p.parse_args(argv)
    if a.command in ("machine", "codehash"):
        info = machine.describe() if a.command == "machine" else codehash.code_hash(a.folders)
        text = json.dumps(info, indent=1)
        print(text)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        return 0
    if hasattr(sys.stdout, "reconfigure"):          # the report has emoji; never crash on a Windows console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(VERSION, file=sys.stderr)
    overrides = dict(kv.split("=", 1) for kv in a.set)
    h = history.load(a.history, {k: parse_value(v) for k, v in overrides.items()})
    results = jmh.read(a.results)
    label = a.label or a.commit[:7]
    runner = CommandRunner(a.rerun_cmd)
    if a.machine == "auto":
        machine_info = machine.describe()
    elif a.machine == "none":
        machine_info = None
    else:
        with open(a.machine, encoding="utf-8") as f:
            machine_info = json.load(f)
    code = None
    if a.code:
        with open(a.code, encoding="utf-8") as f:
            code = json.load(f)
    rows = run_check(h, results, a.commit, label, runner, np.random.default_rng(a.seed), save=not a.no_save,
                     machine_info=machine_info, code=code)
    text = report(rows, h, a.commit, label)
    if not a.quiet:
        print(text)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(text)
    if not a.no_save:
        history.save(h, a.history)
    if a.fail_on_confirmed and any(r["decision"] == "confirmed" for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
