"""Command line:  python -m nasc_ci check --results jmh-result.json --history nasc_history.json
                 --commit <sha> [--label v07] --rerun-cmd "<command template>" [--report report.md]

  --no-save            check only, do not store the version (pull requests: compare, don't record)
  --fail-on-confirmed  exit with code 1 when a slowdown is confirmed (to fail the CI job)
  --set key=value      change a setting, e.g. --set detector=bootstrap --set ttest_tau=3
"""

import argparse
import sys

import numpy as np

from . import VERSION, history, jmh
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
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):          # the report has emoji; never crash on a Windows console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(VERSION, file=sys.stderr)
    overrides = dict(kv.split("=", 1) for kv in a.set)
    h = history.load(a.history, {k: parse_value(v) for k, v in overrides.items()})
    results = jmh.read(a.results)
    label = a.label or a.commit[:7]
    runner = CommandRunner(a.rerun_cmd)
    rows = run_check(h, results, a.commit, label, runner, np.random.default_rng(a.seed), save=not a.no_save)
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
