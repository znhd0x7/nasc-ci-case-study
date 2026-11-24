"""Re-runs a benchmark for one NASC confirmation batch.

The command is a template; NASC-CI fills in
  {rev}        commit to measure (the candidate's, or the parent's for the reference side)
  {label}      the version's label in the history (the demo uses it instead of a commit)
  {benchmark}  benchmark name without its [params]
  {iterations} measurement iterations = n_b blocks x 5
  {out}        JSON file the command must write (JMH format)
  {seed}       a different number for every re-run (the demo uses it to make a replay repeatable)
  {period}     a name for this confirmation job; both sides of every batch run in it, back to back
               (only the demo uses it: fake_jmh gives all runs of one period the same machine offset)
Real JMH, for example (the parent's jar is built once in its own folder):
  java -jar build/{rev}/benchmarks.jar "{benchmark}$" -f 1 -wi 10 -i {iterations} -rf json -rff {out}
Demo (no Java needed):
  python demo/fake_jmh.py --label {label} --benchmark "{benchmark}" --forks 1 --iterations {iterations} \
      --period {period} --out {out}
"""

import os
import subprocess
import sys
import tempfile

from . import jmh


class CommandRunner:
    def __init__(self, template, verbose=False, seed=None):
        self.template = template
        self.seed = seed
        self.verbose = verbose
        self.calls = 0

    def measurer(self, benchmark, candidate, reference):
        """-> measure(side, n_blocks) for nasc_confirm. candidate / reference: version entries."""
        method = benchmark.split(" [")[0]

        def measure(side, n_blocks):
            v = candidate if side == "candidate" else reference
            fd, out = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            try:
                cmd = self.template.format(rev=v["commit"], label=v["label"], benchmark=method,
                                           period=f"confirm-{candidate['commit']}",
                                           seed=(self.seed or 0) * 100003 + self.calls,
                                           iterations=n_blocks * jmh.BLOCK_SIZE, out=f"\"{out}\"")
                if self.verbose:
                    print("   re-run:", cmd, file=sys.stderr)
                subprocess.run(cmd, shell=True, check=True)
                self.calls += 1
                return jmh.mean_of(out, benchmark)
            finally:
                os.remove(out)
        return measure
