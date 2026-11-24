"""check_benchmarks v1 (24 Nov 2026): are all case-study benchmarks in a JMH jar's list?
python3 check_benchmarks.py <output of 'java -jar <jmh jar> -l'> <position> <sha>
Prints a Markdown line for the job summary; exits 1 if a benchmark is missing."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
wanted = [l.strip() for l in (HERE / "benchmarks.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
listed = {l.strip() for l in Path(sys.argv[1]).read_text().splitlines()}
pos, sha = sys.argv[2], sys.argv[3]
missing = [b for b in wanted if b not in listed]
if missing:
    print(f"| {pos} | `{sha}` | ❌ {len(wanted) - len(missing)} of {len(wanted)}; missing: {', '.join(missing)} |")
    sys.exit(1)
print(f"| {pos} | `{sha}` | ✅ all {len(wanted)} benchmarks in the JMH jar |")
