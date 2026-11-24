"""Which machine a job ran on (v0.3, 24 Nov 2026): CPU model, CPU count, clock source, OS.

Cloud runners of the same label can land on different CPU types, and the pilot saw a ~60 %
speed difference between two of them, so every recorded version stores this next to its results.
Nothing personal is read: no host name, user name or path.

  python -m nasc_ci machine            prints it as JSON
"""

import os
import platform
import subprocess


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _cpuinfo_field(text, name):
    for line in text.splitlines():
        if line.split(":")[0].strip() == name:
            return line.split(":", 1)[1].strip()
    return None


def describe():
    """-> {"cpu_model", "cpus", "cpu_mhz", "clocksource", "system", "kernel", "arch"}"""
    cpuinfo = _read("/proc/cpuinfo")
    model = _cpuinfo_field(cpuinfo, "model name")
    if model is None and platform.system() == "Darwin":
        try:
            model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            model = None
    clock = _read("/sys/devices/system/clocksource/clocksource0/current_clocksource").strip() or None
    mhz = _cpuinfo_field(cpuinfo, "cpu MHz")
    return {"cpu_model": model or platform.processor() or "unknown",
            "cpus": os.cpu_count(),
            "cpu_mhz": float(mhz) if mhz else None,
            "clocksource": clock,
            "system": platform.system(),
            "kernel": platform.release(),
            "arch": platform.machine()}


def cpu_of(machine):
    """Short CPU label of a stored machine dict (None if unknown)."""
    return (machine or {}).get("cpu_model")
