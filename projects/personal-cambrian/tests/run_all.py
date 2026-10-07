#!/usr/bin/env python3
"""Run every test module and report one pass/fail summary (the project's CI gate).

Each test file is a zero-install standalone runner that prints "N/M passed" and
exits non-zero on failure; this just shells out to each and tallies.

    python3 tests/run_all.py
"""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    files = sorted(f for f in glob.glob(os.path.join(HERE, "test_*.py")))
    total_pass = total = 0
    failed = []
    print(f"running {len(files)} test modules\n" + "-" * 48)
    for path in files:
        name = os.path.basename(path)
        r = subprocess.run([sys.executable, path], capture_output=True, text=True)
        m = re.search(r"(\d+)/(\d+) passed", r.stdout)
        if m:
            p, t = int(m.group(1)), int(m.group(2))
            total_pass += p
            total += t
            status = "ok" if r.returncode == 0 else "FAIL"
            print(f"  {name:24s} {p:>3}/{t:<3} {status}")
        else:
            status = "ERROR"
            print(f"  {name:24s}   ?    ERROR")
            print(r.stdout[-500:])
            print(r.stderr[-500:])
        if r.returncode != 0:
            failed.append(name)
    print("-" * 48)
    print(f"TOTAL  {total_pass}/{total} passed across {len(files)} modules")
    if failed:
        print("FAILED modules: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
