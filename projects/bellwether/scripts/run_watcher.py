"""Run the filing watcher.

Usage:
  python scripts/run_watcher.py backfill   # seed recent history (no doc download)
  python scripts/run_watcher.py poll        # one real-time sweep (+ downloads)
  python scripts/run_watcher.py watch        # loop forever, every N minutes
"""

from __future__ import annotations

import sys
import time

from bellwether import db, watcher
from bellwether.config import ROOT
from bellwether.logging_setup import get_logger

log = get_logger("bellwether.run_watcher")
INTERVAL_MIN = 15  # poll cadence for `watch`


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    conn = db.init_db()

    if cmd == "backfill":
        watcher.backfill(conn)
    elif cmd == "poll":
        watcher.poll_once(conn)
    elif cmd == "watch":
        log.info("watching every %d min — Ctrl-C to stop", INTERVAL_MIN)
        while True:
            try:
                watcher.poll_once(conn)
            except Exception as e:
                log.error("poll error: %s", e)
            time.sleep(INTERVAL_MIN * 60)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
