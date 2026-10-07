"""Local production daemon.

Runs the SEC watcher poll and the LLM pipeline processing in a loop every 15 minutes.
"""

from __future__ import annotations

import time
import subprocess
import sys
from pathlib import Path

# Add root directory to python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bellwether.logging_setup import get_logger

log = get_logger("bellwether.local_daemon")
INTERVAL_MIN = 15

def main() -> None:
    log.info("Starting local production daemon (polling every %d minutes)...", INTERVAL_MIN)
    while True:
        log.info("--- Starting Sweep ---")
        try:
            # 1. Run watcher poll
            log.info("Running watcher poll...")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "run_watcher.py"), "poll"], check=True)
            
            # 2. Run backlog process
            log.info("Running pipeline backlog processing...")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "process_backlog.py"), "--limit", "50"], check=True)
            
            log.info("Sweep completed successfully.")
        except subprocess.CalledProcessError as e:
            log.error("Subprocess execution failed: %s", e)
        except Exception as e:
            log.error("Unexpected error during sweep: %s", e)
            
        log.info("Sleeping for %d minutes...", INTERVAL_MIN)
        time.sleep(INTERVAL_MIN * 60)

if __name__ == "__main__":
    main()
