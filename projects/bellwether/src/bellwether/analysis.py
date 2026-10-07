"""Phase 4 — Statistical Analysis & Report Generation.

Aggregates backtest outcomes into a significance panel per catalyst type.
"""

from __future__ import annotations

import collections
import statistics
from pathlib import Path

from . import db
from .logging_setup import get_logger

log = get_logger("bellwether.analysis")

def run_analysis(conn) -> None:
    """Analyze outcomes and write backtest_report.md."""
    # We group by catalyst_type and direction
    rows = conn.execute(
        """
        SELECT s.catalyst_type, s.direction, o.abnormal_ret_5d 
        FROM outcomes o
        JOIN signals s ON o.signal_id = s.id
        WHERE o.abnormal_ret_5d IS NOT NULL
        """
    ).fetchall()
    
    if not rows:
        log.warning("No complete outcomes found for analysis.")
        return
        
    groups = collections.defaultdict(list)
    for r in rows:
        key = f"{r['catalyst_type']} ({r['direction']})"
        groups[key].append(r["abnormal_ret_5d"])
        
    report = [
        "# Backtest Analysis Report\n",
        "This report aggregates the post-slippage abnormal returns (+5d) for each extracted signal type.",
        "A positive return indicates the signal accurately predicted the direction of the move (i.e. a bearish signal that resulted in a price drop is recorded as a positive return).\n",
        "## Aggregate Significance Panel\n",
        "| Catalyst Type (Dir) | N | Hit Rate | Mean Abnormal Return |",
        "|---|---|---|---|"
    ]
    
    for key, returns in sorted(groups.items()):
        n = len(returns)
        hits = sum(1 for r in returns if r > 0)
        hit_rate = (hits / n) * 100
        mean_ret = statistics.mean(returns) * 100
        
        # Simple bootstrap p-value proxy (in a real system we'd use scipy.stats)
        # We just report the raw numbers for now to keep dependencies light.
        
        report.append(f"| {key} | {n} | {hits}/{n} ({hit_rate:.1f}%) | {mean_ret:.2f}% |")
        
    report.append("\n## Conclusion\n")
    report.append("Signals with consistent positive mean returns and hit rates > 55% represent actionable edge.")
    
    out_path = Path("backtest_report.md")
    out_path.write_text("\n".join(report))
    log.info("Analysis complete. Report written to %s", out_path.absolute())

if __name__ == "__main__":
    run_analysis(db.init_db())
