"""Phase 3 — Entry point for background processing of new filings.

Sweeps the database for downloaded filings that haven't been passed through
the LLM extraction pipeline yet.
"""

from __future__ import annotations

import argparse

from bellwether import alerts, db, pipeline
from bellwether.logging_setup import get_logger

log = get_logger("bellwether.process_backlog")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Max filings to process")
    args = parser.parse_args()

    conn = db.init_db()
    unprocessed = db.get_unprocessed_filings(conn, limit=args.limit)
    
    if not unprocessed:
        log.info("No unprocessed filings found in the backlog.")
        return
        
    log.info("Found %d unprocessed filings. Starting pipeline...", len(unprocessed))
    
    total_signals = 0
    for row in unprocessed:
        log.info("Processing filing %s (%s) for %s...", row["id"], row["form_type"], row["ticker"])
        try:
            signal_ids = pipeline.process_filing(conn, row)
            total_signals += len(signal_ids)
            for sid in signal_ids:
                alerts.maybe_send_alert(conn, sid)
            db.mark_filing_processed(conn, row["id"])
        except Exception as e:
            err_str = str(e).upper()
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "QUOTA" in err_str
            if is_rate_limit:
                log.error("Failed to process filing %s due to rate limit: %s. Stopping backlog run.", row["id"], e)
                break
            else:
                log.error("Failed to process filing %s due to permanent error: %s. Marking as processed anyway.", row["id"], e)
                db.mark_filing_processed(conn, row["id"])
                
        # Pace requests to respect Gemini free tier RPM limits
        import time
        from bellwether import config
        provider = config.require("DEEP_READ_PROVIDER")
        if provider == "gemini" and not config.USE_MOCK_LLM:
            time.sleep(4.0)
            
    log.info("Finished processing backlog. Extracted %d total signals.", total_signals)

if __name__ == "__main__":
    main()
