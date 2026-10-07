"""
Braindance Dream Machine - Main Application Entrypoint
Launches the full interactive Web Studio dashboard.
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    port = 7860
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    print(f"🌌 Starting Braindance Dream Machine Dashboard on http://127.0.0.1:{port}...")
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False, log_level="info")
