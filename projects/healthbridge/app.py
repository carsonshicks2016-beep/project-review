"""
HealthBridge AI - compatibility launcher

Running `python3 app.py` now starts the upgraded templated application
while preserving the familiar default port.
"""

from app_v2_templated import app


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print("=" * 64)
    print("HealthBridge AI Structured Edition")
    print("Deterministic analysis, confidence scoring, and methodology view")
    print(f"Running at http://127.0.0.1:{port}")
    print("=" * 64)
    app.run(debug=False, host="127.0.0.1", port=port)
