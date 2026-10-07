"""Launch the dashboard:  python3 -m dashboard  [--host H] [--port P]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args()
    import uvicorn
    print(f"Personal Cambrian — Mission Control  ->  http://{args.host}:{args.port}")
    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
