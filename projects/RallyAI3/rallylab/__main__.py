import argparse
import uvicorn

parser = argparse.ArgumentParser(description="Local Rally Research Lab")
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
uvicorn.run("rallylab.api:app", host="127.0.0.1", port=args.port)
