from __future__ import annotations

import argparse
import json

from .progress import ProgressStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python-dojo", description="Run Python Dojo.")
    sub = parser.add_subparsers(dest="command")
    web = sub.add_parser("web", help="Start the local browser app.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8797)
    sub.add_parser("term", help="Start the terminal dojo.")
    sub.add_parser("health", help="Print local health information.")
    sub.add_parser("reset-progress", help="Reset local learner progress.")
    args = parser.parse_args(argv)

    if args.command in (None, "web"):
        from .server import create_app

        create_app().run(host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8797), debug=False)
        return 0
    if args.command == "term":
        from .terminal import run_terminal

        return run_terminal()
    if args.command == "health":
        from .tutor import TutorCoach

        store = ProgressStore()
        payload = {
            "ok": True,
            "db_path": str(store.path),
            "progress": store.state().as_dict(),
            "settings": store.settings().__dict__,
            "ollama_available": TutorCoach(store.settings()).ollama_available(),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "reset-progress":
        print(json.dumps(ProgressStore().reset().as_dict(), indent=2))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
