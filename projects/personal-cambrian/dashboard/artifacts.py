"""Artifact service: index + safely serve the run/render outputs as a gallery."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ["renders", "runs"]
_KINDS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
          ".mp4": "video", ".webm": "video", ".json": "json", ".nwk": "text",
          ".jsonl": "text", ".txt": "text"}


def kind_of(path: str) -> str:
    return _KINDS.get(os.path.splitext(path)[1].lower(), "file")


def list_artifacts(limit: int = 300) -> list:
    out = []
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in _KINDS:
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                out.append({"path": rel, "name": fn, "kind": kind_of(fn),
                            "size": st.st_size, "mtime": st.st_mtime,
                            "dir": os.path.relpath(dirpath, ROOT)})
    out.sort(key=lambda a: a["mtime"], reverse=True)
    return out[:limit]


def safe_path(rel: str):
    """Resolve a relative path under ROOT, refusing traversal outside it."""
    full = os.path.realpath(os.path.join(ROOT, rel))
    if not full.startswith(os.path.realpath(ROOT) + os.sep):
        return None
    return full if os.path.isfile(full) else None


def artifacts_modified_since(start: float, dirs=("renders", "runs")) -> list:
    """Files created/modified since `start` -- used to attach a job's outputs."""
    out = []
    for d in dirs:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in _KINDS:
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    if os.stat(full).st_mtime >= start - 0.5:
                        out.append(os.path.relpath(full, ROOT))
                except OSError:
                    pass
    return sorted(out)
