#!/usr/bin/env python3
"""Read-only ntfy phone alerts for Fable Five training.

Examples:
    SUPRA_NTFY_TOPIC=pydrive787b python3 tools/notify_fable5_ntfy.py
    python3 tools/notify_fable5_ntfy.py --topic pydrive787b --heartbeat-minutes 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MILESTONES = (0.50, 0.75, 0.90, 0.93, 0.96, 1.00)
CRASH_PAT = re.compile(r"(traceback|exception|nan|runtimeerror|segmentation fault)", re.I)
DEFAULT_HEARTBEAT_MINUTES = 10.0


def _topic_url(server: str, topic: str) -> str:
    return server.rstrip("/") + "/" + urllib.parse.quote(topic.strip(), safe="")


def send_ntfy(server: str, topic: str, title: str, message: str,
              priority: str = "default", tags: str = "", dry_run: bool = False) -> bool:
    if dry_run:
        print(f"[dry-run] {title}: {message}")
        return True
    cmd = [
        "curl", "-fsS", "--max-time", "10",
        "-H", f"Title: {title}",
        "-H", f"Priority: {priority}",
    ]
    if tags:
        cmd += ["-H", f"Tags: {tags}"]
    cmd += ["--data-binary", "@-", _topic_url(server, topic)]
    try:
        subprocess.run(cmd, input=message, text=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        # Notifications are an observability sidecar. A transient DNS/TLS/ntfy
        # outage must not kill the only watcher that can report recovery later.
        detail = getattr(exc, "stderr", None) or str(exc)
        print(f"[ntfy] send failed ({title}): {str(detail).strip()}",
              file=sys.stderr, flush=True)
        return False


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def find_training_processes() -> list[str]:
    try:
        p = subprocess.run(["pgrep", "-fl", "run.py.*--fable"],
                           text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, check=False)
    except Exception:
        return []
    return [line.strip() for line in p.stdout.splitlines()
            if line.strip() and "supervise_fable5.py" not in line]


def _fmt(value, digits: int = 3, missing: str = "?") -> str:
    if value is None:
        return missing
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _pct(value, digits: int = 1, missing: str = "?") -> str:
    if value is None:
        return missing
    try:
        return f"{float(value) * 100.0:.{digits}f}%"
    except Exception:
        return str(value)


def _mtime(path: Path | None) -> str:
    if not path or not path.exists():
        return "missing"
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
    except Exception:
        return "unknown"


def _format_terms(terms: dict | None) -> str:
    if not terms:
        return "none"
    items = sorted(terms.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return ", ".join(f"{k}:{v}" for k, v in items[:6])


def _format_worst(worst: list | None) -> str:
    if not worst:
        return "none"
    out = []
    for item in worst[:5]:
        if not isinstance(item, dict):
            continue
        sector = item.get("sector", "?")
        pace = _fmt(item.get("pace"), 3)
        clean = "clean" if item.get("clean") else "bad"
        reason = item.get("reason") or "no_reason"
        out.append(f"s{sector}:{pace}/{clean}/{reason}")
    return ", ".join(out) or "none"


def _format_stage(name: str, ev: dict) -> str:
    if not ev:
        return f"{name}: missing"
    lap = ev.get("lap_time")
    lap_s = f" lap={_fmt(lap, 2)}s" if lap else ""
    return (
        f"{name}: metric={_fmt(ev.get('metric'))} "
        f"progress={_pct(ev.get('progress_frac'))} "
        f"clean={ev.get('clean_sectors', '?')}/{ev.get('sector_count', 16)} "
        f"chain={ev.get('clean_chain', '?')} pace={_fmt(ev.get('pace_ratio'))}"
        f"{lap_s}"
    )


def rich_manifest_summary(manifest: dict, procs: list[str] | None = None,
                          manifest_path: Path | None = None,
                          pit_path: Path | None = None,
                          run_log_path: Path | None = None) -> str:
    ev = manifest.get("latest_eval") or {}
    stages = manifest.get("stages") or {}
    auto = manifest.get("auto") or {}
    pit = manifest.get("pit") or {}
    hof = manifest.get("hall_of_fame") or {}
    procs = procs or []

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stage = manifest.get("current_stage") or ev.get("stage") or "unknown"
    rec = manifest.get("recommended_next_action") or ev.get("recommendation") or "none"
    status = "running" if procs else "not detected"
    proc_line = procs[0] if procs else "none"

    lines = [
        f"Snapshot: {now}",
        f"Process: {status} | {proc_line[:160]}",
        f"Pipeline: {manifest.get('pipeline', '?')} | track={manifest.get('track', '?')} | car={manifest.get('car', '?')}",
        f"Stage: {stage} | recommendation={rec}",
        f"Checkpoint: active={manifest.get('active_checkpoint')} | best={manifest.get('best_checkpoint')} | resumed_from={manifest.get('resumed_from')}",
        f"Files: manifest_mtime={_mtime(manifest_path)} | pit_mtime={_mtime(pit_path)} | run_log={run_log_path.name if run_log_path else 'none'} mtime={_mtime(run_log_path)}",
        f"Reward: {manifest.get('fable_reward_version')} | envelope_scale={_fmt(manifest.get('envelope_scale'), 3)} | theoretical={_fmt(manifest.get('theoretical_lap'), 1)}s",
        "",
        "Latest Eval:",
        f"  metric={_fmt(ev.get('metric'))} | progress={_pct(ev.get('progress_frac'))} ({_fmt(ev.get('max_progress_m'), 0)}m) | laps={_fmt(ev.get('laps'))}",
        f"  clean={ev.get('clean_sectors', '?')}/{ev.get('sector_count', 16)} | chain={ev.get('clean_chain', '?')} | terminal={_fmt(ev.get('terminal_rate'))} | offtrack={_fmt(ev.get('offtrack_seconds'), 2)}s",
        f"  pace={_fmt(ev.get('pace_ratio'))} | mean_speed={_fmt(ev.get('mean_speed'), 2)}m/s | lap_time={_fmt(ev.get('lap_time'), 2, 'none')} | style={ev.get('lap_style', '?')}",
        f"  worst={_format_worst(ev.get('worst_sectors'))}",
        f"  terms={_format_terms(ev.get('termination_counts'))}",
    ]

    latest_log = ev.get("log")
    if latest_log:
        lines.append(f"  log={str(latest_log)[-500:]}")

    lines += ["", "Stage Bests:"]
    for name in ("foundation", "flow", "finish", "fast", "frontier"):
        if name in stages:
            lines.append("  " + _format_stage(name, stages.get(name) or {}))

    if auto:
        hist = auto.get("history") or []
        lines += [
            "",
            f"Auto: prefix={auto.get('prefix')} spent={auto.get('spent')}/{auto.get('total')} history={len(hist)}",
        ]
        for h in hist[-4:]:
            bits = [str(h.get("stage", "?"))]
            for key in ("segment", "attempt", "iters", "metric", "progress_frac",
                        "recommendation", "scale", "next_scale", "skipped",
                        "gated", "soft_advanced"):
                if key in h:
                    val = h.get(key)
                    if key in ("metric", "progress_frac", "scale", "next_scale"):
                        val = _fmt(val, 3)
                    bits.append(f"{key}={val}")
            lines.append("  " + " | ".join(bits))

    if pit:
        decisions = pit.get("decisions") or []
        lines += [
            "",
            f"Pit: stage={pit.get('stage')} run={pit.get('run_name')} reseeds={pit.get('reseeds')} since_best={pit.get('reseeds_since_best')} consolidations={pit.get('consolidations')}",
        ]
        for d in decisions[-5:]:
            if isinstance(d, dict):
                lines.append(
                    "  "
                    f"{d.get('t', '?')} {d.get('stage', '?')} "
                    f"metric={_fmt(d.get('metric'))} best={_fmt(d.get('best'))} "
                    f"decision={d.get('decision')} reason={str(d.get('reason', ''))[:180]}"
                )

    if hof:
        lines.append("")
        lines.append("Hall Of Fame:")
        for key, val in sorted(hof.items()):
            if isinstance(val, dict):
                lines.append(
                    f"  {key}: metric={_fmt(val.get('metric'))} "
                    f"progress={_pct(val.get('progress_frac'))} "
                    f"clean={val.get('clean_sectors', '?')}/{val.get('sector_count', 16)} "
                    f"lap={_fmt(val.get('lap_time'), 2, 'none')}"
                )
            else:
                lines.append(f"  {key}: {val}")

    return "\n".join(lines)


def manifest_summary(manifest: dict) -> str:
    ev = manifest.get("latest_eval") or {}
    stage = manifest.get("current_stage") or ev.get("stage") or "unknown"
    metric = ev.get("metric")
    progress = ev.get("progress_frac")
    clean = ev.get("clean_sectors")
    sectors = ev.get("sector_count") or 16
    rec = manifest.get("recommended_next_action") or ev.get("recommendation")
    parts = [f"stage={stage}"]
    if metric is not None:
        parts.append(f"metric={float(metric):.3f}")
    if progress is not None:
        parts.append(f"progress={float(progress) * 100:.1f}%")
    if clean is not None:
        parts.append(f"clean={clean}/{sectors}")
    if rec:
        parts.append(str(rec))
    return " | ".join(parts)


def classify_pit(line: str) -> tuple[str, str, str, str] | None:
    if "[pit]" not in line:
        return None
    priority = "default"
    tags = "racing_car"
    if "new best banked" in line:
        title = "Fable new best"
        priority = "high"
        tags = "trophy,racing_car"
    elif "RESEED" in line:
        title = "Fable reseed"
        tags = "recycle,racing_car"
    elif "CONSOLIDATE" in line:
        title = "Fable consolidate"
        priority = "high"
        tags = "wrench,racing_car"
    else:
        return None
    msg = line.strip()
    return title, msg, priority, tags


def classify_run_log(line: str) -> tuple[str, str, str, str] | None:
    s = line.strip()
    if not s:
        return None
    low = s.lower()
    if CRASH_PAT.search(s):
        return "Fable possible error", s[-900:], "urgent", "warning,racing_car"
    if "STALL:" in s or "FATAL" in s or "restart budget exhausted" in low:
        return "Fable supervisor alert", s[-900:], "urgent", "warning,racing_car"
    if "PAUSED low disk" in s:
        return "Fable paused: low disk", s[-900:], "urgent", "warning,racing_car"
    if "disk recovered" in low:
        return "Fable disk recovered", s[-900:], "high", "white_check_mark,racing_car"
    if "unexpected stop" in low or ("trainer exit=" in low and "exit=0" not in low):
        return "Fable trainer restarting", s[-900:], "high", "recycle,racing_car"
    if "gate" in low and ("advance" in low or "stopping" in low or "banked" in low):
        return "Fable gate", s[-900:], "high", "checkered_flag,racing_car"
    if "soft-advances" in low or "auto ladder done" in low:
        return "Fable ladder", s[-900:], "high", "checkered_flag,racing_car"
    if "[interrupted]" in low or "saving checkpoint" in low:
        return "Fable checkpoint", s[-900:], "default", "floppy_disk,racing_car"
    return None


class Tailer:
    def __init__(self, path: Path, start_at_end: bool = True):
        self.path = path
        self.pos = 0
        self.exists = False
        if start_at_end and path.exists():
            self.pos = path.stat().st_size
            self.exists = True

    def poll(self) -> list[str]:
        if not self.path.exists():
            self.exists = False
            return []
        size = self.path.stat().st_size
        if not self.exists or size < self.pos:
            self.pos = size
            self.exists = True
            return []
        if size == self.pos:
            return []
        with self.path.open("r", errors="replace") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        return data.splitlines()


def current_run_log(logpath_file: Path) -> Path | None:
    try:
        raw = logpath_file.read_text().strip()
    except Exception:
        return None
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else Path.cwd() / p


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default=os.environ.get("SUPRA_NTFY_TOPIC"),
                    help="ntfy topic, or set SUPRA_NTFY_TOPIC")
    ap.add_argument("--server", default=os.environ.get("SUPRA_NTFY_SERVER", "https://ntfy.sh"))
    ap.add_argument("--manifest", default=str(ROOT / "fable5_ring_pipeline.json"))
    ap.add_argument("--pit-log", default=str(ROOT / "fable5_pit_log.txt"))
    ap.add_argument("--logpath-file", default=str(ROOT / "fable5_supervisor.logpath"),
                    help="file containing the current stdout log path, if present")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--heartbeat-minutes", type=float, default=DEFAULT_HEARTBEAT_MINUTES,
                    help="0 disables periodic summaries")
    ap.add_argument("--stale-minutes", type=float, default=30.0,
                    help="urgent alert when a running supervisor checkpoint is this old")
    ap.add_argument("--once", action="store_true",
                    help="send one current summary and exit")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.topic:
        print("error: pass --topic or set SUPRA_NTFY_TOPIC", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest)
    pit_path = Path(args.pit_log)
    pit_tail = Tailer(pit_path)
    run_log_tail: Tailer | None = None
    run_log_path: Path | None = None
    last_stage = None
    last_rec = None
    best_progress = 0.0
    sent_milestones: set[float] = set()
    was_running: bool | None = None
    stale_alerted = False
    heartbeat_s = max(0.0, args.heartbeat_minutes * 60.0)
    next_heartbeat = time.time() + heartbeat_s if heartbeat_s else 0.0

    manifest = read_json(manifest_path)
    procs = find_training_processes()
    startup = (rich_manifest_summary(manifest, procs, manifest_path, pit_path, run_log_path)
               if manifest else "manifest not found yet")
    send_ntfy(args.server, args.topic, "Fable watcher started", startup,
              "default", "eyes,racing_car", args.dry_run)
    initial_progress = float((manifest.get("latest_eval") or {}).get(
        "progress_frac") or 0.0)
    best_progress = initial_progress
    sent_milestones = {m for m in MILESTONES if initial_progress >= m}
    if args.once:
        return 0

    while True:
        now_log_path = current_run_log(Path(args.logpath_file))
        if now_log_path and now_log_path != run_log_path:
            run_log_path = now_log_path
            run_log_tail = Tailer(run_log_path)

        manifest = read_json(manifest_path)
        procs = find_training_processes()
        supervisor_state = (read_json(run_log_path.parent / "supervisor_state.json")
                            if run_log_path else {})
        full_status = rich_manifest_summary(manifest, procs, manifest_path,
                                            pit_path, run_log_path)

        for line in pit_tail.poll():
            classified = classify_pit(line)
            if classified:
                title, msg, priority, tags = classified
                send_ntfy(args.server, args.topic, title,
                          f"{msg}\n\n{full_status}", priority, tags,
                          dry_run=args.dry_run)

        if run_log_tail is not None:
            for line in run_log_tail.poll():
                classified = classify_run_log(line)
                if classified:
                    title, msg, priority, tags = classified
                    send_ntfy(args.server, args.topic, title,
                              f"{msg}\n\n{full_status}", priority, tags,
                              dry_run=args.dry_run)

        ev = manifest.get("latest_eval") or {}
        stage = manifest.get("current_stage") or ev.get("stage")
        rec = manifest.get("recommended_next_action") or ev.get("recommendation")
        progress = float(ev.get("progress_frac") or 0.0)

        if stage and last_stage and stage != last_stage:
            send_ntfy(args.server, args.topic, "Fable stage changed",
                      full_status, "high",
                      "checkered_flag,racing_car", args.dry_run)
            sent_milestones.clear()
            best_progress = 0.0
        if rec and last_rec and rec != last_rec:
            send_ntfy(args.server, args.topic, "Fable recommendation changed",
                      full_status, "high",
                      "clipboard,racing_car", args.dry_run)

        if progress > best_progress:
            best_progress = progress
            for milestone in MILESTONES:
                if progress >= milestone and milestone not in sent_milestones:
                    sent_milestones.add(milestone)
                    send_ntfy(args.server, args.topic,
                              f"Fable reached {milestone * 100:.0f}%",
                              full_status, "high",
                              "checkered_flag,racing_car", args.dry_run)

        running = bool(procs)
        if was_running is not None and was_running and not running:
            status = supervisor_state.get("status")
            if status == "complete":
                send_ntfy(args.server, args.topic, "Fable training complete",
                          full_status, "high", "checkered_flag,racing_car",
                          args.dry_run)
            elif status == "stopped":
                send_ntfy(args.server, args.topic, "Fable training stopped",
                          full_status, "default", "stop_sign,racing_car",
                          args.dry_run)
            else:
                send_ntfy(args.server, args.topic, "Fable process stopped",
                          full_status, "urgent",
                          "warning,racing_car", args.dry_run)
        was_running = running
        last_stage = stage or last_stage
        last_rec = rec or last_rec

        # The supervisor writes an authoritative checkpoint heartbeat every
        # 15 seconds. Alert once on a stall and once when it clears; do not spam
        # every polling interval while the trainer is already being recovered.
        age = float(supervisor_state.get("checkpoint_age_seconds") or 0.0)
        stale = (running and args.stale_minutes > 0
                 and supervisor_state.get("status") == "running"
                 and age >= args.stale_minutes * 60.0)
        if stale and not stale_alerted:
            send_ntfy(args.server, args.topic, "Fable checkpoint stalled",
                      f"Checkpoint heartbeat is {age / 60.0:.1f} minutes old.\n\n"
                      f"{full_status}", "urgent", "warning,racing_car",
                      args.dry_run)
            stale_alerted = True
        elif stale_alerted and running and not stale:
            send_ntfy(args.server, args.topic, "Fable heartbeat recovered",
                      full_status, "high", "white_check_mark,racing_car",
                      args.dry_run)
            stale_alerted = False
        elif stale_alerted and not running:
            stale_alerted = False

        if heartbeat_s and time.time() >= next_heartbeat:
            send_ntfy(args.server, args.topic, "Fable heartbeat", full_status,
                      "low", "blue_heart,racing_car", args.dry_run)
            next_heartbeat = time.time() + heartbeat_s

        time.sleep(max(2.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
