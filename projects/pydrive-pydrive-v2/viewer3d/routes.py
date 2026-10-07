"""Flask routes and WebSocket bridge for the standalone 3D viewers.

/3d/         -> v2 (the from-scratch rebuild, VIEWER3D_V2_PLAN.md)
/3d/legacy/  -> v1, frozen (viewer3d/legacy/)
Both share the same per-connection physics bridge at /api/3d/ws.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flask import send_from_directory
from flask_sock import Sock

from .session import SimSession
from supra.sound import SpatialAudioMixer
import numpy as np


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"                      # v2
LEGACY_STATIC = HERE / "legacy" / "static"    # frozen v1


def register_viewer3d(app):
    """Attach standalone 3D viewer routes to an existing Flask app."""
    sock = Sock(app)

    def nocache(resp):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/3d/")
    def viewer3d_index():
        return nocache(send_from_directory(STATIC, "index.html"))

    @app.route("/3d/static/<path:filename>")
    def viewer3d_static(filename):
        return nocache(send_from_directory(STATIC, filename))

    @app.route("/3d/legacy/")
    def viewer3d_legacy_index():
        return nocache(send_from_directory(LEGACY_STATIC, "index.html"))

    @app.route("/3d/legacy/static/<path:filename>")
    def viewer3d_legacy_static(filename):
        return nocache(send_from_directory(LEGACY_STATIC, filename))

    @sock.route("/api/3d/ws")
    def viewer3d_ws(ws):
        session = SimSession()
        audio = SpatialAudioMixer()
        camera_cut = False

        last_send = 0.0
        send_hz = 30.0
        ws.send(json.dumps({"type": "event", "event": "ready"}))

        try:
            while True:
                try:
                    raw = ws.receive(timeout=0.001)
                except TimeoutError:
                    raw = None
                except Exception:
                    break

                if raw:
                    try:
                        data = json.loads(raw)
                        kind = data.get("type")
                        if kind == "start":
                            session.configure(
                                car=data.get("car", "supra"),
                                track_name=data.get("track", "club"),
                                seed=int(data.get("seed", 7)),
                                style=data.get("style"),
                                difficulty=float(data.get("difficulty", 0.5)),
                                length=(float(data["length"]) if data.get("length") else None),
                                checkpoint=data.get("checkpoint"),
                            )
                            audio.stop()
                            audio.start([session.veh])
                            camera_cut = True
                            ws.send(json.dumps(session.track_payload()))
                            # AI watch feedback: who's driving, or the loud
                            # reason nobody is (pre-hills checkpoint, etc.)
                            if getattr(session, "agent_error", None):
                                ws.send(json.dumps({
                                    "type": "event", "event": "notice",
                                    "level": "warn",
                                    "message": f"AI load failed: {session.agent_error}"}))
                            elif session.agent and session.agent_meta:
                                m = session.agent_meta
                                ws.send(json.dumps({
                                    "type": "event", "event": "notice",
                                    "message": (f"🧠 {m['name']} driving "
                                                f"({m['mode']}, {m['updates']}u)")}))
                            ws.send(json.dumps(session.state_payload()))
                        elif kind == "input":
                            session.driver.update_from_payload(data)
                        elif kind == "reset":
                            session.reset()
                            audio.reset([session.veh])
                            camera_cut = True
                            ws.send(json.dumps({"type": "event", "event": "reset"}))
                            ws.send(json.dumps(session.state_payload()))
                        elif kind == "camera":
                            camera_cut = True
                            ws.send(json.dumps({"type": "event", "event": "camera",
                                                "mode": data.get("mode", "chase")}))
                        elif kind == "mute":
                            muted = audio.toggle_mute()
                            ws.send(json.dumps({"type": "event", "event": "mute", "muted": muted}))
                    except Exception as e:
                        try:
                            ws.send(json.dumps({"type": "error", "message": str(e)}))
                        except Exception:
                            break

                session.tick_wall()
                # live-training watch: the trainer rewrites its rolling
                # checkpoint each iteration; swap the brain in mid-drive
                if session.maybe_reload_agent():
                    m = session.agent_meta or {}
                    ws.send(json.dumps({
                        "type": "event", "event": "notice",
                        "message": (f"🧠 brain updated → {m.get('updates', '?')}u"
                                    + (f" · eval {m['metric']:.2f}"
                                       if m.get("metric") else ""))}))
                if audio.ok:
                    vx = session.veh.speed * np.cos(session.veh.yaw)
                    vy = session.veh.speed * np.sin(session.veh.yaw)
                    session.veh._audio_brake = session.b_in
                    session.veh._audio_clutch = (session.gearbox.clutch if session.agent else session.driver.clutch)
                    audio.update((session.veh.x, session.veh.y), (vx, vy), session.veh.yaw,
                                 [session.veh], [session.t_in], brakes=[session.b_in],
                                 clutches=[session.veh._audio_clutch], perspective=1,
                                 camera_cut=camera_cut)
                    camera_cut = False

                now = time.perf_counter()
                if now - last_send >= 1.0 / send_hz:
                    try:
                        ws.send(json.dumps(session.state_payload()))
                    except Exception:
                        break
                    last_send = now
                time.sleep(0.001)
        finally:
            audio.stop()
