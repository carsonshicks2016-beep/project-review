#!/usr/bin/env python3
"""Tiny Raspberry Pi server for the LR4 Valet Copilot prototype.

It serves the static dashboard and exposes a same-origin `/api/llm/brief`
endpoint. The browser talks to this server; this server talks to Ollama on
the Pi, avoiding phone/tablet localhost and CORS problems.
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request


SYSTEM_FALLBACK = (
    "You are the local valet copilot for Carson's 2015 Land Rover LR4 HSE. "
    "Be concise, calm, safety-minded, and do not recommend sending vehicle commands."
)


def fallback_brief(packet: dict) -> dict:
    obd = packet.get("obd", [])
    hot = [item for item in obd if item.get("risk") != "good"][:3]
    focus = ", ".join(item.get("label", "sensor").lower() for item in hot) or "coolant, voltage, and fuel trims"
    line = f"Hello, Carson. I am reviewing {focus}; the LR4 is within baseline with a few deltas worth watching."
    note = "Pi proxy fallback used because the local model did not answer."
    return {"line": line, "note": note, "text": f"{line} NOTE: {note}", "fallback": True}


def parse_model_text(text: str) -> dict:
    clean = " ".join((text or "").split())
    if "NOTE:" in clean:
        line, note = clean.split("NOTE:", 1)
    else:
        line, note = clean, "Local model generated this brief from the current OBD and IMU packet."
    return {"line": line.strip("\"' "), "note": note.strip(), "text": clean}


class ValetHandler(SimpleHTTPRequestHandler):
    server_version = "LR4Valet/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "model": self.server.model,
                    "ollama": self.server.ollama,
                    "message": "LR4 valet proxy online",
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/llm/brief":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        payload = self.read_json()
        packet = payload.get("packet", {})
        try:
            text = self.call_ollama(payload)
            self.send_json(parse_model_text(text))
        except Exception as exc:  # Keep the car UI graceful if the model is cold/offline.
            result = fallback_brief(packet)
            result["error"] = str(exc)
            self.send_json(result)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def call_ollama(self, payload: dict) -> str:
        model = payload.get("model") or self.server.model
        system = payload.get("system") or SYSTEM_FALLBACK
        prompt = payload.get("prompt") or json.dumps(payload.get("packet", {}), indent=2)
        body = json.dumps(
            {
                "model": model,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 90},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.server.ollama.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.server.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc
        return data.get("message", {}).get("content", "")

    def send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LR4 Valet Copilot and proxy local LLM requests.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--model", default=os.environ.get("LR4_LLM_MODEL", "llama3.2:1b"))
    parser.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ValetHandler)
    server.model = args.model
    server.ollama = args.ollama
    server.timeout_seconds = args.timeout
    print(f"LR4 Valet Copilot serving http://{args.host}:{args.port}")
    print(f"Proxying Ollama at {args.ollama} with model {args.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLR4 Valet Copilot server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
