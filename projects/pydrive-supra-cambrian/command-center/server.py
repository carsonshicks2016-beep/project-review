import os
import signal
import subprocess
import threading
import time
from flask import Flask, Response, jsonify, request, send_from_directory, send_file

app = Flask(__name__, static_folder=None)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# State: pid -> subprocess.Popen object
active_procs = {}

@app.route("/")
def index():
    return send_file(os.path.join(HERE, "dom.html"))

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(HERE, "static"), path)

@app.route("/api/launch", methods=["POST"])
def launch():
    data = request.json
    gens = int(data.get("gens", 3))
    pop = int(data.get("pop", 4))
    track = data.get("track", "club")
    out = data.get("out", "evolution_log.json")
    iters = int(data.get("iters", 25))
    
    cmd = [
        "python3", "tools/evolve_car.py",
        "--gens", str(gens),
        "--pop", str(pop),
        "--track", track,
        "--out", out,
        "--iters", str(iters)
    ]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid  # Put in own process group so we can SIGINT it and its children
    )
    
    active_procs[str(proc.pid)] = {
        "proc": proc,
        "cmd": " ".join(cmd),
        "start": time.time()
    }
    
    return jsonify({"pid": str(proc.pid)})

@app.route("/api/stream/<pid>")
def stream(pid):
    if pid not in active_procs:
        return jsonify({"error": "No such process"}), 404
        
    proc = active_procs[pid]["proc"]
    
    def generate():
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue
                
            yield f"data: {line.strip()}\n\n"
            
        # Check exit code
        if proc.returncode == 0:
            yield "data: [PROCESS EXITED CLEANLY]\n\n"
        else:
            yield f"data: [PROCESS EXITED WITH ERROR CODE {proc.returncode}]\n\n"
            
        if pid in active_procs:
            del active_procs[pid]
            
    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/status")
def status():
    # Cleanup dead procs
    dead = []
    for p, info in active_procs.items():
        if info["proc"].poll() is not None:
            dead.append(p)
    for p in dead:
        del active_procs[p]
        
    res = []
    for p, info in active_procs.items():
        res.append({
            "pid": p,
            "cmd": info["cmd"],
            "uptime": int(time.time() - info["start"])
        })
    return jsonify({"runs": res})

@app.route("/api/stop/<pid>", methods=["POST"])
def stop(pid):
    if pid not in active_procs:
        return jsonify({"error": "No such process"}), 404
        
    proc = active_procs[pid]["proc"]
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except:
            pass
            
    if pid in active_procs:
        del active_procs[pid]
        
    return jsonify({"status": "killed"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8770)
