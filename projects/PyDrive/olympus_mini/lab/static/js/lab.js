// Olympus Mini Laboratory Dashboard Controller

let currentTask = "sprint";
let currentCam = "side";
let isManualMode = false;
let isPaused = false;
let orbitHistory = [];
const MAX_ORBIT_POINTS = 120;

// Initialize when DOM loads
document.addEventListener("DOMContentLoaded", () => {
    initControls();
    initKeyListeners();
    startTelemetryLoop();
});

function initControls() {
    // Discipline Switchers
    document.querySelectorAll(".btn-task").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".btn-task").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentTask = btn.dataset.task;
            sendControlCommand("set_task", { task: currentTask });
        });
    });

    // Camera Switchers
    document.querySelectorAll(".btn-cam").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".btn-cam").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentCam = btn.dataset.cam;
            sendControlCommand("set_camera", { camera: currentCam });
        });
    });

    // Play/Pause Button
    const playBtn = document.getElementById("btn-play-pause");
    playBtn.addEventListener("click", () => {
        sendControlCommand("toggle_pause", {});
    });

    // Reset Button
    document.getElementById("btn-reset").addEventListener("click", () => {
        sendControlCommand("reset", {});
        orbitHistory = [];
    });

    // Manual Mode Toggle
    const modeBtn = document.getElementById("btn-mode-toggle");
    modeBtn.addEventListener("click", () => {
        isManualMode = !isManualMode;
        modeBtn.classList.toggle("active", isManualMode);
        modeBtn.innerText = isManualMode ? "🕹️ MANUAL DRIVE: ON" : "🤖 AUTONOMOUS AI";
        document.getElementById("manual-controls-panel").style.display = isManualMode ? "flex" : "none";
        sendControlCommand("set_mode", { mode: isManualMode ? "manual" : "auto" });
    });

    // Manual Sliders
    const leanSlider = document.getElementById("slider-lean");
    const cadenceSlider = document.getElementById("slider-cadence");
    const thrustSlider = document.getElementById("slider-thrust");

    const updateManualAction = () => {
        if (!isManualMode) return;
        fetch("/api/manual_action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                lean: parseFloat(leanSlider.value),
                cadence: parseFloat(cadenceSlider.value),
                thrust: parseFloat(thrustSlider.value),
                jump: 0.0
            })
        });
        document.getElementById("val-lean").innerText = leanSlider.value;
        document.getElementById("val-cadence").innerText = cadenceSlider.value + " Hz";
        document.getElementById("val-thrust").innerText = Math.round(thrustSlider.value * 100) + "%";
    };

    leanSlider.addEventListener("input", updateManualAction);
    cadenceSlider.addEventListener("input", updateManualAction);
    thrustSlider.addEventListener("input", updateManualAction);

    // Record Clip Button
    const recordBtn = document.getElementById("btn-record");
    recordBtn.addEventListener("click", () => {
        recordBtn.innerText = "⏺️ RECORDING (3s)...";
        recordBtn.disabled = true;
        fetch("/api/record_clip", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                setTimeout(() => {
                    recordBtn.innerText = "✅ CLIP SAVED";
                    setTimeout(() => {
                        recordBtn.innerText = "📹 CAPTURE GIF";
                        recordBtn.disabled = false;
                    }, 2500);
                }, 3500);
            });
    });

    // Training Controls
    const trainBtn = document.getElementById("btn-toggle-train");
    trainBtn.addEventListener("click", () => {
        const isTraining = trainBtn.dataset.training === "true";
        if (!isTraining) {
            const steps = parseInt(document.getElementById("train-steps").value) || 20000;
            const lr = parseFloat(document.getElementById("train-lr").value) || 0.0003;
            fetch("/api/train/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ task: currentTask, steps: steps, lr: lr })
            }).then(() => {
                trainBtn.dataset.training = "true";
                trainBtn.innerText = "⏹️ STOP TRAINING";
                trainBtn.classList.add("active");
            });
        } else {
            fetch("/api/train/stop", { method: "POST" }).then(() => {
                trainBtn.dataset.training = "false";
                trainBtn.innerText = "🚀 START PPO TRAINING";
                trainBtn.classList.remove("active");
            });
        }
    });
}

function sendControlCommand(cmd, payload) {
    fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd, ...payload })
    });
}

function initKeyListeners() {
    window.addEventListener("keydown", (e) => {
        if (!isManualMode) return;
        const lean = document.getElementById("slider-lean");
        const cadence = document.getElementById("slider-cadence");
        let updated = false;
        
        if (e.key === "w" || e.key === "ArrowUp") {
            lean.value = (parseFloat(lean.value) + 0.05).toFixed(2);
            updated = true;
        } else if (e.key === "s" || e.key === "ArrowDown") {
            lean.value = (parseFloat(lean.value) - 0.05).toFixed(2);
            updated = true;
        } else if (e.key === "d" || e.key === "ArrowRight") {
            cadence.value = (parseFloat(cadence.value) + 0.1).toFixed(1);
            updated = true;
        } else if (e.key === "a" || e.key === "ArrowLeft") {
            cadence.value = (parseFloat(cadence.value) - 0.1).toFixed(1);
            updated = true;
        } else if (e.code === "Space") {
            fetch("/api/manual_action", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ jump: 0.5 })
            });
            setTimeout(() => {
                fetch("/api/manual_action", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ jump: 0.0 })
                });
            }, 250);
        }
        
        if (updated) {
            document.getElementById("val-lean").innerText = lean.value;
            document.getElementById("val-cadence").innerText = cadence.value + " Hz";
            fetch("/api/manual_action", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    lean: parseFloat(lean.value),
                    cadence: parseFloat(cadence.value)
                })
            });
        }
    });
}

function startTelemetryLoop() {
    setInterval(() => {
        fetch("/api/telemetry")
            .then(res => res.json())
            .then(data => {
                updateUI(data);
            })
            .catch(err => {});
    }, 50);
}

function updateUI(t) {
    if (!t) return;
    
    // Play/Pause button text
    const playBtn = document.getElementById("btn-play-pause");
    if (playBtn) {
        playBtn.innerText = t.is_paused ? "▶️ PLAY" : "⏸️ PAUSE";
    }

    // Top status
    document.getElementById("live-fps").innerText = `${t.fps || 0} FPS`;
    document.getElementById("active-task-pill").innerText = (t.task || "sprint").toUpperCase();

    // HUD overlay on video
    document.getElementById("hud-task").innerText = (t.task || "").toUpperCase();
    document.getElementById("hud-speed").innerText = (t.vx || 0.0).toFixed(2);
    document.getElementById("hud-dist").innerText = (t.dist || 0.0).toFixed(2);
    document.getElementById("hud-height").innerText = (t.height || 0.0).toFixed(2);
    document.getElementById("hud-step").innerText = t.step || 0;

    // Metric cards
    document.getElementById("metric-speed").innerText = (t.vx || 0.0).toFixed(2);
    document.getElementById("metric-dist").innerText = (t.dist || 0.0).toFixed(2);
    document.getElementById("metric-height").innerText = (t.height || 0.0).toFixed(2);
    document.getElementById("metric-phase").innerText = (t.phase || 0.0).toFixed(2);

    // Event-specific metrics
    const hurdleMetric = document.getElementById("metric-hurdles-box");
    if (hurdleMetric) {
        hurdleMetric.style.display = (t.task === "hurdle") ? "block" : "none";
        document.getElementById("metric-hurdles").innerText = t.hurdles_cleared || 0;
    }
    const vaultMetric = document.getElementById("metric-vault-box");
    if (vaultMetric) {
        vaultMetric.style.display = (t.task === "vault") ? "block" : "none";
        document.getElementById("metric-peak-z").innerText = (t.peak_z || 0.0).toFixed(2);
    }

    // Render Foot Heatmap
    if (t.foot_contacts) {
        renderFootHeatmap(t.foot_contacts);
    }

    // Render Knee Phase-Plane Orbit
    if (t.knee_angle !== undefined && t.knee_vel !== undefined) {
        orbitHistory.push({ angle: t.knee_angle, vel: t.knee_vel });
        if (orbitHistory.length > MAX_ORBIT_POINTS) {
            orbitHistory.shift();
        }
        renderPhaseOrbit();
    }

    // Render Training Chart if active
    if (t.training) {
        const trainStatus = document.getElementById("train-status-text");
        if (t.training.is_training) {
            trainStatus.innerText = `Training... Step: ${t.training.step}/${t.training.total_steps} | Return: ${t.training.mean_reward}`;
        } else {
            trainStatus.innerText = "PPO Trainer Idle. Ready to launch.";
        }
        if (t.training.history && t.training.history.length > 0) {
            renderTrainingChart(t.training.history);
        }
    }
}

// 1. Biomechanical Foot Heatmap
function renderFootHeatmap(contacts) {
    const leftCanvas = document.getElementById("canvas-foot-left");
    const rightCanvas = document.getElementById("canvas-foot-right");
    if (!leftCanvas || !rightCanvas) return;

    drawFootSole(leftCanvas, [contacts[0], contacts[1], contacts[2], contacts[3]], true);
    drawFootSole(rightCanvas, [contacts[4], contacts[5], contacts[6], contacts[7]], false);
}

function drawFootSole(canvas, padStates, isLeft) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Sole outline
    ctx.strokeStyle = "#273347";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(10, 10, w - 20, h - 20, [25, 25, 15, 15]);
    ctx.stroke();

    // 4 Contact Pads: [toe_in, toe_out, heel_in, heel_out]
    const pads = isLeft
        ? [{ x: 50, y: 35, s: padStates[0] }, { x: 25, y: 35, s: padStates[1] }, { x: 45, y: 105, s: padStates[2] }, { x: 25, y: 105, s: padStates[3] }]
        : [{ x: 25, y: 35, s: padStates[0] }, { x: 50, y: 35, s: padStates[1] }, { x: 25, y: 105, s: padStates[2] }, { x: 45, y: 105, s: padStates[3] }];

    pads.forEach(pad => {
        ctx.beginPath();
        ctx.arc(pad.x, pad.y, 10, 0, 2 * Math.PI);
        if (pad.s > 0.5) {
            ctx.fillStyle = "#00ffaa";
            ctx.shadowColor = "#00ffaa";
            ctx.shadowBlur = 10;
        } else {
            ctx.fillStyle = "#161d27";
            ctx.shadowBlur = 0;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
    });
}

// 2. Knee Limit-Cycle Phase-Plane Orbit Plot
function renderPhaseOrbit() {
    const canvas = document.getElementById("canvas-phase-orbit");
    if (!canvas || orbitHistory.length < 2) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Grid center lines
    ctx.strokeStyle = "#1a2230";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.stroke();

    // Map: angle [-2.5, 0.2] rad -> x [0, w], vel [-12, 12] rad/s -> y [h, 0]
    const mapX = (a) => ((a + 2.5) / 2.7) * w;
    const mapY = (v) => h - ((v + 12.0) / 24.0) * h;

    // Draw trajectory trail
    for (let i = 1; i < orbitHistory.length; i++) {
        const p1 = orbitHistory[i - 1];
        const p2 = orbitHistory[i];
        const alpha = (i / orbitHistory.length).toFixed(2);

        ctx.strokeStyle = `rgba(255, 119, 0, ${alpha})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(mapX(p1.angle), mapY(p1.vel));
        ctx.lineTo(mapX(p2.angle), mapY(p2.vel));
        ctx.stroke();
    }

    // Draw current head point
    const head = orbitHistory[orbitHistory.length - 1];
    ctx.beginPath();
    ctx.arc(mapX(head.angle), mapY(head.vel), 4, 0, 2 * Math.PI);
    ctx.fillStyle = "#00f0ff";
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
}

// 3. Training Progress Curves
function renderTrainingChart(history) {
    const canvas = document.getElementById("canvas-training");
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Find min and max rewards
    const rewards = history.map(h => h.reward);
    const minR = Math.min(...rewards, 0);
    const maxR = Math.max(...rewards, 10);
    const range = Math.max(1.0, maxR - minR);

    // Draw Reward Line
    ctx.strokeStyle = "#00ffaa";
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((pt, idx) => {
        const x = (idx / (history.length - 1)) * (w - 20) + 10;
        const y = h - ((pt.reward - minR) / range) * (h - 30) - 15;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
}
