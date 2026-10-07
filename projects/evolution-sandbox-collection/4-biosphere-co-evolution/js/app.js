/**
 * app.js
 * Core Simulation Manager. Handles canvas rendering, simulation loops,
 * UI controls, telemetry graphs, and integration between physics, GA, and PPO.
 */

// Simulation parameters
const canvas = document.getElementById('sim-canvas');
const ctx = canvas.getContext('2d');

// Configure canvas display resolution
canvas.width = 800;
canvas.height = 550;

// Simulation state
let rabbits = [];
let foxes = [];
let carrots = [];
let history = []; // Lotka-Volterra population records
let stepCounter = 0;
let isPlaying = true;
let followedEntity = null;

// Statistics trackers
let totalCrashes = 0;
let totalCatches = 0;

// Initialize PPO Learner
const ppoLearner = new PPOLearner(10, 16, 4);

// Setup ecosystem values
function initEcosystem(rCount = 35, fCount = 6, cCount = 50) {
    rabbits = [];
    foxes = [];
    carrots = [];
    history = [];
    stepCounter = 0;
    followedEntity = null;
    totalCrashes = 0;
    totalCatches = 0;
    
    ppoLearner.clear();

    spawnRabbits(rCount);
    spawnFoxes(fCount);
    spawnCarrots(cCount);
    
    // Initial stats reset
    document.getElementById('ppo-crashes-val').textContent = '0';
    document.getElementById('ppo-catches-val').textContent = '0';
}

function spawnRabbits(count) {
    for (let i = 0; i < count; i++) {
        const x = Math.random() * (canvas.width - 40) + 20;
        const y = Math.random() * (canvas.height - 40) + 20;
        const r = RabbitGA.createRandomRabbit(x, y);
        rabbits.push(r);
    }
}

function spawnFoxes(count) {
    for (let i = 0; i < count; i++) {
        const x = Math.random() * (canvas.width - 40) + 20;
        const y = Math.random() * (canvas.height - 40) + 20;
        const f = new Fox(x, y);
        f.brain = ppoLearner.globalBrain;
        foxes.push(f);
    }
}

function spawnCarrots(count) {
    const maxCarrots = 150;
    for (let i = 0; i < count; i++) {
        if (carrots.length >= maxCarrots) break;
        const x = Math.random() * (canvas.width - 20) + 10;
        const y = Math.random() * (canvas.height - 20) + 10;
        carrots.push(new Carrot(x, y));
    }
}

// Simulation Core Update Logic
function updateSimulation() {
    stepCounter++;

    // 1. Spawning Carrots depending on active Season
    const season = document.getElementById('season-select').value;
    let spawnRate = 20; // Default
    let seasonText = "SPRING";

    if (season === 'spring') {
        spawnRate = 8;
        seasonText = "SPRING";
    } else if (season === 'summer') {
        spawnRate = 20;
        seasonText = "SUMMER";
    } else if (season === 'winter') {
        spawnRate = 50;
        seasonText = "WINTER";
    }

    document.getElementById('season-badge').textContent = seasonText;
    document.getElementById('season-badge').className = "status-value " + (season === 'winter' ? 'highlight-orange' : 'highlight-green');

    if (stepCounter % spawnRate === 0) {
        spawnCarrots(1);
    }

    // 2. Read sliders from GUI
    const gaMutationRate = parseFloat(document.getElementById('ga-mutation-slider').value) / 100;
    const gaBirthEnergy = parseFloat(document.getElementById('ga-energy-slider').value);
    
    const ppoLR = parseFloat(document.getElementById('ppo-lr-slider').value) * 0.0001;
    const ppoEntropy = parseFloat(document.getElementById('ppo-entropy-slider').value) * 0.001;

    const rCatch = parseFloat(document.getElementById('r-catch-slider').value);
    const rStarve = parseFloat(document.getElementById('r-starve-slider').value) / 100;
    const rApproach = parseFloat(document.getElementById('r-approach-slider').value) / 100;

    // 3. Update Rabbit Population (Genetic Algorithm)
    const deadRabbits = [];
    const childrenRabbits = [];

    for (const rabbit of rabbits) {
        rabbit.update(carrots, foxes, canvas.width, canvas.height);
        
        if (rabbit.isDead) {
            deadRabbits.push(rabbit);
            continue;
        }

        // Reproduction Check
        if (rabbit.energy >= gaBirthEnergy) {
            // Find a nearby mating partner
            let partner = null;
            let bestMatingDist = 55;
            for (const other of rabbits) {
                if (other === rabbit || other.isDead || other.energy < gaBirthEnergy) continue;
                const dist = Math.hypot(other.x - rabbit.x, other.y - rabbit.y);
                if (dist < bestMatingDist) {
                    bestMatingDist = dist;
                    partner = other;
                }
            }

            let child = null;
            if (partner) {
                // Sexual crossover reproduction
                child = RabbitGA.reproduceSexually(rabbit, partner, gaMutationRate);
            } else {
                // Asexual division reproduction
                child = RabbitGA.reproduceAsexually(rabbit, gaMutationRate);
            }
            childrenRabbits.push(child);
        }
    }

    // Remove dead rabbits and push newborns
    rabbits = rabbits.filter(r => !r.isDead && !deadRabbits.includes(r));
    rabbits.push(...childrenRabbits);

    // Safeguard extinction
    if (rabbits.length === 0) {
        spawnRabbits(5);
    }

    // Filter dead carrots
    carrots = carrots.filter(c => !c.isDead);

    // 4. Update Fox Population (PPO training)
    for (const fox of foxes) {
        // Record details from before step for approach reward calculations
        const lastTarget = fox.targetLock;
        let lastTargetDist = null;
        if (lastTarget && !lastTarget.isDead) {
            lastTargetDist = Math.hypot(lastTarget.x - fox.x, lastTarget.y - fox.y);
        }

        const result = fox.update(rabbits, foxes, canvas.width, canvas.height);

        if (fox.isDead) {
            // Log terminal step reward
            ppoLearner.recordStep(fox.id, fox.lastState, fox.lastAction, fox.lastActionProb, fox.lastValue, -12.0, true);
            
            // Respawn fox at random position
            fox.x = Math.random() * (canvas.width - 40) + 20;
            fox.y = Math.random() * (canvas.height - 40) + 20;
            fox.vx = 0;
            fox.vy = 0;
            fox.energy = 250;
            fox.isDead = false;
            fox.targetLock = null;
            fox.lastTargetDist = null;
            continue;
        }

        // Calculate custom reward shaping
        let stepReward = -rStarve; // Starvation penalty

        if (result.caughtRabbit) {
            stepReward += rCatch;
            totalCatches++;
        }

        if (result.wallHit) {
            stepReward -= 0.6; // Wall collision penalty
            totalCrashes++;
        }

        // Dense Approach reward
        if (fox.targetLock && !fox.targetLock.isDead) {
            const currentDist = Math.hypot(fox.targetLock.x - fox.x, fox.targetLock.y - fox.y);
            if (lastTargetDist !== null && lastTarget === fox.targetLock) {
                const diff = lastTargetDist - currentDist;
                stepReward += rApproach * diff; // Positive reward if closer, negative if farther
            }
        }

        // Record non-terminal experience transition
        ppoLearner.recordStep(fox.id, result.state, result.action, fox.lastActionProb, fox.lastValue, stepReward, false);
    }

    // 5. Run online PPO update if experiences buffer matches threshold
    if (ppoLearner.globalMemory.length >= ppoLearner.updateInterval) {
        ppoLearner.bootstrapActive(foxes);
        ppoLearner.train(ppoLR, ppoEntropy);
    }

    // 6. Record population stats for chart drawing (every 1 second / 60 ticks)
    if (stepCounter % 60 === 0) {
        history.push({
            carrots: carrots.length,
            rabbits: rabbits.length,
            foxes: foxes.length
        });
        if (history.length > 80) {
            history.shift(); // Keep standard view length
        }
        drawTelemetryChart();
    }

    // Update Telemetry numbers
    document.getElementById('rabbit-count').textContent = rabbits.length;
    document.getElementById('fox-count').textContent = foxes.length;
    document.getElementById('carrot-count').textContent = carrots.length;
    document.getElementById('ppo-crashes-val').textContent = totalCrashes;
    document.getElementById('ppo-catches-val').textContent = totalCatches;

    updateAdaptationsPanel();
    updateNeuralVisualizer();
}

// Simulation Draw Logic
function drawSimulation() {
    // Clear canvas
    ctx.fillStyle = '#0f172a'; // Slate-900 background
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid backdrop
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.015)';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }

    const showVision = document.getElementById('show-vision-chk').checked;
    const showTarget = document.getElementById('show-target-chk').checked;

    // Draw Carrots
    for (const carrot of carrots) {
        carrot.draw(ctx);
    }

    // Draw Rabbits
    for (const rabbit of rabbits) {
        rabbit.draw(ctx, showVision, showTarget);
    }

    // Draw Foxes
    for (const fox of foxes) {
        fox.draw(ctx, showVision, showTarget);
    }

    // Highlight Followed Entity
    if (followedEntity && !followedEntity.isDead) {
        ctx.save();
        ctx.strokeStyle = followedEntity instanceof Fox ? '#F97316' : '#38BDF8';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.arc(followedEntity.x, followedEntity.y, followedEntity.radius + 10, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }
}

// Lotka-Volterra history telemetry drawing
function drawTelemetryChart() {
    const chartCanvas = document.getElementById('telemetry-chart');
    if (!chartCanvas) return;
    const cctx = chartCanvas.getContext('2d');

    const w = chartCanvas.width = chartCanvas.clientWidth;
    const h = chartCanvas.height = chartCanvas.clientHeight;

    cctx.clearRect(0, 0, w, h);

    if (history.length < 2) return;

    // Find highest peak in history
    let maxVal = 10;
    for (const pt of history) {
        if (pt.carrots > maxVal) maxVal = pt.carrots;
        if (pt.rabbits > maxVal) maxVal = pt.rabbits;
        if (pt.foxes > maxVal) maxVal = pt.foxes;
    }
    maxVal = Math.ceil(maxVal * 1.1); // Add headroom

    // Plot helper
    const drawLine = (key, strokeColor, fillColor) => {
        cctx.beginPath();
        for (let i = 0; i < history.length; i++) {
            const x = (i / (history.length - 1)) * w;
            const y = h - (history[i][key] / maxVal) * h;
            if (i === 0) cctx.moveTo(x, y);
            else cctx.lineTo(x, y);
        }
        cctx.strokeStyle = strokeColor;
        cctx.lineWidth = 2;
        cctx.stroke();

        // Fill below
        cctx.lineTo(w, h);
        cctx.lineTo(0, h);
        cctx.closePath();
        cctx.fillStyle = fillColor;
        cctx.fill();
    };

    // Draw Area layers
    drawLine('carrots', '#10B981', 'rgba(16, 185, 129, 0.04)');
    drawLine('rabbits', '#38BDF8', 'rgba(56, 189, 248, 0.04)');
    drawLine('foxes', '#F97316', 'rgba(249, 115, 22, 0.04)');
}

// Update GA adaptation averages
function updateAdaptationsPanel() {
    if (rabbits.length === 0) return;

    let sumSpeed = 0;
    let sumVision = 0;
    let sumSize = 0;

    for (const r of rabbits) {
        sumSpeed += r.genes.speedGene;
        sumVision += r.genes.visionGene;
        sumSize += r.genes.sizeGene;
    }

    const avgSpeed = sumSpeed / rabbits.length;
    const avgVision = sumVision / rabbits.length;
    const avgSize = sumSize / rabbits.length;

    document.getElementById('trait-speed-val').textContent = avgSpeed.toFixed(2) + 'x';
    document.getElementById('trait-vision-val').textContent = (avgVision * 140).toFixed(0) + 'px';
    document.getElementById('trait-size-val').textContent = (avgSize * 8).toFixed(1) + 'px';

    // Map [0.5, 2.0] range to width percentages
    const speedPct = Math.max(0, Math.min(100, ((avgSpeed - 0.5) / 1.5) * 100));
    const visionPct = Math.max(0, Math.min(100, ((avgVision - 0.5) / 1.5) * 100));
    const sizePct = Math.max(0, Math.min(100, ((avgSize - 0.5) / 1.5) * 100));

    document.getElementById('trait-speed-fill').style.width = speedPct + '%';
    document.getElementById('trait-vision-fill').style.width = visionPct + '%';
    document.getElementById('trait-size-fill').style.width = sizePct + '%';
}

// Draw Fox Brain Activations inside footer
function updateNeuralVisualizer() {
    // If followedEntity is a Fox, use it. Otherwise default to first fox.
    let fox = null;
    if (followedEntity && followedEntity instanceof Fox && !followedEntity.isDead) {
        fox = followedEntity;
    } else if (foxes.length > 0) {
        fox = foxes[0];
    }

    if (!fox || !fox.lastState) {
        return; // No active fox data
    }

    // Read probabilities of policy activations
    const { probs } = ppoLearner.globalBrain.forward(fox.lastState);

    const actions = ['left', 'right', 'accel', 'coast'];
    for (let i = 0; i < actions.length; i++) {
        const p = probs[i];
        const pct = (p * 100).toFixed(0) + '%';
        const fillBar = document.getElementById(`net-act-${actions[i]}`);
        const textLabel = document.getElementById(`net-act-${actions[i]}-val`);
        if (fillBar && textLabel) {
            fillBar.style.width = (p * 100) + '%';
            textLabel.textContent = pct;
        }
    }
}

// Slider changes reflection
function bindSliders() {
    const sync = (sliderId, textId, formatFn) => {
        const slider = document.getElementById(sliderId);
        const txt = document.getElementById(textId);
        slider.addEventListener('input', () => {
            txt.textContent = formatFn(slider.value);
        });
    };

    sync('speed-slider', 'speed-val', v => v + 'x');
    sync('ga-mutation-slider', 'ga-mutation-val', v => v + '%');
    sync('ga-energy-slider', 'ga-energy-val', v => v);
    sync('ppo-lr-slider', 'ppo-lr-val', v => (v * 0.0001).toFixed(4));
    sync('ppo-entropy-slider', 'ppo-entropy-val', v => (v / 1000).toFixed(3));
    sync('r-catch-slider', 'r-catch-val', v => '+' + parseFloat(v).toFixed(1));
    sync('r-starve-slider', 'r-starve-val', v => '-' + (v / 100).toFixed(2));
    sync('r-approach-slider', 'r-approach-val', v => '+' + (v / 100).toFixed(2));
}

// Hook UI Button Event Listeners
function bindButtons() {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            document.getElementById('tab-' + btn.getAttribute('data-tab')).classList.remove('hidden');
        });
    });

    // Spawners
    document.getElementById('spawn-rabbit-btn').addEventListener('click', () => spawnRabbits(10));
    document.getElementById('spawn-fox-btn').addEventListener('click', () => spawnFoxes(5));
    document.getElementById('spawn-carrot-btn').addEventListener('click', () => spawnCarrots(25));
    document.getElementById('clear-all-btn').addEventListener('click', () => {
        rabbits = [];
        foxes = [];
        carrots = [];
        followedEntity = null;
        ppoLearner.clear();
    });

    // Play/Pause
    const playPauseBtn = document.getElementById('play-pause-btn');
    playPauseBtn.addEventListener('click', () => {
        isPlaying = !isPlaying;
        if (isPlaying) {
            playPauseBtn.innerHTML = '<span class="btn-icon">⏸️</span> Pause Sim';
            playPauseBtn.className = "primary-btn";
        } else {
            playPauseBtn.innerHTML = '<span class="btn-icon">▶️</span> Resume Sim';
            playPauseBtn.className = "primary-btn resume-style"; // Styled via green if paused
        }
    });

    // Reset Ecosystem
    document.getElementById('reset-btn').addEventListener('click', () => {
        initEcosystem(35, 6, 50);
    });

    // Click/Double-click entity targeting on simulation canvas
    canvas.addEventListener('dblclick', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = ((e.clientX - rect.left) / rect.width) * canvas.width;
        const mouseY = ((e.clientY - rect.top) / rect.height) * canvas.height;

        let nearest = null;
        let minDist = 45; // Max click targeting range

        // Check rabbits
        for (const r of rabbits) {
            const d = Math.hypot(r.x - mouseX, r.y - mouseY);
            if (d < minDist) {
                minDist = d;
                nearest = r;
            }
        }
        // Check foxes
        for (const f of foxes) {
            const d = Math.hypot(f.x - mouseX, f.y - mouseY);
            if (d < minDist) {
                minDist = d;
                nearest = f;
            }
        }

        followedEntity = nearest;
    });
}

// Start simulation on load
window.addEventListener('load', () => {
    bindSliders();
    bindButtons();
    initEcosystem(35, 6, 50);

    // Kickoff run loops
    function animationLoop() {
        if (isPlaying) {
            const speedMultiplier = parseInt(document.getElementById('speed-slider').value) || 1;
            for (let s = 0; s < speedMultiplier; s++) {
                updateSimulation();
            }
        }
        drawSimulation();
        requestAnimationFrame(animationLoop);
    }
    requestAnimationFrame(animationLoop);
});
