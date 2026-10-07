/**
 * Darwin's Doodles - Simulation Coordinator
 */

document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------
    // App State
    // ----------------------------------------------------
    const physics = new PhysicsWorld();
    const ga = new GeneticAlgorithm();

    let population = [];
    let generationCount = 1;
    let simTimer = 0.0;
    
    // History & Records
    let bestFitnessHistory = [];
    let avgFitnessHistory = [];
    let hallOfFame = []; // Array of { gen, score, dna, type }
    
    // User Settings
    let activePreset = 'biped';
    let simSpeed = 1;
    let isPlaying = true;
    let generationDuration = 10; // seconds
    let popSize = 30;
    let mutationRate = 0.15;
    let elitismCount = 2;
    let selectionMethod = 'tournament';
    
    // Custom Creator Editor State
    let isEditorMode = false;
    let editorTool = 'joint'; // 'joint', 'bone', 'muscle', 'erase'
    let editorNodes = [];      // { x, y, radius, mass }
    let editorConstraints = []; // { a, b, ratioA, ratioB, isMuscle }
    let editorSelectedNodeIdx = null;
    let editorSelectedBoneIdx = null;
    let editorSelectedBoneRatio = 0.5;
    let editorMousePos = { x: 0, y: 0 };
    
    // Simulation Coordinates
    const spawnX = 200;
    const spawnY = 320;
    
    // Camera
    const camera = {
        x: 0,
        y: 0,
        zoom: 1.0,
        targetZoom: 1.0,
        autoTrack: true,
        lastInteractionTime: 0
    };
    
    // Interactivity
    let hoveredCreature = null;
    let replayingChamp = null; // { dna, type, creatureInstance }
    const timeStep = 1 / 60;   // fixed dt for physics

    // ----------------------------------------------------
    // DOM Elements
    // ----------------------------------------------------
    const simCanvas = document.getElementById('sim-canvas');
    const simCtx = simCanvas.getContext('2d');
    
    const telemetryCanvas = document.getElementById('telemetry-chart');
    const telemetryCtx = telemetryCanvas.getContext('2d');
    
    const waveformCanvas = document.getElementById('waveform-canvas');
    const waveformCtx = waveformCanvas.getContext('2d');
    
    // Labels
    const genCountLabel = document.getElementById('gen-count');
    const bestFitnessLabel = document.getElementById('best-fitness-val');
    const simTimerLabel = document.getElementById('sim-timer');
    const avgFitnessLabel = document.getElementById('avg-fitness-val');
    const peakVelLabel = document.getElementById('peak-vel-val');
    
    // Sliders & Controls
    const speedSlider = document.getElementById('speed-slider');
    const speedVal = document.getElementById('speed-val');
    const durationSlider = document.getElementById('duration-slider');
    const durationVal = document.getElementById('duration-val');
    const popSlider = document.getElementById('pop-slider');
    const popVal = document.getElementById('pop-val');
    const mutationSlider = document.getElementById('mutation-slider');
    const mutationVal = document.getElementById('mutation-val');
    const elitismSlider = document.getElementById('elitism-slider');
    const elitismVal = document.getElementById('elitism-val');
    
    const terrainSelect = document.getElementById('terrain-select');
    const selectionSelect = document.getElementById('selection-select');
    
    // Buttons
    const playPauseBtn = document.getElementById('play-pause-btn');
    const resetBtn = document.getElementById('reset-btn');
    const instantBtn = document.getElementById('instant-btn');
    
    const presetBtns = document.querySelectorAll('.preset-btn');
    const speedPresetBtns = document.querySelectorAll('.speed-preset-btn');
    
    const genomeGrid = document.getElementById('genome-grid');
    const hallOfFameList = document.getElementById('hall-of-fame-list');

    // ----------------------------------------------------
    // Canvas Resizing & Camera Handling
    // ----------------------------------------------------
    function resizeCanvas() {
        const rect = simCanvas.parentElement.getBoundingClientRect();
        simCanvas.width = rect.width;
        simCanvas.height = rect.height;
        
        telemetryCanvas.width = telemetryCanvas.parentElement.clientWidth;
        telemetryCanvas.height = telemetryCanvas.parentElement.clientHeight;
        
        waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
        waveformCanvas.height = waveformCanvas.parentElement.clientHeight;
        
        drawTelemetryChart();
        drawWaveformChart();
    }
    window.addEventListener('resize', resizeCanvas);

    // Zooming & Dragging
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };
    
    simCanvas.addEventListener('mousedown', (e) => {
        if (isEditorMode) {
            handleEditorClick(e);
            return;
        }
        isDragging = true;
        dragStart.x = e.clientX - camera.x;
        dragStart.y = e.clientY - camera.y;
        camera.autoTrack = false;
        camera.lastInteractionTime = Date.now();
    });

    window.addEventListener('mousemove', (e) => {
        if (isEditorMode) {
            handleEditorMouseMove(e);
            return;
        }
        if (!isDragging) {
            // Check for hovered creature
            detectHover(e);
            return;
        }
        camera.x = e.clientX - dragStart.x;
        camera.y = e.clientY - dragStart.y;
        camera.lastInteractionTime = Date.now();
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
    });

    simCanvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (isEditorMode) return; // Lock zoom in editor mode
        
        const zoomIntensity = 0.05;
        const mouseX = e.clientX - simCanvas.getBoundingClientRect().left;
        const mouseY = e.clientY - simCanvas.getBoundingClientRect().top;

        // Zoom relative to mouse
        const wheel = e.deltaY < 0 ? 1 : -1;
        const zoomFactor = Math.exp(wheel * zoomIntensity);
        
        // Translate back, scale, translate forward
        camera.x -= (mouseX - camera.x) * (zoomFactor - 1);
        camera.y -= (mouseY - camera.y) * (zoomFactor - 1);
        camera.zoom *= zoomFactor;
        camera.zoom = Math.max(0.15, Math.min(3.0, camera.zoom));
        
        camera.autoTrack = false;
        camera.lastInteractionTime = Date.now();
    }, { passive: false });

    function detectHover(e) {
        if (population.length === 0) return;
        const rect = simCanvas.getBoundingClientRect();
        const mouseCanvasX = (e.clientX - rect.left - simCanvas.width / 2 - camera.x) / camera.zoom;
        const mouseCanvasY = (e.clientY - rect.top - simCanvas.height / 2 - camera.y) / camera.zoom;

        let closest = null;
        let closestDist = 45; // Hover activation threshold in pixels
        
        const activeGroup = replayingChamp ? [replayingChamp.creatureInstance] : population;

        for (const c of activeGroup) {
            const com = c.getCenterOfMass();
            const dx = com.x - mouseCanvasX;
            const dy = com.y - mouseCanvasY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < closestDist) {
                closestDist = dist;
                closest = c;
            }
        }
        hoveredCreature = closest;
    }

    // ----------------------------------------------------
    // Preset & Setup
    // ----------------------------------------------------
    function initSimulation() {
        physics.clear();
        population = [];
        hoveredCreature = null;
        replayingChamp = null;
        simTimer = 0.0;
        
        // Generate initial random population
        const tempCreature = CreaturePresets.get(activePreset, 999, spawnX, spawnY);
        const numGenes = ga.getGenomeLengthForCreature(tempCreature);

        for (let i = 0; i < popSize; i++) {
            const creature = CreaturePresets.get(activePreset, i, spawnX, spawnY);
            const genome = ga.createRandomGenome(numGenes);
            ga.applyGenome(creature, genome);
            population.push(creature);
            physics.addCreature(creature);
        }

        // Align camera to spawn point
        camera.x = -spawnX * camera.zoom;
        camera.y = 50;
        camera.autoTrack = true;
        
        // Update stats UI
        genCountLabel.textContent = generationCount;
        bestFitnessLabel.textContent = "0.00m";
        avgFitnessLabel.textContent = "0.00m";
        peakVelLabel.textContent = "0.0m/s";
        simTimerLabel.textContent = "0.0s";
        
        // Setup Genome Visualizer placeholder
        genomeGrid.innerHTML = `<div class="no-selection">Gen 1 initialized. Waiting for evolution...</div>`;
        
        drawWaveformChart();
    }

    // ----------------------------------------------------
    // UI Events
    // ----------------------------------------------------
    // Sliders
    speedSlider.addEventListener('input', (e) => {
        simSpeed = parseInt(e.target.value);
        speedVal.textContent = simSpeed + 'x';
        updateSpeedActiveBtn();
    });

    durationSlider.addEventListener('input', (e) => {
        generationDuration = parseInt(e.target.value);
        durationVal.textContent = generationDuration + 's';
    });

    popSlider.addEventListener('input', (e) => {
        popSize = parseInt(e.target.value);
        popVal.textContent = popSize;
    });

    mutationSlider.addEventListener('input', (e) => {
        mutationRate = parseInt(e.target.value) / 100;
        mutationVal.textContent = (mutationRate * 100).toFixed(0) + '%';
    });

    elitismSlider.addEventListener('input', (e) => {
        elitismCount = parseInt(e.target.value);
        elitismVal.textContent = elitismCount;
    });

    terrainSelect.addEventListener('change', (e) => {
        physics.setTerrain(e.target.value);
        if (replayingChamp) {
            replayingChamp.creatureInstance.reset(0, 0);
        }
        physics.resetAll(0, 0);
        simTimer = 0.0;
    });

    selectionSelect.addEventListener('change', (e) => {
        selectionMethod = e.target.value;
    });

    // Preset Buttons
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.id === 'custom-preset-btn') return; // Handled separately
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activePreset = btn.dataset.preset;
            
            // Wipe histories
            generationCount = 1;
            bestFitnessHistory = [];
            avgFitnessHistory = [];
            
            initSimulation();
            drawTelemetryChart();
        });
    });

    // Speed Preset Buttons
    speedPresetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.id === 'instant-btn') return; // Handled separately
            speedPresetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            simSpeed = parseInt(btn.dataset.speed);
            speedSlider.value = simSpeed;
            speedVal.textContent = simSpeed + 'x';
        });
    });

    function updateSpeedActiveBtn() {
        speedPresetBtns.forEach(b => {
            if (parseInt(b.dataset.speed) === simSpeed) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });
    }

    // Play/Pause
    playPauseBtn.addEventListener('click', () => {
        isPlaying = !isPlaying;
        if (isPlaying) {
            playPauseBtn.innerHTML = '<span class="btn-icon">⏸️</span> Pause Sim';
            playPauseBtn.classList.remove('paused');
        } else {
            playPauseBtn.innerHTML = '<span class="btn-icon">▶️</span> Resume Sim';
            playPauseBtn.classList.add('paused');
        }
    });

    // Reset
    resetBtn.addEventListener('click', () => {
        generationCount = 1;
        bestFitnessHistory = [];
        avgFitnessHistory = [];
        initSimulation();
        drawTelemetryChart();
        
        // Show Toast Notification
        showToast("Simulation database reset to Gen 1");
    });

    // Instant Gen Button
    instantBtn.addEventListener('click', () => {
        if (replayingChamp) replayingChamp = null; // Break out of replay
        
        // Fast forward simulation without rendering
        instantGeneration();
    });

    function showToast(message) {
        let toast = document.querySelector('.toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'toast';
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2500);
    }

    // ----------------------------------------------------
    // Genetic Algorithm Pipeline
    // ----------------------------------------------------
    function getBestCreature(group = population) {
        if (group.length === 0) return null;
        let best = group[0];
        let bestScore = best.getCenterOfMass().x;
        
        for (let i = 1; i < group.length; i++) {
            const score = group[i].getCenterOfMass().x;
            if (score > bestScore) {
                bestScore = score;
                best = group[i];
            }
        }
        return best;
    }

    function evaluateFitness() {
        let sum = 0;
        let peakVelocity = 0;
        let bestOfGen = null;
        let highestFitness = -Infinity;

        // Calculate fitness for all creatures
        population.forEach(c => {
            // Fitness = Distance traveled from spawn point, scaled down
            const com = c.getCenterOfMass();
            const distance = (com.x - spawnX) / 100; // 1 meter = 100 pixels
            
            // Posture validation: penalize if torso collapses close to the ground (prevents tumbling/rolling)
            let penalty = 1.0;
            if (activePreset === 'biped') {
                const torso = c.nodes[0];
                if (torso && torso.y > 425) { // Torso collapsed close to the ground (480)
                    penalty = 0.12;
                }
            } else if (activePreset === 'quadruped') {
                const torsoRear = c.nodes[0];
                const torsoFront = c.nodes[1];
                if ((torsoRear && torsoRear.y > 425) || (torsoFront && torsoFront.y > 425)) {
                    penalty = 0.12;
                }
            }

            c.fitness = distance > 0 ? distance * penalty : distance;

            // Calculate peak velocity of nodes as a telemetry stat
            let maxNodeVel = 0;
            c.nodes.forEach(n => {
                const vx = n.x - n.oldX;
                const vy = n.y - n.oldY;
                const speed = Math.sqrt(vx * vx + vy * vy) * 60 / 100; // m/s
                if (speed > maxNodeVel) maxNodeVel = speed;
            });
            if (maxNodeVel > peakVelocity) peakVelocity = maxNodeVel;

            sum += distance;
            
            if (distance > highestFitness) {
                highestFitness = distance;
                bestOfGen = c;
            }
        });

        const avgFitness = sum / population.length;
        
        return {
            best: bestOfGen,
            bestFitness: highestFitness,
            avgFitness: avgFitness,
            peakVelocity: peakVelocity
        };
    }

    function nextGeneration() {
        const stats = evaluateFitness();
        
        // Log histories
        bestFitnessHistory.push(stats.bestFitness);
        avgFitnessHistory.push(stats.avgFitness);
        
        // Archive champion
        archiveChampion(generationCount, stats.bestFitness, stats.best.dna);

        // UI Stats Update
        bestFitnessLabel.textContent = stats.bestFitness.toFixed(2) + "m";
        avgFitnessLabel.textContent = stats.avgFitness.toFixed(2) + "m";
        peakVelLabel.textContent = stats.peakVelocity.toFixed(1) + "m/s";

        // Draw genome grid for the elite walker
        populateGenomeGrid(stats.best);

        // Transition population
        const temp = CreaturePresets.get(activePreset, 999, spawnX, spawnY);
        const numGenes = ga.getGenomeLengthForCreature(temp);

        population = ga.evolveGeneration(
            population, 
            numGenes, 
            mutationRate, 
            elitismCount, 
            selectionMethod, 
            activePreset, 
            spawnX, 
            spawnY
        );

        // Reset Physics World
        physics.clear();
        population.forEach(c => physics.addCreature(c));
        physics.resetAll(0, 0);
        
        generationCount++;
        genCountLabel.textContent = generationCount;
        simTimer = 0.0;
        
        // Refresh visuals
        drawTelemetryChart();
        drawWaveformChart(stats.best.dna);
    }

    function instantGeneration() {
        const stepsPerFrame = Math.round(generationDuration / timeStep);
        
        // Put UI in loading state briefly
        instantBtn.textContent = "Processing...";
        instantBtn.disabled = true;

        setTimeout(() => {
            // Execute all steps synchronously in one thread block
            for (let i = 0; i < stepsPerFrame; i++) {
                physics.step(timeStep);
            }
            
            // Advance generation immediately
            nextGeneration();
            
            instantBtn.textContent = "Instant Gen";
            instantBtn.disabled = false;
            showToast("Calculated next generation instantly");
        }, 10);
    }

    // ----------------------------------------------------
    // Replay Champions
    // ----------------------------------------------------
    function archiveChampion(gen, score, dna) {
        // Only keep top 15 champions to prevent UI clutter
        if (hallOfFame.some(c => c.gen === gen)) return;

        const champ = {
            gen: gen,
            score: score,
            dna: dna,
            type: activePreset
        };
        
        hallOfFame.push(champ);
        hallOfFame.sort((a, b) => b.score - a.score); // Order by best score
        
        // Re-render list
        renderHallOfFameUI();
    }

    function renderHallOfFameUI() {
        if (hallOfFame.length === 0) {
            hallOfFameList.innerHTML = `<div class="empty-hall-msg">No champions archived. Evolve at least one generation.</div>`;
            return;
        }

        hallOfFameList.innerHTML = '';
        
        // Display top 10 champions
        hallOfFame.slice(0, 12).forEach(champ => {
            const card = document.createElement('div');
            card.className = 'champ-card';
            if (replayingChamp && replayingChamp.gen === champ.gen) {
                card.classList.add('active');
            }
            
            card.innerHTML = `
                <span class="gen-num">Gen ${champ.gen}</span>
                <span class="score">${champ.score.toFixed(2)}m</span>
                <span class="type-tag" style="font-size: 0.55rem; color: var(--text-muted)">(${champ.type})</span>
            `;
            
            card.addEventListener('click', () => {
                triggerReplay(champ);
            });
            
            hallOfFameList.appendChild(card);
        });
    }

    function triggerReplay(champ) {
        if (replayingChamp && replayingChamp.gen === champ.gen) {
            // Toggle off replay
            replayingChamp = null;
            initSimulation();
            renderHallOfFameUI();
            showToast("Returned to live evolution");
            return;
        }

        // Build single creature to simulate
        physics.clear();
        
        const creatureInstance = CreaturePresets.get(champ.type, champ.gen, spawnX, spawnY);
        ga.applyGenome(creatureInstance, champ.dna);
        creatureInstance.color = 'rgba(245, 158, 11, 0.7)'; // Glow amber
        creatureInstance.isBest = true;

        physics.addCreature(creatureInstance);
        physics.resetAll(0, 0);
        
        replayingChamp = {
            gen: champ.gen,
            score: champ.score,
            dna: champ.dna,
            type: champ.type,
            creatureInstance: creatureInstance
        };
        
        simTimer = 0.0;
        camera.autoTrack = true;
        
        populateGenomeGrid(creatureInstance);
        drawWaveformChart(champ.dna);
        renderHallOfFameUI();
        
        showToast(`Replaying Champion of Gen ${champ.gen} (${champ.score.toFixed(2)}m)`);
    }

    // ----------------------------------------------------
    // Genome Matrix UI Population
    // ----------------------------------------------------
    function populateGenomeGrid(creature) {
        genomeGrid.innerHTML = '';
        const muscles = creature.constraints.filter(c => c.isMuscle);
        
        if (muscles.length === 0 || !creature.dna || creature.dna.length === 0) {
            genomeGrid.innerHTML = `<div class="no-selection">No muscles found.</div>`;
            return;
        }

        muscles.forEach((muscle, idx) => {
            const dnaOffset = idx * 4;
            
            // Amp, Freq, Phase, Len
            const cell = document.createElement('div');
            cell.className = 'genome-cell';
            cell.innerHTML = `
                <span class="genome-cell-label">M${idx+1} AMP</span>
                <span class="genome-cell-val">${creature.dna[dnaOffset].toFixed(2)}</span>
            `;
            genomeGrid.appendChild(cell);

            const cell2 = document.createElement('div');
            cell2.className = 'genome-cell';
            cell2.innerHTML = `
                <span class="genome-cell-label">M${idx+1} FRQ</span>
                <span class="genome-cell-val">${creature.dna[dnaOffset+1].toFixed(2)}</span>
            `;
            genomeGrid.appendChild(cell2);

            const cell3 = document.createElement('div');
            cell3.className = 'genome-cell';
            cell3.innerHTML = `
                <span class="genome-cell-label">M${idx+1} PHS</span>
                <span class="genome-cell-val">${creature.dna[dnaOffset+2].toFixed(2)}</span>
            `;
            genomeGrid.appendChild(cell3);

            const cell4 = document.createElement('div');
            cell4.className = 'genome-cell';
            cell4.innerHTML = `
                <span class="genome-cell-label">M${idx+1} LEN</span>
                <span class="genome-cell-val">${creature.dna[dnaOffset+3].toFixed(2)}</span>
            `;
            genomeGrid.appendChild(cell4);
        });
    }

    // ----------------------------------------------------
    // Rendering & Canvas Graphics
    // ----------------------------------------------------
    function render() {
        if (isEditorMode) {
            renderEditor();
            return;
        }
        
        // Clear viewport canvas
        simCtx.clearRect(0, 0, simCanvas.width, simCanvas.height);
        
        // Handle Auto camera tracking
        updateCameraTracking();

        // 1. Begin Transform Matrix
        simCtx.save();
        
        // Shift matrix to center screen, apply zoom, then translate to camera coordinates
        simCtx.translate(simCanvas.width / 2, simCanvas.height / 2);
        simCtx.scale(camera.zoom, camera.zoom);
        simCtx.translate(camera.x, camera.y);

        // 2. Render Environment Grid / Backdrop elements
        drawEnvironmentBackdrop();

        // 3. Render Terrain Ground
        drawTerrain();

        // 4. Render Creatures
        if (replayingChamp) {
            // Draw replaying champion
            drawCreature(replayingChamp.creatureInstance);
        } else {
            // Identify best creature to highlight
            const best = getBestCreature();
            
            // Draw non-best creatures first (so they render behind)
            population.forEach(c => {
                if (c !== best) {
                    c.isBest = false;
                    drawCreature(c);
                }
            });
            
            // Draw best creature on top with highlights
            if (best) {
                best.isBest = true;
                drawCreature(best);
            }
        }

        // Restore context
        simCtx.restore();

        // 5. Draw 2D GUI overlay elements (outside of camera matrix)
        drawOverlayUI();
    }

    function updateCameraTracking() {
        if (!camera.autoTrack) {
            // If user has not dragged or zoomed for 6 seconds, resume tracking best creature
            if (Date.now() - camera.lastInteractionTime > 6000) {
                camera.autoTrack = true;
            }
            return;
        }

        // Identify target to track
        let trackTarget = spawnX;
        const activeGroup = replayingChamp ? [replayingChamp.creatureInstance] : population;
        const best = getBestCreature(activeGroup);
        
        if (best) {
            trackTarget = best.getCenterOfMass().x;
        }

        // Smooth interpolate camera.x to keep target centered:
        // center = canvasWidth/2 + cameraX + trackTarget * zoom => cameraX = -trackTarget
        const targetX = -trackTarget;
        const k = 0.05; // interpolation weight
        camera.x += (targetX - camera.x) * k;
        
        // Smooth zoom interpolation
        camera.zoom += (camera.targetZoom - camera.zoom) * 0.05;
        
        // Keep vertical camera level centering ground
        const groundHeight = physics.terrain.baseHeight;
        const targetY = -groundHeight + 150;
        camera.y += (targetY - camera.y) * 0.05;
    }

    function drawEnvironmentBackdrop() {
        const startVisibleX = -camera.x - simCanvas.width / (2 * camera.zoom);
        const endVisibleX = -camera.x + simCanvas.width / (2 * camera.zoom);
        
        const gridSpacing = 80;
        const startGridIdx = Math.floor(startVisibleX / gridSpacing);
        const endGridIdx = Math.ceil(endVisibleX / gridSpacing);

        // Draw vertical background gridlines
        simCtx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
        simCtx.lineWidth = 1;
        for (let i = startGridIdx; i <= endGridIdx; i++) {
            simCtx.beginPath();
            simCtx.moveTo(i * gridSpacing, -1000);
            simCtx.lineTo(i * gridSpacing, 1000);
            simCtx.stroke();
        }
        
        // Draw distance rulers on grid background
        simCtx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        simCtx.font = '10px Space Mono';
        simCtx.textAlign = 'center';
        
        const rSpacing = 100; // Ticks every 100px (1 meter)
        const startRIdx = Math.floor(startVisibleX / rSpacing);
        const endRIdx = Math.ceil(endVisibleX / rSpacing);
        
        for (let i = startRIdx; i <= endRIdx; i++) {
            const rx = i * rSpacing;
            if (rx < spawnX - 50) continue; // Don't draw before spawn
            
            const meters = (rx - spawnX) / 100;
            const ry = physics.terrain.getHeight(rx) + 22;
            
            simCtx.fillText(meters.toFixed(0) + 'm', rx, ry);
            simCtx.beginPath();
            simCtx.arc(rx, ry - 12, 1.5, 0, Math.PI * 2);
            simCtx.fill();
        }
    }

    function drawTerrain() {
        const startVisibleX = -camera.x - simCanvas.width / (2 * camera.zoom) - 50;
        const endVisibleX = -camera.x + simCanvas.width / (2 * camera.zoom) + 50;
        
        simCtx.beginPath();
        // Start shape on bottom left corner of visible space
        simCtx.moveTo(startVisibleX, 800);
        
        // Trace terrain contour
        const step = 8;
        for (let x = startVisibleX; x <= endVisibleX; x += step) {
            const y = physics.terrain.getHeight(x);
            simCtx.lineTo(x, y);
        }
        
        // Close polygon on bottom right
        simCtx.lineTo(endVisibleX, 800);
        simCtx.closePath();

        // Fill ground with gradient
        const grad = simCtx.createLinearGradient(0, 480, 0, 800);
        grad.addColorStop(0, '#151b30');
        grad.addColorStop(1, '#080b15');
        
        simCtx.fillStyle = grad;
        simCtx.fill();
        
        // Draw bright glowing top soil contour
        simCtx.beginPath();
        for (let x = startVisibleX; x <= endVisibleX; x += step) {
            const y = physics.terrain.getHeight(x);
            if (x === startVisibleX) {
                simCtx.moveTo(x, y);
            } else {
                simCtx.lineTo(x, y);
            }
        }
        simCtx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
        simCtx.lineWidth = 3;
        simCtx.stroke();
    }

    function drawCreature(creature) {
        const isBest = creature.isBest;
        const isReplay = !!replayingChamp;
        
        // Set dynamic visual parameters
        let opacity = isBest ? 1.0 : 0.25;
        
        // 1. Draw Bones (non-muscle constraints)
        creature.constraints.forEach(c => {
            if (!c.isMuscle) {
                simCtx.beginPath();
                simCtx.moveTo(c.nodeA.x, c.nodeA.y);
                simCtx.lineTo(c.nodeB.x, c.nodeB.y);
                
                if (isBest) {
                    // Steel beam double-highlight skeleton style
                    simCtx.strokeStyle = 'rgba(226, 232, 240, 0.15)';
                    simCtx.lineWidth = 5;
                    simCtx.stroke();
                    
                    simCtx.strokeStyle = isReplay ? 'rgba(245, 158, 11, 0.85)' : 'rgba(241, 245, 249, 0.9)';
                    simCtx.lineWidth = 2.2;
                    simCtx.stroke();
                } else {
                    simCtx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
                    simCtx.lineWidth = 1.5;
                    simCtx.stroke();
                }
            }
        });

        // 2. Draw Muscles (pulsating width and color transition based on contraction/extension)
        creature.constraints.forEach(c => {
            if (c.isMuscle) {
                const p1 = c.getAttachmentPoint(c.attachmentA, c.ratioA);
                const p2 = c.getAttachmentPoint(c.attachmentB, c.ratioB);
                
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                const len = Math.sqrt(dx * dx + dy * dy);
                
                // Show contraction state (ratio < 1.0 is contracted, ratio > 1.0 is extended)
                const expansionRatio = len / c.baseLength;
                
                let muscleColor;
                let thickness;
                
                if (isBest) {
                    // Dynamic color interpolation:
                    // Max contraction (e.g. ratio = 0.8) -> bright red (RGB 239, 68, 68)
                    // Max extension (e.g. ratio = 1.2) -> cyan blue (RGB 6, 182, 212)
                    // Neutral (e.g. ratio = 1.0) -> magenta/purple (RGB 168, 85, 247)
                    
                    const minRatio = 0.8;
                    const maxRatio = 1.2;
                    const norm = Math.max(0, Math.min(1, (expansionRatio - minRatio) / (maxRatio - minRatio)));
                    
                    // Interpolate RGB:
                    let r, g, b;
                    if (norm < 0.5) {
                        const t = norm * 2; // 0 to 1
                        r = Math.round(239 + (168 - 239) * t);
                        g = Math.round(68 + (85 - 68) * t);
                        b = Math.round(68 + (247 - 68) * t);
                    } else {
                        const t = (norm - 0.5) * 2; // 0 to 1
                        r = Math.round(168 + (6 - 168) * t);
                        g = Math.round(85 + (182 - 85) * t);
                        b = Math.round(247 + (212 - 247) * t);
                    }
                    
                    muscleColor = `rgba(${r}, ${g}, ${b}, ${isReplay ? 0.85 : 0.75})`;
                    
                    // Thickness: thicker when contracted, thinner when stretched
                    const baseWidth = 5.5;
                    thickness = baseWidth * (1.4 - Math.max(0.3, Math.min(1.1, expansionRatio - 0.2)));
                } else {
                    // Translucent purple/blue for herd members
                    muscleColor = 'rgba(139, 92, 246, 0.25)';
                    thickness = 2.5;
                }

                simCtx.lineCap = 'round';
                simCtx.strokeStyle = muscleColor;
                simCtx.lineWidth = thickness;
                
                simCtx.beginPath();
                simCtx.moveTo(p1.x, p1.y);
                simCtx.lineTo(p2.x, p2.y);
                simCtx.stroke();
                
                // Draw inner white tendon line for champion to represent core tension
                if (isBest) {
                    simCtx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                    simCtx.lineWidth = Math.max(0.5, thickness * 0.22);
                    simCtx.beginPath();
                    simCtx.moveTo(p1.x, p1.y);
                    simCtx.lineTo(p2.x, p2.y);
                    simCtx.stroke();
                }
            }
        });

        // 3. Draw Nodes (Pin Joints)
        creature.nodes.forEach(n => {
            let outerRingColor, innerPinColor, jointRadius;
            
            if (isBest) {
                outerRingColor = 'rgba(15, 23, 42, 0.85)'; // Slate cap
                innerPinColor = isReplay ? 'var(--accent-gold)' : '#f59e0b'; // Gold center
                jointRadius = n.radius;
                
                // Draw outer structural joint ring
                simCtx.fillStyle = outerRingColor;
                simCtx.beginPath();
                simCtx.arc(n.x, n.y, jointRadius + 2.5, 0, Math.PI * 2);
                simCtx.fill();
                
                // Draw metal outline
                simCtx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                simCtx.lineWidth = 1;
                simCtx.stroke();

                // Draw inner brass/gold pin
                simCtx.fillStyle = innerPinColor;
                simCtx.beginPath();
                simCtx.arc(n.x, n.y, jointRadius - 1.5, 0, Math.PI * 2);
                simCtx.fill();

                // Ground contact halo ring
                if (n.onGround) {
                    simCtx.strokeStyle = '#10b981';
                    simCtx.lineWidth = 2.0;
                    simCtx.beginPath();
                    simCtx.arc(n.x, n.y, jointRadius + 2.5, 0, Math.PI * 2);
                    simCtx.stroke();
                }
            } else {
                // simple translucent node
                simCtx.fillStyle = 'rgba(14, 165, 233, 0.35)';
                simCtx.beginPath();
                simCtx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
                simCtx.fill();
            }
        });
        
        // 4. Draw distance pointer flag above champion
        if (isBest && creature.nodes.length > 0) {
            const com = creature.getCenterOfMass();
            const dist = (com.x - spawnX) / 100;
            
            simCtx.fillStyle = isReplay ? 'var(--accent-gold)' : 'var(--accent-emerald)';
            simCtx.font = 'bold 9px Space Mono';
            simCtx.textAlign = 'center';
            simCtx.fillText(dist.toFixed(2) + 'm', com.x, com.y - 25);
            
            // Draw marker arrow pointing down
            simCtx.beginPath();
            simCtx.moveTo(com.x, com.y - 20);
            simCtx.lineTo(com.x - 3, com.y - 15);
            simCtx.lineTo(com.x + 3, com.y - 15);
            simCtx.closePath();
            simCtx.fill();
        }
    }

    function drawOverlayUI() {
        // Draw hovered creature details if applicable
        if (hoveredCreature) {
            const com = hoveredCreature.getCenterOfMass();
            const comScreenX = simCanvas.width / 2 + camera.x + com.x * camera.zoom;
            const comScreenY = simCanvas.height / 2 + camera.y + com.y * camera.zoom;
            
            const dist = (com.x - spawnX) / 100;
            
            simCtx.fillStyle = 'rgba(15, 23, 42, 0.85)';
            simCtx.strokeStyle = hoveredCreature.isBest ? 'var(--accent-gold)' : 'rgba(255,255,255,0.2)';
            simCtx.lineWidth = 1;
            
            const tooltipW = 120;
            const tooltipH = 50;
            const tx = Math.max(10, Math.min(simCanvas.width - tooltipW - 10, comScreenX - tooltipW / 2));
            const ty = Math.max(10, comScreenY - tooltipH - 25);
            
            // Round rect background
            simCtx.beginPath();
            simCtx.roundRect(tx, ty, tooltipW, tooltipH, 6);
            simCtx.fill();
            simCtx.stroke();
            
            simCtx.fillStyle = '#ffffff';
            simCtx.font = 'bold 10px Outfit';
            simCtx.textAlign = 'left';
            simCtx.fillText(`Creature #${hoveredCreature.id}`, tx + 8, ty + 18);
            
            simCtx.fillStyle = 'var(--text-secondary)';
            simCtx.font = '9px Space Mono';
            simCtx.fillText(`Dist: ${dist.toFixed(2)}m`, tx + 8, ty + 32);
            simCtx.fillText(`Nodes: ${hoveredCreature.nodes.length} | Mus: ${hoveredCreature.constraints.filter(c=>c.isMuscle).length}`, tx + 8, ty + 42);
        }

        // Replay watermarks
        if (replayingChamp) {
            simCtx.fillStyle = 'rgba(245, 158, 11, 0.08)';
            simCtx.font = 'bold 36px Outfit';
            simCtx.textAlign = 'center';
            simCtx.fillText("REPLAY CHAMPION", simCanvas.width / 2, 80);
            
            simCtx.fillStyle = 'var(--accent-gold)';
            simCtx.font = '12px Space Mono';
            simCtx.fillText(`Generation ${replayingChamp.gen} Leader | Record: ${replayingChamp.score.toFixed(2)}m`, simCanvas.width / 2, 105);
            
            // Back button overlay
            simCtx.fillStyle = 'rgba(245, 158, 11, 0.15)';
            simCtx.strokeStyle = 'var(--accent-gold)';
            simCtx.beginPath();
            simCtx.roundRect(simCanvas.width / 2 - 80, 125, 160, 26, 6);
            simCtx.fill();
            simCtx.stroke();
            
            simCtx.fillStyle = '#ffffff';
            simCtx.font = '10px Outfit';
            simCtx.textAlign = 'center';
            simCtx.fillText("Click Champ Card to Exit", simCanvas.width / 2, 141);
        }
        
        // Auto-tracking notice
        if (!camera.autoTrack) {
            simCtx.fillStyle = 'var(--text-muted)';
            simCtx.font = '9px Outfit';
            simCtx.textAlign = 'left';
            simCtx.fillText("Manual Cam Lock (Click canvas center to restore track)", 15, simCanvas.height - 15);
            
            // Draw a small icon
            simCtx.fillStyle = 'rgba(255, 255, 255, 0.05)';
            simCtx.beginPath();
            simCtx.arc(simCanvas.width / 2, simCanvas.height - 20, 15, 0, Math.PI*2);
            simCtx.fill();
        }

        // Draw Anatomy Legend
        const lx = 15;
        const ly = 55;
        const lw = 185;
        const lh = 85;
        
        simCtx.fillStyle = 'rgba(15, 23, 42, 0.8)';
        simCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        simCtx.lineWidth = 1;
        simCtx.beginPath();
        simCtx.roundRect(lx, ly, lw, lh, 8);
        simCtx.fill();
        simCtx.stroke();
        
        simCtx.fillStyle = '#ffffff';
        simCtx.font = 'bold 9px Outfit';
        simCtx.textAlign = 'left';
        simCtx.fillText("ANATOMY KEY", lx + 10, ly + 16);
        
        // Bone
        simCtx.strokeStyle = 'rgba(241, 245, 249, 0.85)';
        simCtx.lineWidth = 2.0;
        simCtx.beginPath();
        simCtx.moveTo(lx + 12, ly + 28);
        simCtx.lineTo(lx + 32, ly + 28);
        simCtx.stroke();
        simCtx.fillStyle = 'var(--text-secondary)';
        simCtx.font = '8px Outfit';
        simCtx.fillText("Rigid Bone (Skeleton)", lx + 40, ly + 31);
        
        // Muscle - Contracted (Red)
        simCtx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
        simCtx.lineWidth = 3.5;
        simCtx.beginPath();
        simCtx.moveTo(lx + 12, ly + 41);
        simCtx.lineTo(lx + 32, ly + 41);
        simCtx.stroke();
        simCtx.fillStyle = 'var(--text-secondary)';
        simCtx.fillText("Muscle - Contracted (Red/Thick)", lx + 40, ly + 44);

        // Muscle - Extended (Cyan)
        simCtx.strokeStyle = 'rgba(6, 182, 212, 0.85)';
        simCtx.lineWidth = 1.5;
        simCtx.beginPath();
        simCtx.moveTo(lx + 12, ly + 54);
        simCtx.lineTo(lx + 32, ly + 54);
        simCtx.stroke();
        simCtx.fillStyle = 'var(--text-secondary)';
        simCtx.fillText("Muscle - Extended (Cyan/Thin)", lx + 40, ly + 57);

        // Joint (Pin)
        simCtx.fillStyle = '#f59e0b';
        simCtx.beginPath();
        simCtx.arc(lx + 22, ly + 68, 3.5, 0, Math.PI * 2);
        simCtx.fill();
        simCtx.strokeStyle = '#ffffff';
        simCtx.lineWidth = 0.5;
        simCtx.stroke();
        
        // Ground Contact Dot
        simCtx.fillStyle = '#10b981';
        simCtx.beginPath();
        simCtx.arc(lx + 22, ly + 68, 1.2, 0, Math.PI * 2);
        simCtx.fill();
        
        simCtx.fillStyle = 'var(--text-secondary)';
        simCtx.fillText("Joint node (Green = Ground Touch)", lx + 40, ly + 71);
    }

    // Double click or click center area restores auto-track
    simCanvas.addEventListener('dblclick', () => {
        camera.autoTrack = true;
        camera.targetZoom = 1.0;
        showToast("Camera lock restored to leader");
    });

    // ----------------------------------------------------
    // Canvas Graph Renderers (No Dependencies)
    // ----------------------------------------------------
    function drawTelemetryChart() {
        const w = telemetryCanvas.width;
        const h = telemetryCanvas.height;
        telemetryCtx.clearRect(0, 0, w, h);
        
        if (bestFitnessHistory.length === 0) {
            telemetryCtx.fillStyle = 'rgba(255,255,255,0.2)';
            telemetryCtx.font = '10px Outfit';
            telemetryCtx.textAlign = 'center';
            telemetryCtx.fillText("Waiting for Gen 1 completion...", w / 2, h / 2);
            return;
        }

        const padding = 20;
        const graphW = w - padding * 2;
        const graphH = h - padding * 2;

        // Calculate limits
        let maxVal = -Infinity;
        let minVal = 0; // standard baseline is 0m
        
        for (let i = 0; i < bestFitnessHistory.length; i++) {
            if (bestFitnessHistory[i] > maxVal) maxVal = bestFitnessHistory[i];
            if (avgFitnessHistory[i] < minVal) minVal = avgFitnessHistory[i];
            if (bestFitnessHistory[i] < minVal) minVal = bestFitnessHistory[i];
        }
        
        // Add padding margin to max value
        maxVal = Math.max(2.0, maxVal * 1.15);
        minVal = Math.min(-0.5, minVal * 1.15);
        const valRange = maxVal - minVal;

        // Draw grids
        telemetryCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        telemetryCtx.lineWidth = 1;
        
        // Y grid lines (3 subdivisions)
        for (let i = 0; i <= 3; i++) {
            const ratio = i / 3;
            const y = padding + graphH * (1 - ratio);
            const val = minVal + valRange * ratio;
            
            telemetryCtx.beginPath();
            telemetryCtx.moveTo(padding, y);
            telemetryCtx.lineTo(w - padding, y);
            telemetryCtx.stroke();
            
            telemetryCtx.fillStyle = 'var(--text-muted)';
            telemetryCtx.font = '7px Space Mono';
            telemetryCtx.textAlign = 'right';
            telemetryCtx.fillText(val.toFixed(1) + 'm', padding - 3, y + 2.5);
        }

        // Draw line helper
        function drawLine(history, color, width, isDashed = false) {
            telemetryCtx.strokeStyle = color;
            telemetryCtx.lineWidth = width;
            if (isDashed) telemetryCtx.setLineDash([3, 3]);
            else telemetryCtx.setLineDash([]);
            
            telemetryCtx.beginPath();

            const n = history.length;
            for (let i = 0; i < n; i++) {
                const xRatio = n > 1 ? i / (n - 1) : 0.5;
                const yRatio = valRange > 0 ? (history[i] - minVal) / valRange : 0.5;

                const gx = padding + xRatio * graphW;
                const gy = padding + (1 - yRatio) * graphH;

                if (i === 0) telemetryCtx.moveTo(gx, gy);
                else telemetryCtx.lineTo(gx, gy);
            }
            telemetryCtx.stroke();
            
            // Draw points if limited gen count
            if (n < 25) {
                telemetryCtx.fillStyle = color;
                for (let i = 0; i < n; i++) {
                    const xRatio = n > 1 ? i / (n - 1) : 0.5;
                    const yRatio = valRange > 0 ? (history[i] - minVal) / valRange : 0.5;
                    const gx = padding + xRatio * graphW;
                    const gy = padding + (1 - yRatio) * graphH;
                    
                    telemetryCtx.beginPath();
                    telemetryCtx.arc(gx, gy, 2.5, 0, Math.PI * 2);
                    telemetryCtx.fill();
                }
            }
        }

        // Draw lines
        drawLine(avgFitnessHistory, 'var(--accent-blue)', 1.5, true);
        drawLine(bestFitnessHistory, 'var(--accent-emerald)', 2.0, false);
        
        // Reset line dashes
        telemetryCtx.setLineDash([]);
        
        // Legends
        telemetryCtx.fillStyle = 'var(--accent-emerald)';
        telemetryCtx.font = 'bold 8px Outfit';
        telemetryCtx.textAlign = 'left';
        telemetryCtx.fillText("PEAK", padding + 5, padding + 10);
        
        telemetryCtx.fillStyle = 'var(--accent-blue)';
        telemetryCtx.fillText("AVG", padding + 40, padding + 10);
    }

    function drawWaveformChart(dna = null) {
        const w = waveformCanvas.width;
        const h = waveformCanvas.height;
        waveformCtx.clearRect(0, 0, w, h);

        const activeGroup = replayingChamp ? [replayingChamp.creatureInstance] : population;
        const best = getBestCreature(activeGroup);
        
        // If we have custom dna provided, fetch it, otherwise use best creature's dna
        const targetDna = dna || (best ? best.dna : null);
        
        if (!targetDna || targetDna.length === 0) {
            waveformCtx.fillStyle = 'rgba(255,255,255,0.2)';
            waveformCtx.font = '9px Outfit';
            waveformCtx.textAlign = 'center';
            waveformCtx.fillText("No active DNA parameters loaded", w / 2, h / 2);
            return;
        }

        // Draw multiple sine waveforms representing muscle contractions
        const numMuscles = Math.min(5, Math.floor(targetDna.length / 4));
        const colorPalette = [
            '#10b981', // emerald
            '#0ea5e9', // blue
            '#f59e0b', // gold
            '#ec4899', // pink
            '#8b5cf6'  // purple
        ];

        waveformCtx.lineWidth = 1.5;
        
        // Draw axis
        waveformCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        waveformCtx.beginPath();
        waveformCtx.moveTo(10, h / 2);
        waveformCtx.lineTo(w - 10, h / 2);
        waveformCtx.stroke();

        const ranges = ga.ranges;

        for (let m = 0; m < numMuscles; m++) {
            const offset = m * 4;
            const gAmp = targetDna[offset];
            const gFreq = targetDna[offset + 1];
            const gPhase = targetDna[offset + 2];

            const amp = ranges.amplitude.min + gAmp * (ranges.amplitude.max - ranges.amplitude.min);
            const freq = ranges.frequency.min + gFreq * (ranges.frequency.max - ranges.frequency.min);
            const phase = ranges.phase.min + gPhase * (ranges.phase.max - ranges.phase.min);

            waveformCtx.strokeStyle = colorPalette[m % colorPalette.length];
            waveformCtx.beginPath();

            for (let x = 10; x < w - 10; x++) {
                const ratio = (x - 10) / (w - 20);
                const t = ratio * 2.0; // Show 2 full reference seconds
                const yOffset = amp * Math.sin(t * freq * Math.PI * 2 + phase);
                
                // map [-0.5, 0.5] range to canvas height bounds [padding, height-padding]
                const val = yOffset * 2.0; // scale up slightly for visual effect
                const cy = h / 2 + val * (h / 2.5);

                if (x === 10) waveformCtx.moveTo(x, cy);
                else waveformCtx.lineTo(x, cy);
            }
            waveformCtx.stroke();
        }
        
        waveformCtx.fillStyle = 'var(--text-muted)';
        waveformCtx.font = '6px Space Mono';
        waveformCtx.textAlign = 'left';
        waveformCtx.fillText("0.0s", 10, h - 4);
        waveformCtx.textAlign = 'right';
        waveformCtx.fillText("2.0s Timeline", w - 10, h - 4);
    }

    // ----------------------------------------------------
    // Custom Creator Editor Mechanics
    // ----------------------------------------------------
    let hoveredBone = null; // { index, ratio }

    function findHoveredBone(mouseX, mouseY) {
        let closestBone = null;
        let closestRatio = 0.5;
        let minDistance = 18; // hover detection range in pixels
        
        editorConstraints.forEach((c, idx) => {
            if (c.isMuscle) return; // Cannot attach muscles to other muscles
            const nodeA = editorNodes[c.a];
            const nodeB = editorNodes[c.b];
            if (!nodeA || !nodeB) return;
            
            const dx = nodeB.x - nodeA.x;
            const dy = nodeB.y - nodeA.y;
            const lenSq = dx * dx + dy * dy;
            if (lenSq === 0) return;
            
            // Project mouse position onto bone segment AB
            let t = ((mouseX - nodeA.x) * dx + (mouseY - nodeA.y) * dy) / lenSq;
            t = Math.max(0.05, Math.min(0.95, t)); // clamp to segment (avoid extreme ends)
            
            const px = nodeA.x + t * dx;
            const py = nodeA.y + t * dy;
            
            const dist = Math.sqrt((mouseX - px) * (mouseX - px) + (mouseY - py) * (mouseY - py));
            if (dist < minDistance) {
                minDistance = dist;
                closestBone = idx;
                closestRatio = t;
            }
        });
        
        if (closestBone !== null) {
            return { index: closestBone, ratio: closestRatio };
        }
        return null;
    }

    function handleEditorMouseMove(e) {
        const rect = simCanvas.getBoundingClientRect();
        editorMousePos.x = e.clientX - rect.left - simCanvas.width / 2;
        editorMousePos.y = e.clientY - rect.top - simCanvas.height / 2;
        
        // Update hovered bone
        if (editorTool === 'muscle') {
            hoveredBone = findHoveredBone(editorMousePos.x, editorMousePos.y);
        } else {
            hoveredBone = null;
        }
    }

    function handleEditorClick(e) {
        const rect = simCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left - simCanvas.width / 2;
        const mouseY = e.clientY - rect.top - simCanvas.height / 2;

        if (editorTool === 'joint') {
            // Add joint node
            const existsNear = editorNodes.some(n => {
                const dx = n.x - mouseX;
                const dy = n.y - mouseY;
                return Math.sqrt(dx * dx + dy * dy) < 16;
            });
            if (!existsNear) {
                editorNodes.push({ x: mouseX, y: mouseY, radius: 8, mass: 1.0 });
            }
        } else if (editorTool === 'bone') {
            // Link joints into rigid bones
            let clickedIdx = -1;
            for (let i = 0; i < editorNodes.length; i++) {
                const n = editorNodes[i];
                const dx = n.x - mouseX;
                const dy = n.y - mouseY;
                if (Math.sqrt(dx * dx + dy * dy) < 14) {
                    clickedIdx = i;
                    break;
                }
            }

            if (clickedIdx !== -1) {
                if (editorSelectedNodeIdx === null) {
                    editorSelectedNodeIdx = clickedIdx;
                } else {
                    if (editorSelectedNodeIdx !== clickedIdx) {
                        const connectionExists = editorConstraints.some(c => 
                            !c.isMuscle && 
                            ((c.a === editorSelectedNodeIdx && c.b === clickedIdx) ||
                             (c.a === clickedIdx && c.b === editorSelectedNodeIdx))
                        );
                        if (!connectionExists) {
                            editorConstraints.push({
                                a: editorSelectedNodeIdx,
                                b: clickedIdx,
                                isMuscle: false
                            });
                        }
                    }
                    editorSelectedNodeIdx = null;
                }
            } else {
                editorSelectedNodeIdx = null;
            }
        } else if (editorTool === 'muscle') {
            // Link bones to bones via dynamic muscles
            const hovered = findHoveredBone(mouseX, mouseY);
            if (hovered !== null) {
                if (editorSelectedBoneIdx === null) {
                    editorSelectedBoneIdx = hovered.index;
                    editorSelectedBoneRatio = hovered.ratio;
                } else {
                    if (editorSelectedBoneIdx !== hovered.index) {
                        // Create muscle connection between Bone A and Bone B
                        editorConstraints.push({
                            a: editorSelectedBoneIdx,
                            ratioA: editorSelectedBoneRatio,
                            b: hovered.index,
                            ratioB: hovered.ratio,
                            isMuscle: true
                        });
                    }
                    editorSelectedBoneIdx = null;
                }
            } else {
                editorSelectedBoneIdx = null;
            }
        } else if (editorTool === 'erase') {
            // Erase joint node
            let clickedIdx = -1;
            for (let i = 0; i < editorNodes.length; i++) {
                const n = editorNodes[i];
                const dx = n.x - mouseX;
                const dy = n.y - mouseY;
                if (Math.sqrt(dx * dx + dy * dy) < 14) {
                    clickedIdx = i;
                    break;
                }
            }

            if (clickedIdx !== -1) {
                // Delete all bones/muscles connected to this joint
                // Since muscles connect to bones, if a bone is deleted, we must delete connected muscles too!
                // To make it simple: if a joint is deleted, we find what bones are deleted.
                const deletedBonesIndices = [];
                editorConstraints.forEach((c, idx) => {
                    if (!c.isMuscle && (c.a === clickedIdx || c.b === clickedIdx)) {
                        deletedBonesIndices.push(idx);
                    }
                });

                // Filter out deleted bones and any muscles attached to them
                editorConstraints = editorConstraints.filter((c, idx) => {
                    if (c.isMuscle) {
                        return !deletedBonesIndices.includes(c.a) && !deletedBonesIndices.includes(c.b);
                    } else {
                        return c.a !== clickedIdx && c.b !== clickedIdx;
                    }
                });

                // Shift joint indices in remaining bones
                editorConstraints.forEach(c => {
                    if (!c.isMuscle) {
                        if (c.a > clickedIdx) c.a--;
                        if (c.b > clickedIdx) c.b--;
                    }
                });

                // Shift bone indices in remaining muscles
                // A bone's index is its position in the list.
                // When we filter the list, indices shift. It's safer to reconstruct index mappings!
                // Let's do that dynamically during save or just reconstruct indices now:
                const oldConstraints = [...editorConstraints];
                const remainingBones = oldConstraints.filter(c => !c.isMuscle);
                const remainingMuscles = oldConstraints.filter(c => c.isMuscle);
                
                remainingMuscles.forEach(m => {
                    const boneA = oldConstraints[m.a];
                    const boneB = oldConstraints[m.b];
                    m.a = remainingBones.indexOf(boneA);
                    m.b = remainingBones.indexOf(boneB);
                });
                
                editorConstraints = [...remainingBones, ...remainingMuscles];

                // Delete the node
                editorNodes.splice(clickedIdx, 1);
                editorSelectedNodeIdx = null;
                editorSelectedBoneIdx = null;
            }
        }
    }

    function renderEditor() {
        simCtx.clearRect(0, 0, simCanvas.width, simCanvas.height);
        
        simCtx.save();
        simCtx.translate(simCanvas.width / 2, simCanvas.height / 2);

        // 1. Ground Level
        const editorGroundY = 160;
        simCtx.strokeStyle = 'rgba(16, 185, 129, 0.5)';
        simCtx.lineWidth = 2;
        simCtx.beginPath();
        simCtx.moveTo(-1000, editorGroundY);
        simCtx.lineTo(1000, editorGroundY);
        simCtx.stroke();
        
        simCtx.fillStyle = 'rgba(16, 185, 129, 0.45)';
        simCtx.font = 'bold 9px Outfit';
        simCtx.textAlign = 'left';
        simCtx.fillText("⚠️ SOLID GROUND LEVEL (Build above this line)", -simCanvas.width / 2 + 20, editorGroundY - 6);

        // Center crosshair
        simCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        simCtx.lineWidth = 1;
        simCtx.beginPath();
        simCtx.moveTo(-20, 0); simCtx.lineTo(20, 0);
        simCtx.moveTo(0, -20); simCtx.lineTo(0, 20);
        simCtx.stroke();
        
        simCtx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        simCtx.font = '8px Outfit';
        simCtx.textAlign = 'center';
        simCtx.fillText("Spawning Center (CoM)", 0, -8);

        // 2. Draw draft bones
        editorConstraints.forEach(c => {
            if (!c.isMuscle) {
                const nodeA = editorNodes[c.a];
                const nodeB = editorNodes[c.b];
                if (!nodeA || !nodeB) return;
                
                simCtx.beginPath();
                simCtx.moveTo(nodeA.x, nodeA.y);
                simCtx.lineTo(nodeB.x, nodeB.y);
                
                simCtx.strokeStyle = 'rgba(226, 232, 240, 0.2)';
                simCtx.lineWidth = 5.0;
                simCtx.stroke();
                
                simCtx.strokeStyle = 'rgba(241, 245, 249, 0.9)';
                simCtx.lineWidth = 1.8;
                simCtx.stroke();
            }
        });

        // 3. Draw draft muscles (Bone-to-Bone)
        editorConstraints.forEach(c => {
            if (c.isMuscle) {
                const boneA = editorConstraints[c.a];
                const boneB = editorConstraints[c.b];
                if (!boneA || !boneB) return;
                
                const nodeA = editorNodes[boneA.a];
                const nodeB = editorNodes[boneA.b];
                const nodeC = editorNodes[boneB.a];
                const nodeD = editorNodes[boneB.b];
                if (!nodeA || !nodeB || !nodeC || !nodeD) return;

                // Attachment points
                const p1x = nodeA.x + c.ratioA * (nodeB.x - nodeA.x);
                const p1y = nodeA.y + c.ratioA * (nodeB.y - nodeA.y);
                const p2x = nodeC.x + c.ratioB * (nodeD.x - nodeC.x);
                const p2y = nodeC.y + c.ratioB * (nodeD.y - nodeC.y);

                simCtx.beginPath();
                simCtx.moveTo(p1x, p1y);
                simCtx.lineTo(p2x, p2y);
                
                simCtx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
                simCtx.lineWidth = 4.5;
                simCtx.stroke();
                
                simCtx.strokeStyle = '#ffffff';
                simCtx.lineWidth = 0.8;
                simCtx.stroke();
            }
        });

        // 4. Draw active rubber-band line
        if (editorSelectedNodeIdx !== null && editorTool === 'bone') {
            // Bone rubber-band
            const originNode = editorNodes[editorSelectedNodeIdx];
            if (originNode) {
                simCtx.beginPath();
                simCtx.moveTo(originNode.x, originNode.y);
                simCtx.lineTo(editorMousePos.x, editorMousePos.y);
                simCtx.strokeStyle = 'rgba(241, 245, 249, 0.5)';
                simCtx.lineWidth = 1.8;
                simCtx.setLineDash([4, 4]);
                simCtx.stroke();
                simCtx.setLineDash([]);
            }
        } else if (editorSelectedBoneIdx !== null && editorTool === 'muscle') {
            // Muscle rubber-band from bone to cursor
            const bone = editorConstraints[editorSelectedBoneIdx];
            const nodeA = editorNodes[bone.a];
            const nodeB = editorNodes[bone.b];
            if (nodeA && nodeB) {
                const px = nodeA.x + editorSelectedBoneRatio * (nodeB.x - nodeA.x);
                const py = nodeA.y + editorSelectedBoneRatio * (nodeB.y - nodeA.y);

                simCtx.beginPath();
                simCtx.moveTo(px, py);
                simCtx.lineTo(editorMousePos.x, editorMousePos.y);
                simCtx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
                simCtx.lineWidth = 2.5;
                simCtx.setLineDash([4, 4]);
                simCtx.stroke();
                simCtx.setLineDash([]);
            }
        }

        // 5. Draw hovered bone preview dot
        if (hoveredBone !== null && editorTool === 'muscle') {
            const bone = editorConstraints[hoveredBone.index];
            const nodeA = editorNodes[bone.a];
            const nodeB = editorNodes[bone.b];
            if (nodeA && nodeB) {
                const px = nodeA.x + hoveredBone.ratio * (nodeB.x - nodeA.x);
                const py = nodeA.y + hoveredBone.ratio * (nodeB.y - nodeA.y);

                // Draw pulsing attachment guide dot
                simCtx.fillStyle = 'rgba(16, 185, 129, 0.9)';
                simCtx.beginPath();
                simCtx.arc(px, py, 4, 0, Math.PI * 2);
                simCtx.fill();
                simCtx.strokeStyle = '#ffffff';
                simCtx.lineWidth = 1;
                simCtx.stroke();
            }
        }

        // 6. Draw draft nodes (joints)
        editorNodes.forEach((n, idx) => {
            const isSelected = (idx === editorSelectedNodeIdx && editorTool === 'bone');
            
            simCtx.fillStyle = 'rgba(15, 23, 42, 0.9)';
            simCtx.beginPath();
            simCtx.arc(n.x, n.y, n.radius + 2, 0, Math.PI * 2);
            simCtx.fill();
            
            simCtx.strokeStyle = isSelected ? 'var(--accent-gold)' : 'rgba(255, 255, 255, 0.4)';
            simCtx.lineWidth = isSelected ? 2.0 : 1.0;
            simCtx.stroke();

            simCtx.fillStyle = '#f59e0b';
            simCtx.beginPath();
            simCtx.arc(n.x, n.y, n.radius - 2, 0, Math.PI * 2);
            simCtx.fill();
            
            simCtx.fillStyle = 'rgba(255,255,255,0.4)';
            simCtx.font = '7px Space Mono';
            simCtx.textAlign = 'center';
            simCtx.fillText(idx, n.x, n.y + 14);
        });

        simCtx.restore();

        // 7. Draw Editor HUD Overlay text
        simCtx.fillStyle = '#ffffff';
        simCtx.font = 'bold 11px Outfit';
        simCtx.textAlign = 'left';
        simCtx.fillText("🛠️ CUSTOM CREATURE DESIGN MODE", 15, 30);
        
        simCtx.fillStyle = 'var(--text-secondary)';
        simCtx.font = '9px Outfit';
        
        const bonesCount = editorConstraints.filter(c => !c.isMuscle).length;
        const musclesCount = editorConstraints.filter(c => c.isMuscle).length;
        simCtx.fillText(`Tool: ${editorTool.toUpperCase()} | Joints: ${editorNodes.length} | Bones: ${bonesCount} | Muscles: ${musclesCount}`, 15, 43);
    }

    function saveCustomCreature() {
        if (editorNodes.length < 2) {
            showToast("⚠️ Add at least 2 joints to create a creature!");
            return;
        }
        
        const bones = editorConstraints.filter(c => !c.isMuscle);
        const muscles = editorConstraints.filter(c => c.isMuscle);

        if (bones.length < 1) {
            showToast("⚠️ Connect your joints with at least 1 bone first!");
            return;
        }
        if (muscles.length < 1) {
            showToast("⚠️ Connect your bones with at least 1 muscle!");
            return;
        }

        // 1. Calculate center of mass of nodes to establish spawn offsets relative to center
        let sumX = 0;
        let sumY = 0;
        editorNodes.forEach(n => {
            sumX += n.x;
            sumY += n.y;
        });
        const avgX = sumX / editorNodes.length;
        const avgY = sumY / editorNodes.length;

        // 2. Map nodes to relative dx, dy offsets
        const relativeNodes = editorNodes.map(n => {
            return {
                dx: n.x - avgX,
                dy: n.y - avgY,
                radius: 8,
                mass: 1.0
            };
        });

        // 3. Map constraints: Skeletons (Bones) first, then Muscles.
        // We translate muscle target indices (which referenced full editorConstraints list)
        // to correspond to their filtered index in the bones-only array.
        const templateConstraints = [];

        // Add Bones
        bones.forEach(b => {
            templateConstraints.push({
                a: b.a,
                b: b.b,
                stiffness: 0.85,
                isMuscle: false
            });
        });

        // Add Muscles
        muscles.forEach(m => {
            const originalBoneA = editorConstraints[m.a];
            const originalBoneB = editorConstraints[m.b];
            
            const boneIdxA = bones.indexOf(originalBoneA);
            const boneIdxB = bones.indexOf(originalBoneB);

            templateConstraints.push({
                a: boneIdxA,
                ratioA: m.ratioA,
                b: boneIdxB,
                ratioB: m.ratioB,
                stiffness: 0.8,
                isMuscle: true
            });
        });

        // 4. Register custom template in presets
        CreaturePresets.templates.custom = {
            color: 'rgba(168, 85, 247, 0.45)', // Custom purple theme
            nodes: relativeNodes,
            constraints: templateConstraints
        };

        // 5. Exit Editor Mode
        isEditorMode = false;
        document.getElementById('editor-toolbar').classList.add('hidden');
        simCanvas.parentElement.classList.remove('editor-mode');

        // Set activePreset to custom
        activePreset = 'custom';
        
        // Highlight custom button, remove active from other presets
        presetBtns.forEach(btn => {
            if (btn.id === 'custom-preset-btn') {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Reset GA
        generationCount = 1;
        bestFitnessHistory = [];
        avgFitnessHistory = [];
        
        // Resume simulation
        isPlaying = true;
        playPauseBtn.innerHTML = '<span class="btn-icon">⏸️</span> Pause Sim';
        playPauseBtn.classList.remove('paused');

        initSimulation();
        drawTelemetryChart();
        showToast("🚀 Custom species evolved and deployed!");
    }

    // ----------------------------------------------------
    // Editor Event Listeners
    // ----------------------------------------------------
    const customPresetBtn = document.getElementById('custom-preset-btn');
    const editorToolbar = document.getElementById('editor-toolbar');
    const clearEditorBtn = document.getElementById('clear-editor-btn');
    const saveCreatureBtn = document.getElementById('save-creature-btn');
    const toolBtns = document.querySelectorAll('.tool-btn[data-tool]');

    customPresetBtn.addEventListener('click', () => {
        // Toggle on Editor Mode
        isEditorMode = true;
        editorNodes = [];
        editorConstraints = [];
        editorSelectedNodeIdx = null;
        editorTool = 'joint';
        
        // Update active tool UI classes
        toolBtns.forEach(b => {
            if (b.dataset.tool === 'joint') b.classList.add('active');
            else b.classList.remove('active');
        });

        // Toggle UI
        editorToolbar.classList.remove('hidden');
        simCanvas.parentElement.classList.add('editor-mode');
        
        // Pause physics
        isPlaying = false;
        playPauseBtn.innerHTML = '<span class="btn-icon">▶️</span> Resume Sim';
        playPauseBtn.classList.add('paused');
        
        // Reset camera to center 0,0 and zoom 1x
        camera.x = 0;
        camera.y = 0;
        camera.zoom = 1.0;
        camera.targetZoom = 1.0;
        camera.autoTrack = false;
        
        presetBtns.forEach(b => b.classList.remove('active'));
        customPresetBtn.classList.add('active');

        showToast("🛠️ Place joints above the floor line, then connect with bones/muscles.");
    });

    toolBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            toolBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            editorTool = btn.dataset.tool;
            editorSelectedNodeIdx = null; // reset dragging
        });
    });

    clearEditorBtn.addEventListener('click', () => {
        editorNodes = [];
        editorConstraints = [];
        editorSelectedNodeIdx = null;
        showToast("🧹 Editor canvas cleared");
    });

    saveCreatureBtn.addEventListener('click', () => {
        saveCustomCreature();
    });

    // ----------------------------------------------------
    // Master Simulation Game Loop
    // ----------------------------------------------------
    let lastTime = 0;
    
    function loop(time) {
        if (!lastTime) lastTime = time;
        
        if (isPlaying) {
            // Apply physics steps according to simSpeed multiplier
            for (let s = 0; s < simSpeed; s++) {
                physics.step(timeStep);
                simTimer += timeStep;
                
                // If generation duration threshold hit, calculate evolution next generation
                if (simTimer >= generationDuration && !replayingChamp) {
                    nextGeneration();
                    break;
                }
            }
            
            // Loop replay for champions instead of evolving
            if (replayingChamp && simTimer >= generationDuration) {
                replayingChamp.creatureInstance.reset(0, 0);
                physics.resetAll(0, 0);
                simTimer = 0.0;
            }
        }

        // Render scene
        render();

        // Update Sim Timer UI
        simTimerLabel.textContent = Math.min(generationDuration, simTimer).toFixed(1) + 's';

        // Keep loop repeating
        lastTime = time;
        requestAnimationFrame(loop);
    }

    // ----------------------------------------------------
    // Application Bootstrap
    // ----------------------------------------------------
    resizeCanvas();
    initSimulation();
    requestAnimationFrame(loop);
});
