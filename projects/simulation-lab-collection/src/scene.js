import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// Visual scale of the drone airframe. fpvCamera.js derives its mount offsets from this,
// so changing it here keeps the FPV camera outside the airframe.
export const DRONE_SCALE = 6.4;

// Shared by the scene fog and the point-cloud splat shader so the environment and the
// objects in it recede at the same rate.
export const FOG_DENSITY = 0.003;

// Gate appearance per race state. Cyan is reserved for the racing line and UI chrome;
// amber means "this is the gate the drone is flying at right now".
export const GATE_STATES = {
  upcoming: { body: '#11616f', emissive: '#0b3d47', rim: '#3fd6e8', rimOpacity: 0.7,  glow: 0.0 },
  next:     { body: '#7a4100', emissive: '#ff9d00', rim: '#ffd894', rimOpacity: 1.0,  glow: 0.4 },
  passed:   { body: '#1a222c', emissive: '#080d12', rim: '#39485a', rimOpacity: 0.3,  glow: 0.0 },
  selected: { body: '#7a0038', emissive: '#ff007f', rim: '#ffffff', rimOpacity: 1.0,  glow: 0.4 }
};

// --- Renderer & Scene ---
export function createRenderer() {
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    
    const context = canvas.getContext('webgl2', { antialias: true });
    const renderer = new THREE.WebGLRenderer({
        canvas,
        context,
        powerPreference: 'high-performance',
        antialias: true
    });
    
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    // Enable Cinematic Lighting & Shadows
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    
    return renderer;
}

export function createScene() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#030108'); 
    // Depth cue: distant geometry recedes into the background colour, which is what
    // separates the drone/gates from the LiDAR cloud behind them.
    scene.fog = new THREE.FogExp2('#030108', FOG_DENSITY);
    
    // Core lighting
    const ambient = new THREE.AmbientLight('#1a1a3a', 1.5);
    ambient.name = 'cyber-ambient';
    scene.add(ambient);
    
    // Sun / Moon directional light for shadows
    const dirLight = new THREE.DirectionalLight('#6688ff', 2.0);
    dirLight.name = 'cyber-sun';
    dirLight.position.set(100, 150, 50);
    dirLight.castShadow = true;
    
    // Configure shadow map area to cover the track
    dirLight.shadow.camera.top = 150;
    dirLight.shadow.camera.bottom = -150;
    dirLight.shadow.camera.left = -150;
    dirLight.shadow.camera.right = 150;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 500;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    // Atmospheric Star / Dust Particles
    const particles = new THREE.BufferGeometry();
    const pCount = 2500;
    const pos = new Float32Array(pCount * 3);
    for(let i=0; i < pCount * 3; i++) {
        pos[i] = (Math.random() - 0.5) * 600; // Scatter in a 600m cube
    }
    particles.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const pMat = new THREE.PointsMaterial({ 
        color: '#00ffff', 
        size: 0.6, 
        transparent: true, 
        opacity: 0.3,
        blending: THREE.AdditiveBlending
    });
    const pPoints = new THREE.Points(particles, pMat);
    pPoints.name = 'cyber-particles';
    scene.add(pPoints);
    
    return scene;
}

// --- LiDAR Denoising Filter ---
// Uses a blazing fast O(N) voxel grid to count spatial density and delete isolated points (floaters)
function cleanPointCloud(geometry, voxelSize = 0.1, threshold = 8) {
    const pos = geometry.attributes.position;
    if (!pos) return;
    const N = pos.count;
    
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    
    const dimX = Math.ceil((bbox.max.x - bbox.min.x) / voxelSize) + 1;
    const dimY = Math.ceil((bbox.max.y - bbox.min.y) / voxelSize) + 1;
    const dimZ = Math.ceil((bbox.max.z - bbox.min.z) / voxelSize) + 1;
    
    const totalCells = dimX * dimY * dimZ;
    // Safety check: if scanner threw a point 5 miles away, don't blow up the RAM
    if (totalCells > 60000000) {
        console.warn("Point cloud spread too large for fast voxel cleanup. Skipping denoise.");
        return;
    }
    
    const grid = new Uint16Array(totalCells);
    
    // Pass 1: Count points per voxel
    for (let i = 0; i < N; i++) {
        const x = pos.getX(i);
        const y = pos.getY(i);
        const z = pos.getZ(i);
        
        const vx = Math.floor((x - bbox.min.x) / voxelSize);
        const vy = Math.floor((y - bbox.min.y) / voxelSize);
        const vz = Math.floor((z - bbox.min.z) / voxelSize);
        
        const idx = vx + (vy * dimX) + (vz * dimX * dimY);
        if (grid[idx] < 65535) grid[idx]++;
    }
    
    // Pass 2: Keep only points in dense voxels
    const keepIndices = [];
    for (let i = 0; i < N; i++) {
        const x = pos.getX(i);
        const y = pos.getY(i);
        const z = pos.getZ(i);
        
        const vx = Math.floor((x - bbox.min.x) / voxelSize);
        const vy = Math.floor((y - bbox.min.y) / voxelSize);
        const vz = Math.floor((z - bbox.min.z) / voxelSize);
        
        const idx = vx + (vy * dimX) + (vz * dimX * dimY);
        if (grid[idx] >= threshold) {
            keepIndices.push(i);
        }
    }
    
    const newN = keepIndices.length;
    console.log(`Denoise complete: Removed ${N - newN} floaters (kept ${newN}/${N}).`);
    
    if (newN === N) return; // Nothing removed
    
    // Pass 3: Rebuild geometry
    const newPos = new Float32Array(newN * 3);
    const hasColors = geometry.hasAttribute('color');
    const hasNormals = geometry.hasAttribute('normal');
    
    let newCol = null;
    let oldCol = null;
    let newNorm = null;
    let oldNorm = null;
    
    if (hasColors) {
        newCol = new Float32Array(newN * 3);
        oldCol = geometry.attributes.color;
    }
    if (hasNormals) {
        newNorm = new Float32Array(newN * 3);
        oldNorm = geometry.attributes.normal;
    }
    
    for (let i = 0; i < newN; i++) {
        const oldIdx = keepIndices[i];
        newPos[i*3] = pos.getX(oldIdx);
        newPos[i*3+1] = pos.getY(oldIdx);
        newPos[i*3+2] = pos.getZ(oldIdx);
        
        if (hasColors) {
            newCol[i*3] = oldCol.getX(oldIdx);
            newCol[i*3+1] = oldCol.getY(oldIdx);
            newCol[i*3+2] = oldCol.getZ(oldIdx);
        }
        
        if (hasNormals) {
            newNorm[i*3] = oldNorm.getX(oldIdx);
            newNorm[i*3+1] = oldNorm.getY(oldIdx);
            newNorm[i*3+2] = oldNorm.getZ(oldIdx);
        }
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
    if (hasColors) geometry.setAttribute('color', new THREE.BufferAttribute(newCol, 3));
    if (hasNormals) geometry.setAttribute('normal', new THREE.BufferAttribute(newNorm, 3));
}

// --- Point Cloud Splat Material ---
// PointsMaterial draws flat, unlit, axis-aligned squares. That is why the scan reads as
// confetti rather than a surface: no shading means no form, and hard square edges tile
// visibly wherever splats overlap. This material fixes both, and uses the per-point
// normals the scan already carries (PLYLoader reads nx/ny/nz) to actually light it.
export function createSplatMaterial(opts = {}) {
    return new THREE.ShaderMaterial({
        uniforms: {
            uSize:       { value: opts.size ?? 1.1 },
            uMinPx:      { value: 1.0 },
            uMaxPx:      { value: 22.0 },   // Above this, splats read as marbles not surface
            uViewportH:  { value: window.innerHeight },
            uLightDir:   { value: new THREE.Vector3(0.35, 0.9, 0.25).normalize() },
            uAmbient:    { value: 0.62 },
            uFogColor:   { value: new THREE.Color(opts.fogColor ?? '#030108') },
            uFogDensity: { value: opts.fogDensity ?? 0.003 },
            uOpacity:    { value: 1.0 }
        },
        vertexShader: `
            attribute vec3 color;
            uniform float uSize, uMinPx, uMaxPx, uViewportH, uAmbient;
            uniform vec3 uLightDir;
            varying vec3 vColor;
            varying float vFogDepth;

            void main() {
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_Position = projectionMatrix * mvPosition;

                // World-size to screen-pixels. Point spacing scales the same way, so the
                // overlap ratio stays constant with distance.
                float px = uSize * (projectionMatrix[1][1] * 0.5 * uViewportH) / -mvPosition.z;
                gl_PointSize = clamp(px, uMinPx, uMaxPx);

                // Lambert from the scan's own normals. abs() because scan normals are not
                // consistently oriented -- we want two-sided shading, not black patches.
                vec3 n = normalize(normalMatrix * normal);
                float diff = abs(dot(n, uLightDir));
                vColor = color * (uAmbient + (1.0 - uAmbient) * diff);

                vFogDepth = -mvPosition.z;
            }
        `,
        fragmentShader: `
            uniform vec3 uFogColor;
            uniform float uFogDensity, uOpacity;
            varying vec3 vColor;
            varying float vFogDepth;

            void main() {
                // Round splat instead of a square. Overlapping discs read as a continuous
                // surface; overlapping squares read as a tile grid.
                vec2 d = gl_PointCoord - 0.5;
                float r2 = dot(d, d);
                if (r2 > 0.25) discard;

                // Darken toward the rim so each splat reads as a small dome. This is what
                // stops large close-up splats from looking like flat stickers.
                // Subtle rim darkening only -- enough to give each splat a little form,
                // not enough to read as a bead.
                float dome = 1.0 - smoothstep(0.06, 0.25, r2);
                vec3 col = vColor * (0.94 + 0.06 * dome);

                float f = 1.0 - exp(-uFogDensity * uFogDensity * vFogDepth * vFogDepth);
                col = mix(col, uFogColor, clamp(f, 0.0, 1.0));

                gl_FragColor = vec4(col, uOpacity);

                #include <tonemapping_fragment>
                #include <colorspace_fragment>
            }
        `
    });
}

// --- Environment Recession ---
// The scan is a photographic interior: mid-brightness, high chroma, high frequency.
// Left alone it occupies exactly the value range the drone and gates need. Pulling
// saturation and brightness down turns it back into a backdrop.
function recedeVertexColors(geometry, saturation = 0.85, brightness = 0.9) {
    const col = geometry.attributes.color;
    if (!col) return;
    // Colours are already linear here (PLYLoader converts from sRGB), so use linear luma.
    for (let i = 0; i < col.count; i++) {
        const r = col.getX(i), g = col.getY(i), b = col.getZ(i);
        const lum = r * 0.2126 + g * 0.7152 + b * 0.0722;
        col.setXYZ(
            i,
            (lum + (r - lum) * saturation) * brightness,
            (lum + (g - lum) * saturation) * brightness,
            (lum + (b - lum) * saturation) * brightness
        );
    }
    col.needsUpdate = true;
}

// --- Environment Loader ---
export function loadHousePointCloud(scene, { onProgress, onDone, onError } = {}) {
    const loader = new PLYLoader();
    console.log("Loading LiDAR scan...");
    
    loader.load(
        './assets/ok.ply',
        (geometry) => {
            console.log("LiDAR scan loaded! Processing geometry...");
            
            // Center the geometry at its origin
            geometry.center();
            
            // Run custom density filter to destroy floaters
            cleanPointCloud(geometry, 0.1, 8);
            
            const hasColors = geometry.hasAttribute('color');
            if (hasColors) recedeVertexColors(geometry);
            
            if (!geometry.hasAttribute('normal')) {
                console.warn('Scan has no normals; splat shading will be flat.');
                geometry.computeVertexNormals();
            }

            const material = createSplatMaterial({ size: 1.1, fogDensity: FOG_DENSITY });

            const points = new THREE.Points(geometry, material);
            points.name = 'lidar'; // Looked up by name so the editor can fade it
            
            // Revert rotation
            points.rotation.set(0, 0, 0); 
            
            // The house is currently a dollhouse! Let's scale it up 8x more (Total: 40x)
            points.scale.set(40, 40, 40); 
            
            // Perfectly align the bottom of the house to the ground (Y = 0)
            geometry.computeBoundingBox();
            const bbox = geometry.boundingBox;
            points.position.y = Math.abs(bbox.min.y) * points.scale.y;
            
            scene.add(points);
            console.log("House added to scene!");
            if (onDone) onDone(points);
        },
        (xhr) => {
            if (xhr.lengthComputable && onProgress) onProgress(xhr.loaded / xhr.total);
        },
        (error) => {
            console.error("Error loading PLY:", error);
            if (onError) onError(error);
        }
    );
}

// --- Ground ---
export function createGround() {
    const group = new THREE.Group();
    group.name = 'cyber-ground';
    
    // Solid lifelike floor (Asphalt / Concrete)
    const floorGeo = new THREE.PlaneGeometry(1000, 1000);
    const floorMat = new THREE.MeshStandardMaterial({ 
        color: '#222222', // Dark asphalt
        metalness: 0.1,
        roughness: 0.95
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    group.add(floor);
    
    // High-contrast outer boundary border
    const borderEdges = new THREE.LineSegments(
        new THREE.EdgesGeometry(floorGeo),
        new THREE.LineBasicMaterial({ color: '#00ffff' })
    );
    borderEdges.rotation.x = -Math.PI / 2;
    borderEdges.position.y = 0.1;
    group.add(borderEdges);

    // High-contrast interior spatial grid (helps ground the drone visually)
    const grid = new THREE.GridHelper(1000, 50, '#ffffff', '#555555');
    grid.position.y = 0.05; 
    // Make the grid highly transparent so it doesn't overwhelm, but provides borders
    grid.material.transparent = true;
    grid.material.opacity = 0.3;
    group.add(grid);
    
    return group;
}

// --- Gates ---
// A 3m gate on a 180m circuit is ~19px on screen from a framing camera, which makes
// gates effectively unfindable in the editor. This label is drawn with
// sizeAttenuation:false so it holds a constant screen size at any distance, turning
// each gate into something you can spot and click.
function createGateLabel(index) {
    const size = 128;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2 - 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(8,12,20,0.85)';
    ctx.fill();
    ctx.lineWidth = 6;
    ctx.strokeStyle = '#3fd6e8';
    ctx.stroke();
    
    ctx.fillStyle = '#eaf6ff';
    ctx.font = 'bold 58px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(index), size / 2, size / 2 + 2);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: texture,
        sizeAttenuation: false,
        depthTest: false,
        transparent: true
    }));
    sprite.scale.set(0.05, 0.05, 1);
    sprite.renderOrder = 900;
    sprite.visible = false;
    return sprite;
}

// Rebuilds the numbered label on a gate (index changes when gates are added/removed).
export function setGateLabel(gateGroup, index) {
    const parts = gateGroup.userData.parts;
    if (!parts) return;
    if (parts.label) {
        gateGroup.remove(parts.label);
        parts.label.material.map.dispose();
        parts.label.material.dispose();
    }
    const label = createGateLabel(index);
    label.position.set(0, 4.6, 0);
    label.visible = !!gateGroup.userData.labelsVisible;
    gateGroup.add(label);
    parts.label = label;
}

export function setGateLabelVisible(gateGroup, visible) {
    gateGroup.userData.labelsVisible = visible;
    const parts = gateGroup.userData.parts;
    if (parts && parts.label) parts.label.visible = visible;
}

export function createGateMesh(radius = 3.0, tubeRadius = 0.5) {
    const group = new THREE.Group();
    const s = GATE_STATES.upcoming;
    
    // Main ring. Emissive rather than lit, so gates hold their colour regardless of
    // where the drone is relative to the directional light.
    const torusGeo = new THREE.TorusGeometry(radius, tubeRadius, 12, 48);
    const torusMat = new THREE.MeshStandardMaterial({
        color: s.body,
        emissive: s.emissive,
        emissiveIntensity: 1.0,
        metalness: 0.0,
        roughness: 0.6
    });
    const torus = new THREE.Mesh(torusGeo, torusMat);
    torus.castShadow = true;
    torus.receiveShadow = true;
    group.add(torus);
    
    // A single clean inner rim. The old EdgesGeometry wireframe overlay produced ~100
    // segments per gate, which read as scribble at any distance; one bright circle reads
    // as a gate.
    const rimGeo = new THREE.TorusGeometry(radius - tubeRadius, tubeRadius * 0.16, 8, 48);
    const rimMat = new THREE.MeshBasicMaterial({
        color: s.rim,
        transparent: true,
        opacity: s.rimOpacity
    });
    const rim = new THREE.Mesh(rimGeo, rimMat);
    group.add(rim);
    
    // Additive halo, only switched on for the gate the drone is currently flying at.
    const glowGeo = new THREE.TorusGeometry(radius, tubeRadius * 1.9, 8, 48);
    const glowMat = new THREE.MeshBasicMaterial({
        color: s.emissive,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        fog: false
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.visible = false;
    glow.renderOrder = 5;
    group.add(glow);
    
    group.userData.parts = { torus, rim, glow };
    group.userData.state = 'upcoming';
    
    return group;
}

// Applies a race state ('upcoming' | 'next' | 'passed' | 'selected') to a gate group.
// No-ops when the state is unchanged, so it is safe to call every frame.
export function setGateState(gateGroup, state, force = false) {
    const parts = gateGroup.userData.parts;
    if (!parts) return;
    if (!force && gateGroup.userData.state === state) return;
    
    const s = GATE_STATES[state] || GATE_STATES.upcoming;
    gateGroup.userData.state = state;
    
    parts.torus.material.color.set(s.body);
    parts.torus.material.emissive.set(s.emissive);
    parts.rim.material.color.set(s.rim);
    parts.rim.material.opacity = s.rimOpacity;
    parts.glow.material.color.set(s.emissive);
    parts.glow.material.opacity = s.glow;
    parts.glow.userData.baseOpacity = s.glow;
    parts.glow.visible = s.glow > 0;
    if (!parts.glow.visible) parts.glow.scale.set(1, 1, 1);
}

// Breathes the halo on whichever gate is currently highlighted.
export function pulseGate(gateGroup, time) {
    const parts = gateGroup.userData.parts;
    if (!parts || !parts.glow.visible) return;
    const base = parts.glow.userData.baseOpacity || 0;
    const wave = 0.65 + 0.35 * Math.sin(time * 5.0);
    parts.glow.material.opacity = base * wave;
    const sc = 1.0 + 0.05 * Math.sin(time * 5.0);
    parts.glow.scale.set(sc, sc, 1);
}

// --- Drone ---
export function createDroneMesh() {
    const group = new THREE.Group();
    const s = DRONE_SCALE;
    
    // Realistic FPV Drone Materials (High-Vis)
    const carbonMat = new THREE.MeshStandardMaterial({ color: '#111111', metalness: 0.2, roughness: 0.8 });
    const motorMat = new THREE.MeshStandardMaterial({ color: '#666666', metalness: 0.9, roughness: 0.4 });
    const propMat = new THREE.MeshStandardMaterial({ 
        color: '#ffaa00', // Orange props for visibility
        transparent: true, 
        opacity: 0.5, 
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const camMat = new THREE.MeshStandardMaterial({ color: '#111111', metalness: 0.5, roughness: 0.5 });
    
    // High-Vis Fluorescent Plastic for the top canopy (so it pops out instantly)
    const highVisMat = new THREE.MeshStandardMaterial({
        color: '#ccff00', 
        emissive: '#334400', // slight glow to mimic fluorescent paint
        roughness: 0.2,
        metalness: 0.1
    });
    
    // Bottom Plate (Carbon Fiber)
    const baseGeo = new THREE.BoxGeometry(0.15 * s, 0.02 * s, 0.3 * s);
    const base = new THREE.Mesh(baseGeo, carbonMat);
    base.add(new THREE.LineSegments(new THREE.EdgesGeometry(baseGeo), new THREE.LineBasicMaterial({ color: '#00ffcc' })));
    base.castShadow = true;
    group.add(base);
    
    // Top Plate / Canopy (High-Vis Neon)
    const topGeo = new THREE.BoxGeometry(0.12 * s, 0.04 * s, 0.22 * s);
    const top = new THREE.Mesh(topGeo, highVisMat);
    top.add(new THREE.LineSegments(new THREE.EdgesGeometry(topGeo), new THREE.LineBasicMaterial({ color: '#ffffff' })));
    top.position.y = 0.08 * s;
    top.castShadow = true;
    group.add(top);
    
    // FPV Camera (Angled on the front)
    const camGeo = new THREE.BoxGeometry(0.05 * s, 0.05 * s, 0.05 * s);
    const camera = new THREE.Mesh(camGeo, camMat);
    camera.add(new THREE.LineSegments(new THREE.EdgesGeometry(camGeo), new THREE.LineBasicMaterial({ color: '#00ffcc' })));
    camera.position.set(0, 0.05 * s, -0.15 * s);
    camera.rotation.x = -Math.PI / 6; // 30-degree camera uptilt
    group.add(camera);
    
    // Bright Red Taillight LED (Makes it incredibly easy to track from behind)
    const ledGeo = new THREE.BoxGeometry(0.06 * s, 0.02 * s, 0.02 * s);
    const ledMat = new THREE.MeshBasicMaterial({ color: '#ff0000' });
    const led = new THREE.Mesh(ledGeo, ledMat);
    led.position.set(0, 0.08 * s, 0.11 * s);
    group.add(led);
    
    // Carbon Fiber Arms (X-Frame)
    const armGeo = new THREE.BoxGeometry(0.04 * s, 0.02 * s, 0.42 * s);
    const armEdgesMat = new THREE.LineBasicMaterial({ color: '#00ffcc' });
    
    const arm1 = new THREE.Mesh(armGeo, carbonMat);
    arm1.add(new THREE.LineSegments(new THREE.EdgesGeometry(armGeo), armEdgesMat));
    arm1.rotation.y = Math.PI / 4;
    group.add(arm1);
    
    const arm2 = new THREE.Mesh(armGeo, carbonMat);
    arm2.add(new THREE.LineSegments(new THREE.EdgesGeometry(armGeo), armEdgesMat));
    arm2.rotation.y = -Math.PI / 4;
    group.add(arm2);
    
    // Motors & Props
    const mDistX = 0.15 * s;
    const mDistZ = 0.15 * s;
    const motorGeo = new THREE.CylinderGeometry(0.03 * s, 0.03 * s, 0.04 * s, 12);
    const propGeo = new THREE.CylinderGeometry(0.12 * s, 0.12 * s, 0.005 * s, 16);
    
    const positions = [
        [-mDistX, mDistZ], [mDistX, mDistZ], [-mDistX, -mDistZ], [mDistX, -mDistZ]
    ];
    
    positions.forEach(pos => {
        // Motor bell
        const motor = new THREE.Mesh(motorGeo, motorMat);
        motor.position.set(pos[0], 0.02 * s, pos[1]);
        motor.castShadow = true;
        group.add(motor);
        
        // Spinning prop disc
        const prop = new THREE.Mesh(propGeo, propMat);
        prop.position.set(pos[0], 0.045 * s, pos[1]);
        group.add(prop);
    });
    
    return group;
}

// --- Track Line ---
export function createTrackLine(curve) {
    // A clean, bright cyan tube acting as a physical racing rope/guide
    // Radius increased to 0.15 (about 3-4x thicker than the old 0.04)
    const geometry = new THREE.TubeGeometry(curve, 400, 0.15, 8, true);
    const material = new THREE.MeshBasicMaterial({ 
        color: '#00e5ff', 
        transparent: true,
        opacity: 0.35, // A hint of the ideal line, not a competitor to the gates
        depthWrite: false
    });
    const tube = new THREE.Mesh(geometry, material);
    return tube;
}
