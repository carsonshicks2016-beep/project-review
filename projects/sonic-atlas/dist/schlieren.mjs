/**
 * Mach Lab — NASA AirBOS-Style Background-Oriented Schlieren & Shadowgraph Density Gradient Renderer
 * Computes optical density gradients |grad rho| around supersonic vehicles.
 * Renders refractive luminescence, dynamic shock shimmer, expansion fans (canopy, wing roots),
 * tail recompression shocks, ambient background grid distortion, and scientific optical filters.
 */

// Clamp utility
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/**
 * Optical and scientific color palettes for Schlieren visualization
 */
export const SCHLIEREN_PALETTES = {
  amber: {
    id: 'amber',
    name: 'NASA Wind Tunnel Amber',
    bg: '#120b04',
    bgField: 'rgba(28, 16, 6, 0.95)',
    grid: 'rgba(180, 110, 30, 0.18)',
    gridDistorted: 'rgba(245, 158, 11, 0.45)',
    shockGlow: '#f59e0b',
    shockCore: '#fef3c7',
    expansion: 'rgba(40, 20, 8, 0.65)',
    expansionTint: 'rgba(120, 50, 10, 0.25)',
    exhaust: 'rgba(251, 146, 60, 0.70)',
    aircraft: '#e2d5c3',
    aircraftOutline: '#785338',
    hud: '#fbbf24',
    hudDim: '#92602e'
  },
  'cyber-cyan': {
    id: 'cyber-cyan',
    name: 'Cyber-Cyan Optical Gradient',
    bg: '#040b12',
    bgField: 'rgba(6, 18, 30, 0.95)',
    grid: 'rgba(34, 211, 238, 0.16)',
    gridDistorted: 'rgba(56, 189, 248, 0.48)',
    shockGlow: '#00f0ff',
    shockCore: '#e0faff',
    expansion: 'rgba(10, 28, 48, 0.70)',
    expansionTint: 'rgba(14, 116, 144, 0.22)',
    exhaust: 'rgba(56, 189, 248, 0.75)',
    aircraft: '#d8f4f8',
    aircraftOutline: '#1e4e66',
    hud: '#22d3ee',
    hudDim: '#155e75'
  },
  monochrome: {
    id: 'monochrome',
    name: 'NASA Knife-Edge Monochrome',
    bg: '#242424',
    bgField: 'rgba(42, 42, 42, 0.95)',
    grid: 'rgba(200, 200, 200, 0.18)',
    gridDistorted: 'rgba(255, 255, 255, 0.55)',
    shockGlow: '#e8e8e8',
    shockCore: '#ffffff',
    expansion: 'rgba(14, 14, 14, 0.75)',
    expansionTint: 'rgba(0, 0, 0, 0.35)',
    exhaust: 'rgba(190, 190, 190, 0.65)',
    aircraft: '#f0f0f0',
    aircraftOutline: '#101010',
    hud: '#dcdcdc',
    hudDim: '#707070'
  },
  'airbos-desert': {
    id: 'airbos-desert',
    name: 'NASA AirBOS Flight Test (Edwards AFB)',
    bg: '#0f2438',
    bgField: 'rgba(15, 36, 56, 0.95)',
    grid: 'rgba(217, 180, 126, 0.22)',
    gridDistorted: 'rgba(245, 210, 150, 0.60)',
    shockGlow: '#ffeed0',
    shockCore: '#ffffff',
    expansion: 'rgba(8, 20, 32, 0.65)',
    expansionTint: 'rgba(180, 140, 90, 0.20)',
    exhaust: 'rgba(255, 170, 70, 0.75)',
    aircraft: '#f7fafc',
    aircraftOutline: '#2d3748',
    hud: '#e2e8f0',
    hudDim: '#94a3b8'
  }
};

/**
 * Calculates shock wave geometry and angles for a given Mach number and vehicle configuration.
 */
export function getShockGeometry(state = {}) {
  const m = Math.max(0.1, Number(state.mach ?? 2.0));
  const isSupersonic = m > 1.0;
  const aircraft = state.aircraft || 'custom';

  // Mach cone half angle: mu = asin(1 / M)
  const mu = isSupersonic ? Math.asin(1 / m) : null;
  const muDeg = mu ? (mu * 180) / Math.PI : null;

  // Slender nose wedge angle theta varies by aircraft
  let thetaNose = 0.14; // ~8 degrees
  if (aircraft === 'x59') thetaNose = 0.07; // ultra-slender needle nose
  else if (aircraft === 'concorde') thetaNose = 0.10;
  else if (aircraft === 'sr71') thetaNose = 0.11;
  else if (aircraft === 'f16') thetaNose = 0.16;

  // Oblique shock wave angle beta > mu
  const beta = isSupersonic ? Math.min(Math.PI * 0.48, mu + (0.6 * thetaNose) / Math.sqrt(m * m - 1 + 0.05)) : null;
  const betaDeg = beta ? (beta * 180) / Math.PI : null;

  // Tail recompression shock angle (slightly steeper due to blunt expansion recovery)
  const betaTail = isSupersonic ? Math.min(Math.PI * 0.48, mu + (0.45 * thetaNose) / Math.sqrt(m * m - 1 + 0.05)) : null;

  return {
    mach: m,
    isSupersonic,
    aircraft,
    mu,
    muDeg,
    beta,
    betaDeg,
    betaTail,
    thetaNose
  };
}

/**
 * Computes the optical density gradient |grad rho| and optical deflection at a point (x, y).
 * Models bow shock, canopy shock, expansion fans, wing shock, and tail recompression.
 *
 * @param {number} x X coordinate in canvas space
 * @param {number} y Y coordinate in canvas space
 * @param {Object} state Simulation and vehicle state
 * @returns {Object} { gradX, gradY, magnitude, rho, deflectionX, deflectionY }
 */
export function computeDensityGradient(x, y, state = {}) {
  const m = Math.max(0.1, Number(state.mach ?? 2.0));
  const geom = getShockGeometry(state);

  const px = Number(state.px ?? 300);
  const py = Number(state.py ?? 150);
  const L = Math.max(40, Number(state.length ?? 160));
  const time = Number(state.time ?? 0);

  // Normalized coordinates relative to aircraft nose
  // Aircraft flies left-to-right: nose is at x = px + 0.5*L, tail at x = px - 0.5*L
  const xNose = px + 0.5 * L;
  const yCenter = py;

  const dx = xNose - x; // distance behind nose (>0 is downstream)
  const dy = y - yCenter; // vertical distance from centerline
  const absDy = Math.abs(dy);

  let gradX = 0;
  let gradY = 0;
  let rho = 1.0; // Ambient density ratio

  // Boundary clamping for numerical stability
  if (Math.abs(dx) > 10000 || Math.abs(dy) > 10000 || !Number.isFinite(dx) || !Number.isFinite(dy)) {
    return { gradX: 0, gradY: 0, magnitude: 0, rho: 1.0, deflectionX: 0, deflectionY: 0 };
  }

  if (m <= 1.0) {
    // Subsonic / Transonic flow field: smooth compressible density variation
    const betaSub = Math.sqrt(Math.max(0.01, 1 - m * m));
    const distSq = dx * dx + betaSub * betaSub * dy * dy + 400;
    const dist = Math.sqrt(distSq);

    const amp = 0.8 * (m * m);
    rho += amp * (dx / dist);
    gradX = amp * (1 / dist - (dx * dx) / (dist * dist * dist));
    gradY = -amp * (dx * betaSub * betaSub * dy) / (dist * dist * dist);

    // Transonic local normal shock if M ≈ 0.90–1.00
    if (m >= 0.88 && dx > 0.25 * L && dx < 0.35 * L && absDy < 0.25 * L) {
      const shockDx = dx - 0.30 * L;
      const normalShock = Math.exp(-Math.pow(shockDx / 3.0, 2)) * Math.exp(-absDy / (0.15 * L));
      gradX += normalShock * 3.5;
    }
  } else {
    // Supersonic flow field: shock discontinuities and expansion fans
    const beta = geom.beta;
    const tanBeta = Math.tan(beta);
    const cosBeta = Math.cos(beta);
    const sinBeta = Math.sin(beta);

    // Shock wave decay with distance (Whitham r^-0.75 or r^-0.5)
    const distFactor = Math.pow(1 + absDy / (0.6 * L), -0.65);

    // 1. Bow Shock: originates at nose (dx = 0)
    if (dx >= -4) {
      // Perpendicular distance to bow shock line: dx * tanBeta - absDy = 0
      // Line: y - yCenter = +/- (xNose - x) * tanBeta => dx * tanBeta - absDy = 0
      const dPerpBow = (dx * tanBeta - absDy) * cosBeta;
      const sigmaBow = 2.4; // shock thickness in pixels

      if (Math.abs(dPerpBow) < 20) {
        const bowGrad = Math.exp(-Math.pow(dPerpBow / sigmaBow, 2)) * distFactor * (m * 1.5);
        // Gradient vector perpendicular to shock line
        const signY = dy >= 0 ? -1 : 1;
        gradX += bowGrad * sinBeta;
        gradY += bowGrad * cosBeta * signY;
        rho += 0.5 * bowGrad;
      }
    }

    // 2. Canopy Compression Shock (behind nose at dx ≈ 0.18*L)
    const dxCanopy = dx - 0.18 * L;
    if (dxCanopy >= -4 && geom.aircraft !== 'x59') {
      const tanBetaCanopy = Math.tan(beta * 1.05);
      const dPerpCanopy = (dxCanopy * tanBetaCanopy - (dy < 0 ? -dy - 0.04 * L : absDy)) * cosBeta;
      if (Math.abs(dPerpCanopy) < 14 && dy < 0.05 * L) {
        const canopyGrad = Math.exp(-Math.pow(dPerpCanopy / 2.2, 2)) * distFactor * (m * 0.9);
        gradX += canopyGrad * sinBeta;
        gradY -= canopyGrad * cosBeta;
      }
    }

    // 3. Wing Leading Edge Shock (dx ≈ 0.45*L)
    const dxWing = dx - 0.45 * L;
    if (dxWing >= -4) {
      const dPerpWing = (dxWing * tanBeta - absDy) * cosBeta;
      if (Math.abs(dPerpWing) < 16) {
        const wingGrad = Math.exp(-Math.pow(dPerpWing / 2.2, 2)) * distFactor * (m * 0.85);
        const signY = dy >= 0 ? -1 : 1;
        gradX += wingGrad * sinBeta;
        gradY += wingGrad * cosBeta * signY;
      }
    }

    // 4. Prandtl-Meyer Expansion Fans (canopy crest dx ≈ 0.30*L, wing shoulder dx ≈ 0.55*L)
    // Density decreases continuously along expansion fan (gradX < 0)
    for (const expDx of [0.30 * L, 0.58 * L]) {
      const dCenter = dx - expDx;
      if (dCenter > 0 && absDy < 1.8 * L) {
        const fanAngle = Math.atan2(absDy, dCenter);
        // Fan spans between beta and Mach angle mu
        if (fanAngle >= geom.mu * 0.9 && fanAngle <= beta * 1.2) {
          const expIntensity = Math.exp(-absDy / (0.8 * L)) * (m * 0.6);
          gradX -= expIntensity * 0.4;
          gradY -= expIntensity * 0.3 * (dy >= 0 ? 1 : -1);
          rho -= 0.25 * expIntensity;
        }
      }
    }

    // 5. Tail Recompression Shock (dx ≈ 0.95*L)
    const dxTail = dx - 0.95 * L;
    if (dxTail >= -4) {
      const tanBetaTail = Math.tan(geom.betaTail);
      const cosBetaTail = Math.cos(geom.betaTail);
      const sinBetaTail = Math.sin(geom.betaTail);
      const dPerpTail = (dxTail * tanBetaTail - absDy) * cosBetaTail;

      if (Math.abs(dPerpTail) < 22) {
        const tailGrad = Math.exp(-Math.pow(dPerpTail / 2.5, 2)) * distFactor * (m * 1.25);
        const signY = dy >= 0 ? -1 : 1;
        gradX += tailGrad * sinBetaTail;
        gradY += tailGrad * cosBetaTail * signY;
        rho += 0.4 * tailGrad;
      }
    }

    // 6. Supersonic Exhaust Plume Wake (dx > L, near centerline absDy < 0.15*L)
    if (dx > L && absDy < 0.25 * L) {
      const wakeDx = dx - L;
      const wakeCore = Math.exp(-Math.pow(absDy / (0.06 * L), 2));
      // Periodic shock diamonds in underexpanded jet plume
      const diamondFreq = (2.0 * Math.PI) / (0.18 * L);
      const diamond = Math.cos(diamondFreq * wakeDx);
      const plumeGrad = wakeCore * Math.exp(-wakeDx / (1.5 * L)) * (1.2 + 0.8 * diamond);
      gradX += plumeGrad * 0.5;
      gradY += plumeGrad * (dy >= 0 ? -0.8 : 0.8);
    }
  }

  // Atmospheric Shimmer turbulence along shock fronts
  if (state.shimmer !== false && time > 0) {
    const shimScale = 0.06;
    const shimTurb = 0.12 * Math.sin(x * shimScale - 3.2 * time) * Math.cos(y * shimScale + 2.1 * time);
    gradX += gradX * shimTurb;
    gradY += gradY * shimTurb;
  }

  // Calculate total gradient magnitude |grad rho|
  const rawMagnitude = Math.sqrt(gradX * gradX + gradY * gradY);
  // Clamped magnitude for robust optical mapping
  const magnitude = clamp(rawMagnitude, 0, 10.0);

  // Optical Ray Deflection (Gladstone-Dale: epsilon = K * grad rho)
  const opticalConstant = 1.8;
  const deflectionX = clamp(gradX * opticalConstant, -15, 15);
  const deflectionY = clamp(gradY * opticalConstant, -15, 15);

  return {
    gradX,
    gradY,
    magnitude,
    rho: clamp(rho, 0.2, 5.0),
    deflectionX,
    deflectionY
  };
}

/**
 * Calculates optical ray deflection vector for background grid refraction.
 */
export function rayDeflection(x, y, state = {}) {
  const g = computeDensityGradient(x, y, state);
  return {
    dx: g.deflectionX,
    dy: g.deflectionY,
    magnitude: g.magnitude
  };
}

/**
 * Draws the aircraft silhouette corresponding to preset geometry.
 */
function drawAircraftBody(ctx, px, py, L, aircraft, palette) {
  ctx.save();
  ctx.translate(px, py);

  ctx.fillStyle = palette.aircraft;
  ctx.strokeStyle = palette.aircraftOutline;
  ctx.lineWidth = 1.5;

  ctx.beginPath();

  if (aircraft === 'x59') {
    // NASA X-59 QueSST: signature long needle nose, canards, delta wing, T-tail
    const noseX = 0.5 * L;
    const tailX = -0.5 * L;
    ctx.moveTo(noseX, 0); // Needle nose tip
    ctx.lineTo(noseX - 0.25 * L, -0.015 * L); // Ultra-long slender nose cone
    ctx.lineTo(noseX - 0.32 * L, -0.06 * L); // Canard leading edge
    ctx.lineTo(noseX - 0.35 * L, -0.02 * L); // Canard trailing edge
    ctx.lineTo(noseX - 0.45 * L, -0.035 * L); // Flush canopy area (camera screen)
    ctx.lineTo(noseX - 0.55 * L, -0.16 * L); // Delta wing leading edge
    ctx.lineTo(noseX - 0.85 * L, -0.16 * L); // Wing tip
    ctx.lineTo(noseX - 0.82 * L, -0.04 * L); // Wing trailing edge
    ctx.lineTo(tailX + 0.08 * L, -0.07 * L); // T-tail vertical fin
    ctx.lineTo(tailX, -0.07 * L); // Tail tip
    ctx.lineTo(tailX, 0.07 * L);
    ctx.lineTo(tailX + 0.08 * L, 0.07 * L);
    ctx.lineTo(noseX - 0.82 * L, 0.04 * L);
    ctx.lineTo(noseX - 0.85 * L, 0.16 * L);
    ctx.lineTo(noseX - 0.55 * L, 0.16 * L);
    ctx.lineTo(noseX - 0.45 * L, 0.035 * L);
    ctx.lineTo(noseX - 0.35 * L, 0.02 * L);
    ctx.lineTo(noseX - 0.32 * L, 0.06 * L);
    ctx.lineTo(noseX - 0.25 * L, 0.015 * L);
  } else if (aircraft === 'concorde') {
    // Concorde: slender ogive delta
    const noseX = 0.5 * L;
    const tailX = -0.5 * L;
    ctx.moveTo(noseX, 0);
    ctx.quadraticCurveTo(noseX - 0.2 * L, -0.03 * L, noseX - 0.45 * L, -0.04 * L);
    ctx.quadraticCurveTo(noseX - 0.6 * L, -0.18 * L, noseX - 0.85 * L, -0.18 * L);
    ctx.lineTo(noseX - 0.88 * L, -0.03 * L);
    ctx.lineTo(tailX, -0.03 * L);
    ctx.lineTo(tailX, 0.03 * L);
    ctx.lineTo(noseX - 0.88 * L, 0.03 * L);
    ctx.lineTo(noseX - 0.85 * L, 0.18 * L);
    ctx.quadraticCurveTo(noseX - 0.6 * L, 0.18 * L, noseX - 0.45 * L, 0.04 * L);
    ctx.quadraticCurveTo(noseX - 0.2 * L, 0.03 * L, noseX, 0);
  } else if (aircraft === 'sr71') {
    // SR-71 Blackbird: distinctive blended chines, twin nacelles
    const noseX = 0.5 * L;
    const tailX = -0.5 * L;
    ctx.moveTo(noseX, 0);
    ctx.lineTo(noseX - 0.3 * L, -0.04 * L);
    ctx.lineTo(noseX - 0.45 * L, -0.08 * L);
    ctx.lineTo(noseX - 0.5 * L, -0.14 * L); // Nacelle
    ctx.lineTo(noseX - 0.75 * L, -0.14 * L);
    ctx.lineTo(noseX - 0.8 * L, -0.08 * L);
    ctx.lineTo(tailX, -0.03 * L);
    ctx.lineTo(tailX, 0.03 * L);
    ctx.lineTo(noseX - 0.8 * L, 0.08 * L);
    ctx.lineTo(noseX - 0.75 * L, 0.14 * L);
    ctx.lineTo(noseX - 0.5 * L, 0.14 * L);
    ctx.lineTo(noseX - 0.45 * L, 0.08 * L);
    ctx.lineTo(noseX - 0.3 * L, 0.04 * L);
  } else {
    // Standard / F-16: sleek supersonic jet
    const noseX = 0.5 * L;
    const tailX = -0.5 * L;
    ctx.moveTo(noseX, 0);
    ctx.lineTo(noseX - 0.2 * L, -0.03 * L);
    ctx.lineTo(noseX - 0.3 * L, -0.06 * L); // Canopy
    ctx.lineTo(noseX - 0.45 * L, -0.035 * L);
    ctx.lineTo(noseX - 0.55 * L, -0.18 * L); // Wing
    ctx.lineTo(noseX - 0.75 * L, -0.18 * L);
    ctx.lineTo(noseX - 0.7 * L, -0.03 * L);
    ctx.lineTo(tailX, -0.03 * L);
    ctx.lineTo(tailX, 0.03 * L);
    ctx.lineTo(noseX - 0.7 * L, 0.03 * L);
    ctx.lineTo(noseX - 0.75 * L, 0.18 * L);
    ctx.lineTo(noseX - 0.55 * L, 0.18 * L);
    ctx.lineTo(noseX - 0.45 * L, 0.035 * L);
    ctx.lineTo(noseX - 0.3 * L, 0.06 * L);
    ctx.lineTo(noseX - 0.2 * L, 0.03 * L);
  }

  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.restore();
}

/**
 * Main Schlieren & Shadowgraph Density Gradient Renderer
 * Implements NASA AirBOS-style optical visualization.
 *
 * @param {CanvasRenderingContext2D|Object} ctx Canvas 2D rendering context or mock
 * @param {number} width Canvas width in pixels
 * @param {number} height Canvas height in pixels
 * @param {Object} state Simulation state { mach, px, py, length, palette, backgroundDistortion, shimmer, time }
 */
export function renderSchlieren(ctx, width, height, state = {}) {
  if (!ctx) return;

  const w = width || 800;
  const h = height || 400;

  const paletteKey = state.palette || 'amber';
  const palette = SCHLIEREN_PALETTES[paletteKey] || SCHLIEREN_PALETTES.amber;

  const m = Math.max(0.1, Number(state.mach ?? 2.0));
  const isSupersonic = m > 1.0;
  const px = Number(state.px ?? w * 0.48);
  const py = Number(state.py ?? h * 0.45);
  const L = Math.max(40, Number(state.length ?? Math.min(180, w * 0.25)));
  const aircraft = state.aircraft || 'custom';
  const backgroundDistortion = state.backgroundDistortion !== false;
  const time = Number(state.time ?? 0);

  const geom = getShockGeometry({ mach: m, aircraft });

  // 1. Draw Background Optical Chamber
  if (ctx.fillStyle !== undefined) {
    ctx.fillStyle = palette.bg;
    ctx.fillRect(0, 0, w, h);

    // Subtle optical vignette
    if (ctx.createRadialGradient) {
      const vig = ctx.createRadialGradient(w * 0.5, h * 0.5, h * 0.2, w * 0.5, h * 0.5, w * 0.65);
      vig.addColorStop(0, 'rgba(0,0,0,0)');
      vig.addColorStop(1, 'rgba(0,0,0,0.55)');
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, w, h);
    }
  }

  // 2. Ambient Background Grid (AirBOS Reference Grid with Optical Refraction Distortion)
  const gridStep = Math.max(20, Math.round(L * 0.18));
  ctx.save();
  ctx.lineWidth = 1;

  if (backgroundDistortion) {
    // Refracted grid lines: trace segments through density gradient field
    ctx.strokeStyle = palette.gridDistorted;

    // Vertical grid lines with lateral optical refraction
    for (let gx = gridStep; gx < w; gx += gridStep) {
      ctx.beginPath();
      let first = true;
      for (let gy = 0; gy <= h; gy += 10) {
        const defl = rayDeflection(gx, gy, { mach: m, px, py, length: L, aircraft, time });
        const drawX = gx + defl.dx;
        const drawY = gy + defl.dy;
        if (first) {
          ctx.moveTo(drawX, drawY);
          first = false;
        } else {
          ctx.lineTo(drawX, drawY);
        }
      }
      ctx.stroke();
    }

    // Horizontal grid lines with longitudinal optical refraction
    for (let gy = gridStep; gy < h; gy += gridStep) {
      ctx.beginPath();
      let first = true;
      for (let gx = 0; gx <= w; gx += 12) {
        const defl = rayDeflection(gx, gy, { mach: m, px, py, length: L, aircraft, time });
        const drawX = gx + defl.dx;
        const drawY = gy + defl.dy;
        if (first) {
          ctx.moveTo(drawX, drawY);
          first = false;
        } else {
          ctx.lineTo(drawX, drawY);
        }
      }
      ctx.stroke();
    }
  } else {
    // Undistorted baseline reference grid
    ctx.strokeStyle = palette.grid;
    ctx.beginPath();
    for (let gx = gridStep; gx < w; gx += gridStep) {
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, h);
    }
    for (let gy = gridStep; gy < h; gy += gridStep) {
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
    }
    ctx.stroke();
  }
  ctx.restore();

  // 3. Supersonic Flow Field Features (Expansion Fans & Refractive Luminescence)
  if (isSupersonic) {
    const noseX = px + 0.5 * L;
    const tailX = px - 0.5 * L;
    const beta = geom.beta;
    const tanBeta = Math.tan(beta);
    const leftEdge = 0;

    // A. Prandtl-Meyer Expansion Fans (canopy crest & wing root)
    // Renders smooth expansion gradient shading between initial and terminal Mach lines
    ctx.save();
    for (const expDx of [0.32 * L, 0.55 * L]) {
      const expOriginX = noseX - expDx;
      const fanAngle1 = geom.mu * 0.95;
      const fanAngle2 = beta * 1.15;

      const fanLen = Math.max(w, h) * 1.2;

      // Upper expansion fan
      ctx.fillStyle = palette.expansionTint;
      ctx.beginPath();
      ctx.moveTo(expOriginX, py - 0.03 * L);
      ctx.lineTo(expOriginX - fanLen * Math.cos(fanAngle1), py - fanLen * Math.sin(fanAngle1));
      ctx.lineTo(expOriginX - fanLen * Math.cos(fanAngle2), py - fanLen * Math.sin(fanAngle2));
      ctx.closePath();
      ctx.fill();

      // Lower expansion fan
      ctx.beginPath();
      ctx.moveTo(expOriginX, py + 0.03 * L);
      ctx.lineTo(expOriginX - fanLen * Math.cos(fanAngle1), py + fanLen * Math.sin(fanAngle1));
      ctx.lineTo(expOriginX - fanLen * Math.cos(fanAngle2), py + fanLen * Math.sin(fanAngle2));
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();

    // B. Bow Shock Wave Luminescence & Shimmer
    ctx.save();
    const renderShockLine = (originX, originY, angle, isUpper, strength = 1.0) => {
      const signY = isUpper ? -1 : 1;
      const tanA = Math.tan(angle);

      // Multi-pass glow for refractive luminescence
      const passes = [
        { width: 9 * strength, alpha: 0.18, color: palette.shockGlow },
        { width: 4 * strength, alpha: 0.45, color: palette.shockGlow },
        { width: 1.5 * strength, alpha: 0.95, color: palette.shockCore }
      ];

      for (const pass of passes) {
        ctx.beginPath();
        ctx.lineWidth = pass.width;
        ctx.strokeStyle = pass.color;
        ctx.globalAlpha = pass.alpha;

        ctx.moveTo(originX, originY);

        const steps = 40;
        const totalDx = originX - leftEdge;
        for (let i = 1; i <= steps; i++) {
          const curDx = (i / steps) * totalDx;
          const curX = originX - curDx;

          // Weak curvature towards Mach angle mu with distance
          const curvedTan = tanA * Math.pow(1 + curDx / (2.5 * L), -0.08);

          // Dynamic refractive shimmer along shock boundary
          const shim = state.shimmer !== false && time > 0
            ? 1.5 * Math.sin(0.08 * curX - 4.0 * time) * Math.cos(0.05 * curDx)
            : 0;

          const curY = originY + signY * curDx * curvedTan + shim;
          ctx.lineTo(curX, curY);
        }
        ctx.stroke();
      }
    };

    // Render Bow Shock (Upper & Lower)
    renderShockLine(noseX, py, beta, true, 1.2);
    renderShockLine(noseX, py, beta, false, 1.2);

    // Render Canopy Shock (if not X-59 quiet needle)
    if (aircraft !== 'x59') {
      const canopyX = noseX - 0.20 * L;
      renderShockLine(canopyX, py - 0.05 * L, beta * 1.04, true, 0.75);
    } else {
      // X-59 signature soft nose ramp & canard compression waves
      const canardX = noseX - 0.32 * L;
      renderShockLine(canardX, py - 0.04 * L, beta * 0.96, true, 0.55);
      renderShockLine(canardX, py + 0.04 * L, beta * 0.96, false, 0.55);
    }

    // Render Wing Leading Edge Shock
    const wingX = noseX - 0.48 * L;
    renderShockLine(wingX, py - 0.06 * L, beta, true, 0.85);
    renderShockLine(wingX, py + 0.06 * L, beta, false, 0.85);

    // Render Tail Recompression Shock Wave
    const tailShockX = tailX + 0.05 * L;
    renderShockLine(tailShockX, py - 0.03 * L, geom.betaTail, true, 1.05);
    renderShockLine(tailShockX, py + 0.03 * L, geom.betaTail, false, 1.05);

    // C. Supersonic Engine Exhaust Plume & Mach Diamonds
    ctx.globalAlpha = 0.85;
    const plumeLen = Math.min(w * 0.35, 1.8 * L);
    const diamondSpacing = 0.16 * L;
    const numDiamonds = Math.floor(plumeLen / diamondSpacing);

    for (let d = 1; d <= numDiamonds; d++) {
      const dX = tailX - d * diamondSpacing;
      const dW = diamondSpacing * 0.45 * Math.pow(0.85, d);
      const dH = 0.07 * L * Math.pow(0.88, d);

      ctx.beginPath();
      ctx.moveTo(dX - dW, py);
      ctx.lineTo(dX, py - dH);
      ctx.lineTo(dX + dW, py);
      ctx.lineTo(dX, py + dH);
      ctx.closePath();
      ctx.fillStyle = palette.exhaust;
      ctx.fill();
      ctx.strokeStyle = palette.shockCore;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    ctx.restore();
  }

  // 4. Vehicle Silhouette
  drawAircraftBody(ctx, px, py, L, aircraft, palette);

  // 5. Sleek Scientific Instrumentation HUD
  ctx.save();
  ctx.font = '11px "DM Sans", -apple-system, BlinkMacSystemFont, monospace';
  ctx.fillStyle = palette.hud;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';

  const margin = 16;
  ctx.fillText('OPTICAL SCHLIEREN · NASA AirBOS DENSITY GRADIENT |∇ρ|', margin, margin);

  ctx.font = '10px "DM Sans", -apple-system, BlinkMacSystemFont, monospace';
  ctx.fillStyle = palette.hudDim;
  ctx.fillText(
    `FILTER: ${palette.name.toUpperCase()} · PALETTE [${paletteKey.toUpperCase()}] · GRID REFRACTION: ${backgroundDistortion ? 'ON' : 'OFF'}`,
    margin,
    margin + 16
  );

  // Flight condition badge
  const badgeX = w - margin;
  ctx.textAlign = 'right';
  ctx.fillStyle = palette.hud;
  ctx.fillText(`MACH ${m.toFixed(2)} · ${isSupersonic ? 'SUPERSONIC SHOCK FIELD' : 'SUBSONIC FLOW'}`, badgeX, margin);

  if (isSupersonic && geom.muDeg) {
    ctx.fillStyle = palette.hudDim;
    ctx.fillText(`MACH ANGLE μ: ${geom.muDeg.toFixed(1)}° · BOW SHOCK β: ${geom.betaDeg.toFixed(1)}°`, badgeX, margin + 16);
  }

  // Density Gradient Scale Bar
  const barW = 120;
  const barH = 8;
  const barX = margin;
  const barY = h - margin - barH - 12;

  ctx.textAlign = 'left';
  ctx.fillStyle = palette.hudDim;
  ctx.fillText('OPTICAL SENSITIVITY |∇ρ|', barX, barY - 12);

  if (ctx.createLinearGradient) {
    const barGrad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
    barGrad.addColorStop(0, 'rgba(0,0,0,0.4)');
    barGrad.addColorStop(0.5, palette.shockGlow);
    barGrad.addColorStop(1, palette.shockCore);
    ctx.fillStyle = barGrad;
    ctx.fillRect(barX, barY, barW, barH);
  }
  ctx.strokeStyle = palette.hudDim;
  ctx.lineWidth = 1;
  ctx.strokeRect(barX, barY, barW, barH);

  ctx.fillText('0', barX, barY + barH + 2);
  ctx.textAlign = 'right';
  ctx.fillText('HIGH (SHOCK)', barX + barW, barY + barH + 2);

  ctx.restore();
}
