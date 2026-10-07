// Procedural PS1 Retro Pixel Art Sprite Generator

export class SpriteManager {
  constructor() {
    this.cache = new Map();
    this.generateAllSprites();
  }

  get(name) {
    return this.cache.get(name);
  }

  generateAllSprites() {
    // 1. Scenery & props
    this.createPineTree();
    this.createPalmTree();
    this.createGuardrails();
    this.createStreetLamp();
    this.createTunnelArch();
    this.createCheckpointArch();
    this.createTurnSign();
    this.createRockCliff();
    this.createBillboards();

    // 2. Traffic cars
    this.createTrafficSedan();
    this.createTrafficSports();
    this.createTrafficVan();

    // 3. Player car states
    this.createPlayerCarSprites();

    // 4. Parallax backgrounds
    this.createBackgroundLayers();
  }

  createCanvas(w, h) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    return { canvas, ctx, w, h };
  }

  createPineTree() {
    const { canvas, ctx, w, h } = this.createCanvas(48, 80);
    // Trunk
    ctx.fillStyle = '#3a2312';
    ctx.fillRect(20, 60, 8, 20);
    ctx.fillStyle = '#24160a';
    ctx.fillRect(25, 60, 3, 20);

    // Foliage tiers
    const drawTier = (y, rW, rH, dark, light) => {
      ctx.fillStyle = dark;
      ctx.beginPath();
      ctx.moveTo(24, y - rH);
      ctx.lineTo(24 - rW, y);
      ctx.lineTo(24 + rW, y);
      ctx.closePath();
      ctx.fill();

      // Highlight on left side
      ctx.fillStyle = light;
      ctx.beginPath();
      ctx.moveTo(24, y - rH);
      ctx.lineTo(24 - rW, y);
      ctx.lineTo(24, y);
      ctx.closePath();
      ctx.fill();
    };

    drawTier(62, 22, 26, '#17361b', '#26542c');
    drawTier(44, 18, 24, '#1e4423', '#316938');
    drawTier(26, 14, 20, '#26542c', '#3e8448');
    drawTier(12, 8, 14, '#316938', '#4fa85c');

    this.cache.set('pine_tree', canvas);
  }

  createPalmTree() {
    const { canvas, ctx, w, h } = this.createCanvas(56, 88);
    // Trunk
    ctx.fillStyle = '#59442b';
    ctx.beginPath();
    ctx.moveTo(26, 88);
    ctx.quadraticCurveTo(24, 50, 30, 24);
    ctx.lineTo(34, 24);
    ctx.quadraticCurveTo(30, 50, 32, 88);
    ctx.closePath();
    ctx.fill();

    // Trunk ridges
    ctx.fillStyle = '#362716';
    for (let y = 30; y < 85; y += 6) {
      ctx.fillRect(26, y, 6, 2);
    }

    // Fronds
    const drawFrond = (ox, oy, cpx, cpy, ex, ey) => {
      ctx.strokeStyle = '#2d6a36';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.quadraticCurveTo(cpx, cpy, ex, ey);
      ctx.stroke();

      ctx.strokeStyle = '#439950';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.quadraticCurveTo(cpx, cpy, ex, ey);
      ctx.stroke();
    };

    drawFrond(32, 24, 10, 10, 2, 28);
    drawFrond(32, 24, 48, 8, 54, 26);
    drawFrond(32, 24, 20, 2, 12, 14);
    drawFrond(32, 24, 44, 2, 48, 14);
    drawFrond(32, 24, 30, -4, 32, 8);

    this.cache.set('palm_tree', canvas);
  }

  createGuardrails() {
    // Left guardrail
    const { canvas: cL, ctx: ctxL } = this.createCanvas(24, 36);
    ctxL.fillStyle = '#505663';
    ctxL.fillRect(10, 14, 5, 22);
    ctxL.fillStyle = '#8e96a4';
    ctxL.fillRect(0, 8, 24, 10);
    ctxL.fillStyle = '#c5cdd9';
    ctxL.fillRect(0, 10, 24, 3);
    this.cache.set('guardrail_l', cL);

    // Right guardrail
    const { canvas: cR, ctx: ctxR } = this.createCanvas(24, 36);
    ctxR.fillStyle = '#505663';
    ctxR.fillRect(9, 14, 5, 22);
    ctxR.fillStyle = '#8e96a4';
    ctxR.fillRect(0, 8, 24, 10);
    ctxR.fillStyle = '#c5cdd9';
    ctxR.fillRect(0, 10, 24, 3);
    this.cache.set('guardrail_r', cR);
  }

  createStreetLamp() {
    const { canvas, ctx } = this.createCanvas(36, 90);
    // Pole
    ctx.fillStyle = '#3a3f4d';
    ctx.fillRect(8, 20, 4, 70);
    // Arm
    ctx.fillRect(8, 16, 20, 4);
    // Fixture
    ctx.fillStyle = '#656e82';
    ctx.fillRect(22, 14, 12, 6);
    // Light bulb glow
    ctx.fillStyle = '#00f0ff';
    ctx.fillRect(24, 20, 8, 3);
    this.cache.set('street_lamp', canvas);
  }

  createTunnelArch() {
    const { canvas, ctx, w, h } = this.createCanvas(240, 90);
    // Concrete arch
    ctx.fillStyle = '#1e212b';
    ctx.fillRect(0, 0, w, 24);
    ctx.fillRect(0, 24, 34, 66);
    ctx.fillRect(w - 34, 24, 34, 66);

    // Warning hazard stripes on lintel
    for (let x = 34; x < w - 34; x += 18) {
      ctx.fillStyle = '#ffb800';
      ctx.beginPath();
      ctx.moveTo(x, 14);
      ctx.lineTo(x + 10, 14);
      ctx.lineTo(x + 4, 24);
      ctx.lineTo(x - 6, 24);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = '#111';
      ctx.beginPath();
      ctx.moveTo(x + 10, 14);
      ctx.lineTo(x + 18, 14);
      ctx.lineTo(x + 12, 24);
      ctx.lineTo(x + 4, 24);
      ctx.closePath();
      ctx.fill();
    }

    // Tunnel sodium lights
    ctx.fillStyle = '#ff9900';
    ctx.fillRect(50, 22, 14, 3);
    ctx.fillRect(115, 22, 14, 3);
    ctx.fillRect(180, 22, 14, 3);

    this.cache.set('tunnel_arch', canvas);
  }

  createCheckpointArch() {
    const { canvas, ctx, w, h } = this.createCanvas(220, 80);
    // Truss posts
    ctx.fillStyle = '#3a4052';
    ctx.fillRect(8, 18, 16, 62);
    ctx.fillRect(w - 24, 18, 16, 62);

    // Overhead truss
    ctx.fillStyle = '#252936';
    ctx.fillRect(0, 0, w, 26);
    ctx.fillStyle = '#ff3344';
    ctx.fillRect(4, 3, w - 8, 20);

    // Checkpoint banner text
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('★ CHECKPOINT ★', w / 2, 17);

    this.cache.set('checkpoint_arch', canvas);
  }

  createTurnSign() {
    const { canvas, ctx } = this.createCanvas(32, 48);
    // Pole
    ctx.fillStyle = '#555';
    ctx.fillRect(14, 26, 4, 22);
    // Yellow diamond
    ctx.fillStyle = '#ffcc00';
    ctx.beginPath();
    ctx.moveTo(16, 2);
    ctx.lineTo(30, 16);
    ctx.lineTo(16, 30);
    ctx.lineTo(2, 16);
    ctx.closePath();
    ctx.fill();
    // Arrow
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(22, 16);
    ctx.lineTo(10, 16);
    ctx.lineTo(15, 10);
    ctx.stroke();
    this.cache.set('sign_turn', canvas);
  }

  createRockCliff() {
    const { canvas, ctx } = this.createCanvas(48, 48);
    ctx.fillStyle = '#3b3e47';
    ctx.beginPath();
    ctx.moveTo(8, 48);
    ctx.lineTo(2, 28);
    ctx.lineTo(14, 12);
    ctx.lineTo(32, 8);
    ctx.lineTo(46, 26);
    ctx.lineTo(42, 48);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#575b68';
    ctx.beginPath();
    ctx.moveTo(14, 12);
    ctx.lineTo(32, 8);
    ctx.lineTo(24, 28);
    ctx.lineTo(2, 28);
    ctx.closePath();
    ctx.fill();

    this.cache.set('rock_cliff', canvas);
    this.cache.set('ocean_rock', canvas);
  }

  createBillboards() {
    // 1. Ridge Racing Billboard
    const { canvas: c1, ctx: ctx1, w: w1, h: h1 } = this.createCanvas(80, 50);
    ctx1.fillStyle = '#444';
    ctx1.fillRect(16, 38, 4, 12);
    ctx1.fillRect(60, 38, 4, 12);
    ctx1.fillStyle = '#151722';
    ctx1.fillRect(0, 0, w1, 38);
    ctx1.fillStyle = '#ea0029';
    ctx1.fillRect(3, 3, w1 - 6, 32);
    ctx1.fillStyle = '#fff';
    ctx1.font = 'bold 9px monospace';
    ctx1.textAlign = 'center';
    ctx1.fillText('RIDGE RUNNER', w1 / 2, 18);
    ctx1.fillStyle = '#fffa38';
    ctx1.font = '8px monospace';
    ctx1.fillText('★ PS1 ARCADE ★', w1 / 2, 29);
    this.cache.set('billboard_ridge', c1);
    this.cache.set('beach_sign', c1);

    // 2. Cyber neon billboard
    const { canvas: c2, ctx: ctx2, w: w2, h: h2 } = this.createCanvas(80, 50);
    ctx2.fillStyle = '#333';
    ctx2.fillRect(16, 38, 4, 12);
    ctx2.fillRect(60, 38, 4, 12);
    ctx2.fillStyle = '#0b0e18';
    ctx2.fillRect(0, 0, w2, 38);
    ctx2.strokeStyle = '#00f0ff';
    ctx2.strokeRect(2, 2, w2 - 4, 34);
    ctx2.fillStyle = '#00f0ff';
    ctx2.font = 'bold 9px monospace';
    ctx2.textAlign = 'center';
    ctx2.fillText('NEO TOKYO', w2 / 2, 17);
    ctx2.fillStyle = '#ff0077';
    ctx2.fillText('MIDNIGHT 98', w2 / 2, 28);
    this.cache.set('cyber_billboard', c2);
    this.cache.set('neon_sign_1', c2);
    this.cache.set('neon_sign_2', c1);
  }

  createTrafficSedan() {
    const { canvas, ctx } = this.createCanvas(44, 30);
    // Body
    ctx.fillStyle = '#3a6ea5';
    ctx.fillRect(4, 12, 36, 14);
    // Cabin
    ctx.fillStyle = '#223d60';
    ctx.fillRect(8, 4, 28, 10);
    // Rear window
    ctx.fillStyle = '#101c2e';
    ctx.fillRect(11, 6, 22, 7);
    // Tires
    ctx.fillStyle = '#111';
    ctx.fillRect(2, 20, 6, 8);
    ctx.fillRect(36, 20, 6, 8);
    // Taillights
    ctx.fillStyle = '#e63946';
    ctx.fillRect(5, 15, 6, 4);
    ctx.fillRect(33, 15, 6, 4);
    // License plate
    ctx.fillStyle = '#fffae0';
    ctx.fillRect(18, 19, 8, 4);

    this.cache.set('car_sedan', canvas);
  }

  createTrafficSports() {
    const { canvas, ctx } = this.createCanvas(46, 28);
    // Sleek low body
    ctx.fillStyle = '#d90429';
    ctx.fillRect(4, 11, 38, 13);
    // Cabin
    ctx.fillStyle = '#7a0014';
    ctx.fillRect(10, 5, 26, 8);
    // Rear window
    ctx.fillStyle = '#1a0508';
    ctx.fillRect(13, 6, 20, 6);
    // Spoiler
    ctx.fillStyle = '#111';
    ctx.fillRect(4, 8, 38, 3);
    ctx.fillRect(8, 10, 3, 3);
    ctx.fillRect(35, 10, 3, 3);
    // Wheels
    ctx.fillRect(2, 18, 6, 8);
    ctx.fillRect(38, 18, 6, 8);
    // Taillight bar
    ctx.fillStyle = '#ff4d6d';
    ctx.fillRect(6, 14, 34, 3);
    // Twin exhaust
    ctx.fillStyle = '#bbb';
    ctx.fillRect(12, 23, 4, 2);
    ctx.fillRect(30, 23, 4, 2);

    this.cache.set('car_sports', canvas);
  }

  createTrafficVan() {
    const { canvas, ctx } = this.createCanvas(48, 38);
    // Tall van body
    ctx.fillStyle = '#e5e5e5';
    ctx.fillRect(4, 6, 40, 26);
    // Top roof shade
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(6, 4, 36, 4);
    // Rear windows
    ctx.fillStyle = '#222';
    ctx.fillRect(8, 10, 14, 10);
    ctx.fillRect(26, 10, 14, 10);
    // Wheels
    ctx.fillStyle = '#111';
    ctx.fillRect(2, 26, 6, 10);
    ctx.fillRect(40, 26, 6, 10);
    // Vertical tail lamps
    ctx.fillStyle = '#c1121f';
    ctx.fillRect(4, 14, 3, 12);
    ctx.fillRect(41, 14, 3, 12);

    this.cache.set('car_van', canvas);
  }

  createPlayerCarSprites() {
    // 1. Straight (Neutral)
    this.cache.set('player_straight', this.renderPlayerCarFrame(0, false, false));
    this.cache.set('player_straight_brake', this.renderPlayerCarFrame(0, true, false));

    // 2. Turning Left / Right (gentle tilt)
    this.cache.set('player_left', this.renderPlayerCarFrame(-0.5, false, false));
    this.cache.set('player_left_brake', this.renderPlayerCarFrame(-0.5, true, false));
    this.cache.set('player_right', this.renderPlayerCarFrame(0.5, false, false));
    this.cache.set('player_right_brake', this.renderPlayerCarFrame(0.5, true, false));

    // 3. Drifting Hard Left / Right
    this.cache.set('player_drift_left', this.renderPlayerCarFrame(-1.0, false, true));
    this.cache.set('player_drift_right', this.renderPlayerCarFrame(1.0, false, true));
    this.cache.set('player_drift_left_brake', this.renderPlayerCarFrame(-1.0, true, true));
    this.cache.set('player_drift_right_brake', this.renderPlayerCarFrame(1.0, true, true));
  }

  renderPlayerCarFrame(tilt, isBraking, isDrifting) {
    const { canvas, ctx, w, h } = this.createCanvas(58, 38);
    const midX = w / 2;
    const skew = tilt * 4;

    // Tires (Chunky low-poly PS1 style)
    ctx.fillStyle = '#0f1013';
    ctx.fillRect(3 + skew * 0.3, 22, 9, 14);
    ctx.fillRect(46 + skew * 0.3, 22, 9, 14);

    // Main Chassis Body (Iconic 90s Sports Coupe White/Panda AE86 & Kamata Fiera vibe)
    ctx.fillStyle = '#f2f4f8';
    ctx.beginPath();
    ctx.moveTo(8 + skew, 16);
    ctx.lineTo(50 + skew, 16);
    ctx.lineTo(54 + skew * 0.5, 30);
    ctx.lineTo(4 + skew * 0.5, 30);
    ctx.closePath();
    ctx.fill();

    // Bottom Bumper / Diffuser
    ctx.fillStyle = '#181920';
    ctx.fillRect(4 + skew * 0.5, 27, 50, 6);

    // Rear Roof / Cabin greenhouse
    ctx.fillStyle = '#e2e5ec';
    ctx.beginPath();
    ctx.moveTo(14 + skew * 1.4, 7);
    ctx.lineTo(44 + skew * 1.4, 7);
    ctx.lineTo(48 + skew, 16);
    ctx.lineTo(10 + skew, 16);
    ctx.closePath();
    ctx.fill();

    // Dark Tinted Rear Window
    ctx.fillStyle = '#131922';
    ctx.beginPath();
    ctx.moveTo(16 + skew * 1.4, 8);
    ctx.lineTo(42 + skew * 1.4, 8);
    ctx.lineTo(46 + skew, 15);
    ctx.lineTo(12 + skew, 15);
    ctx.closePath();
    ctx.fill();

    // Rear Wing / Spoiler
    ctx.fillStyle = '#1c1e24';
    ctx.fillRect(6 + skew * 1.2, 12, 46, 3);
    ctx.fillRect(10 + skew * 1.1, 14, 3, 3);
    ctx.fillRect(45 + skew * 1.1, 3, 3, 3);

    // Iconic Japanese 90s Quad Taillights
    if (isBraking) {
      // Vivid bright neon brake glow
      ctx.fillStyle = '#ff1133';
      ctx.shadowColor = '#ff0033';
      ctx.shadowBlur = 8;
      ctx.fillRect(8 + skew, 18, 12, 6);
      ctx.fillRect(38 + skew, 18, 12, 6);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(12 + skew, 20, 4, 2);
      ctx.fillRect(42 + skew, 20, 4, 2);
      ctx.shadowBlur = 0;
    } else {
      // Normal running taillights
      ctx.fillStyle = '#b3001b';
      ctx.fillRect(8 + skew, 18, 12, 5);
      ctx.fillRect(38 + skew, 18, 12, 5);
      ctx.fillStyle = '#ff7b00'; // Amber turn section
      ctx.fillRect(8 + skew, 18, 3, 5);
      ctx.fillRect(47 + skew, 18, 3, 5);
    }

    // Japanese License Plate
    ctx.fillStyle = '#e9ecef';
    ctx.fillRect(23 + skew * 0.7, 21, 12, 6);
    ctx.fillStyle = '#225522';
    ctx.fillRect(26 + skew * 0.7, 23, 6, 2);

    // Chrome Dual Exhaust Pipes with backfire sparks
    ctx.fillStyle = '#adb5bd';
    ctx.fillRect(12 + skew * 0.5, 31, 5, 3);
    ctx.fillRect(19 + skew * 0.5, 31, 5, 3);
    ctx.fillStyle = '#111';
    ctx.fillRect(13 + skew * 0.5, 32, 3, 2);
    ctx.fillRect(20 + skew * 0.5, 32, 3, 2);

    if (isDrifting) {
      // Exhaust pop flame
      ctx.fillStyle = '#00f0ff';
      ctx.fillRect(13 + skew * 0.5, 34, 3, 4);
      ctx.fillStyle = '#ffaa00';
      ctx.fillRect(20 + skew * 0.5, 34, 3, 5);
    }

    return canvas;
  }

  createBackgroundLayers() {
    // 1. Far Mountains Silhouette (320x100)
    const { canvas: cM, ctx: ctxM, w: wM, h: hM } = this.createCanvas(320, 100);
    ctxM.fillStyle = '#1a1b28';
    ctxM.beginPath();
    ctxM.moveTo(0, hM);
    let my = 60;
    for (let x = 0; x <= wM; x += 16) {
      my += (Math.sin(x * 0.04) * 12) + (Math.sin(x * 0.08) * 8);
      ctxM.lineTo(x, Math.max(20, Math.min(85, my)));
    }
    ctxM.lineTo(wM, hM);
    ctxM.closePath();
    ctxM.fill();
    this.cache.set('bg_mountains', cM);

    // 2. City Skyline (320x100)
    const { canvas: cC, ctx: ctxC, w: wC, h: hC } = this.createCanvas(320, 100);
    ctxC.fillStyle = '#0e111d';
    for (let x = 0; x < wC; x += 12 + Math.floor(Math.sin(x) * 4)) {
      const bh = 30 + (Math.abs(Math.sin(x * 3.7)) * 55);
      ctxC.fillRect(x, hC - bh, 14, bh);
      // Windows
      ctxC.fillStyle = (x % 3 === 0) ? '#ffcc00' : '#00f0ff';
      for (let wy = hC - bh + 6; wy < hC - 8; wy += 8) {
        if (Math.sin(x + wy) > 0.1) {
          ctxC.fillRect(x + 3, wy, 2, 3);
          ctxC.fillRect(x + 8, wy, 2, 3);
        }
      }
      ctxC.fillStyle = '#0e111d';
    }
    this.cache.set('bg_city', cC);
  }
}
