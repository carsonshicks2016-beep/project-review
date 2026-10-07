// 2D Faux-3D Rasterizer & PS1 Retro Post-Processing Pipeline

import { ROAD_CONSTANTS } from './road.js';

export class PS1Renderer {
  constructor(canvas, road, playerCar, trafficSystem, spriteManager) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.road = road;
    this.player = playerCar;
    this.traffic = trafficSystem;
    this.sprites = spriteManager;

    // Native PS1 retro resolution
    this.width = 320;
    this.height = 240;
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    // Offscreen render buffer
    this.bufferCanvas = document.createElement('canvas');
    this.bufferCanvas.width = this.width;
    this.bufferCanvas.height = this.height;
    this.bCtx = this.bufferCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    this.bCtx.imageSmoothingEnabled = false;

    // Visual options
    this.enableDither = true;
    this.enableJitter = false;
    this.enableFog = true;

    // Sky parallax offset
    this.skyOffset = 0;

    // 4x4 Bayer Dithering Matrix (0 to 15)
    this.bayer4x4 = [
      0,  8,  2, 10,
      12, 4, 14,  6,
      3, 11,  1,  9,
      15, 7, 13,  5
    ];
  }

  toggleDither() {
    this.enableDither = !this.enableDither;
    return this.enableDither;
  }

  toggleJitter() {
    this.enableJitter = !this.enableJitter;
    return this.enableJitter;
  }

  project(p, cameraX, cameraY, cameraZ, cameraDepth) {
    p.camera.x = (p.world.x || 0) - cameraX;
    p.camera.y = (p.world.y || 0) - cameraY;
    p.camera.z = (p.world.z || 0) - cameraZ;

    p.screen.scale = cameraDepth / p.camera.z;
    p.screen.x = Math.round((this.width / 2) + (p.screen.scale * p.camera.x * this.width / 2));
    p.screen.y = Math.round((this.height / 2) - (p.screen.scale * p.camera.y * this.height / 2));
    p.screen.w = Math.round(p.screen.scale * ROAD_CONSTANTS.ROAD_WIDTH * this.width / 2);

    if (this.enableJitter) {
      // PS1 sub-pixel vertex wobble
      p.screen.x = Math.floor(p.screen.x);
      p.screen.y = Math.floor(p.screen.y);
    }
  }

  render(dt) {
    const ctx = this.bCtx;
    const road = this.road;
    const player = this.player;

    const baseSegment = road.findSegment(player.z);
    const basePercent = (player.z % ROAD_CONSTANTS.SEGMENT_LENGTH) / ROAD_CONSTANTS.SEGMENT_LENGTH;
    const totalSegments = road.segments.length;

    // Camera height based on mode
    let camHeight = ROAD_CONSTANTS.CAMERA_HEIGHT;
    let camZOffset = 0;
    if (player.cameraMode === 'cockpit') {
      camHeight = 580;
      camZOffset = 180;
    } else if (player.cameraMode === 'bumper') {
      camHeight = 280;
      camZOffset = 300;
    }

    const playerWorldY = baseSegment.p1.world.y + (baseSegment.p2.world.y - baseSegment.p1.world.y) * basePercent;
    const cameraX = player.x * ROAD_CONSTANTS.ROAD_WIDTH;
    const cameraY = camHeight + playerWorldY;
    const cameraZ = player.z - (player.cameraMode === 'chase' ? 0 : camZOffset);

    // 1. Render Sky & Parallax Background
    this.renderBackground(ctx, baseSegment.curve, player.speed);

    // 2. Project and Render Road Segments (Painter's algorithm front-to-back with crest clipping)
    let maxY = this.height;
    let xAccum = 0;
    let dx = -(baseSegment.curve * basePercent);

    const visibleSegments = [];

    for (let n = 0; n < ROAD_CONSTANTS.DRAW_DISTANCE; n++) {
      const segIndex = (baseSegment.index + n) % totalSegments;
      const segment = road.segments[segIndex];
      const looped = segIndex < baseSegment.index;

      // World coordinates
      const z1 = segment.p1.world.z + (looped ? road.totalLength : 0);
      const z2 = segment.p2.world.z + (looped ? road.totalLength : 0);

      segment.p1.camera.z = z1 - cameraZ;
      segment.p2.camera.z = z2 - cameraZ;

      // Behind camera clip
      if (segment.p1.camera.z <= ROAD_CONSTANTS.CAMERA_DEPTH) continue;

      // Project curve displacement
      segment.p1.world.x = xAccum;
      xAccum += dx;
      dx += segment.curve;
      segment.p2.world.x = xAccum;

      this.project(segment.p1, cameraX, cameraY, cameraZ, ROAD_CONSTANTS.CAMERA_DEPTH);
      this.project(segment.p2, cameraX, cameraY, cameraZ, ROAD_CONSTANTS.CAMERA_DEPTH);

      // Occlusion check for blind hill crests
      if (segment.p1.camera.z <= ROAD_CONSTANTS.CAMERA_DEPTH || segment.p2.screen.y >= maxY) {
        continue;
      }

      // Compute distance fog factor
      const zRatio = (segment.p1.camera.z / (ROAD_CONSTANTS.DRAW_DISTANCE * ROAD_CONSTANTS.SEGMENT_LENGTH));
      segment.fog = Math.min(1, Math.max(0, Math.pow(zRatio, 1.4)));

      this.renderSegment(ctx, segment);
      maxY = segment.p2.screen.y;

      visibleSegments.push(segment);
    }

    // 3. Render Sprites and Traffic Back-to-Front
    for (let n = visibleSegments.length - 1; n >= 0; n--) {
      const segment = visibleSegments[n];

      // Draw roadside scenery
      for (const spriteItem of segment.sprites) {
        this.renderSprite(ctx, spriteItem, segment);
      }

      // Draw AI traffic cars
      for (const car of this.traffic.cars) {
        const carSeg = road.findSegment(car.z);
        if (carSeg.index === segment.index) {
          this.renderTrafficCar(ctx, car, segment);
        }
      }
    }

    // 4. Render Tire Smoke Particles
    this.renderTireSmoke(ctx, cameraX, cameraY, cameraZ);

    // 5. Render Player Vehicle (if Chase Cam)
    if (player.cameraMode === 'chase') {
      this.renderPlayerVehicle(ctx);
    } else if (player.cameraMode === 'cockpit') {
      this.renderCockpit(ctx);
    }

    // 6. Post-Processing (Bayer Dithering)
    if (this.enableDither) {
      this.applyBayerDither(ctx);
    }

    // Copy to screen canvas
    this.ctx.drawImage(this.bufferCanvas, 0, 0);
  }

  renderBackground(ctx, curve, speed) {
    const sky = this.road.theme.sky;
    const grad = ctx.createLinearGradient(0, 0, 0, this.height * 0.55);
    grad.addColorStop(0, sky.top);
    grad.addColorStop(0.65, sky.mid);
    grad.addColorStop(1, sky.bot);

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this.width, this.height * 0.55);

    // Mountain / City Parallax
    this.skyOffset += (curve * 0.4) + (this.player.steer * 0.6);
    const bgSprite = this.road.theme.id === 'yokohama' 
      ? this.sprites.get('bg_city') 
      : this.sprites.get('bg_mountains');

    if (bgSprite) {
      const bgW = bgSprite.width;
      const bgH = bgSprite.height;
      const shift = Math.floor(this.skyOffset) % bgW;
      const y = Math.floor(this.height * 0.55 - bgH + 15);

      ctx.drawImage(bgSprite, -shift, y);
      ctx.drawImage(bgSprite, bgW - shift, y);
      if (shift > 0) {
        ctx.drawImage(bgSprite, -bgW - shift, y);
      }
    }
  }

  renderSegment(ctx, segment) {
    const p1 = segment.p1.screen;
    const p2 = segment.p2.screen;
    const col = segment.colors;
    const fog = this.enableFog ? segment.fog : 0;
    const fogCol = this.road.theme.fog;

    // Grass / Verge
    ctx.fillStyle = this.blendFog(col.grass, fogCol, fog);
    ctx.fillRect(0, p2.y, this.width, p1.y - p2.y);

    // Rumble Curb
    const rW1 = p1.w * 1.18;
    const rW2 = p2.w * 1.18;
    ctx.fillStyle = this.blendFog(col.rumble, fogCol, fog);
    this.drawTrapezoid(ctx, p1.x - rW1, p1.y, p1.x + rW1, p1.y, p2.x + rW2, p2.y, p2.x - rW2, p2.y);

    // Road Surface
    ctx.fillStyle = this.blendFog(col.road, fogCol, fog);
    this.drawTrapezoid(ctx, p1.x - p1.w, p1.y, p1.x + p1.w, p1.y, p2.x + p2.w, p2.y, p2.x - p2.w, p2.y);

    // Lane Dividers
    if (col.lane) {
      const lW1 = p1.w * 0.035;
      const lW2 = p2.w * 0.035;
      ctx.fillStyle = this.blendFog(col.lane, fogCol, fog);
      // Center stripe
      this.drawTrapezoid(ctx, p1.x - lW1, p1.y, p1.x + lW1, p1.y, p2.x + lW2, p2.y, p2.x - lW2, p2.y);
      // Left lane stripe
      const off1 = p1.w * 0.45;
      const off2 = p2.w * 0.45;
      this.drawTrapezoid(ctx, p1.x - off1 - lW1, p1.y, p1.x - off1 + lW1, p1.y, p2.x - off2 + lW2, p2.y, p2.x - off2 - lW2, p2.y);
      // Right lane stripe
      this.drawTrapezoid(ctx, p1.x + off1 - lW1, p1.y, p1.x + off1 + lW1, p1.y, p2.x + off2 + lW2, p2.y, p2.x + off2 - lW2, p2.y);
    }
  }

  drawTrapezoid(ctx, x1, y1, x2, y2, x3, y3, x4, y4) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x3, y3);
    ctx.lineTo(x4, y4);
    ctx.closePath();
    ctx.fill();
  }

  renderSprite(ctx, spriteItem, segment) {
    const img = this.sprites.get(spriteItem.type);
    if (!img) return;

    const isArch = spriteItem.type === 'tunnel_arch' || spriteItem.type === 'checkpoint_arch';
    let destW, destH, destX;
    if (isArch) {
      destW = segment.p1.screen.w * 2.2;
      destH = (img.height / img.width) * destW;
      destX = segment.p1.screen.x;
    } else {
      const isTree = spriteItem.type.includes('tree');
      const isBillboard = spriteItem.type.includes('billboard') || spriteItem.type.includes('sign');
      const factor = isTree ? 0.7 : (isBillboard ? 0.85 : 0.5);
      destW = segment.p1.screen.w * factor;
      destH = (img.height / img.width) * destW;
      destX = segment.p1.screen.x + (segment.p1.screen.w * spriteItem.offset);
    }

    if (destW < 3 || destH < 3) return;

    const renderX = destX - (destW / 2);
    const renderY = segment.p1.screen.y - destH;

    // Check screen bounds
    if (renderX + destW < 0 || renderX > this.width) return;

    ctx.drawImage(img, Math.floor(renderX), Math.floor(renderY), Math.floor(destW), Math.floor(destH));
  }

  renderTrafficCar(ctx, car, segment) {
    const img = this.sprites.get(car.spriteType);
    if (!img) return;

    const destW = segment.p1.screen.w * 0.42;
    const destH = (img.height / img.width) * destW;
    const destX = segment.p1.screen.x + (segment.p1.screen.w * car.x);

    if (destW < 4 || destH < 4) return;

    const renderX = destX - (destW / 2);
    const renderY = segment.p1.screen.y - destH;

    if (renderX + destW < 0 || renderX > this.width) return;

    ctx.drawImage(img, Math.floor(renderX), Math.floor(renderY), Math.floor(destW), Math.floor(destH));
  }

  renderTireSmoke(ctx, cameraX, cameraY, cameraZ) {
    for (const p of this.player.smokeParticles) {
      const relZ = p.z - cameraZ;
      if (relZ <= ROAD_CONSTANTS.CAMERA_DEPTH) continue;

      const scale = ROAD_CONSTANTS.CAMERA_DEPTH / relZ;
      const sx = (this.width / 2) + (scale * (p.x * ROAD_CONSTANTS.ROAD_WIDTH - cameraX) * this.width / 2);
      const sy = (this.height / 2) - (scale * (p.y - cameraY) * this.height / 2);
      const sSize = p.size * scale * (this.width / 2) * 0.08;

      if (sSize > 1) {
        ctx.fillStyle = `rgba(230, 235, 245, ${p.alpha * 0.7})`;
        ctx.fillRect(Math.floor(sx - sSize / 2), Math.floor(sy - sSize / 2), Math.floor(sSize), Math.floor(sSize));
      }
    }
  }

  renderPlayerVehicle(ctx) {
    const p = this.player;
    const isBraking = p.speed > 0 && (p.steer !== 0 ? p.isDrifting : false);

    let spriteKey = 'player_straight';
    if (p.isDrifting) {
      spriteKey = p.steer < 0 ? 'player_drift_left' : 'player_drift_right';
    } else if (p.steer < -0.2) {
      spriteKey = 'player_left';
    } else if (p.steer > 0.2) {
      spriteKey = 'player_right';
    }

    const img = this.sprites.get(spriteKey);
    if (!img) return;

    const carW = 68;
    const carH = 44;
    const carX = Math.floor((this.width / 2) - (carW / 2) + (p.steer * 8));
    const bounce = Math.sin(Date.now() * 0.02) * (p.speed / p.maxSpeed) * 1.5;
    const carY = Math.floor(this.height - carH - 10 + bounce);

    ctx.drawImage(img, carX, carY, carW, carH);
  }

  renderCockpit(ctx) {
    // Hood silhouette
    ctx.fillStyle = '#111318';
    ctx.beginPath();
    ctx.moveTo(30, this.height);
    ctx.lineTo(90, this.height - 35);
    ctx.lineTo(this.width - 90, this.height - 35);
    ctx.lineTo(this.width - 30, this.height);
    ctx.closePath();
    ctx.fill();

    // Hood scoop
    ctx.fillStyle = '#0a0a0d';
    ctx.fillRect(this.width / 2 - 20, this.height - 30, 40, 6);
  }

  applyBayerDither(ctx) {
    // 4x4 Bayer Dithering for authentic PS1 15-bit color banding
    const imgData = ctx.getImageData(0, 0, this.width, this.height);
    const data = imgData.data;
    const bayer = this.bayer4x4;

    for (let y = 0; y < this.height; y++) {
      const bayerY = (y & 3) * 4;
      for (let x = 0; x < this.width; x++) {
        const idx = (y * this.width + x) * 4;
        const threshold = (bayer[bayerY + (x & 3)] - 7.5) * 2.2;

        // Quantize RGB to 5 bits (32 levels per channel, PS1 style)
        data[idx] = Math.max(0, Math.min(255, Math.floor((data[idx] + threshold) / 8) * 8));
        data[idx + 1] = Math.max(0, Math.min(255, Math.floor((data[idx + 1] + threshold) / 8) * 8));
        data[idx + 2] = Math.max(0, Math.min(255, Math.floor((data[idx + 2] + threshold) / 8) * 8));
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  blendFog(hexCol, hexFog, factor) {
    if (factor <= 0 || !hexFog) return hexCol;
    if (factor >= 1) return hexFog;

    const c1 = this.hexToRgb(hexCol);
    const c2 = this.hexToRgb(hexFog);
    const r = Math.round(c1.r + (c2.r - c1.r) * factor);
    const g = Math.round(c1.g + (c2.g - c1.g) * factor);
    const b = Math.round(c1.b + (c2.b - c1.b) * factor);
    return `rgb(${r},${g},${b})`;
  }

  hexToRgb(hex) {
    const clean = hex.replace('#', '');
    const bigint = parseInt(clean, 16);
    return {
      r: (bigint >> 16) & 255,
      g: (bigint >> 8) & 255,
      b: bigint & 255
    };
  }
}
