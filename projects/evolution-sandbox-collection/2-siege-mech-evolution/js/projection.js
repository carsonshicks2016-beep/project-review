/**
 * projection.js — 3D-to-2D Projection Utility Library
 * Siege Mech Evolution
 *
 * Provides a Camera class with yaw/pitch orbit, perspective projection,
 * smooth follow interpolation, plus depth-sorting and color helpers.
 */

/* ------------------------------------------------------------------ */
/*  Camera                                                            */
/* ------------------------------------------------------------------ */

/**
 * Orbital camera with perspective projection.
 * Supports smooth follow-target interpolation and yaw/pitch orbit.
 */
class Camera {
  constructor() {
    /** Yaw (rotation around Y) in radians */
    this.yaw = -0.2;

    /** Pitch (rotation around X) in radians */
    this.pitch = -0.2;

    /** Perspective focal length */
    this.focalLength = 600;

    /** Orbit target coordinates */
    this.targetX = 0;
    this.targetY = 15;
    this.targetZ = 0;

    /** Distance from target */
    this.distance = 280;

    /** Speed of smooth target interpolation */
    this.followSpeed = 0.06;

    /** Calculated camera world position */
    this.x = 0;
    this.y = 0;
    this.z = 0;
    this.updatePosition();
  }

  /* -------------------------------- helpers ----------------------- */

  /**
   * Recalculate camera position (x, y, z) based on target, yaw/pitch, and distance.
   */
  updatePosition() {
    const cosP = Math.cos(this.pitch);
    const sinP = Math.sin(this.pitch);
    const cosY = Math.cos(this.yaw);
    const sinY = Math.sin(this.yaw);

    this.x = this.targetX + sinY * cosP * this.distance;
    this.y = this.targetY - sinP * this.distance;
    this.z = this.targetZ - cosY * cosP * this.distance;
  }

  /**
   * Smoothly interpolate the orbit target coordinates toward (tx, ty, tz).
   */
  updateFollow(tx, ty, tz) {
    const s = this.followSpeed;
    this.targetX += (tx - this.targetX) * s;
    this.targetY += (ty - this.targetY) * s;
    this.targetZ += (tz - this.targetZ) * s;
    this.updatePosition();
  }

  /* -------------------------------- project ----------------------- */

  /**
   * Project a single 3-D world point to 2-D screen coordinates.
   *
   * Pipeline:
   *  1. Translate by –camera position
   *  2. Rotate around Y (yaw)
   *  3. Rotate around X (pitch)
   *  4. Perspective divide
   *
   * @param {number} wx  World X
   * @param {number} wy  World Y
   * @param {number} wz  World Z
   * @param {number} cw  Canvas width  (pixels)
   * @param {number} ch  Canvas height (pixels)
   * @returns {{ x: number, y: number, z: number, visible: boolean }}
   */
  project(wx, wy, wz, cw, ch) {
    // 1. Camera-space translation
    let dx = wx - this.x;
    let dy = wy - this.y;
    let dz = wz - this.z;

    // 2. Yaw rotation (around Y)
    const cosY = Math.cos(this.yaw);
    const sinY = Math.sin(this.yaw);
    const rx = dx * cosY + dz * sinY;
    const ry = dy;
    const rz = -dx * sinY + dz * cosY;

    // 3. Pitch rotation (around X)
    const cosP = Math.cos(this.pitch);
    const sinP = Math.sin(this.pitch);
    const lx = rx;
    const ly = ry * cosP - rz * sinP;
    const lz = ry * sinP + rz * cosP;

    // 4. Perspective divide
    const visible = lz > 10;
    const iz = visible ? this.focalLength / lz : 0;
    const screenX = lx * iz + cw * 0.5;
    const screenY = -ly * iz + ch * 0.5; // flip Y so up is up

    return { x: screenX, y: screenY, z: lz, visible };
  }

  /**
   * Project a line segment (two endpoints) and return both projected
   * points plus an average depth for depth-sort ordering.
   *
   * @returns {{ a: object, b: object, depth: number }}
   */
  projectLine(x1, y1, z1, x2, y2, z2, cw, ch) {
    const a = this.project(x1, y1, z1, cw, ch);
    const b = this.project(x2, y2, z2, cw, ch);
    return { a, b, depth: (a.z + b.z) * 0.5 };
  }
}

/* ------------------------------------------------------------------ */
/*  Depth-sorted rendering                                            */
/* ------------------------------------------------------------------ */

/**
 * Sort an array of draw-call descriptors by depth (farthest first)
 * and execute each draw function in order.
 *
 * @param {{ depth: number, drawFn: Function }[]} drawCalls
 */
function depthSort(drawCalls) {
  drawCalls.sort((a, b) => b.depth - a.depth);
  for (let i = 0; i < drawCalls.length; i++) {
    drawCalls[i].drawFn();
  }
}

/* ------------------------------------------------------------------ */
/*  Color helpers                                                     */
/* ------------------------------------------------------------------ */

/**
 * Parse a hex colour string (#RGB or #RRGGBB) into [r, g, b].
 * @param {string} hex
 * @returns {number[]}
 */
function _hexToRgb(hex) {
  let h = hex.replace('#', '');
  if (h.length === 3) {
    h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  }
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16)
  ];
}

/**
 * Convert [r, g, b] (0–255) back to a hex string.
 * @param {number[]} rgb
 * @returns {string}
 */
function _rgbToHex(rgb) {
  const r = Math.max(0, Math.min(255, Math.round(rgb[0])));
  const g = Math.max(0, Math.min(255, Math.round(rgb[1])));
  const b = Math.max(0, Math.min(255, Math.round(rgb[2])));
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

/**
 * Linearly interpolate between two hex colors.
 *
 * @param {string} color1  Start colour (hex)
 * @param {string} color2  End colour   (hex)
 * @param {number} t       Interpolation factor 0→1
 * @returns {string}       Interpolated hex colour
 */
function lerpColor(color1, color2, t) {
  const c1 = _hexToRgb(color1);
  const c2 = _hexToRgb(color2);
  return _rgbToHex([
    c1[0] + (c2[0] - c1[0]) * t,
    c1[1] + (c2[1] - c1[1]) * t,
    c1[2] + (c2[2] - c1[2]) * t
  ]);
}
