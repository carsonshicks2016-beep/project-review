// The voxel Mk4 Supra. Deliberately chunkier than the world: boxes with
// per-face shading variation (tops darker maroon, sides bright red), a
// basket-handle wing, glowing rectangle taillights, white-cap wheels, decals
// from small canvas textures. The whole body merges into ONE mesh; lights,
// beams, decals and wheels are the only extra objects.
// Car local space: +X forward, +Y up, length ~4.5 m.
import * as THREE from "../vendor/three.module.min.js";
import { hash01 } from "./noise.js";

const TAG_NONE = null; // car materials live for the whole session

// decal config — flip off / edit freely
const DECALS = {
  banner: "SUPRA",
  doorNumber: "7",
  tailText: "SUPRA",
};

/* ── Voxel body builder ───────────────────────────────────────── */

class VoxBuffer {
  constructor() {
    this.pos = [];
    this.col = [];
    this._c = new THREE.Color();
    this._n = 0;
  }

  box(cx, cy, cz, sx, sy, sz, hex, opts = {}) {
    const hx = sx / 2;
    const hy = sy / 2;
    const hz = sz / 2;
    // per-face brightness: voxel charm = visible panel shading
    const faces = [
      { n: [0, 1, 0], b: opts.top ?? 0.80 },    // top (darker deck)
      { n: [0, -1, 0], b: 0.55 },               // bottom
      { n: [1, 0, 0], b: 0.96 },                // nose
      { n: [-1, 0, 0], b: 0.90 },               // tail
      { n: [0, 0, 1], b: 1.0 },                 // sides (brightest)
      { n: [0, 0, -1], b: 1.0 },
    ];
    const corners = (nx, ny, nz) => {
      // returns 4 corners of the face with normal (nx,ny,nz), CCW outward
      const u = nx !== 0 ? [0, 0, 1] : [1, 0, 0];
      const v = ny !== 0 ? [0, 0, 1] : [0, 1, 0];
      const c = [cx + nx * hx, cy + ny * hy, cz + nz * hz];
      const U = [u[0] * (nx !== 0 ? hz : hx), 0, u[2] * (nx !== 0 ? hz : hx)];
      const V = [v[0] * hx * 0, v[1] * (ny !== 0 ? hz : hy), v[2] * (ny !== 0 ? hz : 0)];
      // simpler: explicit per axis
      let a;
      let b;
      if (nx !== 0) {
        a = [0, hy, 0];
        b = [0, 0, hz];
      } else if (ny !== 0) {
        a = [hx, 0, 0];
        b = [0, 0, hz];
      } else {
        a = [hx, 0, 0];
        b = [0, hy, 0];
      }
      return [
        [c[0] - a[0] - b[0], c[1] - a[1] - b[1], c[2] - a[2] - b[2]],
        [c[0] + a[0] - b[0], c[1] + a[1] - b[1], c[2] + a[2] - b[2]],
        [c[0] + a[0] + b[0], c[1] + a[1] + b[1], c[2] + a[2] + b[2]],
        [c[0] - a[0] + b[0], c[1] - a[1] + b[1], c[2] - a[2] + b[2]],
      ];
    };
    for (const f of faces) {
      const [p0, p1, p2, p3] = corners(f.n[0], f.n[1], f.n[2]);
      const jitter = 0.97 + hash01(this._n * 7.31 + f.n[0] * 3 + f.n[1] * 5 + f.n[2] * 7) * 0.06;
      this._c.set(hex).multiplyScalar(f.b * jitter * (opts.scale ?? 1));
      // two tris, outward winding fixed by normal sign
      const quad = f.n[0] + f.n[1] + f.n[2] > 0
        ? [p0, p1, p2, p0, p2, p3]
        : [p0, p2, p1, p0, p3, p2];
      for (const p of quad) this.pos.push(p[0], p[1], p[2]);
      for (let k = 0; k < 6; k++) this.col.push(this._c.r, this._c.g, this._c.b);
      this._n++;
    }
  }

  build(material) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    geo.computeVertexNormals();
    return new THREE.Mesh(geo, material);
  }
}

/* ── Textures ─────────────────────────────────────────────────── */

export function radialTexture() {
  const s = 128;
  const canvas = document.createElement("canvas");
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 4, 64, 64, 62);
  g.addColorStop(0, "rgba(255,255,255,0.95)");
  g.addColorStop(0.45, "rgba(255,255,255,0.40)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function textTexture(text, w, h, { bg = null, fg = "#f2f3f5", font = "bold 30px Menlo, monospace" } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (bg) {
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
  }
  ctx.fillStyle = fg;
  ctx.font = font;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2 + 1);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function roundelTexture(num) {
  const s = 128;
  const canvas = document.createElement("canvas");
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext("2d");
  ctx.beginPath();
  ctx.arc(64, 64, 56, 0, Math.PI * 2);
  ctx.fillStyle = "#e9ebec";
  ctx.fill();
  ctx.lineWidth = 6;
  ctx.strokeStyle = "#15171a";
  ctx.stroke();
  ctx.fillStyle = "#15171a";
  ctx.font = "bold 64px Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(num), 64, 68);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function decalPlane(tex, w, h) {
  return new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({
      map: tex,
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
    }),
  );
}

/* ── The car ──────────────────────────────────────────────────── */

const RED = 0xb50d1f;
const RED_DARK = 0x6e0812;
const GLASS = 0x07090d;
const TRIM = 0x0a0b0c;

export function createVoxelSupra(mood) {
  const group = new THREE.Group();
  group.name = "supra-v2-voxel";

  /* body: one merged voxel mesh */
  const buf = new VoxBuffer();
  // hull + nose
  buf.box(0, 0.55, 0, 4.50, 0.50, 1.82, RED);
  buf.box(2.18, 0.46, 0, 0.26, 0.30, 1.68, RED);
  buf.box(2.30, 0.32, 0, 0.12, 0.12, 1.78, TRIM);          // splitter
  // hood + cowl
  buf.box(1.25, 0.86, 0, 1.95, 0.14, 1.66, RED, { top: 0.74 });
  buf.box(0.28, 0.94, 0, 0.34, 0.24, 1.60, RED, { top: 0.78 });
  // canopy (two stepped slabs, all black glass like the pics)
  buf.box(-0.42, 1.12, 0, 1.95, 0.32, 1.58, GLASS, { top: 0.9 });
  buf.box(-0.58, 1.38, 0, 1.28, 0.22, 1.38, GLASS, { top: 0.85 });
  // rear deck + ducktail + bumper
  buf.box(-1.70, 0.90, 0, 1.15, 0.22, 1.66, RED, { top: 0.76 });
  buf.box(-2.16, 1.00, 0, 0.26, 0.16, 1.70, RED, { top: 0.80 });
  buf.box(-2.21, 0.58, 0, 0.20, 0.42, 1.80, RED);
  buf.box(-2.31, 0.76, 0, 0.06, 0.24, 1.52, TRIM);          // tail bar
  // haunches + fenders + skirts
  buf.box(-1.35, 0.68, 0.92, 1.45, 0.36, 0.14, RED, { scale: 1.05 });
  buf.box(-1.35, 0.68, -0.92, 1.45, 0.36, 0.14, RED, { scale: 1.05 });
  buf.box(1.45, 0.72, 0.89, 0.95, 0.22, 0.10, RED);
  buf.box(1.45, 0.72, -0.89, 0.95, 0.22, 0.10, RED);
  buf.box(0.05, 0.36, 0.88, 2.60, 0.16, 0.10, RED_DARK);
  buf.box(0.05, 0.36, -0.88, 2.60, 0.16, 0.10, RED_DARK);
  // mirrors
  buf.box(0.46, 1.24, 1.00, 0.16, 0.10, 0.16, TRIM);
  buf.box(0.46, 1.24, -1.00, 0.16, 0.10, 0.16, TRIM);
  // exhaust
  buf.box(-2.34, 0.38, -0.48, 0.14, 0.16, 0.20, 0x42464a);
  // basket-handle wing
  buf.box(-1.98, 1.08, 0.82, 0.30, 0.44, 0.08, TRIM);
  buf.box(-1.98, 1.08, -0.82, 0.30, 0.44, 0.08, TRIM);
  buf.box(-2.06, 1.32, 0, 0.36, 0.08, 1.86, TRIM, { top: 0.95 });

  // slight emissive keeps the toy-red readable on shaded faces and at night
  const bodyMat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    color: 0xffffff,
    emissive: 0x200408,
  });
  mood.bindColor(bodyMat, "color", "carTint", TAG_NONE);
  const body = buf.build(bodyMat);
  body.castShadow = true;
  group.add(body);

  /* taillights + headlights (unlit, mood-bound) */
  const tailMat = new THREE.MeshBasicMaterial({ color: 0xff2230 });
  mood.bindColor(tailMat, "color", "tailLight", TAG_NONE);
  const headMat = new THREE.MeshBasicMaterial({ color: 0xffe9b8 });
  mood.bindColor(headMat, "color", "headLight", TAG_NONE);
  const lampGeo = new THREE.BoxGeometry(0.05, 0.17, 0.46);
  for (const z of [0.52, -0.52]) {
    const tail = new THREE.Mesh(lampGeo, tailMat);
    tail.position.set(-2.345, 0.76, z);
    group.add(tail);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.14, 0.34), headMat);
    head.position.set(2.33, 0.64, z + (z > 0 ? 0.06 : -0.06));
    group.add(head);
  }

  const radial = radialTexture();

  /* taillight halos: one Points draw call */
  const tailGlowGeo = new THREE.BufferGeometry();
  tailGlowGeo.setAttribute("position", new THREE.Float32BufferAttribute(
    [-2.40, 0.76, 0.52, -2.40, 0.76, -0.52], 3));
  const tailGlowMat = new THREE.PointsMaterial({
    map: radial,
    color: 0xff2630,
    size: 1.5,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.8,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  mood.bindNumber(tailGlowMat, "opacity", "tailGlowOpacity", TAG_NONE);
  const tailGlow = new THREE.Points(tailGlowGeo, tailGlowMat);
  tailGlow.frustumCulled = false;
  group.add(tailGlow);

  /* headlight pool on the road ahead (no beam cones: end-on they stack into
     an ugly faceted dome, and the inspiration shows lit road, not shafts) */
  const poolGeo = new THREE.PlaneGeometry(8.5, 3.6);
  poolGeo.rotateX(-Math.PI / 2);
  const poolMat = new THREE.MeshBasicMaterial({
    map: radial,
    color: 0xffd28a,
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  mood.bindNumber(poolMat, "opacity", "headPoolOpacity", TAG_NONE);
  const pool = new THREE.Mesh(poolGeo, poolMat);
  pool.position.set(6.3, 0.06, 0);
  pool.renderOrder = 7;
  group.add(pool);

  /* contact shadow blob */
  const blobGeo = new THREE.PlaneGeometry(5.2, 2.4);
  blobGeo.rotateX(-Math.PI / 2);
  const blobMat = new THREE.MeshBasicMaterial({
    map: radial,
    color: 0x000000,
    transparent: true,
    opacity: 0.4,
    depthWrite: false,
  });
  mood.bindNumber(blobMat, "opacity", "contactShadow", TAG_NONE);
  const blob = new THREE.Mesh(blobGeo, blobMat);
  blob.position.set(0, 0.035, 0);
  blob.renderOrder = 6;
  group.add(blob);

  /* decals */
  if (DECALS.banner) {
    const banner = decalPlane(
      textTexture(DECALS.banner, 256, 40, { bg: "#101214", font: "bold 30px Menlo, monospace" }),
      1.42, 0.20);
    banner.position.set(0.565, 1.20, 0);
    banner.rotation.y = Math.PI / 2;
    group.add(banner);
  }
  if (DECALS.doorNumber) {
    const tex = roundelTexture(DECALS.doorNumber);
    for (const z of [0.945, -0.945]) {
      const roundel = decalPlane(tex, 0.46, 0.46);
      roundel.position.set(0.15, 0.72, z);
      roundel.rotation.y = z > 0 ? 0 : Math.PI;
      group.add(roundel);
    }
  }
  if (DECALS.tailText) {
    const badge = decalPlane(
      textTexture(DECALS.tailText, 128, 28, { font: "bold 20px Menlo, monospace" }),
      0.52, 0.115);
    badge.position.set(-2.346, 0.99, 0);
    badge.rotation.y = -Math.PI / 2;
    group.add(badge);
  }

  /* wheels: dark tires with white voxel caps */
  const tireMat = new THREE.MeshLambertMaterial({ color: 0x0b0d0f });
  const capMat = new THREE.MeshLambertMaterial({ color: 0xc9ccd0 });
  mood.bindColor(capMat, "color", "wheelCap", TAG_NONE);
  const tireGeo = new THREE.CylinderGeometry(0.37, 0.37, 0.30, 14);
  tireGeo.rotateX(Math.PI / 2);

  const wheels = [
    [1.45, 0.37, 0.93, true],
    [1.45, 0.37, -0.93, true],
    [-1.40, 0.37, 0.93, false],
    [-1.40, 0.37, -0.93, false],
  ].map(([x, y, z, steer]) => {
    const pivot = new THREE.Group();
    pivot.position.set(x, y, z);
    pivot.userData.steer = steer;
    const roll = new THREE.Group();
    const tire = new THREE.Mesh(tireGeo, tireMat);
    tire.castShadow = true;
    roll.add(tire);
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.30, 0.04), capMat);
    cap.position.z = z > 0 ? 0.17 : -0.17;
    roll.add(cap);
    pivot.userData.roll = roll;
    pivot.add(roll);
    group.add(pivot);
    return pivot;
  });

  return { group, wheels, blob };
}
