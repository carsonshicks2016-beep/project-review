// Valley mist: soft alpha-gradient banks seeded into the valley pockets the
// corridor's side signal carves out. All banks merge into ONE transparent
// mesh (crossed vertical quads + flat floor sheets) so the whole system costs
// a single draw call and zero per-frame work. Color/opacity are mood-bound.
import * as THREE from "../vendor/three.module.min.js";
import { hash01 } from "./noise.js";

const TAG = "world";

function mistTexture() {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 6, 64, 64, 62);
  g.addColorStop(0, "rgba(255,255,255,0.85)");
  g.addColorStop(0.55, "rgba(255,255,255,0.42)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

class QuadBuffer {
  constructor() {
    this.pos = [];
    this.uv = [];
    this.index = [];
    this._n = 0;
  }

  _emit(corners) {
    for (const c of corners) this.pos.push(c.x, c.y, c.z);
    this.uv.push(0, 0, 1, 0, 0, 1, 1, 1);
    const b = this._n * 4;
    this.index.push(b, b + 1, b + 2, b + 1, b + 3, b + 2);
    this._n++;
  }

  // vertical plane centered at pos, rotated to `yaw`
  wall(pos, yaw, w, h) {
    const rx = Math.cos(yaw) * w * 0.5;
    const rz = -Math.sin(yaw) * w * 0.5;
    this._emit([
      new THREE.Vector3(pos.x - rx, pos.y - h * 0.5, pos.z - rz),
      new THREE.Vector3(pos.x + rx, pos.y - h * 0.5, pos.z + rz),
      new THREE.Vector3(pos.x - rx, pos.y + h * 0.5, pos.z - rz),
      new THREE.Vector3(pos.x + rx, pos.y + h * 0.5, pos.z + rz),
    ]);
  }

  // horizontal sheet centered at pos
  floor(pos, w, d) {
    this._emit([
      new THREE.Vector3(pos.x - w * 0.5, pos.y, pos.z - d * 0.5),
      new THREE.Vector3(pos.x + w * 0.5, pos.y, pos.z - d * 0.5),
      new THREE.Vector3(pos.x - w * 0.5, pos.y, pos.z + d * 0.5),
      new THREE.Vector3(pos.x + w * 0.5, pos.y, pos.z + d * 0.5),
    ]);
  }

  build(material) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(this.uv, 2));
    geo.setIndex(this.index);
    return new THREE.Mesh(geo, material);
  }
}

export function buildAtmosphere(corridor, mood) {
  const group = new THREE.Group();
  group.name = "atmosphere-v2";
  const seed = corridor.seed;

  const mat = new THREE.MeshBasicMaterial({
    map: mistTexture(),
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    color: 0x5a6b7d,
    opacity: 0.3,
  });
  mood.bindColor(mat, "color", "mist", TAG);
  mood.bindNumber(mat, "opacity", "mistOpacity", TAG);

  const buf = new QuadBuffer();
  const out = new THREE.Vector3();
  const n = corridor.center.length;
  let banks = 0;

  for (let i = 4; i < n - 4 && banks < 60; i += 11) {
    for (const sideSign of [1, -1]) {
      const cliff = corridor.cliff[i];
      const valleyness = sideSign > 0 ? Math.max(0, -cliff) : Math.max(0, cliff);
      if (valleyness < 0.5) continue;
      const h1 = hash01(i * 7.71 + seed + sideSign * 17);
      if (h1 < 0.35) continue;

      const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
      const ctr = corridor.center[i];
      out.set(edge.x - ctr.x, 0, edge.z - ctr.z).normalize();

      const a = corridor.center[Math.max(0, i - 1)];
      const b = corridor.center[Math.min(n - 1, i + 1)];
      const roadYaw = Math.atan2(-(b.z - a.z), b.x - a.x);

      const off = 16 + hash01(i * 3.3 + seed) * 46;
      const pos = edge.clone().addScaledVector(out, off);
      pos.y = edge.y - 3.0 - hash01(i * 5.9 + seed) * 7.0;

      const w = 34 + hash01(i * 9.1 + seed) * 50;
      const hgt = 9 + hash01(i * 11.7 + seed) * 12;
      const yaw = roadYaw + (h1 - 0.5) * 1.2;

      buf.wall(pos, yaw, w, hgt);
      const pos2 = pos.clone().addScaledVector(out, 8 + h1 * 10);
      pos2.y -= 1.5;
      buf.wall(pos2, yaw + 0.9, w * 0.8, hgt * 0.85);

      if (banks % 3 === 0) {
        const fpos = edge.clone().addScaledVector(out, off + 18);
        fpos.y = edge.y - 6.5 - hash01(i * 13.3 + seed) * 6.0;
        buf.floor(fpos, w * 1.6, w * 1.1);
      }
      banks++;
    }
  }

  const mesh = buf.build(mat);
  mesh.renderOrder = 6;
  group.add(mesh);
  return group;
}
