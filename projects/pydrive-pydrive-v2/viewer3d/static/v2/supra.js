// Hero car loader. Loads the curated low-poly A80 glb (CC-BY G1itchz, recolored
// — see assets-src/CREDITS.md) and wires it into the viewer's car frame:
// auto-fit to ~4.5 m with the nose at +X, wheels wrapped in steer/roll pivots,
// plus the shared night-glow effects (taillight halo, headlight road pool,
// contact shadow). `?car=voxel` falls back to the original boxy build, and so
// does any load failure — the viewer never ends up with no car.
//
// Returns { group, wheels, blob } synchronously (same contract app.js expects);
// the glb populates `group` and `wheels` when it finishes loading.
import * as THREE from "../vendor/three.module.min.js";
import { GLTFLoader } from "../vendor/GLTFLoader.js";
import { createVoxelSupra, radialTexture } from "./supra_voxel.js";

const GLB_URL = "/3d/static/assets/supra_a80.glb";
const TARGET_LEN = 4.5;          // metres, nose-to-tail (matches the voxel car)

export function createSupra(mood) {
  const params = new URLSearchParams(location.search);
  if (params.get("car") === "voxel") return createVoxelSupra(mood);

  const group = new THREE.Group();
  const wheels = [];
  const radial = radialTexture();

  // ---- shared night-glow effects (car-local; independent of the body mesh) --
  const tailGlowGeo = new THREE.BufferGeometry();
  tailGlowGeo.setAttribute("position", new THREE.Float32BufferAttribute(
    [-2.30, 0.78, 0.55, -2.30, 0.78, -0.55], 3));
  const tailGlowMat = new THREE.PointsMaterial({
    map: radial, color: 0xff2630, size: 1.4, sizeAttenuation: true,
    transparent: true, opacity: 0.8, depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  mood.bindNumber(tailGlowMat, "opacity", "tailGlowOpacity", null);
  const tailGlow = new THREE.Points(tailGlowGeo, tailGlowMat);
  tailGlow.frustumCulled = false;
  group.add(tailGlow);

  const poolGeo = new THREE.PlaneGeometry(8.5, 3.6);
  poolGeo.rotateX(-Math.PI / 2);
  const poolMat = new THREE.MeshBasicMaterial({
    map: radial, color: 0xffd28a, transparent: true, opacity: 0.3,
    depthWrite: false, blending: THREE.AdditiveBlending,
  });
  mood.bindNumber(poolMat, "opacity", "headPoolOpacity", null);
  const pool = new THREE.Mesh(poolGeo, poolMat);
  pool.position.set(6.3, 0.06, 0);
  pool.renderOrder = 7;
  group.add(pool);

  // contact shadow — app.js detaches this to the world and scales it with air
  const blobGeo = new THREE.PlaneGeometry(5.2, 2.4);
  blobGeo.rotateX(-Math.PI / 2);
  const blobMat = new THREE.MeshBasicMaterial({
    map: radial, color: 0x000000, transparent: true, opacity: 0.4,
    depthWrite: false,
  });
  mood.bindNumber(blobMat, "opacity", "contactShadow", null);
  const blob = new THREE.Mesh(blobGeo, blobMat);
  blob.position.set(0, 0.035, 0);
  blob.renderOrder = 6;
  group.add(blob);

  // ---- async glb load + fit ------------------------------------------------
  new GLTFLoader().load(GLB_URL, (gltf) => {
    fitAndWire(gltf.scene, group, wheels, mood);
  }, undefined, (err) => {
    console.warn("supra glb load failed, using voxel fallback:", err);
    const v = createVoxelSupra(mood);
    group.add(v.group);
    for (const w of v.wheels) wheels.push(w);
  });

  return { group, wheels, blob };
}

function fitAndWire(scene, group, wheels, mood) {
  // The glb arrives (Blender Y-up convert) nose +Z, up +Y, left +X. Rotate it
  // into the viewer's car frame: nose +X, up +Y, left -Z (= sim +Y left).
  const holder = new THREE.Group();
  holder.add(scene);
  holder.rotation.y = Math.PI / 2;            // +Z nose -> +X
  holder.updateWorldMatrix(true, true);

  // fit: scale to TARGET_LEN along X, centre the wheelbase on X/Z, sit on ground
  let box = new THREE.Box3().setFromObject(holder);
  const size = new THREE.Vector3(); box.getSize(size);
  const s = TARGET_LEN / size.x;
  holder.scale.setScalar(s);
  holder.updateWorldMatrix(true, true);
  box = new THREE.Box3().setFromObject(holder);
  const ctr = new THREE.Vector3(); box.getCenter(ctr);
  holder.position.x -= ctr.x;
  holder.position.z -= ctr.z;
  holder.position.y -= box.min.y;             // wheels on the ground (y=0)
  holder.updateWorldMatrix(true, true);

  // pull the four wheels out into steer/roll pivots
  const names = ["wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR"];
  for (const name of names) {
    const wheel = holder.getObjectByName(name);
    if (!wheel) continue;
    const hub = new THREE.Vector3();
    wheel.getWorldPosition(hub);
    group.worldToLocal(hub);
    const steer = new THREE.Group();
    steer.position.copy(hub);
    steer.userData.steer = name === "wheel_FL" || name === "wheel_FR";
    const roll = new THREE.Group();
    steer.add(roll);
    steer.userData.roll = roll;
    roll.attach(wheel);                       // keeps world transform, re-pivots
    group.add(steer);
    wheels.push(steer);
  }

  // material polish: glossy paint, glowing taillights + headlights, shadows
  holder.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      if (!m) continue;
      if (m.name === "paint") {
        m.metalness = 0.5; m.roughness = 0.32;
      } else if (/Material\.013|tail/i.test(m.name)) {
        m.emissive = new THREE.Color(0xff1822);
        mood.bindNumber(m, "emissiveIntensity", "tailEmissive", null);
      } else if (/Material\.012|Material\.015|head/i.test(m.name)) {
        m.emissive = new THREE.Color(0xfff2d8);     // headlight lens glow
        mood.bindNumber(m, "emissiveIntensity", "headEmissive", null);
      }
    }
  });
  group.add(holder);
  addHeadlights(group, mood);
}

// Real headlights: two SpotLights from the nose lighting the road ahead at
// night (children of the car group, so they follow pitch/roll over crests).
function addHeadlights(group, mood) {
  // decay kept low (0.6) so the beam actually carries down the road instead of
  // dying in the first couple of metres; cones aimed onto the asphalt ~7 m out.
  for (const side of [1, -1]) {
    const lamp = new THREE.SpotLight(0xfff0d6, 0.0, 40, 0.5, 0.55, 0.6);
    lamp.position.set(2.05, 0.6, side * 0.5);
    const target = new THREE.Object3D();
    target.position.set(9, -1.4, side * 0.85);
    group.add(target);
    lamp.target = target;
    group.add(lamp);
    mood.bindNumber(lamp, "intensity", "headlightPower", null);
  }
}
