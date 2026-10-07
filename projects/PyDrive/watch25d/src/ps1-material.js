import * as THREE from "three";

/**
 * Hard-ish PS1 look: flat / vertex colors, nearest-neighbor textures,
 * strong affine-ish vertex snap + UV wobble. No PBR, no bloom.
 */

const SNAP_VERT = /* glsl */ `
varying vec3 vColor;
varying float vFogDepth;
uniform float uSnap;
uniform vec2 uRes;
uniform float uTime;
uniform float uWobble;

void main() {
  vColor = color;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vFogDepth = -mvPosition.z;
  vec4 clip = projectionMatrix * mvPosition;
  if (uSnap > 0.5) {
    // Coarser snap grid ≈ early PS1 / V-Rally raster.
    vec2 grid = max(uRes, vec2(1.0)) * 0.5;
    vec3 ndc = clip.xyz / max(clip.w, 1e-6);
    ndc.xy = floor(ndc.xy * grid + 0.5) / grid;
    clip = vec4(ndc * clip.w, clip.w);
  }
  gl_Position = clip;
}
`;

const SNAP_FRAG = /* glsl */ `
varying vec3 vColor;
varying float vFogDepth;
uniform vec3 fogColor;
uniform float fogNear;
uniform float fogFar;
uniform float fogDensity;
uniform bool fogExp2;
uniform vec3 uTint;

void main() {
  vec3 col = vColor * uTint;
  float fogFactor;
  if (fogExp2) {
    float f = fogDensity * vFogDepth;
    fogFactor = 1.0 - exp(-f * f);
  } else {
    fogFactor = smoothstep(fogNear, fogFar, vFogDepth);
  }
  col = mix(col, fogColor, clamp(fogFactor, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}
`;

/** Textured curb: snap + affine-ish UV wobble (presentation only). */
const TEX_SNAP_VERT = /* glsl */ `
varying vec2 vUv;
varying float vFogDepth;
uniform float uSnap;
uniform vec2 uRes;
uniform float uTime;
uniform float uWobble;

void main() {
  vUv = uv;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vFogDepth = -mvPosition.z;
  // Affine-ish UV crawl — cheap PS1 texture swim, not perspective-correct.
  if (uWobble > 0.001) {
    float w = uWobble * (0.004 + 0.002 * sin(uTime * 2.1 + position.x * 0.07));
    vUv += vec2(
      sin(mvPosition.z * 0.11 + uTime * 1.3) * w,
      cos(mvPosition.x * 0.09 - uTime * 0.9) * w * 0.6
    );
  }
  vec4 clip = projectionMatrix * mvPosition;
  if (uSnap > 0.5) {
    vec2 grid = max(uRes, vec2(1.0)) * 0.5;
    vec3 ndc = clip.xyz / max(clip.w, 1e-6);
    ndc.xy = floor(ndc.xy * grid + 0.5) / grid;
    clip = vec4(ndc * clip.w, clip.w);
  }
  gl_Position = clip;
}
`;

const TEX_SNAP_FRAG = /* glsl */ `
varying vec2 vUv;
varying float vFogDepth;
uniform sampler2D map;
uniform vec3 fogColor;
uniform float fogNear;
uniform float fogFar;
uniform float fogDensity;
uniform bool fogExp2;
uniform vec3 uTint;

void main() {
  vec3 col = texture2D(map, vUv).rgb * uTint;
  float fogFactor;
  if (fogExp2) {
    float f = fogDensity * vFogDepth;
    fogFactor = 1.0 - exp(-f * f);
  } else {
    fogFactor = smoothstep(fogNear, fogFar, vFogDepth);
  }
  col = mix(col, fogColor, clamp(fogFactor, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}
`;

function baseFogUniforms() {
  return {
    fogColor: { value: new THREE.Color(0x7a8a82) },
    fogNear: { value: 40 },
    fogFar: { value: 180 },
    fogDensity: { value: 0.012 },
    fogExp2: { value: true },
    uTint: { value: new THREE.Color(1, 1, 1) },
  };
}

/** Flat vertex-color material with strong PS1 vertex snap + fog. */
export function makePs1VertexMaterial({ snap = true } = {}) {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uSnap: { value: snap ? 1 : 0 },
      uRes: { value: new THREE.Vector2(240, 180) },
      uTime: { value: 0 },
      uWobble: { value: 0 },
      ...baseFogUniforms(),
    },
    vertexShader: SNAP_VERT,
    fragmentShader: SNAP_FRAG,
    vertexColors: true,
    side: THREE.DoubleSide,
  });
  mat.userData.ps1 = true;
  return mat;
}

/** Nearest-neighbor map + snap + UV wobble (curbs). */
export function makePs1TexturedMaterial(map, { snap = true, wobble = 1 } = {}) {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      map: { value: map },
      uSnap: { value: snap ? 1 : 0 },
      uRes: { value: new THREE.Vector2(240, 180) },
      uTime: { value: 0 },
      uWobble: { value: wobble },
      ...baseFogUniforms(),
    },
    vertexShader: TEX_SNAP_VERT,
    fragmentShader: TEX_SNAP_FRAG,
    side: THREE.DoubleSide,
  });
  mat.userData.ps1 = true;
  return mat;
}

/** Solid flat color (no lighting) — body panels, Armco posts. */
export function makePs1Flat(color, { fog = true } = {}) {
  return new THREE.MeshBasicMaterial({
    color,
    fog,
  });
}

/** Tiny nearest-neighbor stripe texture for curbs (red/white). */
export function makeCurbStripeTexture() {
  const data = new Uint8Array([
    220, 40, 40, 255,
    236, 236, 230, 255,
  ]);
  const tex = new THREE.DataTexture(data, 2, 1);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

export function syncPs1Fog(materials, fog) {
  if (!fog) return;
  const isExp2 = fog.isFogExp2 === true;
  for (const mat of materials) {
    if (!mat?.uniforms?.fogColor) continue;
    mat.uniforms.fogColor.value.copy(fog.color);
    mat.uniforms.fogExp2.value = isExp2;
    if (isExp2) {
      mat.uniforms.fogDensity.value = fog.density;
    } else {
      mat.uniforms.fogNear.value = fog.near;
      mat.uniforms.fogFar.value = fog.far;
    }
  }
}

/** Coarser snap grid = crunchier PS1 raster (was ~0.45× viewport). */
export function syncPs1Resolution(materials, width, height) {
  const w = Math.max(120, Math.floor(width * 0.28));
  const h = Math.max(90, Math.floor(height * 0.28));
  for (const mat of materials) {
    if (mat?.uniforms?.uRes) mat.uniforms.uRes.value.set(w, h);
  }
}

export function syncPs1Time(materials, timeSec) {
  for (const mat of materials) {
    if (mat?.uniforms?.uTime) mat.uniforms.uTime.value = timeSec;
  }
}

export function syncPs1Tint(materials, tint) {
  const c = tint instanceof THREE.Color ? tint : new THREE.Color(tint);
  for (const mat of materials) {
    if (mat?.uniforms?.uTint) mat.uniforms.uTint.value.copy(c);
  }
}
