import * as THREE from "./vendor/three.module.min.js";

/* ── Day / Night palette system ─────────────────────────────── */

export const PALETTE_NIGHT = {
  sky: 0x0a0e1a,
  fog: 0x141d2a,
  moon: 0xb8c0bd,
  lamp: 0xffb257,
  lampPool: 0xd99754,
  road: 0x343536,
  shoulder: 0x4a4337,
  terrain: 0x202d22,
  rock: 0x4f504d,
  pine: 0x17251b,
  pineDark: 0x09110d,
  guardrail: 0x9aa098,
  sign: 0xd7e2ce,
  body: 0xc81024,
  bodyDark: 0x4b0610,
  glass: 0x20314c,
  tire: 0x111416,
  rim: 0xbab5a2,
  amber: 0xffcf66,
  tail: 0xff2633,
};

export const PALETTE_DAY = {
  sky: 0x6ba3d6,
  fog: 0x9dbad6,
  moon: 0xfff4d6,
  lamp: 0xffb257,
  lampPool: 0xd99754,
  road: 0x505254,
  shoulder: 0x6a5f4a,
  terrain: 0x3a5a3a,
  rock: 0x6f706d,
  pine: 0x1f3525,
  pineDark: 0x0f1d14,
  guardrail: 0xc0c8c0,
  sign: 0xd7e2ce,
  body: 0xc81024,
  bodyDark: 0x4b0610,
  glass: 0x405878,
  tire: 0x111416,
  rim: 0xbab5a2,
  amber: 0xffcf66,
  tail: 0xff2633,
};

// Mutable active palette – night by default
export const PALETTE = { ...PALETTE_NIGHT };

export const MOUNTAIN_PASS = {
  renderScale: 0.56,
  colorSteps: 13.0,
  ditherStrength: 0.13,
  scanlineStrength: 0.022,
  noiseStrength: 0.010,
  vignetteStrength: 0.080,
  chromaticStrength: 0.20,
  fogDensity: 0.0032,
  exposure: 1.26,
  shadowLift: 0.095,
  midtoneBoost: 0.17,
  roadSurfaceLift: 0.18,
  carSurfaceOffset: -0.01,
  cameraFov: 66,
};

/* ── Pixel textures ────────────────────────────────────────── */

function makePixelTexture(size, draw) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  draw(ctx, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  if ("colorSpace" in tex) tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createMountainPassTextures() {
  const road = makePixelTexture(128, (ctx, s) => {
    ctx.fillStyle = "#41413d";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "#585850";
    for (let y = 0; y < s; y += 28) ctx.fillRect(0, y, s, 2);
    ctx.fillStyle = "#292a27";
    ctx.fillRect(13, 24, 48, 14);
    ctx.fillRect(78, 68, 37, 20);
    ctx.fillStyle = "#77786f";
    ctx.fillRect(4, 55, 88, 2);
    ctx.fillRect(34, 109, 70, 2);
    ctx.fillStyle = "rgba(188,194,182,0.18)";
    for (let y = 8; y < s; y += 34) ctx.fillRect(0, y, s, 1);
    ctx.fillStyle = "rgba(11,13,13,0.46)";
    for (let i = 0; i < 32; i++) ctx.fillRect((i * 31) % s, (i * 19) % s, 14 + (i % 3) * 12, 2);
    for (let i = 0; i < 170; i++) {
      const v = 58 + ((i * 37) % 42);
      ctx.fillStyle = `rgb(${v + 2},${v + 2},${v})`;
      ctx.fillRect((i * 13) % s, (i * 29) % s, 1 + (i % 3), 1);
    }
  });
  road.repeat.set(1.0, 11);

  const shoulder = makePixelTexture(128, (ctx, s) => {
    ctx.fillStyle = "#504838";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "#665b46";
    for (let y = 0; y < s; y += 18) ctx.fillRect(0, y, s, 2);
    ctx.fillStyle = "#2b3029";
    ctx.fillRect(7, 12, 45, 14);
    ctx.fillRect(70, 62, 48, 18);
    ctx.fillStyle = "#7c7059";
    for (let i = 0; i < 52; i++) ctx.fillRect((i * 17) % s, (i * 43) % s, 14 + (i % 4) * 6, 3);
    for (let i = 0; i < 210; i++) {
      const g = 54 + ((i * 19) % 34);
      ctx.fillStyle = `rgb(${g + 17},${g + 13},${g + 5})`;
      ctx.fillRect((i * 7) % s, (i * 23) % s, 2 + (i % 3), 2);
    }
  });
  shoulder.repeat.set(1.0, 10);

  const terrain = makePixelTexture(128, (ctx, s) => {
    ctx.fillStyle = "#243326";
    ctx.fillRect(0, 0, s, s);
    for (let y = 0; y < s; y += 16) {
      for (let x = 0; x < s; x += 16) {
        const k = ((x * 3 + y * 5) % 42);
        ctx.fillStyle = `rgb(${31 + k / 7},${43 + k / 2},${32 + k / 5})`;
        ctx.fillRect(x, y, 16, 16);
      }
    }
    ctx.fillStyle = "rgba(6,13,10,0.42)";
    for (let x = 0; x < s; x += 16) ctx.fillRect(x, 0, 1, s);
    for (let y = 0; y < s; y += 16) ctx.fillRect(0, y, s, 1);
    ctx.fillStyle = "rgba(12,22,16,0.48)";
    ctx.fillRect(18, 20, 42, 28);
    ctx.fillRect(84, 72, 34, 42);
  });
  terrain.repeat.set(8, 8);

  const rock = makePixelTexture(128, (ctx, s) => {
    ctx.fillStyle = "#4d4f4d";
    ctx.fillRect(0, 0, s, s);
    for (let y = 0; y < s; y += 18) {
      ctx.fillStyle = y % 36 === 0 ? "#646660" : "#343837";
      ctx.fillRect(0, y, s, 3);
    }
    ctx.fillStyle = "#282c2b";
    for (let i = 0; i < 26; i++) ctx.fillRect((i * 23) % s, (i * 37) % s, 22 + (i % 4) * 9, 5);
    ctx.fillStyle = "rgba(205,205,185,0.10)";
    ctx.fillRect(0, 9, s, 1);
    ctx.fillRect(0, 73, s, 1);
  });
  rock.repeat.set(3, 3);

  const pine = makePixelTexture(64, (ctx, s) => {
    ctx.fillStyle = "#132018";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "#243829";
    for (let y = 0; y < s; y += 9) ctx.fillRect(0, y, s, 2);
    ctx.fillStyle = "#08100c";
    for (let x = 0; x < s; x += 11) ctx.fillRect(x, 0, 3, s);
  });

  const car = makePixelTexture(32, (ctx, s) => {
    ctx.fillStyle = "#c81024";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "rgba(255,105,78,0.30)";
    ctx.fillRect(3, 0, 4, s);
    ctx.fillStyle = "rgba(74,0,14,0.35)";
    ctx.fillRect(18, 0, 8, s);
    ctx.fillStyle = "rgba(220,240,255,0.14)";
    for (let y = 2; y < s; y += 9) ctx.fillRect(0, y, s, 1);
  });
  car.repeat.set(2, 2);

  const sign = makePixelTexture(64, (ctx, s) => {
    ctx.fillStyle = "#1d2f34";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "#c3eee8";
    ctx.fillRect(7, 9, 50, 7);
    ctx.fillRect(7, 24, 34, 5);
    ctx.fillRect(7, 39, 44, 5);
    ctx.fillStyle = "#0d1113";
    ctx.fillRect(0, 0, s, 3);
    ctx.fillRect(0, s - 3, s, 3);
  });

  const guardrail = makePixelTexture(32, (ctx, s) => {
    ctx.fillStyle = "#8c9a9b";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "#d4e0dd";
    ctx.fillRect(0, 5, s, 4);
    ctx.fillStyle = "#4d5a5c";
    for (let x = 0; x < s; x += 8) ctx.fillRect(x, 0, 2, s);
  });

  const tire = makePixelTexture(32, (ctx, s) => {
    ctx.fillStyle = "#111416";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "#252b2f";
    for (let y = 0; y < s; y += 5) ctx.fillRect(0, y, s, 2);
  });

  const glass = makePixelTexture(32, (ctx, s) => {
    ctx.fillStyle = "#20314c";
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = "rgba(170,205,255,0.22)";
    ctx.fillRect(0, 0, s, 5);
    ctx.fillStyle = "rgba(0,0,0,0.42)";
    ctx.fillRect(0, 16, s, 8);
  });

  return { road, shoulder, terrain, rock, pine, car, sign, guardrail, tire, glass };
}

/* ── Material helpers ──────────────────────────────────────── */

function mountainLambert(options = {}) {
  return new THREE.MeshLambertMaterial({
    color: options.color ?? 0xffffff,
    map: options.map ?? null,
    emissive: options.emissive ?? 0x000000,
    emissiveIntensity: options.emissiveIntensity ?? 0,
    transparent: options.transparent ?? false,
    opacity: options.opacity ?? 1,
    side: options.side ?? THREE.FrontSide,
    polygonOffset: options.polygonOffset ?? false,
    polygonOffsetFactor: options.polygonOffsetFactor ?? 0,
    polygonOffsetUnits: options.polygonOffsetUnits ?? 0,
    flatShading: true,
  });
}

function mountainBasic(options = {}) {
  return new THREE.MeshBasicMaterial({
    color: options.color ?? 0xffffff,
    map: options.map ?? null,
    transparent: options.transparent ?? false,
    opacity: options.opacity ?? 1,
    side: options.side ?? THREE.FrontSide,
    depthWrite: options.depthWrite ?? true,
    blending: options.blending ?? THREE.NormalBlending,
    polygonOffset: options.polygonOffset ?? false,
    polygonOffsetFactor: options.polygonOffsetFactor ?? 0,
    polygonOffsetUnits: options.polygonOffsetUnits ?? 0,
  });
}

export function createMountainPassMaterials(textures) {
  return {
    road: mountainLambert({
      color: 0x8a867a,
      map: textures.road,
      emissive: 0x20201d,
      emissiveIntensity: 0.52,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -4,
    }),
    shoulder: mountainLambert({
      color: 0x84755e,
      map: textures.shoulder,
      emissive: 0x17140f,
      emissiveIntensity: 0.30,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    }),
    terrain: mountainLambert({ color: 0x40543f, map: textures.terrain, emissive: 0x101b13, emissiveIntensity: 0.28 }),
    rock: mountainLambert({ color: 0x696a65, map: textures.rock, emissive: 0x181b1a, emissiveIntensity: 0.24 }),
    pineNeedles: mountainLambert({ color: PALETTE.pine, map: textures.pine, emissive: 0x030805, emissiveIntensity: 0.14 }),
    pineTrunk: mountainLambert({ color: 0x392c20, emissive: 0x080503, emissiveIntensity: 0.10 }),
    body: mountainLambert({ color: PALETTE.body, map: textures.car, emissive: 0x210005, emissiveIntensity: 0.26 }),
    bodyDark: mountainLambert({ color: PALETTE.bodyDark, map: textures.car, emissive: 0x120003, emissiveIntensity: 0.24 }),
    glass: mountainLambert({ color: 0x222d37, map: textures.glass, emissive: 0x03080b, emissiveIntensity: 0.14, transparent: true, opacity: 0.84 }),
    black: mountainLambert({ color: 0x050707 }),
    tire: mountainLambert({ color: PALETTE.tire, map: textures.tire }),
    tread: mountainLambert({ color: 0x15191b, map: textures.tire }),
    rim: mountainLambert({ color: PALETTE.rim, emissive: 0x141819, emissiveIntensity: 0.18 }),
    lamp: new THREE.MeshBasicMaterial({ color: PALETTE.amber }),
    streetLamp: new THREE.MeshBasicMaterial({ color: PALETTE.lamp }),
    lampGlow: mountainBasic({
      color: PALETTE.lamp,
      transparent: true,
      opacity: 0.46,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    }),
    lightPool: mountainBasic({
      color: PALETTE.lampPool,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    }),
    headlightPool: mountainBasic({
      color: 0xffd28a,
      transparent: true,
      opacity: 0.20,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      polygonOffset: true,
      polygonOffsetFactor: -11,
      polygonOffsetUnits: -11,
    }),
    tail: new THREE.MeshBasicMaterial({ color: PALETTE.tail }),
    guardrail: mountainLambert({ color: PALETTE.guardrail, map: textures.guardrail, emissive: 0x101417, emissiveIntensity: 0.18 }),
    pole: mountainLambert({ color: 0x737d80 }),
    sign: mountainBasic({ color: 0xffffff, map: textures.sign }),
    ridge: mountainLambert({ color: 0x242927, map: textures.rock, emissive: 0x050807, emissiveIntensity: 0.18 }),
    reflectorWarm: mountainBasic({ color: 0xffb257 }),
    reflectorCool: mountainBasic({ color: 0xd6d8c8 }),
    roadPatch: mountainBasic({
      color: 0x111413,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -8,
      polygonOffsetUnits: -8,
    }),
    roadSheen: mountainBasic({
      color: 0xe3d5ac,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      polygonOffset: true,
      polygonOffsetFactor: -9,
      polygonOffsetUnits: -9,
    }),
    edgeLine: new THREE.LineBasicMaterial({ color: 0xd9d8ca, transparent: true, opacity: 0.94 }),
    centerLine: new THREE.LineBasicMaterial({ color: 0xe1c66b, transparent: true, opacity: 0.96 }),

    /* ── NEW: mountain peaks + snow ─────────────────────────── */
    mountainDark: mountainLambert({
      color: 0x2a2d2a,
      emissive: 0x050807,
      emissiveIntensity: 0.12,
    }),
    mountainMid: mountainLambert({
      color: 0x3d423d,
      map: textures.rock,
      emissive: 0x0a0d0b,
      emissiveIntensity: 0.16,
    }),
    snowCap: mountainLambert({
      color: 0xd8e4ec,
      emissive: 0x3a4550,
      emissiveIntensity: 0.35,
    }),

    /* ── NEW: valley fog planes ─────────────────────────────── */
    fogPlane: new THREE.MeshBasicMaterial({
      color: 0x2a3845,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
    }),
    fogPlaneLight: new THREE.MeshBasicMaterial({
      color: 0x3a4855,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    }),
  };
}

/* ── Sky dome ──────────────────────────────────────────────── */

const SKY_VERT = `
  varying vec3 vWorldDir;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldDir = normalize(worldPos.xyz);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SKY_FRAG = `
  precision highp float;
  varying vec3 vWorldDir;
  uniform float timeOfDay;  // 0 = night, 1 = day
  uniform vec3 moonDir;

  // Pseudo-random for stars
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  // Value noise
  float noise2d(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }

  void main() {
    vec3 dir = normalize(vWorldDir);
    float height = dir.y;  // -1 bottom to +1 top
    float horizonBlend = smoothstep(-0.08, 0.55, height);

    // ── Night sky ──
    vec3 nightZenith  = vec3(0.028, 0.040, 0.085);
    vec3 nightMid     = vec3(0.045, 0.065, 0.125);
    vec3 nightHorizon = vec3(0.085, 0.110, 0.170);
    vec3 nightHaze    = vec3(0.110, 0.135, 0.195);
    vec3 nightColor;
    if (height > 0.35) {
      nightColor = mix(nightMid, nightZenith, smoothstep(0.35, 0.85, height));
    } else if (height > 0.0) {
      nightColor = mix(nightHorizon, nightMid, smoothstep(0.0, 0.35, height));
    } else {
      nightColor = mix(nightHaze, nightHorizon, smoothstep(-0.15, 0.0, height));
    }

    // Stars (night)
    vec2 starUV = dir.xz / max(0.001, dir.y + 0.15) * 120.0;
    float starNoise = hash(floor(starUV));
    float starMask = step(0.993, starNoise) * smoothstep(0.05, 0.35, height);
    float starBright = starNoise * starNoise * 1.8;
    float starTwinkle = 0.7 + 0.3 * sin(starNoise * 6283.0);
    vec3 starColor = mix(vec3(0.85, 0.90, 1.0), vec3(1.0, 0.92, 0.75), step(0.5, fract(starNoise * 7.0)));
    nightColor += starColor * starBright * starMask * starTwinkle;

    // Moon (night)
    float moonDist = length(dir - moonDir);
    float moonDisc = smoothstep(0.035, 0.028, moonDist);
    float moonGlow = smoothstep(0.35, 0.04, moonDist) * 0.18;
    float moonHalo = smoothstep(0.85, 0.08, moonDist) * 0.06;
    nightColor += vec3(0.82, 0.86, 0.92) * moonDisc;
    nightColor += vec3(0.45, 0.52, 0.65) * moonGlow;
    nightColor += vec3(0.20, 0.26, 0.38) * moonHalo;

    // ── Day sky ──
    vec3 dayZenith  = vec3(0.22, 0.44, 0.82);
    vec3 dayMid     = vec3(0.38, 0.60, 0.88);
    vec3 dayHorizon = vec3(0.72, 0.82, 0.92);
    vec3 dayHaze    = vec3(0.82, 0.86, 0.90);
    vec3 dayColor;
    if (height > 0.30) {
      dayColor = mix(dayMid, dayZenith, smoothstep(0.30, 0.80, height));
    } else if (height > 0.0) {
      dayColor = mix(dayHorizon, dayMid, smoothstep(0.0, 0.30, height));
    } else {
      dayColor = mix(dayHaze, dayHorizon, smoothstep(-0.12, 0.0, height));
    }

    // Sun glow (day) — soft glow near moon position repurposed as sun
    vec3 sunDir = normalize(vec3(moonDir.x, max(moonDir.y, 0.15), moonDir.z));
    float sunDist = length(dir - sunDir);
    float sunGlow = smoothstep(0.65, 0.05, sunDist) * 0.35;
    float sunHalo = smoothstep(1.2, 0.10, sunDist) * 0.12;
    dayColor += vec3(1.0, 0.95, 0.80) * sunGlow;
    dayColor += vec3(0.95, 0.88, 0.70) * sunHalo;

    // Cloud hints (day only — subtle streaks)
    float cloud1 = noise2d(dir.xz / max(0.001, abs(dir.y) + 0.3) * 8.0);
    float cloud2 = noise2d(dir.xz / max(0.001, abs(dir.y) + 0.3) * 16.0);
    float cloudMask = smoothstep(0.48, 0.72, cloud1 * 0.6 + cloud2 * 0.4);
    cloudMask *= smoothstep(0.02, 0.25, height) * smoothstep(0.80, 0.40, height);
    dayColor = mix(dayColor, vec3(0.92, 0.94, 0.96), cloudMask * 0.38);

    // ── Blend day/night ──
    vec3 sky = mix(nightColor, dayColor, timeOfDay);

    // Atmospheric haze band at horizon for both modes
    float hazeBand = exp(-abs(height) * 14.0) * 0.12;
    vec3 hazeCol = mix(vec3(0.10, 0.13, 0.20), vec3(0.75, 0.80, 0.88), timeOfDay);
    sky += hazeCol * hazeBand;

    // Below-horizon darken
    sky *= smoothstep(-0.30, 0.0, height) * 0.35 + 0.65;

    gl_FragColor = vec4(sky, 1.0);
  }
`;

export function buildSkyDome(scene) {
  const skyGeo = new THREE.SphereGeometry(2400, 40, 28);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: true,
    uniforms: {
      timeOfDay: { value: 0.0 },
      moonDir: { value: new THREE.Vector3(-0.35, 0.55, -0.45).normalize() },
    },
    vertexShader: SKY_VERT,
    fragmentShader: SKY_FRAG,
  });
  const skyMesh = new THREE.Mesh(skyGeo, skyMat);
  skyMesh.renderOrder = -100;
  skyMesh.name = "sky-dome";
  scene.add(skyMesh);
  return skyMesh;
}

/* ── Scene setup (returns lighting refs for day/night toggle) ── */

export function setupMountainPassScene(scene, renderer) {
  scene.background = null; // sky dome handles background
  scene.fog = new THREE.FogExp2(new THREE.Color(PALETTE.fog), MOUNTAIN_PASS.fogDensity);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.NoToneMapping;
  if ("outputColorSpace" in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;

  const hemi = new THREE.HemisphereLight(0xb9b8a8, 0x141b15, 1.08);
  hemi.name = "hemi";
  scene.add(hemi);

  const moon = new THREE.DirectionalLight(PALETTE.moon, 0.72);
  moon.name = "moon";
  moon.position.set(-120, 130, 58);
  moon.castShadow = true;
  moon.shadow.mapSize.set(1024, 1024);
  moon.shadow.camera.left = -280;
  moon.shadow.camera.right = 280;
  moon.shadow.camera.top = 280;
  moon.shadow.camera.bottom = -280;
  scene.add(moon);

  const stormFill = new THREE.DirectionalLight(0x6f756d, 0.34);
  stormFill.name = "stormFill";
  stormFill.position.set(140, 46, -130);
  scene.add(stormFill);

  const ambient = new THREE.AmbientLight(0x3d433b, 0.32);
  ambient.name = "ambient";
  scene.add(ambient);

  return { hemi, moon, stormFill, ambient };
}

/* ── Day / night toggle ────────────────────────────────────── */

export function setTimeOfDay(isDay, { skyDome, lighting, scene, post, fogPlanes }) {
  const t = isDay ? 1.0 : 0.0;

  // Sky dome
  if (skyDome && skyDome.material.uniforms) {
    skyDome.material.uniforms.timeOfDay.value = t;
  }

  // Fog
  const fogColor = isDay ? PALETTE_DAY.fog : PALETTE_NIGHT.fog;
  const fogDensity = isDay ? 0.0020 : 0.0032;
  if (scene.fog) {
    scene.fog.color.set(fogColor);
    scene.fog.density = fogDensity;
  }

  // Lighting
  if (lighting) {
    if (isDay) {
      lighting.hemi.color.set(0xe8e4d8);
      lighting.hemi.groundColor.set(0x4a5a3a);
      lighting.hemi.intensity = 1.65;
      lighting.moon.color.set(0xfff4d6);
      lighting.moon.intensity = 1.85;
      lighting.stormFill.color.set(0xa5b0a8);
      lighting.stormFill.intensity = 0.72;
      lighting.ambient.color.set(0x8a9a82);
      lighting.ambient.intensity = 0.68;
    } else {
      lighting.hemi.color.set(0xb9b8a8);
      lighting.hemi.groundColor.set(0x141b15);
      lighting.hemi.intensity = 1.08;
      lighting.moon.color.set(PALETTE_NIGHT.moon);
      lighting.moon.intensity = 0.72;
      lighting.stormFill.color.set(0x6f756d);
      lighting.stormFill.intensity = 0.34;
      lighting.ambient.color.set(0x3d433b);
      lighting.ambient.intensity = 0.32;
    }
  }

  // Post-processing tint
  if (post && post.uniforms) {
    if (isDay) {
      post.uniforms.tint.value.set(1.02, 1.01, 0.98);
      post.uniforms.exposure.value = 1.38;
      post.uniforms.shadowLift.value = 0.065;
      post.uniforms.vignetteStrength.value = 0.065;
    } else {
      post.uniforms.tint.value.set(0.96, 0.98, 1.06);
      post.uniforms.exposure.value = 1.26;
      post.uniforms.shadowLift.value = 0.095;
      post.uniforms.vignetteStrength.value = 0.080;
    }
  }

  // Fog planes
  if (fogPlanes && fogPlanes.length) {
    for (const fp of fogPlanes) {
      if (!fp.material) continue;
      if (isDay) {
        fp.material.color.set(0x9aabbf);
        fp.material.opacity = fp.userData.dayOpacity ?? 0.18;
      } else {
        fp.material.color.set(fp.userData.nightColor ?? 0x2a3845);
        fp.material.opacity = fp.userData.nightOpacity ?? 0.22;
      }
    }
  }

  // Update mutable PALETTE
  const src = isDay ? PALETTE_DAY : PALETTE_NIGHT;
  for (const k of Object.keys(src)) PALETTE[k] = src[k];
}

/* ── Post-processing ───────────────────────────────────────── */

export function createMountainPassPost() {
  const target = new THREE.WebGLRenderTarget(1, 1, {
    depthBuffer: true,
    stencilBuffer: false,
  });
  target.texture.magFilter = THREE.NearestFilter;
  target.texture.minFilter = THREE.NearestFilter;
  target.texture.generateMipmaps = false;

  const uniforms = {
    tDiffuse: { value: target.texture },
    resolution: { value: new THREE.Vector2(1, 1) },
    time: { value: 0 },
    colorSteps: { value: MOUNTAIN_PASS.colorSteps },
    ditherStrength: { value: MOUNTAIN_PASS.ditherStrength },
    scanlineStrength: { value: MOUNTAIN_PASS.scanlineStrength },
    noiseStrength: { value: MOUNTAIN_PASS.noiseStrength },
    vignetteStrength: { value: MOUNTAIN_PASS.vignetteStrength },
    chromaticStrength: { value: MOUNTAIN_PASS.chromaticStrength },
    exposure: { value: MOUNTAIN_PASS.exposure },
    shadowLift: { value: MOUNTAIN_PASS.shadowLift },
    midtoneBoost: { value: MOUNTAIN_PASS.midtoneBoost },
    tint: { value: new THREE.Vector3(0.96, 0.98, 1.06) },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    depthTest: false,
    depthWrite: false,
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D tDiffuse;
      uniform vec2 resolution;
      uniform float time;
      uniform float colorSteps;
      uniform float ditherStrength;
      uniform float scanlineStrength;
      uniform float noiseStrength;
      uniform float vignetteStrength;
      uniform float chromaticStrength;
      uniform float exposure;
      uniform float shadowLift;
      uniform float midtoneBoost;
      uniform vec3 tint;

      float rand(vec2 co) {
        return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
      }

      float bayer4(vec2 p) {
        vec2 q = mod(floor(p), 4.0);
        float x = q.x;
        float y = q.y;
        float v = 0.0;
        if (y < 0.5) {
          if (x < 0.5) v = 0.0; else if (x < 1.5) v = 8.0; else if (x < 2.5) v = 2.0; else v = 10.0;
        } else if (y < 1.5) {
          if (x < 0.5) v = 12.0; else if (x < 1.5) v = 4.0; else if (x < 2.5) v = 14.0; else v = 6.0;
        } else if (y < 2.5) {
          if (x < 0.5) v = 3.0; else if (x < 1.5) v = 11.0; else if (x < 2.5) v = 1.0; else v = 9.0;
        } else {
          if (x < 0.5) v = 15.0; else if (x < 1.5) v = 7.0; else if (x < 2.5) v = 13.0; else v = 5.0;
        }
        return (v + 0.5) / 16.0;
      }

      void main() {
        vec2 px = 1.0 / resolution;
        float edge = smoothstep(0.20, 0.84, distance(vUv, vec2(0.5)));
        vec2 ca = vec2(px.x * chromaticStrength * edge, 0.0);
        vec3 color = texture2D(tDiffuse, vUv).rgb;
        color.r = texture2D(tDiffuse, vUv + ca).r;
        color.b = texture2D(tDiffuse, vUv - ca).b;

        color *= exposure;
        float luma = dot(color, vec3(0.299, 0.587, 0.114));
        color += shadowLift * (1.0 - smoothstep(0.0, 0.50, luma));
        color = mix(color, sqrt(max(color, vec3(0.0))), midtoneBoost);
        color *= tint;
        color = mix(vec3(dot(color, vec3(0.299, 0.587, 0.114))), color, 1.04);

        float d = (bayer4(gl_FragCoord.xy) - 0.5) * ditherStrength;
        color = floor(color * colorSteps + d) / colorSteps;

        float scan = 1.0 - scanlineStrength * step(0.5, fract(gl_FragCoord.y * 0.5));
        float n = (rand(gl_FragCoord.xy + time * 29.0) - 0.5) * noiseStrength;
        float vig = smoothstep(0.94, 0.24, distance(vUv, vec2(0.5)));
        color = color * scan + n;
        color *= mix(1.0 - vignetteStrength, 1.04, vig);
        gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
      }
    `,
  });

  const postScene = new THREE.Scene();
  postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  return { target, scene: postScene, camera, uniforms };
}
