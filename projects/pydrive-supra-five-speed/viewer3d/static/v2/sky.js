// The v2 sky: one dome shader for both moods. The gradient colors are
// mood-lerped on the CPU (bound THREE.Color uniforms); night extras (sparse
// stars, moon disc/halo) and day extras (sun, cumulus puffs) cross-fade with
// the mood mix uniform. Custom ShaderMaterial => not tone-mapped; colors here
// are authored final.
import * as THREE from "../vendor/three.module.min.js";

const VERT = `
  varying vec3 vDir;
  void main() {
    vDir = (modelMatrix * vec4(position, 1.0)).xyz - cameraPosition;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAG = `
  precision highp float;
  varying vec3 vDir;
  uniform vec3 topColor;
  uniform vec3 horizonColor;
  uniform vec3 bottomColor;
  uniform float mixDN;     // 0 = night, 1 = day
  uniform vec3 moonDir;
  uniform vec3 sunDir;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

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

  float fbm(vec2 p) {
    return noise2d(p) * 0.55
         + noise2d(p * 2.31 + 17.7) * 0.30
         + noise2d(p * 5.13 + 41.3) * 0.15;
  }

  void main() {
    vec3 dir = normalize(vDir);
    float h = dir.y;

    vec3 col = h >= 0.0
      ? mix(horizonColor, topColor, smoothstep(0.02, 0.55, h))
      : mix(horizonColor, bottomColor, smoothstep(-0.02, -0.30, h));

    float nightW = 1.0 - mixDN;

    // ── Night: sparse stars + moon ──
    if (nightW > 0.001) {
      vec2 starUV = dir.xz / max(0.02, dir.y + 0.18) * 140.0;
      float sn = hash(floor(starUV));
      float mask = step(0.9966, sn) * smoothstep(0.04, 0.30, h);
      float t = clamp((sn - 0.9966) / 0.0034, 0.0, 1.0);
      float bright = 0.5 + t * t * 2.2;
      vec3 starCol = mix(vec3(0.88, 0.92, 1.0), vec3(1.0, 0.94, 0.80),
                         step(0.5, fract(sn * 7.0)));
      col += starCol * mask * bright * nightW;

      float md = distance(dir, moonDir);
      float disc = smoothstep(0.045, 0.036, md);
      float glow = smoothstep(0.30, 0.03, md) * 0.16;
      float halo = smoothstep(0.95, 0.10, md) * 0.05;
      col += (vec3(0.93, 0.95, 0.99) * disc
            + vec3(0.55, 0.62, 0.75) * glow
            + vec3(0.25, 0.31, 0.43) * halo) * nightW;
    }

    // ── Day: sun + cumulus puffs ──
    if (mixDN > 0.001) {
      float sd = distance(dir, sunDir);
      float sdisc = smoothstep(0.030, 0.024, sd);
      float sglow = smoothstep(0.45, 0.04, sd) * 0.22;
      float shalo = smoothstep(1.10, 0.10, sd) * 0.05;
      col += (vec3(1.0, 0.98, 0.92) * sdisc * 1.2
            + vec3(1.0, 0.92, 0.72) * sglow
            + vec3(0.85, 0.78, 0.62) * shalo) * mixDN;

      // planar projection => clouds foreshorten toward the horizon
      vec2 cuv = dir.xz / max(0.10, dir.y + 0.06) * 1.6;
      float cf = fbm(cuv);
      float puff = smoothstep(0.58, 0.70, cf);
      float band = smoothstep(0.03, 0.15, h) * smoothstep(0.90, 0.35, h);
      vec3 cloudCol = mix(vec3(0.74, 0.79, 0.84), vec3(0.99, 0.99, 0.97),
                          smoothstep(0.58, 0.82, cf));
      col = mix(col, cloudCol, puff * band * 0.92 * mixDN);
    }

    // horizon haze band, both moods
    float hazeBand = exp(-abs(h) * 11.0) * 0.10;
    vec3 hazeCol = mix(vec3(0.16, 0.20, 0.28), vec3(0.72, 0.78, 0.82), mixDN);
    col += hazeCol * hazeBand;

    gl_FragColor = vec4(col, 1.0);
  }
`;

export function buildSky(scene, mood) {
  const uniforms = {
    topColor: { value: new THREE.Color(0x0b1220) },
    horizonColor: { value: new THREE.Color(0x2a3a50) },
    bottomColor: { value: new THREE.Color(0x10161f) },
    mixDN: { value: 0 },
    moonDir: { value: new THREE.Vector3(-120, 140, 60).normalize() },
    sunDir: { value: new THREE.Vector3(110, 160, -70).normalize() },
  };
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(2600, 40, 24),
    new THREE.ShaderMaterial({
      uniforms,
      vertexShader: VERT,
      fragmentShader: FRAG,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
    }),
  );
  mesh.renderOrder = -100;
  mesh.frustumCulled = false;
  mesh.name = "sky-v2";
  scene.add(mesh);

  mood.bindColor(uniforms.topColor, "value", "skyTop");
  mood.bindColor(uniforms.horizonColor, "value", "skyHorizon");
  mood.bindColor(uniforms.bottomColor, "value", "skyBottom");
  mood.bindNumber(uniforms.mixDN, "value", "skyMix");
  return mesh;
}
