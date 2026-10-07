import * as THREE from "three";
import { condition } from "./conditions.js";

/**
 * Procedural sky dome + fog + sun/ambient.
 * Shadow volume is recentered on the subject every frame so the car
 * actually casts a readable shadow on asphalt.
 */
export class SkyRig {
  constructor(scene) {
    this.scene = scene;
    this.scene.background = new THREE.Color(0xbcd6de);

    this._sunOffset = new THREE.Vector3(55, 95, 35);
    this._fillOffset = new THREE.Vector3(-40, 35, -25);
    this._tmp = new THREE.Vector3();

    this.sun = new THREE.DirectionalLight(0xfff3dc, 2.6);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0002;
    this.sun.shadow.normalBias = 0.035;
    this.sun.shadow.radius = 2.5;
    // Legacy-scale ortho volume so contact/verge shadows are not a tight square.
    const cam = this.sun.shadow.camera;
    cam.near = 1;
    cam.far = 900;
    cam.left = -70;
    cam.right = 70;
    cam.top = 70;
    cam.bottom = -70;

    this.fill = new THREE.DirectionalLight(0xb8cdd8, 0.12);
    this.fill.castShadow = false;

    this.ambient = new THREE.HemisphereLight(0x9fb8c4, 0x2a3228, 0.42);
    this.scene.add(this.sun);
    this.scene.add(this.sun.target);
    this.scene.add(this.fill);
    this.scene.add(this.fill.target);
    this.scene.add(this.ambient);
    this.scene.fog = new THREE.FogExp2(0xbcd6de, 0.0014);

    const geo = new THREE.SphereGeometry(8000, 48, 24);
    this.material = new THREE.ShaderMaterial({
      name: "observatory_sky",
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
      fog: false,
      uniforms: {
        zenith: { value: new THREE.Color(0x2e6ea8) },
        horizon: { value: new THREE.Color(0xbcd6de) },
        stars: { value: 0 },
        sunDirection: { value: new THREE.Vector3(0.4, 0.7, 0.3).normalize() },
        sunColor: { value: new THREE.Color(0xfff3dc) },
        cloudAmount: { value: 0.42 },
        cloudTint: { value: new THREE.Color(0xffffff) },
      },
      vertexShader: `
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_Position.z = gl_Position.w;
        }
      `,
      fragmentShader: `
        uniform vec3 zenith;
        uniform vec3 horizon;
        uniform float stars;
        uniform vec3 sunDirection;
        uniform vec3 sunColor;
        uniform float cloudAmount;
        uniform vec3 cloudTint;
        varying vec3 vDir;

        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
        float vnoise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
            mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
            f.y);
        }
        float fbm(vec2 p) {
          float v = 0.0;
          float a = 0.5;
          for (int i = 0; i < 5; i++) {
            v += a * vnoise(p);
            p *= 2.03;
            a *= 0.5;
          }
          return v;
        }

        // Posterize with a soft edge — matches the banding the world uses so
        // sky and ground read as one graphic system.
        float skyBand(float x, float bands, float softness) {
          float s = clamp(x, 0.0, 1.0) * bands;
          float i = floor(s);
          float f = fract(s);
          return (i + smoothstep(0.5 - softness, 0.5 + softness, f)) / bands;
        }

        void main() {
          float elev = clamp(vDir.y, -0.15, 1.0);

          // Base gradient, stepped into a few broad bands.
          float h = smoothstep(-0.05, 0.72, elev);
          h = skyBand(pow(h, 1.15), 7.0, 0.16);
          vec3 col = mix(horizon, zenith, h);

          float sunDot = max(dot(normalize(vDir), normalize(sunDirection)), 0.0);

          // Broad forward-scatter halo, then the disc itself.
          col += sunColor * pow(sunDot, 8.0) * 0.18;
          col += sunColor * pow(sunDot, 220.0) * 0.55;
          float disc = smoothstep(0.9987, 0.9994, sunDot);
          col = mix(col, sunColor * 1.6, disc);

          // Flattened cloud band: project onto a plane above the viewer so the
          // cells stretch toward the horizon instead of pinching at the zenith.
          if (cloudAmount > 0.01 && elev > 0.02) {
            vec2 cp = vDir.xz / max(vDir.y, 0.06);
            float n = fbm(cp * 0.9);
            // Hard-edged shapes with a second inner step, so clouds read as
            // drawn forms with a lit core rather than as soft photographic haze.
            float threshold = 0.52 - cloudAmount * 0.30;
            float cover = smoothstep(threshold, threshold + 0.05, n);
            float core = smoothstep(threshold + 0.09, threshold + 0.14, n);
            cover *= smoothstep(0.02, 0.30, elev);
            core *= smoothstep(0.02, 0.30, elev);
            float lit = 0.72 + 0.28 * pow(sunDot, 3.0);
            col = mix(col, cloudTint * lit * 0.88, cover * 0.85);
            col = mix(col, cloudTint * lit, core * 0.75);
          }

          float under = smoothstep(0.0, -0.12, elev);
          col = mix(col, horizon * 0.72, under * 0.55);

          // Round point stars. Celling raw vDir.xz stretches the cells into
          // vertical slivers as they approach the zenith, so divide out the
          // elevation first and place a disc inside each cell.
          if (stars > 0.01 && elev > 0.05) {
            vec2 sp = vDir.xz / max(abs(vDir.y), 0.22) * 30.0;
            vec2 cell = floor(sp);
            vec2 f = fract(sp);
            float pick = hash(cell);
            if (pick > 0.982) {
              vec2 centre = vec2(hash(cell + 3.17), hash(cell + 7.71));
              float d = length(f - centre);
              float point = smoothstep(0.10, 0.0, d);
              float twinkle = 0.55 + 0.45 * hash(cell + 13.3);
              col += vec3(point) * stars * twinkle * smoothstep(0.05, 0.32, elev);
            }
          }
          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -1000;
    this.scene.add(this.mesh);
    this.id = "clear-day";
  }

  apply(id) {
    const c = condition(id);
    this.id = c.id;
    this.material.uniforms.zenith.value.setHex(c.zenith);
    this.material.uniforms.horizon.value.setHex(c.horizon);
    this.material.uniforms.stars.value = c.stars;
    this.sun.color.setHex(c.sun);
    this.sun.intensity = c.sunIntensity;
    const alt = Math.max(0.08, c.sunAltitude);
    // Directional offset from the subject (world metres).
    this._sunOffset.set(
      Math.cos(alt * Math.PI) * 70,
      Math.max(18, Math.sin(alt * Math.PI) * 110),
      45,
    );
    // The dome draws its own disc/halo, so it has to agree with the light.
    this.material.uniforms.sunDirection.value.copy(this._sunOffset).normalize();
    this.material.uniforms.sunColor.value.setHex(c.sun);
    this.material.uniforms.cloudAmount.value = c.cloudAmount ?? 0;
    this.material.uniforms.cloudTint.value.setHex(c.cloudTint ?? 0xffffff);
    this.fill.color.setHex(c.light === "night" ? 0x6a7a9a : 0xb8cdd8);
    this.fill.intensity = c.light === "night" ? 0.08 : c.light === "dusk" ? 0.10 : 0.12;
    this.ambient.color.setHex(c.ambient);
    this.ambient.intensity = c.ambientIntensity;
    this.ambient.groundColor.setHex(
      c.light === "night" ? 0x0a1018 : c.light === "dusk" ? 0x2a2228 : 0x243028,
    );
    this.scene.fog.color.setHex(c.fogColor);
    this.scene.fog.density = c.fogDensity;
    this.scene.background.setHex(c.horizon);
    // Night: softer shadows / lower radius so they don't smear into noise.
    this.sun.castShadow = c.light !== "night" || c.sunIntensity > 0.25;
    this.sun.shadow.radius = c.light === "day" ? 2.2 : 3.2;
    return c;
  }

  /**
   * Recenter sky + sun + shadow frustum on the subject (usually the car).
   * Passing the camera instead makes the shadow map miss the vehicle.
   */
  follow(subject) {
    this.mesh.position.copy(subject);

    this.sun.position.copy(subject).add(this._sunOffset);
    this.sun.target.position.copy(subject);
    this.sun.target.updateMatrixWorld();
    this.sun.updateMatrixWorld();

    this.fill.position.copy(subject).add(this._fillOffset);
    this.fill.target.position.copy(subject);
    this.fill.target.updateMatrixWorld();

    const cam = this.sun.shadow.camera;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
  }

  /** Quality hook — shrink shadow map on medium tier. */
  setShadowQuality(level) {
    const size = level === "low" ? 512 : level === "medium" ? 1024 : 2048;
    if (this.sun.shadow.mapSize.x !== size) {
      this.sun.shadow.mapSize.set(size, size);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
    }
    this.sun.castShadow = level !== "low";
  }
}
