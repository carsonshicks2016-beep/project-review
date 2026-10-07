import * as THREE from 'three';

// Eye-Dome Lighting.
//
// A screen-space pass that darkens a pixel in proportion to how much *closer* its
// neighbours are. The result is a dark crease wherever the surface recedes behind a
// nearer silhouette, which is the depth cue a point cloud otherwise has no way to give
// you: unlike a mesh it has no continuous surface for light to fall across.
//
// This is the technique Potree uses. It works on depth alone, so it costs one extra
// full-screen pass and nothing per-point.
export class EyeDomeLighting {
  constructor(renderer, { strength = 5.0, radius = 4.0 } = {}) {
    this.renderer = renderer;
    this.enabled = true;

    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.target = this._createTarget(size.x, size.y);

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: this.target.texture },
        tDepth: { value: this.target.depthTexture },
        uTexel: { value: new THREE.Vector2(1 / size.x, 1 / size.y) },
        uRadius: { value: radius },
        uStrength: { value: strength },
        uGain: { value: 1.45 },
        uLens: { value: 0.0 },      // 0 = off, 1 = full FPV lens look
        uNear: { value: 0.1 },
        uFar: { value: 1000.0 }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform sampler2D tDepth;
        uniform vec2 uTexel;
        uniform float uRadius, uStrength, uNear, uFar, uGain, uLens;
        varying vec2 vUv;

        // Depth buffer stores a non-linear value; EDL needs real eye-space distance.
        float eyeDepth(vec2 uv) {
          float d = texture2D(tDepth, uv).x;
          float ndc = d * 2.0 - 1.0;
          return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
        }

        void main() {
          // FPV cams are ~150 degrees with real barrel distortion. Bending the sample
          // coordinates outward from centre fakes it convincingly at almost no cost.
          vec2 c = vUv - 0.5;
          float r2 = dot(c, c);
          // Sample inward at the edges so edge content stretches outward: that is the
          // fisheye bulge. Sampling outward instead just reads off the frame and leaves
          // black corners.
          vec2 uv = vUv - c * r2 * 0.30 * uLens;

          vec4 color = texture2D(tDiffuse, uv);
          float zc = eyeDepth(uv);

          // Leave the empty background alone -- shading it just adds a grey haze.
          if (zc >= uFar * 0.999) {
            gl_FragColor = color;
            return;
          }

          // Log-space so the response is scale-invariant: a 10cm step reads the same at
          // 2m as a 1m step does at 20m.
          float sum = 0.0;
          for (int i = 0; i < 8; i++) {
            float ang = float(i) * 0.7853981634;
            vec2 off = vec2(cos(ang), sin(ang)) * uRadius * uTexel;
            float zn = eyeDepth(uv + off);
            sum += max(0.0, log2(zc) - log2(zn));
          }

          float shade = exp(-(sum / 8.0) * uStrength);
          vec3 outCol = min(color.rgb * shade * uGain, vec3(1.0));

          // Vignette. Goggle footage always falls off at the corners.
          float vig = 1.0 - uLens * smoothstep(0.12, 0.62, r2) * 0.65;
          outCol *= vig;

          gl_FragColor = vec4(outCol, color.a);
        }
      `,
      depthTest: false,
      depthWrite: false
    });

    // Full-screen triangle driven straight in clip space by the vertex shader.
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.quad.frustumCulled = false;
    this.quadScene = new THREE.Scene();
    this.quadScene.add(this.quad);
    this.quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  _createTarget(w, h) {
    const target = new THREE.WebGLRenderTarget(Math.max(1, w), Math.max(1, h), {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true
    });
    // The scene pass writes tone-mapped sRGB into the target, so the EDL pass is a
    // straight multiply with no further colour conversion.
    target.texture.colorSpace = THREE.SRGBColorSpace;
    target.depthTexture = new THREE.DepthTexture(Math.max(1, w), Math.max(1, h));
    target.depthTexture.format = THREE.DepthFormat;
    target.depthTexture.type = THREE.UnsignedIntType;
    return target;
  }

  setSize(w, h) {
    const dpr = this.renderer.getPixelRatio();
    const pw = Math.max(1, Math.floor(w * dpr));
    const ph = Math.max(1, Math.floor(h * dpr));
    this.target.setSize(pw, ph);
    this.material.uniforms.uTexel.value.set(1 / pw, 1 / ph);
  }

  // Ramps the lens look in and out so switching to FPV does not snap.
  setLens(amount, dt) {
    const u = this.material.uniforms.uLens;
    const k = 1 - Math.exp(-dt / 0.15);
    u.value += (amount - u.value) * k;
  }

  render(scene, camera) {
    const renderer = this.renderer;

    if (!this.enabled) {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      return;
    }

    renderer.setRenderTarget(this.target);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);

    this.material.uniforms.uNear.value = camera.near;
    this.material.uniforms.uFar.value = camera.far;
    renderer.render(this.quadScene, this.quadCamera);
  }

  dispose() {
    this.target.dispose();
    this.material.dispose();
    this.quad.geometry.dispose();
  }
}
