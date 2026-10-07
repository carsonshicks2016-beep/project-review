import * as THREE from "three";
import { Pass, FullScreenQuad } from "three/addons/postprocessing/Pass.js";

/**
 * Screen-space ink outline — the load-bearing element of the stylized look.
 *
 * Runs a normal+depth prepass with `scene.overrideMaterial`, then a Sobel over
 * both buffers. Depth edges catch silhouettes (car against road, rail against
 * hill); normal edges catch interior creases (wheel arches, the shoulder line
 * along the bodywork) that depth alone cannot see.
 *
 * MeshNormalMaterial is used for the prepass rather than a hand-written shader
 * so three keeps handling instancing (the forest) and every other vertex
 * variant for free; depth comes from an attached DepthTexture.
 *
 * Edges fade with distance and into fog, otherwise the horizon turns into a
 * scribble of aliased hairlines at speed.
 */
export class InkOutlinePass extends Pass {
  constructor(scene, camera, options = {}) {
    super();
    this.scene = scene;
    this.camera = camera;
    this.resolutionScale = options.resolutionScale ?? 1.0;

    this.normalMaterial = new THREE.MeshNormalMaterial();
    const depthTexture = new THREE.DepthTexture(1, 1);
    depthTexture.type = THREE.UnsignedIntType;
    this.prepass = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthTexture,
      depthBuffer: true,
    });

    this.uniforms = {
      tDiffuse: { value: null },
      tNormal: { value: this.prepass.texture },
      tDepth: { value: depthTexture },
      uTexel: { value: new THREE.Vector2(1, 1) },
      uColor: { value: new THREE.Color(options.color ?? 0x0a1218) },
      uStrength: { value: options.strength ?? 0.85 },
      uThickness: { value: options.thickness ?? 1.0 },
      uDepthEdge: { value: options.depthEdge ?? 0.9 },
      uNormalEdge: { value: options.normalEdge ?? 0.55 },
      uNear: { value: camera.near },
      uFar: { value: camera.far },
      uFadeStart: { value: options.fadeStart ?? 120.0 },
      uFadeEnd: { value: options.fadeEnd ?? 900.0 },
    };

    this.material = new THREE.ShaderMaterial({
      name: "supra_ink_outline",
      uniforms: this.uniforms,
      vertexShader: /* glsl */`
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */`
        uniform sampler2D tDiffuse;
        uniform sampler2D tNormal;
        uniform sampler2D tDepth;
        uniform vec2 uTexel;
        uniform vec3 uColor;
        uniform float uStrength;
        uniform float uThickness;
        uniform float uDepthEdge;
        uniform float uNormalEdge;
        uniform float uNear;
        uniform float uFar;
        uniform float uFadeStart;
        uniform float uFadeEnd;
        varying vec2 vUv;

        float linearDepth(vec2 uv) {
          float d = texture2D(tDepth, uv).x;
          // Perspective depth -> view-space metres.
          float ndc = d * 2.0 - 1.0;
          return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
        }

        void main() {
          vec4 base = texture2D(tDiffuse, vUv);
          vec2 o = uTexel * uThickness;

          float dC = linearDepth(vUv);
          float d1 = linearDepth(vUv + vec2( o.x,  0.0));
          float d2 = linearDepth(vUv + vec2(-o.x,  0.0));
          float d3 = linearDepth(vUv + vec2( 0.0,  o.y));
          float d4 = linearDepth(vUv + vec2( 0.0, -o.y));

          // Scale the depth threshold with distance so a 1 m step reads the same
          // at 10 m and at 200 m instead of banding the whole far field.
          float tolerance = uDepthEdge * max(1.0, dC * 0.045);
          float depthDiff = max(max(abs(d1 - dC), abs(d2 - dC)),
                                max(abs(d3 - dC), abs(d4 - dC)));
          float depthEdge = smoothstep(tolerance * 0.5, tolerance, depthDiff);

          vec3 nC = texture2D(tNormal, vUv).xyz * 2.0 - 1.0;
          vec3 n1 = texture2D(tNormal, vUv + vec2( o.x,  0.0)).xyz * 2.0 - 1.0;
          vec3 n2 = texture2D(tNormal, vUv + vec2(-o.x,  0.0)).xyz * 2.0 - 1.0;
          vec3 n3 = texture2D(tNormal, vUv + vec2( 0.0,  o.y)).xyz * 2.0 - 1.0;
          vec3 n4 = texture2D(tNormal, vUv + vec2( 0.0, -o.y)).xyz * 2.0 - 1.0;
          float nd = min(min(dot(nC, n1), dot(nC, n2)), min(dot(nC, n3), dot(nC, n4)));
          float normalEdge = smoothstep(uNormalEdge, uNormalEdge * 0.55, nd);

          float edge = max(depthEdge, normalEdge * 0.85);
          // Distance fade, and never draw on the sky (depth at far plane).
          edge *= 1.0 - smoothstep(uFadeStart, uFadeEnd, dC);
          edge *= step(dC, uFar * 0.92);

          gl_FragColor = vec4(mix(base.rgb, uColor, edge * uStrength), base.a);
        }
      `,
    });
    this._quad = new FullScreenQuad(this.material);
    this._size = new THREE.Vector2(1, 1);
  }

  setSize(width, height) {
    this._size.set(width, height);
    const w = Math.max(1, Math.floor(width * this.resolutionScale));
    const h = Math.max(1, Math.floor(height * this.resolutionScale));
    this.prepass.setSize(w, h);
    this.uniforms.uTexel.value.set(1 / w, 1 / h);
  }

  setQuality(level) {
    this.enabled = level !== "low";
    this.resolutionScale = level === "medium" ? 0.6 : 1.0;
    this.setSize(this._size.x, this._size.y);
  }

  render(renderer, writeBuffer, readBuffer) {
    this.uniforms.uNear.value = this.camera.near;
    this.uniforms.uFar.value = this.camera.far;

    const previousTarget = renderer.getRenderTarget();
    const previousOverride = this.scene.overrideMaterial;
    const previousBackground = this.scene.background;

    // Hide anything flagged `noOutline` for the prepass. The terrain
    // containment shell is a translucent gap-filler that stretches far past the
    // hills; inked, its top edge draws a hard hairline straight across the sky.
    const hidden = [];
    this.scene.traverse((node) => {
      if (node.visible && node.userData && node.userData.noOutline) {
        node.visible = false;
        hidden.push(node);
      }
    });

    // The sky dome would otherwise stamp its own normals over everything.
    this.scene.background = null;
    this.scene.overrideMaterial = this.normalMaterial;
    renderer.setRenderTarget(this.prepass);
    renderer.clear();
    renderer.render(this.scene, this.camera);
    this.scene.overrideMaterial = previousOverride;
    this.scene.background = previousBackground;
    renderer.setRenderTarget(previousTarget);
    for (const node of hidden) node.visible = true;

    this.uniforms.tDiffuse.value = readBuffer.texture;
    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
    } else {
      renderer.setRenderTarget(writeBuffer);
      if (this.clear) renderer.clear();
    }
    this._quad.render(renderer);
  }

  dispose() {
    this.prepass.dispose();
    this.material.dispose();
    this.normalMaterial.dispose();
    this._quad.dispose();
  }
}
