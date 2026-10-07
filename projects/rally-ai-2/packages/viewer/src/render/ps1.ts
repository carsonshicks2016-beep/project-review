/**
 * The early-3D presentation pass.
 *
 * The scene's deliberately simple geometry should carry the style. A previous
 * version rendered everything into a fixed 320x240 target, enlarged it with
 * nearest-neighbour filtering, then added Bayer dithering and 5-bit colour
 * quantisation. That made the image read as enlarged pixel art before the eye
 * could see any of the modelling.
 *
 * This pass now renders at the canvas' native drawing-buffer size and performs
 * only the colour-space conversion needed by Three's linear render target. The
 * class name and toggle remain for compatibility and for clean A/B testing.
 * Geometry can retain a very small fixed-point wobble without degrading the
 * whole frame.
 */

import {
  LinearFilter,
  Mesh,
  OrthographicCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  Vector2,
  WebGLRenderTarget,
  type Camera,
  type Material,
  type WebGLRenderer,
} from "three";

/** Legacy logical framebuffer exports, retained for downstream compatibility. */
export const INTERNAL_WIDTH = 320;
export const INTERNAL_HEIGHT = 240;

/**
 * Snap grid for vertex jitter, in normalised device space.
 *
 * At a typical 960-1440 px-wide 4:3 canvas this is around one display pixel,
 * rather than the 4-9 px jumps produced by the old 160-step grid.
 */
const JITTER_GRID = 720.0;

const VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const FRAG = /* glsl */ `
precision mediump float;
uniform sampler2D tDiffuse;
varying vec2 vUv;

// three's own linear->sRGB transfer, reproduced because this pass has to do it
// by hand. See the note on the render target for why.
vec3 linearToSRGB(vec3 c) {
  return mix(
    pow(c, vec3(0.41666)) * 1.055 - vec3(0.055),
    c * 12.92,
    step(c, vec3(0.0031308))
  );
}

void main() {
  vec3 c = texture2D(tDiffuse, vUv).rgb;
  c = linearToSRGB(c);
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
`;

/**
 * Add vertex snapping to a material.
 *
 * Patched via `onBeforeCompile` rather than by writing bespoke materials, so
 * every mesh in the scene gets it without the scene code knowing about it.
 *
 * The snap happens in clip space **after** the projection, divided through by
 * w, which is what the hardware actually did — snapping in world space instead
 * gives a uniform wobble that does not increase with distance, and the
 * distance-dependence is the whole character of the artefact.
 */
export function applyVertexJitter(material: Material, affine = false): void {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uJitter = { value: JITTER_GRID };
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
         uniform float uJitter;
         varying float vAffineW;`,
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         {
           // Snap in normalised device coordinates, then restore w.
           vec2 grid = vec2(uJitter);
           vec4 snapped = gl_Position;
           snapped.xyz = gl_Position.xyz / gl_Position.w;
           snapped.xy = floor(grid * snapped.xy + 0.5) / grid;
           snapped.xyz *= gl_Position.w;
           gl_Position = snapped;
         }
         vAffineW = gl_Position.w;`,
      );

    if (!affine) return;

    // --- affine texture mapping -------------------------------------------
    //
    // The console had no perspective-correct texture interpolation, so textures
    // visibly swim and shear on surfaces angled away from the camera. On a road
    // stretching to the horizon it is the most obvious artefact of the whole
    // era, and it is the one most "retro" shaders miss.
    //
    // GLSL ES 1.0 has no `noperspective` qualifier, so it is recovered
    // arithmetically. The GPU interpolates a varying V as
    //
    //     V_frag = (sum L_i V_i / w_i) / (sum L_i / w_i)
    //
    // for screen-space barycentrics L. Passing V' = V * w and W' = w gives
    //
    //     V'_frag = (sum L_i V_i) / (sum L_i / w_i)
    //     W'_frag = 1 / (sum L_i / w_i)
    //
    // so V'_frag / W'_frag = sum L_i V_i — linear in SCREEN space, which is
    // exactly the hardware's behaviour.
    shader.vertexShader = shader.vertexShader
      .replace(
        "varying float vAffineW;",
        `varying float vAffineW;
         varying vec2 vAffineUv;`,
      )
      .replace(
        "vAffineW = gl_Position.w;",
        `vAffineW = gl_Position.w;
         vAffineUv = uv * gl_Position.w;`,
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
         varying float vAffineW;
         varying vec2 vAffineUv;`,
      )
      // Every map lookup in the built-in materials goes through vMapUv, so
      // redefining it once covers them all.
      .replace(
        "#include <map_fragment>",
        `{
           vec2 vMapUv = vAffineUv / vAffineW;
           #include <map_fragment>
         }`,
      );
  };
  material.needsUpdate = true;
}

/**
 * Render at native drawing-buffer resolution, then perform a colour-correct
 * pass-through.
 *
 * Call {@link render} instead of `renderer.render`.
 */
export class Ps1Pass {
  private readonly target: WebGLRenderTarget;
  private readonly quadScene = new Scene();
  private readonly quadCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly material: ShaderMaterial;

  enabled = true;

  constructor(private readonly renderer: WebGLRenderer) {
    const { width, height } = this.internalSize();
    this.target = new WebGLRenderTarget(width, height, {
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      format: RGBAFormat,
      depthBuffer: true,
    });
    // NOTE: the target is deliberately left in linear space and the shader
    // converts by hand.
    //
    // three decides a render's output colour space as
    // `target === null ? outputColorSpace : LinearSRGBColorSpace` — it ignores
    // `target.texture.colorSpace` entirely for a non-XR target. So the scene
    // lands here unconverted no matter what this texture claims, and the
    // pass-through quad is a raw ShaderMaterial with no colorspace include of
    // its own, which would write linear values onto an sRGB canvas.
    //
    // The symptom is not an error, but a uniformly dark image. Toggling the
    // pass off with `p` remains the quickest way to distinguish colour-space
    // handling from a scene-material problem.

    this.material = new ShaderMaterial({
      uniforms: {
        tDiffuse: { value: this.target.texture },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      depthTest: false,
      depthWrite: false,
    });
    this.quadScene.add(new Mesh(new PlaneGeometry(2, 2), this.material));
  }

  private internalSize(): { width: number; height: number } {
    const size = this.renderer.getDrawingBufferSize(new Vector2());
    return {
      width: Math.max(1, Math.round(size.x)),
      height: Math.max(1, Math.round(size.y)),
    };
  }

  resize(): void {
    const { width, height } = this.internalSize();
    this.target.setSize(width, height);
  }

  render(scene: Scene, camera: Camera): void {
    if (!this.enabled) {
      this.renderer.setRenderTarget(null);
      this.renderer.render(scene, camera);
      return;
    }
    this.renderer.setRenderTarget(this.target);
    this.renderer.clear();
    this.renderer.render(scene, camera);

    this.renderer.setRenderTarget(null);
    this.renderer.render(this.quadScene, this.quadCamera);
  }
}
