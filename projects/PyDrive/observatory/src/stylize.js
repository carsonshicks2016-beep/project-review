import * as THREE from "three";

/**
 * Committed-stylized shading, layered onto MeshStandardMaterial.
 *
 * The Observatory's world is procedurally generated geometry with no textures,
 * so it will never survive photoreal scrutiny. Instead it commits to a graphic
 * "instrument-grade" read that extends the UI's cold-signal identity: direct
 * light quantized into a few hard steps, plus an ice-cyan fresnel rim that
 * separates the car from the asphalt.
 *
 * This is done with `onBeforeCompile` rather than by swapping in
 * MeshToonMaterial, because world.js already does careful per-mesh material
 * work (styleVisualMesh, softenTerrainMaterial, wetness response) and three's
 * shadow / fog / IBL plumbing has to keep working. We layer, we don't replace.
 *
 * Injection points (order verified against three 0.185's physical shader):
 *   lights_fragment_end -> band `reflectedLight.directDiffuse`, compute rim
 *   opaque_fragment     -> add the rim into `outgoingLight`
 * `geometryNormal` and `geometryViewDir` are both in scope at the first, and
 * are already view-space, which is what the fresnel term wants.
 */

const BAND_FUNCTION = /* glsl */`
  // Quantize with a controllable terminator: softness 0.5 is a linear ramp,
  // small values give hard steps.
  float supraBand(float x, float bands, float softness) {
    float s = clamp(x, 0.0, 1.0) * bands;
    float i = floor(s);
    float f = fract(s);
    float e = smoothstep(0.5 - softness, 0.5 + softness, f);
    return (i + e) / bands;
  }
`;

export const STYLIZE_DEFAULTS = Object.freeze({
  bands: 4.0,
  softness: 0.09,
  // Ceiling of the *lighting factor* (direct light with albedo divided out),
  // which is roughly `sunIntensity / PI * NdotL` and so tops out near 1.
  range: 1.0,
  rimColor: 0x5ee1ff,
  rimStrength: 0.0,
  rimStart: 0.55,
  rimPower: 2.2,
});

let cacheKeySeed = 0;

/**
 * Layer stylized shading onto a material. Safe to call once per material;
 * repeat calls just update the uniforms.
 */
export function stylize(material, options = {}) {
  if (!material || material.isShaderMaterial) return material;
  const settings = { ...STYLIZE_DEFAULTS, ...options };

  if (material.userData.stylizeUniforms) {
    setStylize(material, settings);
    return material;
  }

  const uniforms = {
    uBands: { value: settings.bands },
    uSoftness: { value: settings.softness },
    uRange: { value: settings.range },
    uRimColor: { value: new THREE.Color(settings.rimColor) },
    uRimStrength: { value: settings.rimStrength },
    uRimStart: { value: settings.rimStart },
    uRimPower: { value: settings.rimPower },
  };
  const terrain = settings.terrain || null;
  if (terrain) {
    uniforms.uGrassLow = { value: new THREE.Color(terrain.low) };
    uniforms.uGrassHigh = { value: new THREE.Color(terrain.high) };
    uniforms.uSlopeColor = { value: new THREE.Color(terrain.slope) };
    uniforms.uHeightLow = { value: terrain.heightLow };
    uniforms.uHeightHigh = { value: terrain.heightHigh };
    uniforms.uSlopeStart = { value: terrain.slopeStart ?? 0.25 };
    uniforms.uTerrainSteps = { value: terrain.steps ?? 4.0 };
  }
  material.userData.stylizeUniforms = uniforms;

  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);

    if (terrain) {
      // Landform palette. A single flat green makes the hills read as one dead
      // slab; keying colour to elevation and steepness is what makes them read
      // as sculpted, and quantizing it keeps the result graphic.
      shader.vertexShader = shader.vertexShader
        .replace(
          "void main() {",
          "varying vec3 vTerrainWorld;\nvarying vec3 vTerrainNormal;\nvoid main() {",
        )
        .replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
           vTerrainWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;
           vTerrainNormal = normalize(mat3(modelMatrix) * objectNormal);`,
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "void main() {",
          `varying vec3 vTerrainWorld;
           varying vec3 vTerrainNormal;
           uniform vec3 uGrassLow;
           uniform vec3 uGrassHigh;
           uniform vec3 uSlopeColor;
           uniform float uHeightLow;
           uniform float uHeightHigh;
           uniform float uSlopeStart;
           uniform float uTerrainSteps;
           void main() {`,
        )
        .replace(
          "#include <map_fragment>",
          `#include <map_fragment>
           {
             float h = clamp((vTerrainWorld.y - uHeightLow) / max(1.0, uHeightHigh - uHeightLow), 0.0, 1.0);
             h = supraBand(h, uTerrainSteps, 0.12);
             vec3 ground = mix(uGrassLow, uGrassHigh, h);
             float steep = 1.0 - clamp(normalize(vTerrainNormal).y, 0.0, 1.0);
             float rock = supraBand(smoothstep(uSlopeStart, uSlopeStart + 0.28, steep), 3.0, 0.14);
             diffuseColor.rgb = mix(ground, uSlopeColor, rock);
           }`,
        );
    }

    shader.fragmentShader = shader.fragmentShader
      .replace(
        "void main() {",
        `${BAND_FUNCTION}
        uniform float uBands;
        uniform float uSoftness;
        uniform float uRange;
        uniform vec3 uRimColor;
        uniform float uRimStrength;
        uniform float uRimStart;
        uniform float uRimPower;
        void main() {`,
      )
      .replace(
        "#include <lights_fragment_end>",
        /* glsl */`
        #include <lights_fragment_end>
        float supraRim = 0.0;
        {
          // Divide the albedo out first, so we quantize the *light* and not the
          // surface colour. Banding directDiffuse raw makes the step positions
          // depend on how bright the material happens to be, which crushes dark
          // surfaces and blows out pale ones.
          vec3 albedo = max(diffuseColor.rgb, vec3(1e-4));
          vec3 factor = reflectedLight.directDiffuse / albedo;
          float f = dot(factor, vec3(0.2126, 0.7152, 0.0722));
          if (f > 1e-5) {
            float stepped = supraBand(f / uRange, uBands, uSoftness) * uRange;
            reflectedLight.directDiffuse = albedo * factor * (stepped / f);
          }
          float ndv = clamp(dot(geometryNormal, geometryViewDir), 0.0, 1.0);
          supraRim = pow(smoothstep(uRimStart, 1.0, 1.0 - ndv), uRimPower) * uRimStrength;
        }
        `,
      )
      .replace(
        "#include <opaque_fragment>",
        /* glsl */`
        outgoingLight += uRimColor * supraRim;
        #include <opaque_fragment>
        `,
      );
  };

  // Without this three can hand a stylized material a program compiled for a
  // plain one, because the cache key only covers built-in parameters.
  cacheKeySeed += 1;
  const key = `supra-stylize-${cacheKeySeed}`;
  material.customProgramCacheKey = () => key;
  material.needsUpdate = true;
  return material;
}

/** Update stylize uniforms in place (palette / weather changes). */
export function setStylize(material, patch = {}) {
  const uniforms = material?.userData?.stylizeUniforms;
  if (!uniforms) return;
  if (patch.bands !== undefined) uniforms.uBands.value = patch.bands;
  if (patch.softness !== undefined) uniforms.uSoftness.value = patch.softness;
  if (patch.range !== undefined) uniforms.uRange.value = patch.range;
  if (patch.rimColor !== undefined) uniforms.uRimColor.value.set(patch.rimColor);
  if (patch.rimStrength !== undefined) uniforms.uRimStrength.value = patch.rimStrength;
  if (patch.rimStart !== undefined) uniforms.uRimStart.value = patch.rimStart;
  if (patch.rimPower !== undefined) uniforms.uRimPower.value = patch.rimPower;
}

/** Apply to every material under an object graph. */
export function stylizeTree(root, options = {}) {
  root?.traverse?.((node) => {
    if (!node.isMesh || !node.material) return;
    const list = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of list) stylize(material, options);
  });
  return root;
}

/** Update every stylized material under an object graph. */
export function setStylizeTree(root, patch = {}) {
  root?.traverse?.((node) => {
    if (!node.isMesh || !node.material) return;
    const list = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of list) setStylize(material, patch);
  });
  return root;
}
