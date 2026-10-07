import * as THREE from "three";
import vertexShader from "./shaders/track.vert.glsl?raw";
import fragmentShader from "./shaders/track.frag.glsl?raw";

/**
 * Shader asphalt skin over exact truth-road vertices.
 * Pace / fable_vref is never painted — edges and wear only.
 */
export function createTrackMaterial(uniforms = {}) {
  const material = new THREE.ShaderMaterial({
    name: "observatory_truth_track_shader",
    // UniformsLib.fog is mandatory alongside `fog: true` — a ShaderMaterial
    // does not get the fog uniforms for free, and three throws inside
    // refreshFogUniforms the first time it renders without them.
    uniforms: THREE.UniformsUtils.merge([
      THREE.UniformsLib.fog,
      {
        wetness: { value: 0 },
        // Lifted dry albedo so mid-distance asphalt hazes with FogExp2 instead
        // of crushing toward black before the fog factor has much weight.
        dryColor: { value: new THREE.Color(0x4a5258) },
        wetColor: { value: new THREE.Color(0x1e2830) },
        edgeLineColor: { value: new THREE.Color(0xece8dc) },
        lightDirection: { value: new THREE.Vector3(0.35, 0.85, 0.3).normalize() },
        lightColor: { value: new THREE.Color(0xfff0d4) },
        lightIntensity: { value: 1.45 },
        ambientColor: { value: new THREE.Color(0.42, 0.46, 0.48) },
        ...uniforms,
      },
    ]),
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
    // Fog is required, not optional: without it the road stays a hard black
    // ribbon running to the horizon while the terrain around it hazes out.
    fog: true,
  });
  material.userData.truthBoundary = "shader-only visual skin over exact runtime track vertices";
  return material;
}

export function setTrackWetness(material, wetness) {
  if (material?.uniforms?.wetness) {
    material.uniforms.wetness.value = THREE.MathUtils.clamp(Number(wetness) || 0, 0, 1);
  }
}

export function setTrackPalette(material, { dry, wet, edge, night = false } = {}) {
  if (!material?.uniforms) return;
  if (dry != null) material.uniforms.dryColor.value.setHex(dry);
  if (wet != null) material.uniforms.wetColor.value.setHex(wet);
  if (edge != null) material.uniforms.edgeLineColor.value.setHex(edge);
  if (night) {
    material.uniforms.dryColor.value.setHex(dry ?? 0x1c242a);
    material.uniforms.edgeLineColor.value.setHex(edge ?? 0xc8c4b8);
  }
}

/** Push sun direction + colors from SkyRig into the asphalt shader. */
export function setTrackLighting(material, {
  direction,
  color,
  intensity = 1.4,
  ambient = 0x485056,
} = {}) {
  if (!material?.uniforms) return;
  if (direction) material.uniforms.lightDirection.value.copy(direction).normalize();
  if (color != null) {
    if (typeof color === "number") material.uniforms.lightColor.value.setHex(color);
    else material.uniforms.lightColor.value.copy(color);
  }
  material.uniforms.lightIntensity.value = intensity;
  if (typeof ambient === "number") material.uniforms.ambientColor.value.setHex(ambient);
  else if (ambient?.isColor) material.uniforms.ambientColor.value.copy(ambient);
}
