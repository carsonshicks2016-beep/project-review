import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { resolveVisualCondition } from "./visual-conditions.js";

/**
 * Bloom is a dusk/night/rain accent only. Clear daylight must stay on the
 * direct renderer path — enabling EffectComposer+OutputPass at tiny strengths
 * blacks out the viewport under three@0.185 with ACES on the renderer.
 */
export class CinematicPipeline {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.composer = new EffectComposer(renderer);
    this.renderPass = new RenderPass(scene, camera);
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 0, 0.42, 0.91);
    this.outputPass = new OutputPass();
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(this.outputPass);
    this.quality = "high";
    this.weatherStrength = 0;
    this.enabled = false;
    this.sync();
  }

  setWeatherStrength(value) {
    this.weatherStrength = THREE.MathUtils.clamp(Number(value) || 0, 0, 0.5);
    this.sync();
  }

  setVisualCondition(condition) {
    const resolved = resolveVisualCondition(condition);
    // Match the pre-overhaul gate: clear day → strength 0 → direct render.
    this.setWeatherStrength(
      resolved.lighting === "dusk" ? 0.18
        : resolved.lighting === "night" ? 0.15
          : resolved.precipitation === "rain" ? 0.04
            : 0,
    );
  }

  setQuality(level) {
    this.quality = level;
    this.sync();
  }

  sync() {
    const qualityScale = this.quality === "high" ? 1 : this.quality === "medium" ? 0.45 : 0;
    this.bloomPass.strength = this.weatherStrength * qualityScale;
    this.bloomPass.radius = 0.34;
    this.bloomPass.threshold = 0.88;
    this.enabled = this.bloomPass.strength > 0.015;
  }

  setSize(width, height, pixelRatio) {
    this.composer.setPixelRatio(pixelRatio);
    this.composer.setSize(width, height);
  }

  render(delta) {
    if (this.enabled) {
      // Avoid double tone-mapping when OutputPass is in the chain.
      const previous = this.renderer.toneMapping;
      this.renderer.toneMapping = THREE.NoToneMapping;
      this.composer.render(delta);
      this.renderer.toneMapping = previous;
    } else {
      this.renderer.render(this.renderPass.scene, this.renderPass.camera);
    }
  }
}
