import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

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
  }

  setWeatherStrength(value) {
    this.weatherStrength = THREE.MathUtils.clamp(Number(value) || 0, 0, 0.5);
    this.sync();
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
    if (this.enabled) this.composer.render(delta);
    else this.renderer.render(this.renderPass.scene, this.renderPass.camera);
  }
}
