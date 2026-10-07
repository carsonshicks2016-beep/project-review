import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { FilmPass } from "three/addons/postprocessing/FilmPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { condition } from "./conditions.js";
import { InkOutlinePass } from "./outline.js";

/**
 * Stylized finish. The composer is the ONLY render path — the stylized look
 * needs a screen-space stage for the ink outline and the grade, so there is no
 * direct-render bypass any more.
 *
 * Historical note, because it cost a while to unpick: this used to skip the
 * composer entirely on clear day, with a comment blaming three@0.185 for
 * "blacking out" under ACES. The real cause was the bypass's own workaround —
 * it set `renderer.toneMapping = NoToneMapping` around `composer.render()`.
 * three already disables tone mapping when a material renders into a render
 * target, so OutputPass is what applies it, and OutputPass picks its tone-map
 * define by *reading `renderer.toneMapping` at render time*. Lying to it left
 * the frame un-tonemapped and washed out. Never touch `renderer.toneMapping`
 * around the composer.
 */
export class CinematicPipeline {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    // MSAA inside the composer. The stylized look leans on hard colour
    // boundaries, and post-AA (FXAA/SMAA) smears them; multisampling keeps
    // banding terminators and ink edges crisp.
    const target = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      samples: 4,
    });
    this.composer = new EffectComposer(renderer, target);
    this.renderPass = new RenderPass(scene, camera);
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.18, 0.5, 0.85);
    this.filmPass = new FilmPass(0.04, false);
    this.outputPass = new OutputPass();
    this.outlinePass = new InkOutlinePass(scene, camera);
    this.composer.addPass(this.renderPass);
    // Ink before bloom, so the lines are graphic and do not themselves glow.
    this.composer.addPass(this.outlinePass);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(this.filmPass);
    this.composer.addPass(this.outputPass);
    this.quality = "high";
    this.bloom = 0.18;
    this.grain = 0.04;
    this.sync();
  }

  setCondition(id) {
    const c = condition(id);
    // Bloom is a light-source glow here, not a haze: strongest where there are
    // actual emitters to bloom (dusk sun, night lights, wet specular).
    this.bloom = c.light === "night" ? 0.34
      : c.light === "dusk" ? 0.30
        : c.wetness > 0.5 ? 0.24
          : 0.16;
    this.grain = c.light === "night" ? 0.09
      : c.light === "dusk" ? 0.06
        : c.rain > 0 ? 0.06
          : 0.035;
    this.sync();
  }

  setQuality(level) {
    this.quality = level;
    this.sync();
  }

  sync() {
    const low = this.quality === "low";
    this.bloomPass.strength = low ? 0 : this.bloom;
    this.bloomPass.radius = 0.5;
    this.bloomPass.threshold = 0.82;
    this.filmPass.uniforms.intensity.value = low ? 0 : this.grain;
    this.bloomPass.enabled = !low && this.bloomPass.strength > 0.01;
    this.filmPass.enabled = !low && this.grain > 0.005;
    this.outlinePass.setQuality(this.quality);
  }

  setSize(width, height, pixelRatio) {
    this.composer.setPixelRatio(pixelRatio);
    this.composer.setSize(width, height);
    this.outlinePass.setSize(width * pixelRatio, height * pixelRatio);
  }

  render(delta) {
    this.composer.render(delta);
  }
}
