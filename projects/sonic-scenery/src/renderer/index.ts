/**
 * Agent D — Three.js renderer (Electron renderer process).
 * Brief: docs/tasks/agent-D-renderer.md
 *
 *  - Bootstrap Three.js (WebGL2): scene, camera, lights, fog, bloom post-fx.
 *  - Consume a WorldSpec → terrain + GPU-instanced scatter + particles, and
 *    mount the weather (Agent E) and creature (Agent F) systems.
 *  - Connect to AUDIO_WS_URL; apply WorldSpec.modifierCurves to each AudioFrame
 *    (centroid→brightness, energy→particles, bass→lightning hooks). Runs
 *    gracefully when the WS is offline (self-animating scene, no crash).
 *  - Crossfade/dissolve (~3s) when the WorldSpec is swapped (track change).
 */
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { AUDIO_WS_URL, type WorldSpec } from "../contracts";
import { AudioClient } from "./audioClient";
import { WorldView } from "./worldView";
import { DEMO_WORLD } from "./demoWorld";

const CROSSFADE_SECONDS = 3.0;

export interface RendererHandle {
  /** Swap to a new WorldSpec with a crossfade/dissolve (no hard cut). */
  swapWorld(spec: WorldSpec): void;
  /** Tear everything down. */
  dispose(): void;
}

interface Fade {
  outgoing: WorldView;
  elapsed: number;
}

export function bootstrap(
  canvas: HTMLCanvasElement,
  initial: WorldSpec = DEMO_WORLD,
): RendererHandle {
  // ---- Renderer (WebGL2) ----------------------------------------------------
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  const scene = new THREE.Scene();

  // ---- Camera ---------------------------------------------------------------
  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    2000,
  );
  camera.position.set(0, 24, 90);
  camera.lookAt(0, 6, 0);

  // ---- Lights (re-tinted per world) ----------------------------------------
  const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 0.6);
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(40, 80, 30);
  const ambient = new THREE.AmbientLight(0xffffff, 0.25);
  scene.add(hemi, sun, ambient);

  // ---- Post-processing: bloom ----------------------------------------------
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.7, // strength
    0.7, // radius
    0.85, // threshold
  );
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  composer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  composer.setSize(window.innerWidth, window.innerHeight);

  // ---- World(s) -------------------------------------------------------------
  let current = new WorldView(initial);
  scene.add(current.group);
  let fade: Fade | null = null;

  function applyEnvironment(view: WorldView): void {
    const p = view.palette;
    scene.background = p.sky;
    scene.fog = new THREE.Fog(
      p.fog.getHex(),
      80,
      420 + (1 - view.day) * 120, // night = nearer fog
    );
    hemi.color.copy(p.sky);
    hemi.groundColor.copy(p.secondary);
    hemi.intensity = 0.35 + view.day * 0.5;
    sun.color.copy(p.sun);
    sun.intensity = 0.5 + view.day * 1.3;
    // Sun arcs with time of day; sits low (warm) at dawn/dusk.
    const ang = view.spec.timeOfDay * Math.PI; // 0..pi across the day
    sun.position.set(Math.cos(ang) * 80, 20 + Math.sin(ang) * 90, 40);
    ambient.color.copy(p.ambient);
    ambient.intensity = 0.15 + view.day * 0.15;
    // Brighter biomes (neon) get a touch more bloom.
    bloom.strength = view.spec.biome.includes("edm") ? 1.1 : 0.7;
  }
  applyEnvironment(current);

  // ---- Audio client ---------------------------------------------------------
  const audio = new AudioClient(AUDIO_WS_URL);
  audio.connect();

  // ---- Render loop ----------------------------------------------------------
  const clock = new THREE.Clock();
  let raf = 0;
  let disposed = false;

  function frame(): void {
    if (disposed) return;
    raf = requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.1);
    const elapsed = clock.elapsedTime;
    const now = performance.now();
    const af = audio.frame(now);

    current.update(dt, elapsed, af);

    // Crossfade between the outgoing and incoming worlds.
    if (fade) {
      fade.elapsed += dt;
      const t = THREE.MathUtils.clamp(fade.elapsed / CROSSFADE_SECONDS, 0, 1);
      const e = THREE.MathUtils.smoothstep(t, 0, 1);
      fade.outgoing.update(dt, elapsed, af);
      fade.outgoing.setOpacity(1 - e);
      current.setOpacity(e);
      if (t >= 1) {
        fade.outgoing.dispose();
        current.setOpacity(1);
        fade = null;
      }
    }

    // Slow idle orbit so the standalone scene is never static.
    const orbit = elapsed * 0.05;
    const radius = 95;
    camera.position.x = Math.sin(orbit) * radius;
    camera.position.z = Math.cos(orbit) * radius;
    camera.position.y = current.cameraHeight + 10;
    camera.lookAt(0, 6, 0);

    composer.render();
  }
  frame();

  // ---- Resize ---------------------------------------------------------------
  function onResize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
  }
  window.addEventListener("resize", onResize);

  return {
    swapWorld(spec: WorldSpec): void {
      // If a fade is already running, finish it instantly to avoid stacking.
      if (fade) {
        fade.outgoing.dispose();
        fade = null;
      }
      const incoming = new WorldView(spec);
      incoming.setOpacity(0);
      scene.add(incoming.group);
      fade = { outgoing: current, elapsed: 0 };
      current = incoming;
      applyEnvironment(current);
    },
    dispose(): void {
      disposed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      audio.dispose();
      if (fade) fade.outgoing.dispose();
      current.dispose();
      composer.dispose();
      renderer.dispose();
    },
  };
}
