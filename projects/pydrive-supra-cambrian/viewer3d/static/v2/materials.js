// Mood controller: modules register (object, property, palette-key) bindings,
// and day/night toggles become a smooth lerp across every registered value.
// Replaces legacy's hand-written per-material if-chains with data.
import * as THREE from "../vendor/three.module.min.js";
import { MOODS, V2 } from "./palette.js";

const _a = new THREE.Color();
const _b = new THREE.Color();

export function createMoodController(initial = "night") {
  const colorBindings = [];   // obj[prop] is a THREE.Color (lerped in place)
  const numberBindings = [];  // obj[prop] is a number
  const vec3Bindings = [];    // obj[prop] is a THREE.Vector3, palette holds [x,y,z]
  let from = initial;
  let to = initial;
  let t = 1;

  function apply(mix) {
    const A = MOODS[from];
    const B = MOODS[to];
    for (const { obj, prop, key } of colorBindings) {
      _a.set(A[key]);
      _b.set(B[key]);
      obj[prop].copy(_a.lerp(_b, mix));
    }
    for (const { obj, prop, key } of numberBindings) {
      obj[prop] = A[key] + (B[key] - A[key]) * mix;
    }
    for (const { obj, prop, key } of vec3Bindings) {
      const a = A[key];
      const b = B[key];
      obj[prop].set(
        a[0] + (b[0] - a[0]) * mix,
        a[1] + (b[1] - a[1]) * mix,
        a[2] + (b[2] - a[2]) * mix,
      );
    }
  }

  const mood = {
    get current() {
      return to;
    },
    // tag lets a rebuildable group (the world) drop its bindings on dispose
    bindColor(obj, prop, key, tag = null) {
      colorBindings.push({ obj, prop, key, tag });
    },
    bindNumber(obj, prop, key, tag = null) {
      numberBindings.push({ obj, prop, key, tag });
    },
    bindVec3(obj, prop, key, tag = null) {
      vec3Bindings.push({ obj, prop, key, tag });
    },
    clearTag(tag) {
      for (const arr of [colorBindings, numberBindings, vec3Bindings]) {
        for (let i = arr.length - 1; i >= 0; i--) {
          if (arr[i].tag === tag) arr.splice(i, 1);
        }
      }
    },
    set(name, instant = false) {
      if (!MOODS[name] || (name === to && !instant)) return;
      from = instant ? name : to;
      to = name;
      t = instant ? 1 : 0;
      if (instant) apply(1);
    },
    toggle() {
      mood.set(to === "night" ? "day" : "night");
      return to;
    },
    update(dt) {
      if (t >= 1) return;
      t = Math.min(1, t + dt / V2.moodFade);
      const s = t * t * (3 - 2 * t);
      apply(s);
    },
    refresh() {
      apply(t >= 1 ? 1 : t * t * (3 - 2 * t));
    },
  };
  return mood;
}
