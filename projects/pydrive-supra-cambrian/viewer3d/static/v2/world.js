// World assembler: terrain + mountains + atmosphere + road + furniture.
// This module stays the single place that composes the scene; each system
// lives in its own builder. buildWorld returns an update(carPos) hook for
// systems that follow the car (the lamp light pool).
import * as THREE from "../vendor/three.module.min.js";
import { createCorridor } from "./corridor.js";
import { buildTerrain } from "./terrain.js";
import { buildAtmosphere } from "./atmosphere.js";
import { buildMountains } from "./mountains.js";
import { buildRoad } from "./road.js";
import { buildFurniture } from "./furniture.js";
import { buildVegetation } from "./vegetation.js";

const TAG = "world";

export function buildWorld(trackMsg, mood) {
  const group = new THREE.Group();
  group.name = "world-v2";
  const corridor = createCorridor(trackMsg);

  group.add(buildTerrain(corridor, mood));
  group.add(buildMountains(corridor, mood));
  group.add(buildAtmosphere(corridor, mood));
  group.add(buildRoad(corridor, mood));
  group.add(buildVegetation(corridor, mood));

  const furniture = buildFurniture(corridor, mood);
  group.add(furniture.group);

  return { group, corridor, update: furniture.update };
}

export function disposeWorld(world, mood) {
  mood.clearTag(TAG);
  world.group.traverse((o) => {
    if (o.isMesh || o.isPoints) {
      o.geometry.dispose();
      if (o.material) {
        if (o.material.map) o.material.map.dispose();
        o.material.dispose();
      }
    }
  });
}
