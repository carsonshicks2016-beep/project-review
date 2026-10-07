/**
 * Roadside vegetation and obstacles.
 *
 * Two different things share this file because they share a technique:
 *
 * - **Obstacles** come from the stage document. They are solid, the sim can hit
 *   them, and they are drawn at exactly the `radius` the physics uses. What you
 *   see is what stops the car.
 * - **Scenery** is invented here. The forest wall either side of the corridor
 *   is not in the contract and nothing can collide with it — it exists because
 *   a stage lined with thirty cones reads as a diagram, and a stage lined with
 *   a wall of trees reads as a road through a forest.
 *
 * The distinction matters and is worth keeping straight: **if a tree can hit
 * you it came from the stage; if it cannot, it came from here.** Scenery is
 * always placed beyond the corridor edge plus a margin, so it can never be
 * mistaken for something drivable.
 *
 * Both use a small number of shared, instanced low-poly parts. The forest stays
 * cheap enough to be dense, but its character now comes from silhouettes,
 * facets and lighting rather than enlarged texture pixels.
 */

import {
  BoxGeometry,
  CircleGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DodecahedronGeometry,
  DoubleSide,
  Group,
  IcosahedronGeometry,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  PlaneGeometry,
  Quaternion,
  Vector3,
  type BufferGeometry,
} from "three";

import { headingDir, rightDir, xyz } from "../coords";
import type { CenterlinePoint, Obstacle, Stage } from "../replay";
import { terrainFall } from "./stage";
import { textures } from "./textures";

/** Deliberately hideous: an unhandled obstacle kind should be impossible to miss. */
const UNKNOWN = 0xff00ff;

/** Scenery density and placement, in metres from the corridor edge. */
const SCENERY_SPACING_M = 5.2;
const SCENERY_MIN_OFFSET_M = 2.6;
const SCENERY_DEPTH_M = 56.0;
/** Rows of forest behind the first. Depth is what stops it reading as a fence. */
const SCENERY_ROWS = 5;

function prng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

interface VegetationPart {
  suffix: string;
  geometry: BufferGeometry;
  material: MeshLambertMaterial;
}

function facetedMaterial(colour: number): MeshLambertMaterial {
  return new MeshLambertMaterial({
    color: colour,
    flatShading: true,
  });
}

/**
 * Bake a part's local proportions and offset into shared unit-height geometry.
 *
 * Instance matrices can then remain one uniform scale plus yaw, which preserves
 * the existing meaning of `Placement.scale`: nominal vegetation height.
 */
function shape(
  geometry: BufferGeometry,
  scale: readonly [number, number, number],
  offset: readonly [number, number, number],
): BufferGeometry {
  geometry.scale(scale[0], scale[1], scale[2]);
  geometry.translate(offset[0], offset[1], offset[2]);
  geometry.computeBoundingSphere();
  return geometry;
}

/**
 * Shared low-poly vegetation kits.
 *
 * Conifers use overlapping six-sided crowns rather than one perfect cone.
 * Broadleaves and bushes mix dodecahedra and icosahedra so their outline changes
 * with the camera instead of collapsing to a flat cutout. A random instance yaw
 * supplies variation without multiplying geometry or draw calls.
 */
const CONIFER_PARTS: readonly VegetationPart[] = [
  {
    suffix: "trunk",
    geometry: shape(
      new CylinderGeometry(0.038, 0.055, 0.48, 6),
      [1, 1, 1],
      [0, 0.24, 0],
    ),
    material: facetedMaterial(0x493624),
  },
  {
    suffix: "crown_low",
    geometry: shape(
      new ConeGeometry(0.31, 0.46, 6),
      [1, 1, 0.9],
      [0, 0.50, 0],
    ),
    material: facetedMaterial(0x24472f),
  },
  {
    suffix: "crown_mid",
    geometry: shape(
      new ConeGeometry(0.245, 0.42, 6),
      [1, 1, 0.92],
      [0, 0.68, 0],
    ),
    material: facetedMaterial(0x315a39),
  },
  {
    suffix: "crown_top",
    geometry: shape(
      new ConeGeometry(0.17, 0.36, 6),
      [1, 1, 0.94],
      [0, 0.82, 0],
    ),
    material: facetedMaterial(0x3d6844),
  },
];

const BROADLEAF_PARTS: readonly VegetationPart[] = [
  {
    suffix: "trunk",
    geometry: shape(
      new CylinderGeometry(0.045, 0.065, 0.56, 6),
      [1, 1, 1],
      [0, 0.28, 0],
    ),
    material: facetedMaterial(0x513a26),
  },
  {
    suffix: "crown_core",
    geometry: shape(
      new DodecahedronGeometry(0.34, 0),
      [1.05, 0.80, 0.94],
      [0, 0.73, 0],
    ),
    material: facetedMaterial(0x315b35),
  },
  {
    suffix: "crown_left",
    geometry: shape(
      new IcosahedronGeometry(0.26, 0),
      [1, 0.82, 0.92],
      [-0.25, 0.69, 0.025],
    ),
    material: facetedMaterial(0x3f6b3b),
  },
  {
    suffix: "crown_right",
    geometry: shape(
      new DodecahedronGeometry(0.255, 0),
      [1, 0.86, 0.96],
      [0.25, 0.72, -0.02],
    ),
    material: facetedMaterial(0x527943),
  },
  {
    suffix: "crown_top",
    geometry: shape(
      new IcosahedronGeometry(0.22, 0),
      [0.94, 0.80, 0.92],
      [0.02, 0.87, 0.025],
    ),
    material: facetedMaterial(0x62864b),
  },
];

const BUSH_PARTS: readonly VegetationPart[] = [
  {
    suffix: "core",
    geometry: shape(
      new DodecahedronGeometry(0.42, 0),
      [1.12, 0.94, 0.96],
      [0, 0.48, 0],
    ),
    material: facetedMaterial(0x31572f),
  },
  {
    suffix: "left",
    geometry: shape(
      new IcosahedronGeometry(0.34, 0),
      [1.10, 0.84, 0.95],
      [-0.38, 0.39, 0.025],
    ),
    material: facetedMaterial(0x3d6736),
  },
  {
    suffix: "right",
    geometry: shape(
      new DodecahedronGeometry(0.36, 0),
      [1.10, 0.82, 0.98],
      [0.37, 0.40, -0.025],
    ),
    material: facetedMaterial(0x4b7640),
  },
  {
    suffix: "top",
    geometry: shape(
      new IcosahedronGeometry(0.26, 0),
      [1, 0.88, 0.92],
      [0.04, 0.73, 0.04],
    ),
    material: facetedMaterial(0x5a8248),
  },
];

function headingAt(pts: CenterlinePoint[], i: number): number {
  const a = pts[Math.max(0, i - 1)]!;
  const b = pts[Math.min(pts.length - 1, i + 1)]!;
  return Math.atan2(b.y - a.y, b.x - a.x);
}

interface Placement {
  pos: Vector3;
  scale: number;
  /** Random yaw exposes different facets and breaks up repeated silhouettes. */
  yaw: number;
  /** One of three deterministic colour/atlas slices. */
  variant: 0 | 1 | 2;
  /** Only the last fog rows may collapse to crossed texture cards. */
  far: boolean;
  /** Near vegetation receives a cheap hard polygonal ground shadow. */
  shadow: boolean;
}

/**
 * Lay scenery down both verges.
 *
 * Placed relative to the corridor's own edge at each sample, so it follows the
 * road through corners and never crowds a narrow section — the corridor is
 * 12 m wide at tier 0 and 5.5 m at tier 5, and scenery at a fixed world offset
 * would sit in the road on one and a field away on the other.
 */
function placeScenery(
  stage: Stage,
  seed: number,
): { conifers: Placement[]; broadleaves: Placement[]; bushes: Placement[] } {
  const pts = stage.centerline;
  const conifers: Placement[] = [];
  const broadleaves: Placement[] = [];
  const bushes: Placement[] = [];
  const r = prng(seed);

  const centre = new Vector3();
  const right = new Vector3();
  const forward = new Vector3();

  let nextS = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    if (p.s < nextS) continue;
    nextS = p.s + SCENERY_SPACING_M * (0.55 + r() * 0.9);

    const h = headingAt(pts, i);
    xyz(p.x, p.y, p.z, centre);
    rightDir(h, right);
    headingDir(h, forward);
    const half = p.width / 2;

    for (const side of [-1, 1]) {
      for (let row = 0; row < SCENERY_ROWS; row++) {
        // Front row dense, back rows sparser: the wall should thin out into fog
        // rather than end.
        if (row > 0 && r() > 0.75) continue;
        const depth =
          SCENERY_MIN_OFFSET_M +
          (row / SCENERY_ROWS) * SCENERY_DEPTH_M +
          r() * (SCENERY_DEPTH_M / SCENERY_ROWS);
        const lateral = side * (half + depth);
        const pos = new Vector3()
          .copy(centre)
          .addScaledVector(right, lateral * Math.cos(p.camber))
          .addScaledVector(forward, (r() - 0.5) * SCENERY_SPACING_M * 0.9);
        pos.y =
          centre.y + lateral * Math.sin(p.camber) - terrainFall(depth);

        if (row === 0 && r() > 0.66) {
          bushes.push({
            pos,
            scale: 1.3 + r() * 1.8,
            yaw: r() * Math.PI,
            variant: Math.floor(r() * 3) as 0 | 1 | 2,
            far: false,
            shadow: depth < 13,
          });
        } else {
          const yaw = r() * Math.PI;
          const placement = {
            pos,
            yaw,
            variant: Math.floor(r() * 3) as 0 | 1 | 2,
            far: depth > 38,
            shadow: depth < 16,
          };
          if (r() < 0.52) {
            conifers.push({
              ...placement,
              scale: 6.5 + r() * 8.5,
            });
          } else {
            broadleaves.push({
              ...placement,
              scale: 5.5 + r() * 6.5,
            });
          }
        }
      }
    }
  }
  return { conifers, broadleaves, bushes };
}

function instanceVegetation(
  places: Placement[],
  parts: readonly VegetationPart[],
  name: string,
  includeTrunk = true,
): Group | null {
  if (places.length === 0) return null;

  const group = new Group();
  group.name = name;
  const tex = textures();
  const m = new Matrix4();
  const rot = new Quaternion();
  const scale = new Vector3();
  const pos = new Vector3();
  const up = new Vector3(0, 1, 0);

  for (const part of parts) {
    if (!includeTrunk && part.suffix === "trunk") continue;
    for (const variant of [0, 1, 2] as const) {
      const subset = places.filter(
        (placement) => !placement.far && placement.variant === variant,
      );
      if (subset.length === 0) continue;
      const variantMaterial = part.material.clone();
      variantMaterial.map =
        part.suffix === "trunk"
          ? tex.bark[variant]
          : tex.foliage[variant];
      // The atlas already owns the deep base colour. Lift the old flat-shader
      // tint toward neutral so texture and tint do not multiply into black.
      variantMaterial.color.lerp(new Color(0xffffff), 0.68);
      variantMaterial.color.offsetHSL(
        variant === 0 ? -0.018 : variant === 2 ? 0.014 : 0,
        variant === 1 ? 0.04 : 0,
        variant === 0 ? -0.045 : variant === 2 ? 0.035 : 0,
      );
      variantMaterial.needsUpdate = true;
      const mesh = new InstancedMesh(
        part.geometry,
        variantMaterial,
        subset.length,
      );
      mesh.name = `${name}_${part.suffix}_variant_${variant}`;
      mesh.frustumCulled = false;
      subset.forEach((p, i) => {
        pos.copy(p.pos);
        scale.setScalar(p.scale);
        rot.setFromAxisAngle(up, p.yaw);
        mesh.setMatrixAt(i, m.compose(pos, rot, scale));
      });
      mesh.instanceMatrix.needsUpdate = true;
      group.add(mesh);
    }
  }

  return group.children.length > 0 ? group : null;
}

/**
 * Crossed foliage cards are reserved for the final forest rows, already inside
 * the dense fog band. Everything close enough to inspect remains dimensional.
 */
function buildFarForest(places: Placement[]): Group | null {
  const far = places.filter((placement) => placement.far);
  if (far.length === 0) return null;
  const group = new Group();
  group.name = "far_fog_tree_cards";
  const tex = textures();
  const geometry = new PlaneGeometry(0.72, 1);
  geometry.translate(0, 0.5, 0);
  const matrix = new Matrix4();
  const rotation = new Quaternion();
  const scale = new Vector3();
  const up = new Vector3(0, 1, 0);

  for (const variant of [0, 1, 2] as const) {
    const subset = far.filter((placement) => placement.variant === variant);
    if (subset.length === 0) continue;
    const material = new MeshBasicMaterial({
      map: tex.farTree[variant],
      transparent: true,
      alphaTest: 0.28,
      side: DoubleSide,
      fog: true,
      color: 0xffffff,
    });
    for (const cross of [0, Math.PI / 2]) {
      const mesh = new InstancedMesh(geometry, material, subset.length);
      mesh.name = `far_tree_variant_${variant}_cross_${cross === 0 ? 0 : 1}`;
      mesh.frustumCulled = false;
      subset.forEach((placement, index) => {
        rotation.setFromAxisAngle(up, placement.yaw + cross);
        scale.setScalar(placement.scale);
        mesh.setMatrixAt(
          index,
          matrix.compose(placement.pos, rotation, scale),
        );
      });
      mesh.instanceMatrix.needsUpdate = true;
      group.add(mesh);
    }
  }
  return group;
}

function buildVegetationShadows(places: Placement[]): InstancedMesh | null {
  const shadowed = places.filter(
    (placement) => placement.shadow && !placement.far,
  );
  if (shadowed.length === 0) return null;
  const geometry = new CircleGeometry(1, 6);
  geometry.rotateX(-Math.PI / 2);
  const mesh = new InstancedMesh(
    geometry,
    new MeshBasicMaterial({
      color: 0x182117,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
      fog: true,
    }),
    shadowed.length,
  );
  mesh.name = "nearby_tree_polygon_shadows";
  mesh.renderOrder = 1;
  mesh.frustumCulled = false;
  const matrix = new Matrix4();
  const rotation = new Quaternion();
  const scale = new Vector3();
  const up = new Vector3(0, 1, 0);
  const position = new Vector3();
  shadowed.forEach((placement, index) => {
    position.copy(placement.pos);
    position.y += 0.024;
    rotation.setFromAxisAngle(up, placement.yaw + 0.72);
    scale.set(placement.scale * 0.24, 1, placement.scale * 0.105);
    mesh.setMatrixAt(
      index,
      matrix.compose(position, rotation, scale),
    );
  });
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

/** Small non-colliding verge rocks, kept beyond the road-edge ditch. */
function buildVergeRocks(stage: Stage, seed: number): InstancedMesh | null {
  const points = stage.centerline;
  const random = prng(seed ^ 0x70c5);
  const placements: { pos: Vector3; yaw: number; scale: Vector3 }[] = [];
  const centre = new Vector3();
  const right = new Vector3();
  let nextS = 24;
  for (let i = 1; i < points.length - 1; i++) {
    const point = points[i]!;
    if (point.s < nextS) continue;
    nextS = point.s + 18 + random() * 29;
    if (random() > 0.64) continue;
    const heading = headingAt(points, i);
    xyz(point.x, point.y, point.z, centre);
    rightDir(heading, right);
    const side = random() > 0.5 ? 1 : -1;
    const depth = 1.15 + random() * 1.25;
    const lateral = side * (point.width / 2 + depth);
    const pos = new Vector3()
      .copy(centre)
      .addScaledVector(right, lateral * Math.cos(point.camber));
    pos.y =
      centre.y +
      lateral * Math.sin(point.camber) -
      terrainFall(depth) +
      0.15;
    placements.push({
      pos,
      yaw: random() * Math.PI,
      scale: new Vector3(
        0.28 + random() * 0.34,
        0.22 + random() * 0.25,
        0.30 + random() * 0.40,
      ),
    });
  }
  if (placements.length === 0) return null;

  const mesh = new InstancedMesh(
    new DodecahedronGeometry(0.5, 0),
    new MeshLambertMaterial({
      map: textures().rock,
      color: 0xb7b5a7,
      flatShading: true,
    }),
    placements.length,
  );
  mesh.name = "textured_verge_rocks";
  mesh.frustumCulled = false;
  const matrix = new Matrix4();
  const rotation = new Quaternion();
  const up = new Vector3(0, 1, 0);
  placements.forEach((placement, index) => {
    rotation.setFromAxisAngle(up, placement.yaw);
    mesh.setMatrixAt(
      index,
      matrix.compose(placement.pos, rotation, placement.scale),
    );
  });
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

/** Build everything roadside: stage obstacles, plus invented scenery. */
export function buildRoadside(stage: Stage): Group {
  const root = new Group();
  root.name = "roadside";

  const seed = (stage.generator?.tier ?? 0) * 7919 + stage.centerline.length;
  const { conifers, broadleaves, bushes } = placeScenery(stage, seed);

  const sceneryConifers = instanceVegetation(
    conifers,
    CONIFER_PARTS,
    "scenery_conifers",
  );
  const sceneryBroadleaves = instanceVegetation(
    broadleaves,
    BROADLEAF_PARTS,
    "scenery_broadleaves",
  );
  const sceneryBushes = instanceVegetation(
    bushes,
    BUSH_PARTS,
    "scenery_bushes",
  );
  const farForest = buildFarForest([...conifers, ...broadleaves]);
  const sceneryShadows = buildVegetationShadows([
    ...conifers,
    ...broadleaves,
    ...bushes,
  ]);
  const rocks = buildVergeRocks(stage, seed);

  // --- obstacles from the stage: solid, and drawn at their collision radius ---
  const obstacleTrees = stage.obstacles.filter((o) => o.kind === "tree");
  const others = stage.obstacles.filter((o) => o.kind !== "tree");

  const solid: Placement[] = obstacleTrees.map((o, i) => ({
    pos: xyz(o.x, o.y, o.z),
    scale: o.height,
    yaw: (i * 2.399) % Math.PI,
    variant: (i % 3) as 0 | 1 | 2,
    far: false,
    shadow: true,
  }));
  // The exact-radius obstacle trunk is built separately below. Reuse only the
  // conifer crowns here so a decorative trunk cannot disagree with collision.
  const solidMesh = instanceVegetation(
    solid,
    CONIFER_PARTS,
    "obstacle_trees",
    false,
  );

  // A faceted trunk at exactly the collision radius, so the thing that stops the
  // car has a visible, solid footprint rather than an approximate canopy edge.
  const trunks = buildTrunks(obstacleTrees);

  const obstacleShadows = buildVegetationShadows(solid);
  for (const m of [
    sceneryConifers,
    sceneryBroadleaves,
    sceneryBushes,
    farForest,
    sceneryShadows,
    rocks,
    solidMesh,
    obstacleShadows,
  ]) {
    if (m) root.add(m);
  }
  if (trunks) root.add(trunks);
  if (others.length > 0) root.add(buildMarkers(others));

  return root;
}

function buildTrunks(obstacles: Obstacle[]): InstancedMesh | null {
  if (obstacles.length === 0) return null;
  const mesh = new InstancedMesh(
    new CylinderGeometry(0.5, 0.5, 1, 6),
    new MeshLambertMaterial({
      map: textures().bark[0],
      color: 0xc9b59c,
      flatShading: true,
    }),
    obstacles.length,
  );
  mesh.name = "obstacle_trunks";
  mesh.frustumCulled = false;
  const m = new Matrix4();
  const pos = new Vector3();
  const rot = new Quaternion();
  const scale = new Vector3();
  obstacles.forEach((o, i) => {
    const h = o.height * 0.42;
    xyz(o.x, o.y, o.z + h / 2, pos);
    scale.set(o.radius * 2, h, o.radius * 2);
    mesh.setMatrixAt(i, m.compose(pos, rot, scale));
  });
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

/**
 * Solid obstacles that are not trees, as magenta boxes at their collision size.
 *
 * These kinds are legal in the schema and the generator does not place them
 * yet, so seeing one means either the generator grew a feature or something is
 * wrong — and both are things to notice immediately rather than drive through.
 */
function buildMarkers(obstacles: Obstacle[]): InstancedMesh {
  const mesh = new InstancedMesh(
    new BoxGeometry(1, 1, 1),
    new MeshBasicMaterial({ color: new Color(UNKNOWN) }),
    obstacles.length,
  );
  mesh.name = "obstacles_other";
  mesh.frustumCulled = false;
  const m = new Matrix4();
  const pos = new Vector3();
  const rot = new Quaternion();
  const scale = new Vector3();
  obstacles.forEach((o, i) => {
    xyz(o.x, o.y, o.z + o.height / 2, pos);
    scale.set(o.radius * 2, o.height, o.radius * 2);
    mesh.setMatrixAt(i, m.compose(pos, rot, scale));
  });
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}

/**
 * Sponsor hoardings along the verge.
 *
 * Not in the contract, not solid, pure spectacle — and one of the most
 * recognisable things about the era's stages. They mark the corner, they give
 * the eye something to measure speed against, and they are a large part of why
 * those games read as events rather than as empty countryside.
 *
 * Placed on the OUTSIDE of corners only, which is both where they really go and
 * where they do the most good: they are the thing you are trying not to hit.
 */
export function buildBanners(stage: Stage): Group {
  const tex = textures();
  const group = new Group();
  group.name = "banners";

  const pts = stage.centerline;
  const PANEL_M = 9.0;
  const HEIGHT_M = 1.15;

  const centre = new Vector3();
  const right = new Vector3();
  const panels: { pos: Vector3; yaw: number; s: number }[] = [];

  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i]!;
    const prev = pts[i - 1]!;
    const next = pts[i + 1]!;
    const h0 = Math.atan2(p.y - prev.y, p.x - prev.x);
    const h1 = Math.atan2(next.y - p.y, next.x - p.x);
    let turn = h1 - h0;
    while (turn > Math.PI) turn -= 2 * Math.PI;
    while (turn < -Math.PI) turn += 2 * Math.PI;

    const ds = next.s - prev.s;
    if (ds < 1e-6) continue;
    const curvature = turn / ds;
    // Only real corners get banners. 0.012 /m is about an 83 m radius.
    if (Math.abs(curvature) < 0.012) continue;
    // One panel per PANEL_M of arc, or a tight corner emits a stack of
    // overlapping boards at every sample through it.
    const last = panels[panels.length - 1];
    if (last && p.s - last.s < PANEL_M) continue;

    // Positive curvature turns left, so the outside of the corner is the right.
    const side = curvature > 0 ? 1 : -1;
    const heading = (h0 + h1) / 2;
    xyz(p.x, p.y, p.z, centre);
    rightDir(heading, right);
    const depth = 1.6;
    const lateral = side * (p.width / 2 + depth);
    const pos = new Vector3()
      .copy(centre)
      .addScaledVector(right, lateral * Math.cos(p.camber));
    pos.y =
      centre.y +
      lateral * Math.sin(p.camber) -
      terrainFall(depth) +
      HEIGHT_M / 2;
    panels.push({ pos, yaw: heading, s: p.s });
  }

  if (panels.length > 0) {
    const mesh = new InstancedMesh(
      new PlaneGeometry(PANEL_M, HEIGHT_M),
      new MeshBasicMaterial({ map: tex.banner, side: DoubleSide }),
      panels.length,
    );
    mesh.name = "banner_panels";
    mesh.frustumCulled = false;
    const m = new Matrix4();
    const rot = new Quaternion();
    const scale = new Vector3(1, 1, 1);
    const up = new Vector3(0, 1, 0);
    panels.forEach((p, i) => {
      // A hoarding runs ALONG the verge with its face pointing across the road.
      rot.setFromAxisAngle(up, p.yaw);
      mesh.setMatrixAt(i, m.compose(p.pos, rot, scale));
    });
    mesh.instanceMatrix.needsUpdate = true;
    group.add(mesh);

    // Two visible supports per board keep the panels from reading as floating
    // texture strips when viewed at a shallow chase-camera angle.
    const supports = new InstancedMesh(
      new BoxGeometry(1, 1, 1),
      new MeshBasicMaterial({ color: 0x2a3138 }),
      panels.length * 2,
    );
    supports.name = "banner_supports";
    supports.frustumCulled = false;
    const fwd = new Vector3();
    const supportScale = new Vector3(0.09, 1.3, 0.09);
    let k = 0;
    for (const p of panels) {
      headingDir(p.yaw, fwd);
      for (const side of [-1, 1]) {
        const pos = new Vector3()
          .copy(p.pos)
          .addScaledVector(fwd, side * (PANEL_M / 2 - 0.16));
        pos.y += (1.3 - HEIGHT_M) / 2;
        supports.setMatrixAt(k++, m.compose(pos, new Quaternion(), supportScale));
      }
    }
    supports.instanceMatrix.needsUpdate = true;
    group.add(supports);

    const shadowGeometry = new CircleGeometry(1, 6);
    shadowGeometry.rotateX(-Math.PI / 2);
    const shadows = new InstancedMesh(
      shadowGeometry,
      new MeshBasicMaterial({
        color: 0x182117,
        transparent: true,
        opacity: 0.13,
        depthWrite: false,
        fog: true,
      }),
      panels.length,
    );
    shadows.name = "banner_polygon_shadows";
    shadows.renderOrder = 1;
    shadows.frustumCulled = false;
    const shadowScale = new Vector3(PANEL_M * 0.38, 1, 0.24);
    panels.forEach((panel, index) => {
      const pos = panel.pos.clone();
      pos.y -= HEIGHT_M / 2 - 0.025;
      pos.x += 0.18;
      pos.z -= 0.22;
      rot.setFromAxisAngle(up, panel.yaw + 0.35);
      shadows.setMatrixAt(
        index,
        m.compose(pos, rot, shadowScale),
      );
    });
    shadows.instanceMatrix.needsUpdate = true;
    group.add(shadows);
  }

  const stakes = buildCourseStakes(stage);
  if (stakes) group.add(stakes);
  return group;
}

/**
 * Small red/white course stakes: non-colliding event dressing outside both
 * road edges. They add speed cues on straights where corner banners are absent.
 */
function buildCourseStakes(stage: Stage): InstancedMesh | null {
  const places: { pos: Vector3; colour: Color }[] = [];
  const pts = stage.centerline;
  const centre = new Vector3();
  const right = new Vector3();
  let nextS = 18;
  let n = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    if (p.s < nextS) continue;
    nextS = p.s + 24;
    const h = headingAt(pts, i);
    xyz(p.x, p.y, p.z, centre);
    rightDir(h, right);
    for (const side of [-1, 1]) {
      const depth = 1.0;
      const lateral = side * (p.width / 2 + depth);
      const pos = new Vector3()
        .copy(centre)
        .addScaledVector(right, lateral * Math.cos(p.camber));
      pos.y =
        centre.y +
        lateral * Math.sin(p.camber) -
        terrainFall(depth) +
        0.36;
      places.push({
        pos,
        colour: new Color((n + (side > 0 ? 1 : 0)) % 2 === 0 ? 0xf4f2e8 : 0xd52b2f),
      });
    }
    n++;
  }
  if (places.length === 0) return null;

  const mesh = new InstancedMesh(
    new BoxGeometry(1, 1, 1),
    new MeshBasicMaterial({ color: 0xffffff, vertexColors: true }),
    places.length,
  );
  mesh.name = "course_stakes";
  mesh.frustumCulled = false;
  const m = new Matrix4();
  const q = new Quaternion();
  const scale = new Vector3(0.09, 0.72, 0.09);
  places.forEach((p, i) => {
    mesh.setMatrixAt(i, m.compose(p.pos, q, scale));
    mesh.setColorAt(i, p.colour);
  });
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return mesh;
}

/** A dark blob under the car, so it reads as touching the ground. */
export function buildShadow(): Mesh {
  const tex = textures();
  const m = new Mesh(
    new PlaneGeometry(4.6, 2.2),
    new MeshBasicMaterial({
      map: tex.shadow,
      transparent: true,
      depthWrite: false,
      fog: true,
    }),
  );
  m.name = "car_shadow";
  m.rotation.x = -Math.PI / 2;
  m.renderOrder = 1;
  return m;
}
