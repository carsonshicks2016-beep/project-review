/**
 * Hard-edged grounding shadows.
 *
 * These are intentionally simple polygons rather than a soft modern shadow
 * map. The stage sampler supplies the road plane, so the shapes follow grade
 * and camber without inheriting the car body's pitch and roll.
 */

import {
  BufferGeometry,
  Float32BufferAttribute,
  Group,
  Mesh,
  MeshBasicMaterial,
  Vector3,
} from "three";

import {
  roadQuaternion,
  StageSurface,
  type WheelVisualState,
} from "./surface";

const SHADOW_AWAY_FROM_SUN = new Vector3(38, 0, -46).normalize();

function polygon(points: readonly [number, number][]): BufferGeometry {
  const positions: number[] = [];
  for (let i = 1; i < points.length - 1; i++) {
    for (const index of [0, i, i + 1]) {
      const point = points[index]!;
      positions.push(point[0], 0, point[1]);
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    "position",
    new Float32BufferAttribute(positions, 3),
  );
  geometry.computeVertexNormals();
  return geometry;
}

function shadowMaterial(opacity: number): MeshBasicMaterial {
  return new MeshBasicMaterial({
    color: 0x111513,
    transparent: true,
    opacity,
    depthWrite: false,
    fog: true,
  });
}

export class GroundingShadows {
  readonly root = new Group();

  private readonly bodyMaterial = shadowMaterial(0.31);
  private readonly tyreMaterial = shadowMaterial(0.18);
  private readonly projected = new Vector3();
  private readonly body: Mesh;
  private readonly tyres: Mesh[] = [];

  constructor() {
    this.root.name = "vehicle_grounding_shadows";
    this.root.renderOrder = 1;

    // Slight directional rake keeps it from reading as a black copy of the
    // floorplan. The narrow nose and wider rear echo the saloon silhouette.
    this.body = new Mesh(
      polygon([
        [-2.20, -0.63],
        [-1.58, -0.88],
        [0.94, -0.86],
        [2.24, -0.56],
        [2.40, 0.42],
        [0.92, 0.82],
        [-1.65, 0.82],
        [-2.32, 0.48],
      ]),
      this.bodyMaterial,
    );
    this.body.name = "directional_car_shadow";
    this.body.renderOrder = 1;
    this.root.add(this.body);

    const tyreGeometry = polygon([
      [-0.42, -0.145],
      [0.42, -0.145],
      [0.42, 0.145],
      [-0.42, 0.145],
    ]);
    for (let i = 0; i < 4; i++) {
      const tyre = new Mesh(tyreGeometry, this.tyreMaterial);
      tyre.name = "tyre_contact_shadow";
      tyre.renderOrder = 1;
      this.root.add(tyre);
      this.tyres.push(tyre);
    }
  }

  update(
    carPosition: Vector3,
    s: number,
    airborneHeight: number,
    road: StageSurface,
    wheels: readonly WheelVisualState[],
  ): void {
    const roadFrame = road.sample(s);
    road.project(carPosition, s, this.projected);
    const height = Math.max(0, airborneHeight);
    this.body.position
      .copy(this.projected)
      .addScaledVector(roadFrame.normal, 0.028)
      .addScaledVector(SHADOW_AWAY_FROM_SUN, height * 0.42);
    this.body.quaternion.copy(roadQuaternion(roadFrame));

    // A higher car casts a longer, lighter shape. Tyre contact shadows disappear
    // much earlier than the body shadow, making wheel separation clear in air.
    this.body.scale.set(1 + height * 0.20, 1, 1 + height * 0.075);
    this.bodyMaterial.opacity = 0.31 * Math.max(0.05, 1 - height / 4.1);
    this.tyreMaterial.opacity = 0.18 * Math.max(0, 1 - height / 0.42);
    for (let i = 0; i < this.tyres.length; i++) {
      const wheel = wheels[i];
      const tyre = this.tyres[i]!;
      if (!wheel) {
        tyre.visible = false;
        continue;
      }
      tyre.visible = wheel.contact;
      tyre.position
        .copy(wheel.contactPoint)
        .addScaledVector(wheel.normal, 0.007);
      tyre.quaternion.copy(roadQuaternion({
        point: wheel.contactPoint,
        forward: wheel.forward,
        right: wheel.right,
        normal: wheel.normal,
        width: roadFrame.width,
        camber: roadFrame.camber,
        surface: wheel.surface,
      }));
      tyre.scale.set(1, 1, 1);
    }
  }
}
