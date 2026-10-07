import { materialByIndex } from '../materials';
import type { BeamState, NodeState } from '../types';
import { FLAG_PINNED } from '../types';

/** Allocate empty SoA node buffers. */
export function createNodes(capacity: number): NodeState {
  return {
    count: 0,
    x: new Float32Array(capacity),
    y: new Float32Array(capacity),
    px: new Float32Array(capacity),
    py: new Float32Array(capacity),
    vx: new Float32Array(capacity),
    vy: new Float32Array(capacity),
    invMass: new Float32Array(capacity),
    mass: new Float32Array(capacity),
    floodMass: new Float32Array(capacity),
    material: new Uint8Array(capacity),
    flags: new Uint8Array(capacity),
    compartment: new Uint16Array(capacity),
  };
}

/** Allocate empty SoA beam buffers. */
export function createBeams(capacity: number): BeamState {
  return {
    count: 0,
    a: new Uint32Array(capacity),
    b: new Uint32Array(capacity),
    rest: new Float32Array(capacity),
    compliance: new Float32Array(capacity),
    tensionLimit: new Float32Array(capacity),
    compressionLimit: new Float32Array(capacity),
    ductility: new Float32Array(capacity),
    plasticRest: new Float32Array(capacity),
    alive: new Uint8Array(capacity),
    strain: new Float32Array(capacity),
    flags: new Uint8Array(capacity),
  };
}

/** Append a node; returns index. Throws if capacity exceeded. */
export function addNode(
  nodes: NodeState,
  x: number,
  y: number,
  materialIndex: number,
  mass: number,
  flags = 0,
  compartment = 0,
): number {
  const i = nodes.count;
  if (i >= nodes.x.length) {
    throw new Error(`Node capacity exceeded (${nodes.x.length})`);
  }
  nodes.x[i] = x;
  nodes.y[i] = y;
  nodes.px[i] = x;
  nodes.py[i] = y;
  nodes.vx[i] = 0;
  nodes.vy[i] = 0;
  nodes.mass[i] = mass;
  nodes.floodMass[i] = 0;
  const pinned = (flags & FLAG_PINNED) !== 0;
  const m = mass > 0 ? mass : 1;
  nodes.invMass[i] = pinned ? 0 : 1 / m;
  nodes.material[i] = materialIndex & 0xff;
  nodes.flags[i] = flags & 0xff;
  nodes.compartment[i] = compartment & 0xffff;
  nodes.count = i + 1;
  return i;
}

/** Append a beam; returns index. Throws if capacity exceeded. */
export function addBeam(
  beams: BeamState,
  a: number,
  b: number,
  rest: number,
  compliance: number,
  tensLim: number,
  compLim: number,
  ductility: number,
  flags = 0,
): number {
  const i = beams.count;
  if (i >= beams.a.length) {
    throw new Error(`Beam capacity exceeded (${beams.a.length})`);
  }
  beams.a[i] = a >>> 0;
  beams.b[i] = b >>> 0;
  beams.rest[i] = rest;
  beams.compliance[i] = compliance;
  beams.tensionLimit[i] = tensLim;
  beams.compressionLimit[i] = compLim;
  beams.ductility[i] = ductility;
  beams.plasticRest[i] = rest;
  beams.alive[i] = 1;
  beams.strain[i] = 0;
  beams.flags[i] = flags & 0xff;
  beams.count = i + 1;
  return i;
}

/**
 * Add a beam using averaged material properties of endpoints.
 * `strengthScale` raises break limits and lowers compliance.
 * `densScale` is accepted for API symmetry with vessel builders (unused for beams).
 */
export function addBeamFromMaterials(
  beams: BeamState,
  nodes: NodeState,
  a: number,
  b: number,
  _densScale: number,
  strengthScale = 1,
  flags = 0,
): number {
  const ma = materialByIndex(nodes.material[a]!);
  const mb = materialByIndex(nodes.material[b]!);
  const dx = nodes.x[b]! - nodes.x[a]!;
  const dy = nodes.y[b]! - nodes.y[a]!;
  const rest = Math.hypot(dx, dy);
  const s = strengthScale > 0 ? strengthScale : 1;
  const compliance = ((ma.compliance + mb.compliance) * 0.5) / s;
  const tensLim = ((ma.tensionLimit + mb.tensionLimit) * 0.5) * s;
  const compLim = ((ma.compressionLimit + mb.compressionLimit) * 0.5) * s;
  const ductility = (ma.ductility + mb.ductility) * 0.5;
  return addBeam(beams, a, b, rest, compliance, tensLim, compLim, ductility, flags);
}

/** Recompute invMass from mass + floodMass (honors pinned flag). */
export function refreshInvMass(nodes: NodeState, i: number): void {
  if ((nodes.flags[i]! & FLAG_PINNED) !== 0) {
    nodes.invMass[i] = 0;
    return;
  }
  const m = nodes.mass[i]! + nodes.floodMass[i]!;
  nodes.invMass[i] = m > 1e-8 ? 1 / m : 0;
}

/**
 * Swap-remove beam at index `i` with the last beam.
 * Returns the old last index that moved into `i`, or -1 if none.
 */
export function removeBeamSwap(beams: BeamState, i: number): number {
  const last = beams.count - 1;
  if (i < 0 || i > last) return -1;
  if (i !== last) {
    beams.a[i] = beams.a[last]!;
    beams.b[i] = beams.b[last]!;
    beams.rest[i] = beams.rest[last]!;
    beams.compliance[i] = beams.compliance[last]!;
    beams.tensionLimit[i] = beams.tensionLimit[last]!;
    beams.compressionLimit[i] = beams.compressionLimit[last]!;
    beams.ductility[i] = beams.ductility[last]!;
    beams.plasticRest[i] = beams.plasticRest[last]!;
    beams.alive[i] = beams.alive[last]!;
    beams.strain[i] = beams.strain[last]!;
    beams.flags[i] = beams.flags[last]!;
  }
  beams.count = last;
  return i !== last ? last : -1;
}

/**
 * Swap-remove node at index `i`. Remaps beam endpoints; drops beams that
 * referenced the removed node. Returns the old last node index that moved, or -1.
 */
export function removeNodeSwap(nodes: NodeState, beams: BeamState, i: number): number {
  const last = nodes.count - 1;
  if (i < 0 || i > last) return -1;

  // Drop beams touching the removed node; remap beams that referenced `last`.
  for (let b = beams.count - 1; b >= 0; b--) {
    const aIdx = beams.a[b]!;
    const bIdx = beams.b[b]!;
    if (aIdx === i || bIdx === i) {
      removeBeamSwap(beams, b);
      continue;
    }
    if (aIdx === last) beams.a[b] = i;
    if (bIdx === last) beams.b[b] = i;
  }

  if (i !== last) {
    nodes.x[i] = nodes.x[last]!;
    nodes.y[i] = nodes.y[last]!;
    nodes.px[i] = nodes.px[last]!;
    nodes.py[i] = nodes.py[last]!;
    nodes.vx[i] = nodes.vx[last]!;
    nodes.vy[i] = nodes.vy[last]!;
    nodes.invMass[i] = nodes.invMass[last]!;
    nodes.mass[i] = nodes.mass[last]!;
    nodes.floodMass[i] = nodes.floodMass[last]!;
    nodes.material[i] = nodes.material[last]!;
    nodes.flags[i] = nodes.flags[last]!;
    nodes.compartment[i] = nodes.compartment[last]!;
  }
  nodes.count = last;
  return i !== last ? last : -1;
}
