/** Shared simulation contracts — all modules must honor these shapes. */

export type MaterialId =
  | 'steel'
  | 'keel'
  | 'bulkhead'
  | 'engine'
  | 'ice'
  | 'glass'
  | 'superstructure'
  | 'deck'
  | 'wood'
  | 'funnel'
  | 'compartment'
  | 'rope';

export interface Material {
  id: MaterialId;
  name: string;
  /** Color for rendering (CSS hex) */
  color: string;
  /** Mass density relative to water (~1.0) */
  density: number;
  /** XPBD compliance (lower = stiffer). 0 = rigid. */
  compliance: number;
  tensionLimit: number;
  compressionLimit: number;
  /** 0 = brittle snap, 1 = fully ductile (plastic rest-length update) */
  ductility: number;
  /** Counts as watertight hull skin when intact */
  hullSkin: boolean;
  /** Structural brace / long tie candidate */
  canBrace: boolean;
}

export interface Vec2 {
  x: number;
  y: number;
}

export interface TankBounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** SoA water particle buffers */
export interface WaterState {
  count: number;
  capacity: number;
  x: Float32Array;
  y: Float32Array;
  px: Float32Array;
  py: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  /** Inv mass; 0 = pinned */
  invMass: Float32Array;
  spacing: number;
  restDensity: number;
}

export interface NodeState {
  count: number;
  x: Float32Array;
  y: Float32Array;
  px: Float32Array;
  py: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  invMass: Float32Array;
  /** Base structural mass (before flood water) */
  mass: Float32Array;
  /** Extra mass from flooded water */
  floodMass: Float32Array;
  material: Uint8Array;
  /** Bit flags: 1=pinned, 2=flooded, 4=broken, 8=skin */
  flags: Uint8Array;
  /** Compartment id (0 = none / exterior) */
  compartment: Uint16Array;
}

export interface BeamState {
  count: number;
  a: Uint32Array;
  b: Uint32Array;
  rest: Float32Array;
  compliance: Float32Array;
  tensionLimit: Float32Array;
  compressionLimit: Float32Array;
  ductility: Float32Array;
  /** Current plastic rest length (starts = rest) */
  plasticRest: Float32Array;
  /** 0=broken, 1=ok; strain for overlay */
  alive: Uint8Array;
  strain: Float32Array;
  /** Bit: 1=skin edge, 2=brace/tie */
  flags: Uint8Array;
}

export interface Ship {
  nodes: NodeState;
  beams: BeamState;
  name: string;
}

export type ToolId =
  | 'drag'
  | 'smash'
  | 'cut'
  | 'bomb'
  | 'pin'
  | 'repair'
  | 'addWater'
  | 'removeWater';

export interface SimConfig {
  dt: number;
  substeps: number;
  gravity: number;
  waterViscosity: number;
  surfaceTension: number;
  wind: number;
  waveAmp: number;
  particleTarget: number;
  useGpu: boolean;
  stressOverlay: boolean;
  paused: boolean;
}

export interface SimStats {
  waterCount: number;
  nodeCount: number;
  beamCount: number;
  brokenBeams: number;
  floodedNodes: number;
  fps: number;
  backend: 'cpu' | 'webgpu';
}

export interface VesselPreset {
  id: string;
  name: string;
  /** Build at world position (cx, cy) */
  build: (cx: number, cy: number, densityScale: number, strengthScale: number) => Ship;
}

export const FLAG_PINNED = 1;
export const FLAG_FLOODED = 2;
export const FLAG_BROKEN = 4;
export const FLAG_SKIN = 8;

export const BEAM_SKIN = 1;
export const BEAM_BRACE = 2;
