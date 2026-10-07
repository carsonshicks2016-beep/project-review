import type { ToolId } from '../types';

export interface ToolState {
  active: ToolId;
  brushRadius: number;
  bombPower: number;
  densityScale: number;
  strengthScale: number;
}

export function createToolState(): ToolState {
  return {
    active: 'drag',
    brushRadius: 10,
    bombPower: 1,
    densityScale: 1,
    strengthScale: 1.25,
  };
}
