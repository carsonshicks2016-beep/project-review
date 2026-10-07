import { tryInitWebGPU } from '../../gpu/webgpu-device';

export type WaterBackend = 'cpu' | 'webgpu';

export async function pickWaterBackend(
  preferGpu: boolean,
): Promise<{ backend: WaterBackend; device: GPUDevice | null }> {
  if (!preferGpu) {
    return { backend: 'cpu', device: null };
  }
  const device = await tryInitWebGPU();
  if (!device) {
    return { backend: 'cpu', device: null };
  }
  return { backend: 'webgpu', device };
}
