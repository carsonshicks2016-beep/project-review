/**
 * WebGPU device init helpers for the water / sim path.
 * Returns null when the API or adapter is unavailable — caller falls back to CPU.
 */

/** Minimum capacity we want headroom for (storage buffers, workgroups). */
export const WEBGPU_TARGET_CAPACITY = 131_072;

/** Bytes per particle in the GPU SoA particle struct (8 × f32). */
export const PARTICLE_STRIDE_BYTES = 32;

function particleBufferBytes(capacity: number): number {
  return capacity * PARTICLE_STRIDE_BYTES;
}

/** Grid cell heads for spatial hash (power-of-two cell count). */
function gridBufferBytes(cellCount: number): number {
  return cellCount * 4; // u32 / atomic<u32>
}

/**
 * Request a GPUDevice suitable for PBF compute at ≥131k particles.
 * Returns null if WebGPU is missing, adapter request fails, or limits are too low.
 */
export async function tryInitWebGPU(): Promise<GPUDevice | null> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return null;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) return null;

    const cap = WEBGPU_TARGET_CAPACITY;
    // Particles + lambda + delta(vec2) + cell heads + next links (+ slack)
    const cellCount = nextPow2(Math.max(cap, 65_536));
    const needBinding = Math.max(
      particleBufferBytes(cap),
      gridBufferBytes(cellCount),
      cap * 4, // lambda
      cap * 8, // delta xy
    );
    const needBuffer = particleBufferBytes(cap) + gridBufferBytes(cellCount) * 2 + cap * 16;

    const {
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup,
    } = adapter.limits;

    if (maxStorageBufferBindingSize < needBinding) return null;
    if (maxBufferSize < needBuffer) return null;
    if (maxStorageBuffersPerShaderStage < 6) return null;
    if (maxComputeWorkgroupSizeX < 64) return null;
    if (maxComputeInvocationsPerWorkgroup < 64) return null;

    // Only bump limits that exceed the default floor and stay within adapter caps.
    const requiredLimits: Record<string, number> = {};
    const wantBinding = Math.min(maxStorageBufferBindingSize, Math.max(needBinding, 128 * 1024 * 1024));
    const wantBuffer = Math.min(maxBufferSize, Math.max(needBuffer, 256 * 1024 * 1024));
    if (wantBinding > 0) requiredLimits.maxStorageBufferBindingSize = wantBinding;
    if (wantBuffer > 0) requiredLimits.maxBufferSize = wantBuffer;

    const device = await adapter.requestDevice({
      label: 'floating-sandbox-pbf',
      requiredLimits,
    });

    device.addEventListener('uncapturederror', (ev) => {
      console.error('[WebGPU]', (ev as GPUUncapturedErrorEvent).error);
    });

    return device;
  } catch (err) {
    console.warn('[WebGPU] init failed:', err);
    return null;
  }
}

export function nextPow2(n: number): number {
  let v = 1;
  while (v < n) v <<= 1;
  return v;
}

/** Dispatch workgroup count for 1-D compute over `count` threads. */
export function workgroupCount(count: number, workgroupSize: number): number {
  return Math.max(1, Math.ceil(count / workgroupSize));
}
