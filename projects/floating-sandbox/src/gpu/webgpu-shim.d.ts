/**
 * Minimal WebGPU DOM typings so `tsc` works without `@webgpu/types`.
 * Augments / stubs only what this project uses.
 */

interface GPUAdapterLimits {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxStorageBuffersPerShaderStage: number;
  readonly maxComputeWorkgroupSizeX: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
}

interface GPUAdapter {
  readonly limits: GPUAdapterLimits;
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUDeviceDescriptor {
  label?: string;
  requiredLimits?: Record<string, number>;
}

interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

type GPUBufferMapState = 'unmapped' | 'pending' | 'mapped';

interface GPUBuffer {
  readonly mapState: GPUBufferMapState;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GPUShaderModuleDescriptor {
  label?: string;
  code: string;
}

interface GPUCompilationMessage {
  readonly message: string;
  readonly type: 'error' | 'warning' | 'info';
  readonly lineNum: number;
  readonly linePos: number;
}

interface GPUCompilationInfo {
  readonly messages: readonly GPUCompilationMessage[];
}

interface GPUShaderModule {
  getCompilationInfo(): Promise<GPUCompilationInfo>;
}

interface GPUBindGroupLayoutDescriptor {
  label?: string;
  entries: {
    binding: number;
    visibility: number;
    buffer?: { type?: 'uniform' | 'storage' | 'read-only-storage' };
  }[];
}

interface GPUBindGroupLayout {}

interface GPUPipelineLayoutDescriptor {
  label?: string;
  bindGroupLayouts: GPUBindGroupLayout[];
}

interface GPUPipelineLayout {}

interface GPUComputePipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

interface GPUBindGroup {}

interface GPUCommandEncoder {
  beginComputePass(descriptor?: object): GPUComputePassEncoder;
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GPUCommandBuffer {}

interface GPUQueue {
  submit(commandBuffers: Iterable<GPUCommandBuffer>): void;
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource,
    dataOffset?: number,
    size?: number,
  ): void;
}

interface GPUDevice {
  readonly queue: GPUQueue;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  createComputePipeline(descriptor: {
    label?: string;
    layout: 'auto' | GPUPipelineLayout;
    compute: { module: GPUShaderModule; entryPoint: string };
  }): GPUComputePipeline;
  createBindGroup(descriptor: {
    label?: string;
    layout: GPUBindGroupLayout;
    entries: { binding: number; resource: { buffer: GPUBuffer } }[];
  }): GPUBindGroup;
  createCommandEncoder(descriptor?: object): GPUCommandEncoder;
  addEventListener(type: string, listener: (ev: unknown) => void): void;
  destroy(): void;
}

interface GPU {
  requestAdapter(options?: { powerPreference?: string }): Promise<GPUAdapter | null>;
}

interface Navigator {
  readonly gpu?: GPU;
}

declare const GPUBufferUsage: {
  MAP_READ: number;
  MAP_WRITE: number;
  COPY_SRC: number;
  COPY_DST: number;
  INDEX: number;
  VERTEX: number;
  UNIFORM: number;
  STORAGE: number;
  INDIRECT: number;
  QUERY_RESOLVE: number;
};

declare const GPUMapMode: {
  READ: number;
  WRITE: number;
};

declare const GPUShaderStage: {
  VERTEX: number;
  FRAGMENT: number;
  COMPUTE: number;
};

interface GPUUncapturedErrorEvent {
  error: { message: string };
}
