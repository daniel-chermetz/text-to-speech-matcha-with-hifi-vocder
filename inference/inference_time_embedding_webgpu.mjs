import { timeEmbeddingTensorSpecs, validateTimeEmbeddingWeights, createTimeEmbeddingFrequencies } from './time_embedding_weights.mjs';
export { TIME_EMBEDDING_MODEL, parseTimeEmbeddingWeights, createTimeEmbeddingFrequencies } from './time_embedding_weights.mjs';

const NAMES = ['set_initial_time_embedding', 'matmul', 'add_bias', 'applySilu', 'applyMish'];
const TOKEN = Symbol('TimeEmbeddingWebGPU');

export function extractTimeEmbeddingKernels(timeSource, preSource, flowSource, blasSource) {
  const sourceFor = { set_initial_time_embedding: timeSource, matmul: blasSource,
    add_bias: preSource, applySilu: flowSource, applyMish: flowSource };
  return Object.freeze(Object.fromEntries(NAMES.map(name => {
    const matches = [...sourceFor[name].matchAll(new RegExp('(?:^|\\n)const\\s+' + name + '\\s*=\\s*`([^`]*)`;?', 'g'))];
    if (matches.length !== 1 || matches[0][1].includes('${') || matches[0][1].includes('\\'))
      throw new Error(`Expected one plain shader literal: ${name}`);
    return [name, matches[0][1]];
  })));
}
export async function loadTimeEmbeddingKernels() {
  const sources = await Promise.all(['inference_time_embedding.js', 'inference_pre_conv.js',
    'inference_flow_matching.js', 'blas_gemm.js'].map(async name => {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    return response.text();
  }));
  return extractTimeEmbeddingKernels(...sources);
}
/** Standalone device. The future shared flow device must instead request the
 * transformer's ten storage bindings (requestFlowTransformerDevice).
 */
export async function requestTimeEmbeddingDevice(gpu = globalThis.navigator?.gpu) {
  if (!gpu) throw new Error('WebGPU is unavailable');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available');
  return adapter.requestDevice();
}
export function timeEmbeddingMemoryPlan() {
  const elements = Object.freeze({ frequencies: 80, sinusoidalAndOutput: 1024, hidden: 1024 });
  return Object.freeze({ elements, workspaceBytes: (80 + 1024 + 1024) * 4,
    scalarBytes: 16, // One mutable f32 timestep and immutable u32 dimensions 160,1024,1.
    weightBytes: timeEmbeddingTensorSpecs().reduce((n, s) => n + s.shape.reduce((a, b) => a * b, 1) * 4, 0),
    cudaActivationBytes: (80 + 160 + 1024 + 1024) * 4 });
}
export function validateTimeEmbeddingCapacity(limits) {
  if (limits.maxStorageBuffersPerShaderStage < 6 || limits.maxBindingsPerBindGroup < 6)
    throw new Error('The supplied time embedding GEMM requires six storage bindings');
  if (limits.maxComputeInvocationsPerWorkgroup < 32 || limits.maxComputeWorkgroupSizeX < 32 || limits.maxComputeWorkgroupsPerDimension < 32)
    throw new Error('Device cannot run the supplied time embedding workgroups');
  if (limits.maxBufferSize < 1024 * 1024 * 4 || limits.maxStorageBufferBindingSize < 1024 * 1024 * 4)
    throw new RangeError('Device cannot hold the 4 MiB time linear_2 weight binding');
}
export function validateTimeEmbeddingTimestep(timestep) {
  if (typeof timestep !== 'number' || !Number.isFinite(timestep)) throw new TypeError('timestep must be a finite number');
  const t = Math.fround(timestep);
  if (!Number.isFinite(t) || !Number.isFinite(Math.fround(1000 * t)))
    throw new RangeError('timestep must produce a finite float32 phase scale');
  return t;
}
async function checked(device, operation) {
  const filters = ['internal', 'out-of-memory', 'validation'];
  for (const filter of filters) device.pushErrorScope(filter);
  let value, failure;
  try { value = await operation(); } catch (error) { failure = error; }
  for (let i = filters.length - 1; i >= 0; --i) {
    try { const error = await device.popErrorScope(); if (error && !failure) failure = new Error(`WebGPU ${filters[i]}: ${error.message}`); }
    catch (error) { failure ??= error; }
  }
  if (failure) throw failure;
  return value;
}

/** Standalone processTimeEmbedding(timestep), including final Mish. No flow loop.
 * All activations run through the supplied unedited shaders. Only the static
 * frequency initialization and scalar upload are computed on the host.
 */
export class TimeEmbeddingWebGPU {
  #device; #owned = []; #weights = new Map(); #pipelines = {}; #b = {}; #plan;
  #busy = false; #destroyed = false; #lost; #generation = 0; #lastResult;
  constructor(token) { if (token !== TOKEN) throw new Error('Use TimeEmbeddingWebGPU.create()'); }
  static async create({ device, weights, kernels, frequencies = createTimeEmbeddingFrequencies() }) {
    if (!device) throw new TypeError('A GPUDevice is required');
    validateTimeEmbeddingWeights(weights); validateTimeEmbeddingCapacity(device.limits);
    if (!(frequencies instanceof Float32Array) || frequencies.length !== 80 ||
        !frequencies.every(value => Number.isFinite(value) && value > 0 && value <= 1))
      throw new Error('frequencies must contain 80 finite float32 values in (0,1]');
    for (const name of NAMES) if (typeof kernels?.[name] !== 'string') throw new Error(`Missing shader: ${name}`);
    const self = new TimeEmbeddingWebGPU(TOKEN); self.#device = device; self.#plan = timeEmbeddingMemoryPlan();
    // Snapshot optional host input before the first asynchronous compilation.
    const frequencyValues = frequencies.slice();
    device.lost.then(info => { self.#lost = info; });
    try {
      await checked(device, async () => {
        for (const name of NAMES) {
          const module = device.createShaderModule({ label: name, code: kernels[name] });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
          if (errors.length) throw new Error(`${name}: ${errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n')}`);
          self.#pipelines[name] = await device.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: 'main' } });
        }
        for (const { name } of timeEmbeddingTensorSpecs()) {
          const data = weights.get(name); self.#weights.set(name, self.#buffer(name, data.length, data));
        }
        for (const [name, count] of Object.entries(self.#plan.elements))
          self.#b[name] = self.#buffer(`time embedding ${name}`, count, name === 'frequencies' ? frequencyValues : undefined);
        self.#b.timestep = self.#buffer('time embedding timestep', 1);
        for (const [name, value] of [['inputDim', 160], ['hiddenDim', 1024], ['one', 1]])
          self.#b[name] = self.#buffer(`time embedding ${name}`, 1, new Uint32Array([value]));
        await device.queue.onSubmittedWorkDone();
      });
      return self;
    } catch (error) { self.destroy(); throw error; }
  }
  get memoryPlan() { return this.#plan; }
  #buffer(label, count, data) {
    const buffer = this.#device.createBuffer({ label, size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.#owned.push(buffer); if (data) this.#device.queue.writeBuffer(buffer, 0, data); return buffer;
  }
  #dispatch(pass, name, buffers, count) {
    const pipeline = this.#pipelines[name];
    const group = this.#device.createBindGroup({ label: name, layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(count / 32), 1, 1);
  }
  #idle() {
    if (this.#destroyed) throw new Error('TimeEmbeddingWebGPU has been destroyed');
    if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
    if (this.#busy) throw new Error('Await the current time embedding operation before reusing it');
  }
  async processTimeEmbedding(timestep) {
    this.#idle(); const t = validateTimeEmbeddingTimestep(timestep);
    this.#busy = true; this.#lastResult = undefined; const generation = ++this.#generation;
    try {
      await checked(this.#device, async () => {
        const b = this.#b, w = name => this.#weights.get(`decoder.estimator.time_mlp.${name}`);
        // Reuse one scalar allocation across arbitrary timesteps; prior work has
        // completed and the operation lock excludes an overlapping scalar write.
        this.#device.queue.writeBuffer(b.timestep, 0, new Float32Array([t]));
        const encoder = this.#device.createCommandEncoder({ label: 'processTimeEmbedding: seven dispatches' });
        const pass = encoder.beginComputePass({ label: 'time embedding' });
        const dispatch = (name, buffers, count = 1024) => this.#dispatch(pass, name, buffers, count);
        dispatch('set_initial_time_embedding', [b.sinusoidalAndOutput, b.frequencies, b.timestep], 80);
        dispatch('matmul', [b.hidden, w('linear_1.weight'), b.sinusoidalAndOutput, b.hiddenDim, b.inputDim, b.one]);
        dispatch('add_bias', [b.hidden, w('linear_1.bias'), b.hiddenDim, b.one]);
        dispatch('applySilu', [b.hidden, b.hiddenDim, b.one]);
        // The initial 160-vector is dead; reuse its larger allocation for output.
        // GEMM still has distinct input/output bindings: hidden -> sinusoidalAndOutput.
        dispatch('matmul', [b.sinusoidalAndOutput, w('linear_2.weight'), b.hidden, b.hiddenDim, b.hiddenDim, b.one]);
        dispatch('add_bias', [b.sinusoidalAndOutput, w('linear_2.bias'), b.hiddenDim, b.one]);
        dispatch('applyMish', [b.sinusoidalAndOutput, b.hiddenDim, b.one]);
        pass.end(); this.#device.queue.submit([encoder.finish()]); await this.#device.queue.onSubmittedWorkDone();
      });
      if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
      const output = this.#b.sinusoidalAndOutput;
      this.#lastResult = Object.freeze({ device: this.#device, timestep: t, generation,
        dim: 1024, L: 1, shape: Object.freeze([1024, 1]), layout: 'column-major', output,
        time_embedding_post_2: output, flow_forward: Object.freeze({ time_embedding_post_2: output }),
        includesMish: true });
      return this.#lastResult;
    } finally { this.#busy = false; }
  }
  async run(timestep) { return this.processTimeEmbedding(timestep); }
  async readOutput(result) {
    this.#idle();
    if (!result || result !== this.#lastResult) throw new Error('Expected the current time embedding result; stale or foreign result');
    this.#busy = true; let staging;
    try {
      return await checked(this.#device, async () => {
        staging = this.#device.createBuffer({ label: 'time embedding readback', size: 1024 * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = this.#device.createCommandEncoder(); encoder.copyBufferToBuffer(result.output, 0, staging, 0, staging.size);
        this.#device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); return values;
      });
    } finally { staging?.destroy(); this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current time embedding operation before destroying it');
    for (const buffer of this.#owned) buffer.destroy(); this.#owned.length = 0; this.#lastResult = undefined; this.#destroyed = true;
  }
}
