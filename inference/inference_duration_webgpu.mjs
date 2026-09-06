import { extractPreConvKernels } from './inference_pre_conv_webgpu.mjs';
import { durationTensorSpecs, validateDurationWeights } from './duration_weights.mjs';
export { DURATION_MODEL, parseTextConditioningWeights } from './duration_weights.mjs';

const SHARED = ['im2col_prepare_for_1D_conv', 'add_bias', 'leakyReLU', 'get_col_mean', 'get_col_variance', 'layer_norm', 'matmul'];
const ROUND = 'get_rounded_durations_per_token';
const PREFIX = 'calculate_cummulative_durations_up_to_each_position';
const REPEAT = 'repeat_each_token_by_its_duration_val';
const NAMES = [...SHARED, ROUND, PREFIX, REPEAT];
const TOKEN = Symbol('DurationWebGPU');

export function extractDurationKernels(preSource, durationSource, blasSource) {
  const shared = extractPreConvKernels(preSource, blasSource);
  const result = Object.fromEntries(SHARED.map(name => [name, shared[name]]));
  for (const name of [ROUND, PREFIX, REPEAT]) {
    const matches = [...durationSource.matchAll(new RegExp('(?:^|\\n)const\\s+' + name + '\\s*=\\s*`([^`]*)`;?', 'g'))];
    if (matches.length !== 1 || matches[0][1].includes('${') || matches[0][1].includes('\\'))
      throw new Error(`Expected one plain shader literal: ${name}`);
    result[name] = matches[0][1];
  }
  return Object.freeze(result);
}

export async function loadDurationKernels() {
  const sources = await Promise.all(['inference_pre_conv.js', 'inference_duration.js', 'blas_gemm.js'].map(async name => {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    return response.text();
  }));
  return extractDurationKernels(...sources);
}

export function durationMemoryPlan(maxTokenLength = 1536) {
  if (!Number.isInteger(maxTokenLength) || maxTokenLength < 1 || maxTokenLength > 1536)
    throw new RangeError('maxTokenLength must be in [1,1536]');
  const L = maxTokenLength;
  const elements = { mel: 80 * L, im2col: 768 * L, hidden: 256 * L,
    meanAndLogDurations: L, variance: L, rounded: L, cumulative: L, repeated: 80 * 1024 };
  return Object.freeze({ maxTokenLength, elements: Object.freeze(elements), metadataBytes: 12 * L,
    workspaceBytes: Object.values(elements).reduce((a, b) => a + b, 0) * 4 + 12 * L,
    weightBytes: durationTensorSpecs().reduce((n, s) => n + s.shape.reduce((a, b) => a * b, 1) * 4, 0),
    cudaActivationBytes: (1941 * L + 2 * L + 80 * 1280) * 4 });
}

export function validateDurationCapacity(limits, plan) {
  if (limits.maxStorageBuffersPerShaderStage < 8 || limits.maxBindingsPerBindGroup < 8 ||
      limits.maxComputeInvocationsPerWorkgroup < 32 || limits.maxComputeWorkgroupSizeX < 32)
    throw new Error('The supplied duration workflow needs eight storage bindings and 32-thread workgroups');
  const largest = Math.max(256 * 768, ...Object.values(plan.elements)) * 4;
  if (largest > limits.maxStorageBufferBindingSize || largest > limits.maxBufferSize)
    throw new RangeError(`Device cannot hold a ${largest}-byte duration binding`);
  if (Math.ceil(Math.max(768 * plan.maxTokenLength, 80 * 1024) / 32) > limits.maxComputeWorkgroupsPerDimension)
    throw new RangeError('Device dispatch limit cannot hold the duration workflow');
}

/** Validation only: the original shaders compute rounding and the prefix sum.
 * CPU checks do not substitute another exp/ceil/prefix implementation for GPU output.
 * Read all O(L) metadata so a wrapped u32 prefix cannot masquerade as <=1024 frames.
 */
export function validateDurationMetadata(logDurations, durations, cumulative) {
  if (!(logDurations instanceof Float32Array) || !(durations instanceof Uint32Array) || !(cumulative instanceof Uint32Array) ||
      logDurations.length < 1 || logDurations.length > 1536 || durations.length !== logDurations.length || cumulative.length !== logDurations.length)
    throw new Error('Invalid duration metadata shape or types');
  const L = logDurations.length;
  let sum = 0;
  for (let i = 0; i < L; ++i) {
    if (!Number.isFinite(logDurations[i]) || logDurations[i] > Math.log(0x7fffffff) || durations[i] > 0x7fffffff)
      throw new RangeError(`Token ${i}: predicted duration is outside the valid CUDA signed-int range`);
    if (logDurations[i] >= 0 && durations[i] === 0)
      throw new Error(`Token ${i}: inconsistent zero duration for a nonnegative log prediction`);
    sum += durations[i];
    // CUDA computes (mel_frame_count + 3) before dividing by four.
    if (sum > 0x7ffffffc || cumulative[i] !== sum)
      throw new RangeError(`Token ${i}: cumulative duration overflow or inconsistent GPU prefix`);
  }
  const mel_frame_count = cumulative[L - 1];
  if (mel_frame_count === 0) throw new RangeError('Predicted zero mel frames; no valid expansion can be launched');
  return Object.freeze({ mel_frame_count, compatible_mel_frame_count: Math.ceil(mel_frame_count / 4) * 4 });
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

/** Input: transformer [192,L]. Output: duration-expanded, padded [80,F] conditioning.
 * Input stays intact. Conv outputs/statistics are reused after their last read.
 */
export class DurationWebGPU {
  #device; #plan; #owned = []; #weights = new Map(); #pipelines = {}; #scalars = new Map(); #b = {};
  #metadata; #busy = false; #destroyed = false; #lost; #generation = 0;
  constructor(token) { if (token !== TOKEN) throw new Error('Use DurationWebGPU.create()'); }
  static async create({ device, weights, kernels, maxTokenLength = 1536 }) {
    if (!device) throw new TypeError('A GPUDevice is required');
    validateDurationWeights(weights);
    const plan = durationMemoryPlan(maxTokenLength); validateDurationCapacity(device.limits, plan);
    for (const name of NAMES) if (typeof kernels?.[name] !== 'string') throw new Error(`Missing shader: ${name}`);
    const self = new DurationWebGPU(TOKEN); self.#device = device; self.#plan = plan;
    device.lost.then(info => { self.#lost = info; });
    try {
      await checked(device, async () => {
        for (const name of NAMES) {
          const module = device.createShaderModule({ label: name, code: kernels[name] });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
          if (errors.length) throw new Error(`${name}: ${errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n')}`);
          self.#pipelines[name] = await device.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: 'main' } });
        }
        for (const { name } of durationTensorSpecs()) {
          const data = weights.get(name); self.#weights.set(name, self.#buffer(name, data.length, data));
        }
        for (const [name, count] of Object.entries(plan.elements)) self.#b[name] = self.#buffer(`duration ${name}`, count);
        self.#metadata = device.createBuffer({ label: 'duration metadata readback', size: plan.metadataBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        self.#owned.push(self.#metadata);
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
  #scalar(value, type = 'u32') {
    const key = `${type}:${value}`;
    if (!this.#scalars.has(key)) this.#scalars.set(key, this.#buffer(key, 1, type === 'f32' ? new Float32Array([value]) : new Uint32Array([value])));
    return this.#scalars.get(key);
  }
  #dispatch(pass, name, buffers, elements) {
    const pipeline = this.#pipelines[name];
    const group = this.#device.createBindGroup({ label: name, layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    // The prefix kernel is unguarded and has @workgroup_size(1).
    const groups = name === PREFIX ? 1 : Math.ceil(elements / 32);
    if (!Number.isInteger(groups) || groups < 1 || groups > this.#device.limits.maxComputeWorkgroupsPerDimension)
      throw new RangeError(`Invalid ${name} dispatch`);
    pass.dispatchWorkgroups(groups, 1, 1);
  }
  #encodePrediction(encoder, input, L) {
    const b = this.#b, u = n => this.#scalar(n), f = n => this.#scalar(n, 'f32');
    const pass = encoder.beginComputePass({ label: 'duration prediction: 20 dispatches' });
    const dispatch = (name, buffers, count) => this.#dispatch(pass, name, buffers, count);
    const w = name => this.#weights.get(`encoder.${name}`);
    const gemm = (out, weights, x, rows, contraction) => dispatch('matmul', [out, weights, x, u(rows), u(contraction), u(L)], rows * L);
    const bias = (out, weights, rows) => dispatch('add_bias', [out, weights, u(rows), u(L)], rows * L);
    gemm(b.mel, w('proj_m.weight'), input, 80, 192); bias(b.mel, w('proj_m.bias'), 80);
    for (const layer of [1, 2]) {
      const inputDim = layer === 1 ? 192 : 256;
      // All source values are captured in im2col before hidden is overwritten.
      dispatch('im2col_prepare_for_1D_conv', [b.im2col, layer === 1 ? input : b.hidden, u(3), u(1), u(1), u(inputDim), u(L)], inputDim * 3 * L);
      gemm(b.hidden, w(`proj_w.conv_${layer}.weight`), b.im2col, 256, inputDim * 3);
      bias(b.hidden, w(`proj_w.conv_${layer}.bias`), 256);
      // CUDA duration uses ReLU BEFORE normalization (unlike the pre-conv stage).
      dispatch('leakyReLU', [b.hidden, f(0), u(256 * L)], 256 * L);
      dispatch('get_col_mean', [b.meanAndLogDurations, b.hidden, u(256), u(L)], L);
      dispatch('get_col_variance', [b.variance, b.meanAndLogDurations, b.hidden, u(256), u(L)], L);
      dispatch('layer_norm', [b.hidden, b.variance, b.meanAndLogDurations, w(`proj_w.norm_${layer}.gamma`),
        w(`proj_w.norm_${layer}.beta`), f(1e-4), u(256), u(L)], 256 * L);
    }
    // The final normalization consumed mean; reuse its L floats for log durations.
    gemm(b.meanAndLogDurations, w('proj_w.proj.weight'), b.hidden, 1, 256);
    bias(b.meanAndLogDurations, w('proj_w.proj.bias'), 1);
    dispatch(ROUND, [b.rounded, b.meanAndLogDurations, u(L)], L);
    dispatch(PREFIX, [b.cumulative, b.rounded, u(L)], 1);
    pass.end();
    for (const [i, buffer] of [b.meanAndLogDurations, b.rounded, b.cumulative].entries())
      encoder.copyBufferToBuffer(buffer, 0, this.#metadata, i * L * 4, L * 4);
  }
  #idle() {
    if (this.#destroyed) throw new Error('DurationWebGPU has been destroyed');
    if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
    if (this.#busy) throw new Error('Await the current duration operation before reusing it');
  }
  async runBuffer(input, L) {
    this.#idle();
    if (!Number.isInteger(L) || L < 1 || L > this.#plan.maxTokenLength) throw new RangeError('Invalid duration token length');
    if (!input || input.size < 192 * L * 4 || !(input.usage & GPUBufferUsage.STORAGE)) throw new Error('Expected transformer storage buffer [192,L]');
    this.#busy = true; const generation = ++this.#generation;
    try {
      return await checked(this.#device, async () => {
        const encoder = this.#device.createCommandEncoder({ label: 'inference_duration.cu: predict' });
        this.#encodePrediction(encoder, input, L); this.#device.queue.submit([encoder.finish()]);
        await this.#metadata.mapAsync(GPUMapMode.READ);
        let bytes;
        try { bytes = this.#metadata.getMappedRange().slice(0, 12 * L); } finally { this.#metadata.unmap(); }
        const host = Object.freeze({ log_durations: new Float32Array(bytes, 0, L),
          final_durations: new Uint32Array(bytes, L * 4, L), cummulative_durations: new Uint32Array(bytes, L * 8, L) });
        const counts = validateDurationMetadata(host.log_durations, host.final_durations, host.cummulative_durations);
        const skippedExpansion = counts.compatible_mel_frame_count > 1024;
        if (!skippedExpansion) {
          const expansion = this.#device.createCommandEncoder({ label: 'inference_duration.cu: expand' });
          const padding = counts.compatible_mel_frame_count - counts.mel_frame_count;
          if (padding) expansion.clearBuffer(this.#b.repeated, counts.mel_frame_count * 80 * 4, padding * 80 * 4);
          const pass = expansion.beginComputePass({ label: 'repeat tokens by duration' });
          this.#dispatch(pass, REPEAT, [this.#b.repeated, this.#b.mel, this.#b.cumulative, this.#scalar(L)], counts.mel_frame_count * 80);
          pass.end(); this.#device.queue.submit([expansion.finish()]); await this.#device.queue.onSubmittedWorkDone();
        }
        if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
        const output = skippedExpansion ? null : this.#b.repeated;
        const duration_forward = Object.freeze({ x_mel_dim: this.#b.mel, x_post_conv3: this.#b.meanAndLogDurations,
          final_durations: this.#b.rounded, cummulative_durations: this.#b.cumulative, x_mel_dim_repeat_by_duration: output });
        return Object.freeze({ device: this.#device, dim: 80, inputDim: 192, L, generation, ...counts, skippedExpansion,
          output, shape: output ? Object.freeze([80, counts.compatible_mel_frame_count]) : null,
          validShape: Object.freeze([80, counts.mel_frame_count]), layout: 'column-major', host, duration_forward,
          ...duration_forward });
      });
    } finally { this.#busy = false; }
  }
  async infer_text_token_durations(dim, L, context) {
    if (dim !== 192 || context?.device !== this.#device || context.L !== L)
      throw new Error('Duration and transformer must share device, dim=192, and L');
    return this.runBuffer(context.transformer?.output ?? context.output, L);
  }
  async readOutput(result, { includePadding = true } = {}) {
    this.#idle();
    if (result?.device !== this.#device || result.generation !== this.#generation || result.output !== this.#b.repeated)
      throw new Error('No current expanded duration result (skipped or stale output)');
    this.#busy = true; let staging;
    try {
      return await checked(this.#device, async () => {
        const count = 80 * (includePadding ? result.compatible_mel_frame_count : result.mel_frame_count);
        staging = this.#device.createBuffer({ label: 'duration output readback', size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = this.#device.createCommandEncoder(); encoder.copyBufferToBuffer(result.output, 0, staging, 0, count * 4);
        this.#device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); return data;
      });
    } finally { staging?.destroy(); this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current duration operation before destroying it');
    for (const buffer of this.#owned) buffer.destroy(); this.#owned.length = 0; this.#scalars.clear(); this.#destroyed = true;
  }
}
