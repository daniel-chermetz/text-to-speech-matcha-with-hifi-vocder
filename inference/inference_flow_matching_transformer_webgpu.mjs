import { extractTransformerKernels, requestTextEncoderDevice } from './inference_transformer_webgpu.mjs';
import { flowTransformerTensorSpecs, validateFlowTransformerWeights, FLOW_TRANSFORMER_PREFIXES } from './flow_transformer_weights.mjs';
export { FLOW_TRANSFORMER_MODEL, parseFlowTransformerWeights } from './flow_transformer_weights.mjs';
export { requestTextEncoderDevice as requestFlowTransformerDevice };

const DOT = 'matmul_by_head_implicit_x1_transpose_no_fuse_target';
const MIX = 'matmul_by_head_x1_fused_x2_not_fused_y_fused';
const SHARED = ['add_bias', 'get_col_mean', 'get_col_variance', 'layer_norm', 'add_residual',
  'getHeadDimScaledAttn', 'getAttnHeadsMaxByCol_softmax', 'getAttnHeadsSumByCol_softmax',
  'applySoftmaxToAttnHeads', 'matmul', DOT, MIX];
const NAMES = [...SHARED, 'snakeBeta'];
const TOKEN = Symbol('FlowMatchingTransformerWebGPU');
const round64 = n => Math.ceil(n / 64) * 64;

export function extractFlowTransformerKernels(preSource, transformerSource, flowTransformerSource, blasSource) {
  const shared = extractTransformerKernels(preSource, transformerSource, blasSource);
  const matches = [...flowTransformerSource.matchAll(/(?:^|\n)const\s+snakeBeta\s*=\s*`([^`]*)`;?/g)];
  if (matches.length !== 1 || matches[0][1].includes('${') || matches[0][1].includes('\\'))
    throw new Error('Expected one plain shader literal: snakeBeta');
  return Object.freeze({ ...Object.fromEntries(SHARED.map(name => [name, shared[name]])), snakeBeta: matches[0][1] });
}
export async function loadFlowTransformerKernels() {
  const sources = await Promise.all(['inference_pre_conv.js', 'inference_transformer.js',
    'inference_flow_matching_transformer.js', 'blas_gemm.js'].map(async name => {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    return response.text();
  }));
  return extractFlowTransformerKernels(...sources);
}

export function flowTransformerMemoryPlan(maxFrameLength = 1280, ffnTileSize = 256) {
  if (!Number.isInteger(maxFrameLength) || maxFrameLength < 1 || maxFrameLength > 1280)
    throw new RangeError('maxFrameLength must be in [1,1280]');
  if (!Number.isInteger(ffnTileSize) || ffnTileSize < 1 || ffnTileSize > 256)
    throw new RangeError('ffnTileSize must be in [1,256]');
  const L = maxFrameLength;
  const elements = { normalizedAndProjection: 256 * L, queriesAndContext: 128 * round64(L),
    keys: 128 * L, values: 128 * L, hidden: 1024 * Math.min(L, ffnTileSize),
    scoresAndProbabilities: 2 * L * Math.min(L, 64), exponentials: 2 * L * Math.min(L, 64),
    softmaxMax: 128, softmaxSum: 128, mean: L, variance: L };
  return Object.freeze({ maxFrameLength, ffnTileSize, elements: Object.freeze(elements),
    workspaceBytes: Object.values(elements).reduce((a, b) => a + b, 0) * 4,
    optionalInputCopyBytes: 256 * L * 4,
    weightBytes: flowTransformerTensorSpecs().reduce((n, s) => n + s.shape.reduce((a, b) => a * b, 1) * 4, 0),
    // CUDA already shares this one workspace across all six weight sets.
    cudaActivationBytesAtCapacity: (3080 * L + 8 * L * L) * 4,
    cudaAllocatedActivationBytes: (3080 * 1280 + 8 * 1280 * 1280) * 4 });
}
export function validateFlowTransformerCapacity(limits, plan) {
  if (limits.maxStorageBuffersPerShaderStage < 10 || limits.maxBindingsPerBindGroup < 10)
    throw new Error('The unchanged attention kernel requires ten storage bindings; use requestFlowTransformerDevice()');
  if (limits.maxComputeInvocationsPerWorkgroup < 32 || limits.maxComputeWorkgroupSizeX < 32 || limits.maxComputeWorkgroupStorageSize < 128)
    throw new Error('Device cannot run the supplied 32-thread softmax kernels');
  if (256 % limits.minStorageBufferOffsetAlignment !== 0)
    throw new Error('This binding plan requires storage offset alignment dividing 256 bytes');
  const largest = Math.max(1024 * 256, ...Object.values(plan.elements)) * 4;
  if (largest > limits.maxBufferSize || largest > limits.maxStorageBufferBindingSize)
    throw new RangeError(`Device cannot hold the largest ${largest}-byte flow transformer binding`);
  const L = plan.maxFrameLength;
  const invocations = Math.max(256 * L, 1024 * Math.min(L, plan.ffnTileSize), 2 * L * Math.min(L, 64));
  if (Math.ceil(invocations / 32) > limits.maxComputeWorkgroupsPerDimension || Math.min(L, 64) > limits.maxComputeWorkgroupsPerDimension)
    throw new RangeError('Device dispatch limit is insufficient for the flow transformer');
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

/** One flow transformer call, selecting i=0..5. Not a six-layer stack or Euler loop.
 * In-place by default; both pre-normalization residuals are preserved before use.
 */
export class FlowMatchingTransformerWebGPU {
  #device; #plan; #owned = []; #weights = new Map(); #pipelines = {}; #scalars = new Map(); #b = {};
  #copy; #busy = false; #destroyed = false; #lost; #generation = 0; #lastResult;
  constructor(token) { if (token !== TOKEN) throw new Error('Use FlowMatchingTransformerWebGPU.create()'); }
  static async create({ device, weights, kernels, maxFrameLength = 1280, ffnTileSize = 256 }) {
    if (!device) throw new TypeError('A GPUDevice is required');
    validateFlowTransformerWeights(weights);
    const plan = flowTransformerMemoryPlan(maxFrameLength, ffnTileSize);
    validateFlowTransformerCapacity(device.limits, plan);
    for (const name of NAMES) if (typeof kernels?.[name] !== 'string') throw new Error(`Missing shader: ${name}`);
    const self = new FlowMatchingTransformerWebGPU(TOKEN); self.#device = device; self.#plan = plan;
    device.lost.then(info => { self.#lost = info; });
    try {
      await checked(device, async () => {
        for (const name of NAMES) {
          const module = device.createShaderModule({ label: name, code: kernels[name] });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
          if (errors.length) throw new Error(`${name}: ${errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n')}`);
          self.#pipelines[name] = await device.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: 'main' } });
        }
        for (const { name } of flowTransformerTensorSpecs()) {
          const data = weights.get(name); self.#weights.set(name, self.#buffer(name, data.length, data));
        }
        for (const [name, count] of Object.entries(plan.elements)) self.#b[name] = self.#buffer(`flow transformer ${name}`, count);
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
  #view(buffer, offset = 0, count = buffer.size / 4 - offset) { return { buffer, offset: offset * 4, size: count * 4 }; }
  #dispatch(pass, name, resources, count, grid) {
    const pipeline = this.#pipelines[name];
    const group = this.#device.createBindGroup({ label: name, layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((resource, binding) => ({ binding, resource: resource.buffer ? resource : { buffer: resource } })) });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    const [x, y = 1, z = 1] = grid ?? [Math.ceil(count / 32)];
    if ([x, y, z].some(n => !Number.isInteger(n) || n < 1 || n > this.#device.limits.maxComputeWorkgroupsPerDimension))
      throw new RangeError(`Invalid ${name} dispatch`);
    pass.dispatchWorkgroups(x, y, z);
  }
  #encode(encoder, state, i, L, L_valid) {
    const b = this.#b, norm = b.normalizedAndProjection, q = b.queriesAndContext;
    const u = n => this.#scalar(n), f = n => this.#scalar(n, 'f32');
    const w = suffix => this.#weights.get(`${FLOW_TRANSFORMER_PREFIXES[i]}.${suffix}`);
    // CUDA normalizes OUT OF PLACE. The supplied norm shader is in-place, so
    // copy into reusable scratch while preserving the unnormalized residual.
    encoder.copyBufferToBuffer(state, 0, norm, 0, 256 * L * 4);
    if (L >= 64 && round64(L) > L) encoder.clearBuffer(q, 128 * L * 4, 128 * (round64(L) - L) * 4);
    let pass = encoder.beginComputePass({ label: `flow transformer ${i}: attention` });
    const dispatch = (name, resources, count, grid) => this.#dispatch(pass, name, resources, count, grid);
    const gemm = (out, weights, input, M, K, N) => dispatch('matmul', [out, weights, input, u(M), u(K), u(N)], M * N);
    const bias = (out, weights, dim, n) => dispatch('add_bias', [out, weights, u(dim), u(n)], dim * n);
    const normalize = prefix => {
      dispatch('get_col_mean', [b.mean, norm, u(256), u(L)], L);
      dispatch('get_col_variance', [b.variance, b.mean, norm, u(256), u(L)], L);
      dispatch('layer_norm', [norm, b.variance, b.mean, w(`${prefix}.weight`), w(`${prefix}.bias`), f(1e-5), u(256), u(L)], 256 * L);
    };
    normalize('norm1');
    for (const [projection, out] of [['q', q], ['k', b.keys], ['v', b.values]])
      gemm(out, w(`attn1.to_${projection}.weight`), norm, 128, 256, L);
    // No Q/K/V biases and no RoPE in this CUDA workflow.
    const tile = Math.min(L, 64);
    for (let start = 0; start < L; start += tile) {
      const query = this.#view(q, start * 128, tile * 128);
      dispatch(DOT, [b.scoresAndProbabilities, b.keys, query, u(128), u(128), u(2), u(64), u(L), u(64), u(tile)], 2 * L * tile);
      // Row index reduces algebraically to index % L even for a query tile.
      // With tile=64 the invocation count is exact, so no tail reaches stale data.
      dispatch('getHeadDimScaledAttn', [b.scoresAndProbabilities, f(0.125), u(2), u(L_valid), u(L)], 2 * L * tile);
      if (L < 64) {
        dispatch('getAttnHeadsMaxByCol_softmax', [b.softmaxMax, b.scoresAndProbabilities, u(L)], 0, [2, L]);
        dispatch('getAttnHeadsSumByCol_softmax', [b.softmaxSum, b.exponentials, b.scoresAndProbabilities, b.softmaxMax, u(L)], 0, [2, L]);
        dispatch('applySoftmaxToAttnHeads', [b.scoresAndProbabilities, b.exponentials, b.softmaxSum, u(2), u(L)], 2 * L * L);
      } else {
        for (let head = 0; head < 2; ++head) {
          const scores = this.#view(b.scoresAndProbabilities, head * L * tile, L * tile);
          const exp = this.#view(b.exponentials, head * L * tile, L * tile);
          const max = this.#view(b.softmaxMax, head * 64, tile), sum = this.#view(b.softmaxSum, head * 64, tile);
          dispatch('getAttnHeadsMaxByCol_softmax', [max, scores, u(L)], 0, [1, tile]);
          dispatch('getAttnHeadsSumByCol_softmax', [sum, exp, scores, max, u(L)], 0, [1, tile]);
          dispatch('applySoftmaxToAttnHeads', [scores, exp, sum, u(1), u(L)], L * tile);
        }
      }
      dispatch(MIX, [query, b.values, b.scoresAndProbabilities, u(128), u(64), u(L), u(tile)], 128 * tile);
    }
    // norm1 scratch is dead after Q/K/V. Q now contains the complete context.
    gemm(norm, w('attn1.to_out.0.weight'), q, 256, 128, L);
    bias(norm, w('attn1.to_out.0.bias'), 256, L);
    dispatch('add_residual', [state, norm, u(256), u(L)], 256 * L);
    pass.end();
    // Preserve the UNNORMALIZED attention residual for the final residual add.
    encoder.copyBufferToBuffer(state, 0, norm, 0, 256 * L * 4);
    pass = encoder.beginComputePass({ label: `flow transformer ${i}: feed-forward` });
    normalize('norm3');
    for (let start = 0; start < L; start += this.#plan.ffnTileSize) {
      const n = Math.min(this.#plan.ffnTileSize, L - start);
      const normalized = this.#view(norm, start * 256, n * 256);
      const residual = this.#view(state, start * 256, n * 256);
      gemm(b.hidden, w('ff.net.0.proj.weight'), normalized, 1024, 256, n);
      bias(b.hidden, w('ff.net.0.proj.bias'), 1024, n);
      dispatch('snakeBeta', [b.hidden, w('ff.net.0.alpha'), w('ff.net.0.beta'), u(1024), u(n)], 1024 * n);
      gemm(normalized, w('ff.net.2.weight'), b.hidden, 256, 1024, n);
      bias(normalized, w('ff.net.2.bias'), 256, n);
      dispatch('add_residual', [residual, normalized, u(256), u(n)], 256 * n);
    }
    pass.end();
  }
  #idle() {
    if (this.#destroyed) throw new Error('FlowMatchingTransformerWebGPU has been destroyed');
    if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
    if (this.#busy) throw new Error('Await the current flow transformer operation before reusing it');
  }
  async runBuffer(input, i, L, L_valid = L, { inPlace = true } = {}) {
    this.#idle();
    if (!Number.isInteger(i) || i < 0 || i >= 6) throw new RangeError('Flow transformer index must be in [0,5]');
    if (!Number.isInteger(L) || L < 1 || L > this.#plan.maxFrameLength) throw new RangeError('Invalid flow transformer frame length');
    if (!Number.isInteger(L_valid) || L_valid < 1 || L_valid > L) throw new RangeError('L_valid must be in [1,L]');
    if (typeof inPlace !== 'boolean') throw new TypeError('inPlace must be a boolean');
    if (!input || input.size < 256 * L * 4 || !(input.usage & GPUBufferUsage.STORAGE) || !(input.usage & GPUBufferUsage.COPY_SRC))
      throw new Error('Expected [256,L] GPUBuffer with STORAGE and COPY_SRC for residual-preserving norm copies');
    this.#busy = true; this.#lastResult = undefined; const generation = ++this.#generation;
    try {
      const output = await checked(this.#device, async () => {
        if (!inPlace && !this.#copy) this.#copy = this.#buffer('flow transformer preserved-input working copy', 256 * this.#plan.maxFrameLength);
        // Reject accidental in-place use of the preservation buffer under the
        // preservation option: the requested source would otherwise be overwritten.
        if (!inPlace && input === this.#copy) throw new Error('Preservation requires an input distinct from the reusable output');
        const state = inPlace ? input : this.#copy;
        const encoder = this.#device.createCommandEncoder({ label: `flow_matching_transformer.cu: i=${i}` });
        if (!inPlace) encoder.copyBufferToBuffer(input, 0, state, 0, 256 * L * 4);
        this.#encode(encoder, state, i, L, L_valid);
        this.#device.queue.submit([encoder.finish()]); await this.#device.queue.onSubmittedWorkDone();
        return state;
      });
      if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
      this.#lastResult = Object.freeze({ device: this.#device, i, dim: 256, halfDim: 128, L, L_valid, generation,
        output, ffn_post_linear_2_post_residual: output, shape: Object.freeze([256, L]), layout: 'column-major', inPlace });
      return this.#lastResult;
    } finally { this.#busy = false; }
  }
  /** Positional CUDA-style adapter. Caller supplies the resnet output and lengths.
   * The future Euler loop owns skip copies, output masking, and stage transitions.
   */
  async flow_matching_transformer(x, i, dim, halfDim, L, L_valid, options) {
    if (dim !== 256 || halfDim !== 128) throw new Error('This model requires dim=256 and halfDim=128');
    return this.runBuffer(x, i, L, L_valid, options);
  }
  async readOutput(result) {
    this.#idle();
    if (!result || result !== this.#lastResult) throw new Error('Expected the current flow transformer result; stale or foreign result');
    this.#busy = true; let staging;
    try {
      return await checked(this.#device, async () => {
        staging = this.#device.createBuffer({ label: 'flow transformer readback', size: 256 * result.L * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = this.#device.createCommandEncoder(); encoder.copyBufferToBuffer(result.output, 0, staging, 0, staging.size);
        this.#device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); return values;
      });
    } finally { staging?.destroy(); this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current flow transformer operation before destroying it');
    for (const buffer of this.#owned) buffer.destroy(); this.#owned.length = 0; this.#scalars.clear(); this.#lastResult = undefined; this.#destroyed = true;
  }
}
