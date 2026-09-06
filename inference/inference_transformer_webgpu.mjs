import { extractPreConvKernels } from './inference_pre_conv_webgpu.mjs';
import { transformerTensorSpecs, validateTransformerWeights, createTransformerRoPE } from './transformer_weights.mjs';
export { TRANSFORMER_MODEL, parseTextEncoderWeights } from './transformer_weights.mjs';

const PRE_NAMES = ['im2col_prepare_for_1D_conv', 'add_bias', 'get_col_mean', 'get_col_variance', 'layer_norm', 'leakyReLU', 'add_residual'];
const TRANS_NAMES = ['applyRoPE', 'getHeadDimScaledAttn', 'getAttnHeadsMaxByCol_softmax', 'getAttnHeadsSumByCol_softmax', 'applySoftmaxToAttnHeads'];
const DOT = 'matmul_by_head_implicit_x1_transpose_no_fuse_target';
const MIX = 'matmul_by_head_x1_fused_x2_not_fused_y_fused';
const NAMES = [...PRE_NAMES, ...TRANS_NAMES, 'matmul', DOT, MIX];
const TOKEN = Symbol('TransformerWebGPU');
const round64 = n => Math.ceil(n / 64) * 64;

export function extractTransformerKernels(preSource, transformerSource, blasSource) {
  const pre = extractPreConvKernels(preSource, blasSource);
  const result = Object.fromEntries([...PRE_NAMES, 'matmul'].map(name => [name, pre[name]]));
  for (const name of [...TRANS_NAMES, DOT, MIX]) {
    const source = TRANS_NAMES.includes(name) ? transformerSource : blasSource;
    const matches = [...source.matchAll(new RegExp('(?:^|\\n)const\\s+' + name + '\\s*=\\s*`([^`]*)`;?', 'g'))];
    if (matches.length !== 1 || matches[0][1].includes('${') || matches[0][1].includes('\\'))
      throw new Error(`Expected one plain shader literal: ${name}`);
    result[name] = matches[0][1];
  }
  return Object.freeze(result);
}

export async function loadTransformerKernels() {
  const sources = await Promise.all(['inference_pre_conv.js', 'inference_transformer.js', 'blas_gemm.js'].map(async name => {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    return response.text();
  }));
  return extractTransformerKernels(...sources);
}

/** The unchanged K^T Q shader declares ten storage buffers. Request this before
 * creating BOTH the pre-conv and transformer stages on the same device.
 */
export async function requestTextEncoderDevice(gpu = globalThis.navigator?.gpu) {
  if (!gpu) throw new Error('WebGPU is unavailable');
  const adapter = await gpu.requestAdapter();
  if (!adapter || adapter.limits.maxStorageBuffersPerShaderStage < 10)
    throw new Error('The supplied attention kernel requires an adapter with ten storage buffers per compute stage');
  return adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
}

export function transformerMemoryPlan(maxTokenLength, ffnTileSize = 512) {
  if (!Number.isInteger(maxTokenLength) || maxTokenLength < 1 || maxTokenLength > 1536)
    throw new RangeError('maxTokenLength must be in [1,1536]');
  if (!Number.isInteger(ffnTileSize) || ffnTileSize < 1 || ffnTileSize > 512)
    throw new RangeError('ffnTileSize must be in [1,512]');
  const L = maxTokenLength;
  const buffers = {
    queriesAndContext: 192 * round64(L), keysAndFfnResidual: 192 * L, values: 192 * L,
    hidden: 768 * L, im2col: Math.max(576 * L, 2304 * Math.min(L, ffnTileSize + 2)),
    scoresAndProbabilities: 2 * L * Math.min(L, 64), exponentials: 2 * L * Math.min(L, 64),
    softmaxMax: 128, softmaxSum: 128, mean: L, variance: L, rope: 96 * L,
  };
  return Object.freeze({ maxTokenLength, ffnTileSize, elements: Object.freeze(buffers),
    workspaceBytes: Object.values(buffers).reduce((a, b) => a + b, 0) * 4,
    optionalInputCopyBytes: 192 * L * 4,
    weightBytes: transformerTensorSpecs().reduce((sum, s) => sum + s.shape.reduce((a, b) => a * b, 1) * 4, 0),
    cudaActivationBytes: (6 * (5960 * L + 8 * L * L) + 96 * L) * 4,
  });
}

export function validateTransformerCapacity(limits, plan) {
  if (limits.maxStorageBuffersPerShaderStage < 10 || limits.maxBindingsPerBindGroup < 10)
    throw new Error('Request GPUDevice with requiredLimits: { maxStorageBuffersPerShaderStage: 10 } for the unchanged attention kernel');
  if (limits.maxComputeInvocationsPerWorkgroup < 32 || limits.maxComputeWorkgroupSizeX < 32 || limits.maxComputeWorkgroupStorageSize < 128)
    throw new Error('Device does not support the supplied 32-thread softmax kernels');
  // All sliced column strides and tile starts are multiples of 256 bytes.
  if (256 % limits.minStorageBufferOffsetAlignment !== 0)
    throw new Error('This binding plan requires storage offset alignment dividing 256 bytes');
  const largest = Math.max(768 * 576, ...Object.values(plan.elements)) * 4;
  if (largest > limits.maxBufferSize || largest > limits.maxStorageBufferBindingSize)
    throw new RangeError(`Device cannot hold the largest ${largest}-byte binding`);
  const L = plan.maxTokenLength;
  const invocations = Math.max(768 * L, 576 * L, 2304 * Math.min(L, plan.ffnTileSize + 2), 2 * L * Math.min(L, 64));
  if (Math.ceil(invocations / 32) > limits.maxComputeWorkgroupsPerDimension || Math.min(L, 64) > limits.maxComputeWorkgroupsPerDimension)
    throw new RangeError('Device dispatch limit is insufficient for the requested transformer capacity');
}

async function checked(device, operation) {
  const scopes = ['internal', 'out-of-memory', 'validation'];
  for (const scope of scopes) device.pushErrorScope(scope);
  let value, failure;
  try { value = await operation(); } catch (error) { failure = error; }
  for (let i = scopes.length - 1; i >= 0; --i) {
    try { const error = await device.popErrorScope(); if (error && !failure) failure = new Error(`WebGPU ${scopes[i]}: ${error.message}`); }
    catch (error) { failure ??= error; }
  }
  if (failure) throw failure;
  return value;
}

/** Six-layer, inference-only transformer using only the original WGSL strings.
 * Scratch is reused across layers. runBuffer defaults to consuming input in place.
 */
export class TransformerWebGPU {
  #device; #plan; #owned = []; #weights = new Map(); #pipelines = {}; #scalarCache = new Map();
  #b = {}; #copy; #busy = false; #destroyed = false; #lost;
  constructor(token) { if (token !== TOKEN) throw new Error('Use TransformerWebGPU.create()'); }

  static async create({ device, kernels, weights, maxTokenLength = 1536, ffnTileSize = 512 }) {
    if (!device) throw new TypeError('A GPUDevice is required');
    validateTransformerWeights(weights);
    const plan = transformerMemoryPlan(maxTokenLength, ffnTileSize);
    validateTransformerCapacity(device.limits, plan);
    for (const name of NAMES) if (typeof kernels?.[name] !== 'string') throw new Error(`Missing shader: ${name}`);
    const self = new TransformerWebGPU(TOKEN);
    self.#device = device; self.#plan = plan;
    device.lost.then(info => { self.#lost = info; });
    try {
      await checked(device, async () => {
        for (const name of NAMES) {
          const module = device.createShaderModule({ label: name, code: kernels[name] });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
          if (errors.length) throw new Error(`${name}: ${errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n')}`);
          self.#pipelines[name] = await device.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: 'main' } });
        }
        for (const { name } of transformerTensorSpecs()) {
          const data = weights.get(name); self.#weights.set(name, self.#buffer(name, data.length, data));
        }
        for (const [name, count] of Object.entries(plan.elements))
          self.#b[name] = self.#buffer(name, count, name === 'rope' ? createTransformerRoPE(maxTokenLength) : undefined);
        await device.queue.onSubmittedWorkDone();
      });
      return self;
    } catch (error) { self.destroy(); throw error; }
  }

  get memoryPlan() { return this.#plan; }
  #buffer(label, count, data) {
    const b = this.#device.createBuffer({ label, size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.#owned.push(b);
    if (data) this.#device.queue.writeBuffer(b, 0, data);
    return b;
  }
  #scalar(value, type = 'u32') {
    const key = `${type}:${value}`;
    if (!this.#scalarCache.has(key)) this.#scalarCache.set(key, this.#buffer(key, 1, type === 'u32' ? new Uint32Array([value]) : new Float32Array([value])));
    return this.#scalarCache.get(key);
  }
  #view(buffer, offsetElements = 0, count = buffer.size / 4 - offsetElements) {
    return { buffer, offset: offsetElements * 4, size: count * 4 };
  }
  #dispatch(pass, name, resources, count, grid) {
    const pipeline = this.#pipelines[name];
    const bindGroup = this.#device.createBindGroup({ label: name, layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((resource, binding) => ({ binding, resource: resource.buffer ? resource : { buffer: resource } })) });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    const [x, y = 1, z = 1] = grid ?? [Math.ceil(count / 32)];
    if ([x, y, z].some(n => !Number.isInteger(n) || n < 1 || n > this.#device.limits.maxComputeWorkgroupsPerDimension))
      throw new RangeError(`Invalid ${name} dispatch`);
    pass.dispatchWorkgroups(x, y, z);
  }
  #encodeLayer(encoder, state, L, layer) {
    const b = this.#b, q = b.queriesAndContext, k = b.keysAndFfnResidual, v = b.values;
    // Extra QUERY columns only; key count and softmax denominator remain exactly L.
    const queryEnd = L < 64 ? L : round64(L);
    if (queryEnd > L) encoder.clearBuffer(q, 192 * L * 4, 192 * (queryEnd - L) * 4);
    const pass = encoder.beginComputePass({ label: `transformer layer ${layer}` });
    const u = n => this.#scalar(n), f = n => this.#scalar(n, 'f32');
    const dispatch = (name, resources, count, grid) => this.#dispatch(pass, name, resources, count, grid);
    const weight = name => this.#weights.get(`encoder.encoder.${name}`);
    const gemm = (out, w, input, m, contraction, n) => dispatch('matmul', [out, w, input, u(m), u(contraction), u(n)], m * n);
    const bias = (x, values, dim = 192) => dispatch('add_bias', [x, values, u(dim), u(L)], dim * L);
    const norm = (x, prefix) => {
      dispatch('get_col_mean', [b.mean, x, u(192), u(L)], L);
      dispatch('get_col_variance', [b.variance, b.mean, x, u(192), u(L)], L);
      dispatch('layer_norm', [x, b.variance, b.mean, weight(`${prefix}.gamma`), weight(`${prefix}.beta`), f(1e-4), u(192), u(L)], 192 * L);
    };
    for (const [projection, output] of [['q', q], ['k', k], ['v', v]]) {
      gemm(output, weight(`attn_layers.${layer}.conv_${projection}.weight`), state, 192, 192, L);
      bias(output, weight(`attn_layers.${layer}.conv_${projection}.bias`));
    }
    for (const x of [k, q]) dispatch('applyRoPE', [x, b.rope, u(192), u(96), u(L)], 192 * L);

    const tile = Math.min(L, 64);
    for (let start = 0; start < L; start += tile) {
      const query = this.#view(q, start * 192, tile * 192);
      dispatch(DOT, [b.scoresAndProbabilities, k, query, u(192), u(192), u(2), u(96), u(L), u(96), u(tile)], 2 * L * tile);
      dispatch('getHeadDimScaledAttn', [b.scoresAndProbabilities, f(Math.fround(1 / Math.fround(Math.sqrt(96)))), u(2), u(L), u(L)], 2 * L * tile);
      if (L < 64) {
        dispatch('getAttnHeadsMaxByCol_softmax', [b.softmaxMax, b.scoresAndProbabilities, u(L)], 0, [2, L]);
        dispatch('getAttnHeadsSumByCol_softmax', [b.softmaxSum, b.exponentials, b.scoresAndProbabilities, b.softmaxMax, u(L)], 0, [2, L]);
        dispatch('applySoftmaxToAttnHeads', [b.scoresAndProbabilities, b.exponentials, b.softmaxSum, u(2), u(L)], 2 * L * L);
      } else {
        for (let head = 0; head < 2; ++head) {
          const scores = this.#view(b.scoresAndProbabilities, head * L * tile, L * tile);
          const exp = this.#view(b.exponentials, head * L * tile, L * tile);
          const max = this.#view(b.softmaxMax, head * 64, tile), sum = this.#view(b.softmaxSum, head * 64, tile);
          // Prefix columns of one [L,L] head. L*64 is divisible by 32, so
          // no invocation reaches beyond these smaller, aligned bindings.
          dispatch('getAttnHeadsMaxByCol_softmax', [max, scores, u(L)], 0, [1, tile]);
          dispatch('getAttnHeadsSumByCol_softmax', [sum, exp, scores, max, u(L)], 0, [1, tile]);
          dispatch('applySoftmaxToAttnHeads', [scores, exp, sum, u(1), u(L)], L * tile);
        }
      }
      // This query block has been consumed. Its Q storage now holds V*softmax.
      dispatch(MIX, [query, v, b.scoresAndProbabilities, u(192), u(96), u(L), u(tile)], 192 * tile);
    }

    // K is dead after all query blocks; reuse it for output projection/FFN residual.
    gemm(k, weight(`attn_layers.${layer}.conv_o.weight`), q, 192, 192, L);
    bias(k, weight(`attn_layers.${layer}.conv_o.bias`));
    dispatch('add_residual', [k, state, u(192), u(L)], 192 * L);
    norm(k, `norm_layers_1.${layer}`);
    dispatch('im2col_prepare_for_1D_conv', [b.im2col, k, u(3), u(1), u(1), u(192), u(L)], 576 * L);
    gemm(b.hidden, weight(`ffn_layers.${layer}.conv_1.weight`), b.im2col, 768, 576, L);
    bias(b.hidden, weight(`ffn_layers.${layer}.conv_1.bias`), 768);
    dispatch('leakyReLU', [b.hidden, f(0), u(768 * L)], 768 * L);

    // Every tile includes its real left/right neighbor. Select only the useful
    // im2col columns for GEMM, retaining original boundary padding semantics.
    for (let start = 0; start < L; start += this.#plan.ffnTileSize) {
      const end = Math.min(L, start + this.#plan.ffnTileSize);
      const sourceStart = Math.max(0, start - 1), sourceEnd = Math.min(L, end + 1);
      const sourceL = sourceEnd - sourceStart;
      dispatch('im2col_prepare_for_1D_conv', [b.im2col, this.#view(b.hidden, 768 * sourceStart, 768 * sourceL),
        u(3), u(1), u(1), u(768), u(sourceL)], 2304 * sourceL);
      gemm(this.#view(state, 192 * start, 192 * (end - start)), weight(`ffn_layers.${layer}.conv_2.weight`),
        this.#view(b.im2col, 2304 * (start - sourceStart), 2304 * (end - start)), 192, 2304, end - start);
    }
    bias(state, weight(`ffn_layers.${layer}.conv_2.bias`));
    dispatch('add_residual', [state, k, u(192), u(L)], 192 * L);
    norm(state, `norm_layers_2.${layer}`);
    pass.end();
  }

  #idle() {
    if (this.#destroyed) throw new Error('TransformerWebGPU has been destroyed');
    if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
    if (this.#busy) throw new Error('Await the current transformer operation before reusing it');
  }
  #validateInput(input, L, inPlace) {
    if (!Number.isInteger(L) || L < 1 || L > this.#plan.maxTokenLength) throw new RangeError('Invalid transformer token length');
    if (!input || input.size < 192 * L * 4 || !(input.usage & GPUBufferUsage.STORAGE)) throw new Error('Expected a storage GPUBuffer holding [192,L]');
    if (!inPlace && !(input.usage & GPUBufferUsage.COPY_SRC)) throw new Error('Preserving input requires COPY_SRC usage');
  }

  async runBuffer(input, L, { inPlace = true } = {}) {
    this.#idle(); this.#validateInput(input, L, inPlace); this.#busy = true;
    try {
      const output = await checked(this.#device, async () => {
        if (!inPlace && !this.#copy) this.#copy = this.#buffer('preserved-input working copy', 192 * this.#plan.maxTokenLength);
        const state = inPlace ? input : this.#copy;
        const encoder = this.#device.createCommandEncoder({ label: 'inference_transformer.cu: six layers' });
        if (!inPlace) encoder.copyBufferToBuffer(input, 0, state, 0, 192 * L * 4);
        for (let layer = 0; layer < 6; ++layer) this.#encodeLayer(encoder, state, L, layer);
        this.#device.queue.submit([encoder.finish()]);
        await this.#device.queue.onSubmittedWorkDone();
        return state;
      });
      if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
      return Object.freeze({ device: this.#device, dim: 192, L, output, ffn_post_conv2: output,
        shape: Object.freeze([192, L]), layout: 'column-major', inPlace, layerCount: 6 });
    } finally { this.#busy = false; }
  }

  // Plug directly into the previous runtime's runInference stage callback.
  async run_embedding_transformer_stack(dim, L, context, options) {
    if (dim !== 192 || context?.device !== this.#device || context.L !== L) throw new Error('Transformer and pre-conv must share device, dim=192, and L');
    return this.runBuffer(context.output ?? context.convNet?.[3]?.conv, L, options);
  }

  async readOutput(result) {
    this.#idle();
    if (result?.device !== this.#device) throw new Error('Result belongs to a different device');
    this.#validateInput(result.output, result.L, true);
    if (!(result.output.usage & GPUBufferUsage.COPY_SRC)) throw new Error('Readback requires output COPY_SRC usage');
    this.#busy = true; let staging;
    try {
      return await checked(this.#device, async () => {
        staging = this.#device.createBuffer({ label: 'transformer readback', size: 192 * result.L * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = this.#device.createCommandEncoder();
        encoder.copyBufferToBuffer(result.output, 0, staging, 0, staging.size);
        this.#device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); return data;
      });
    } finally { staging?.destroy(); this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current transformer operation before destroying it');
    for (const buffer of this.#owned) buffer.destroy();
    this.#owned.length = 0; this.#scalarCache.clear(); this.#destroyed = true;
  }
}
