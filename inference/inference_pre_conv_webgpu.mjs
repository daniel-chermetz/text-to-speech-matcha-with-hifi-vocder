import { PRE_CONV_MODEL as MODEL, preConvTensorSpecs, validatePreConvWeights } from './pre_conv_weights.mjs';
export { PRE_CONV_MODEL, parsePreConvWeights } from './pre_conv_weights.mjs';

const PRE_KERNEL_NAMES = Object.freeze([
  'set_x_embeddings', 'im2col_prepare_for_1D_conv', 'add_bias',
  'get_col_mean', 'get_col_variance', 'layer_norm', 'leakyReLU', 'add_residual',
]);
const KERNEL_NAMES = [...PRE_KERNEL_NAMES, 'matmul'];
const WORKGROUP_SIZE = 32; // Taken from the unchanged WGSL, NOT CUDA threadsPerBlock.
const CONSTRUCTION_TOKEN = Symbol('PreConvWebGPU construction');

/** Extract the existing plain template literals without executing their JS.
 * Refuse escapes/interpolation: silently rewriting source would change kernels.
 */
export function extractPreConvKernels(preConvJavaScript, blasJavaScript) {
  function extract(source, name) {
    const pattern = new RegExp('(?:^|\\n)const\\s+' + name + '\\s*=\\s*`([^`]*)`;?', 'g');
    const matches = [...source.matchAll(pattern)];
    if (matches.length !== 1 || matches[0][1].includes('${') || matches[0][1].includes('\\'))
      throw new Error(`Expected one plain, unescaped shader literal: ${name}`);
    return matches[0][1];
  }
  return Object.freeze(Object.fromEntries(KERNEL_NAMES.map(name => [
    name, extract(name === 'matmul' ? blasJavaScript : preConvJavaScript, name),
  ])));
}

export async function loadPreConvKernels({
  preConvURL = new URL('./inference_pre_conv.js', import.meta.url),
  blasURL = new URL('./blas_gemm.js', import.meta.url),
} = {}) {
  const sources = await Promise.all([preConvURL, blasURL].map(async url => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
    return response.text();
  }));
  return extractPreConvKernels(...sources);
}

export function validateTokenIds(tokenIds, maxTokenLength, { matchaProtocol = false } = {}) {
  if (!Array.isArray(tokenIds) && !(tokenIds instanceof Uint32Array) && !(tokenIds instanceof Int32Array))
    throw new TypeError('tokenIds must be an array, Uint32Array, or Int32Array');
  const L = tokenIds.length;
  if (L < 1 || L > maxTokenLength) throw new RangeError(`Token count must be in [1, ${maxTokenLength}]`);
  for (let i = 0; i < L; ++i) {
    if (!Number.isInteger(tokenIds[i]) || tokenIds[i] < 0 || tokenIds[i] >= MODEL.vocabularySize)
      throw new RangeError(`Token ${i} must be an integer in [0, 177]`);
  }
  if (matchaProtocol && (L > MODEL.maximumInputTokenCount || L % 2 !== 1 ||
      tokenIds.some((value, i) => i % 2 === 0 && value !== 0)))
    throw new Error('Matcha requires an odd count <=1536 with blank ID 0 at every even position');
  return Uint32Array.from(tokenIds); // Validation precedes signed-to-unsigned conversion.
}

export function validatePreConvCapacity(limits, maxTokenLength) {
  if (!Number.isInteger(maxTokenLength) || maxTokenLength < 1 || maxTokenLength > MODEL.maximumTokenLength)
    throw new RangeError('maxTokenLength must be an integer in [1, 32767]');
  if (limits.maxStorageBuffersPerShaderStage < 8 || limits.maxBindingsPerBindGroup < 8 ||
      limits.maxComputeInvocationsPerWorkgroup < 32 || limits.maxComputeWorkgroupSizeX < 32)
    throw new Error('Device must support eight storage bindings and 32-thread workgroups');
  const largestBuffer = Math.max(192 * 960, 960 * maxTokenLength) * 4;
  if (largestBuffer > limits.maxStorageBufferBindingSize || largestBuffer > limits.maxBufferSize)
    throw new RangeError(`Device buffer limits cannot hold ${largestBuffer} bytes`);
  const groups = Math.ceil(960 * maxTokenLength / WORKGROUP_SIZE);
  if (groups > limits.maxComputeWorkgroupsPerDimension)
    throw new RangeError(`im2col needs ${groups} x workgroups; device permits ${limits.maxComputeWorkgroupsPerDimension}`);
}

// Error scopes cover creation, dispatch, and readback. The caller owns the device.
async function checked(device, operation) {
  const filters = ['internal', 'out-of-memory', 'validation'];
  for (const filter of filters) device.pushErrorScope(filter);
  let value, failure;
  try { value = await operation(); } catch (error) { failure = error; }
  for (let i = filters.length - 1; i >= 0; --i) {
    try {
      const error = await device.popErrorScope();
      if (error && !failure) failure = new Error(`WebGPU ${filters[i]}: ${error.message}`);
    } catch (error) { failure ??= error; }
  }
  if (failure) throw failure;
  return value;
}

/** Single reusable GPU workspace; use create(), then await runPreConv().
 * All inference arithmetic is performed by the user's nine unchanged shaders.
 * Returned buffers belong to this instance and are reused by the next run.
 */
export class PreConvWebGPU {
  #device;
  #owned = [];
  #pipelines = {};
  #steps = [];
  #scalars = {};
  #buffers;
  #maxTokenLength;
  #busy = false;
  #destroyed = false;
  #lost = null;

  constructor(token) {
    if (token !== CONSTRUCTION_TOKEN) throw new Error('Use await PreConvWebGPU.create(...)');
  }

  static async create({ device, weights, kernels, maxTokenLength = MODEL.maximumInputTokenCount }) {
    if (!device) throw new TypeError('A GPUDevice is required');
    validatePreConvWeights(weights);
    validatePreConvCapacity(device.limits, maxTokenLength);
    for (const name of KERNEL_NAMES) {
      if (typeof kernels?.[name] !== 'string') throw new Error(`Missing shader: ${name}`);
    }
    // Construction token avoids exposing a partially initialized public instance.
    const instance = new PreConvWebGPU(CONSTRUCTION_TOKEN);
    instance.#device = device;
    instance.#maxTokenLength = maxTokenLength;
    device.lost.then(info => { instance.#lost = info; });
    try {
      await checked(device, async () => {
        for (const name of KERNEL_NAMES) {
          const module = device.createShaderModule({ label: name, code: kernels[name] });
          const info = await module.getCompilationInfo();
          const errors = info.messages.filter(message => message.type === 'error');
          if (errors.length) throw new Error(`${name}: ${errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n')}`);
          instance.#pipelines[name] = await device.createComputePipelineAsync({
            label: name, layout: 'auto', compute: { module, entryPoint: 'main' },
          });
        }
        instance.#allocate(weights);
        instance.#buildSteps();
        await device.queue.onSubmittedWorkDone();
      });
      return instance;
    } catch (error) {
      instance.destroy();
      throw error;
    }
  }

  #buffer(label, elements, data) {
    const buffer = this.#device.createBuffer({
      label, size: elements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.#owned.push(buffer);
    if (data) this.#device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  #allocate(weights) {
    const L = this.#maxTokenLength;
    const weightBuffers = new Map(preConvTensorSpecs().map(({ name }) => {
      const data = weights.get(name);
      return [name, this.#buffer(name, data.length, data)];
    }));
    const convNetWeights = Array.from({ length: 3 }, (_, i) => Object.freeze({
      conv: weightBuffers.get(`encoder.prenet.conv_layers.${i}.weight`),
      convBias: weightBuffers.get(`encoder.prenet.conv_layers.${i}.bias`),
      normGammaBias: weightBuffers.get(`encoder.prenet.norm_layers.${i}.gamma`),
      normBetaBias: weightBuffers.get(`encoder.prenet.norm_layers.${i}.beta`),
    }));
    convNetWeights.push(Object.freeze({
      one_on_one_proj: weightBuffers.get('encoder.prenet.proj.weight'),
      convBias: weightBuffers.get('encoder.prenet.proj.bias'),
    }));
    const mean = this.#buffer('meanByCol (shared)', L);
    const variance = this.#buffer('varianceByCol (shared)', L);
    this.#buffers = Object.freeze({
      embedding_weights: weightBuffers.get('encoder.emb.weight'),
      convNetWeights: Object.freeze(convNetWeights),
      original_x: this.#buffer('original_x', 192 * L),
      x_col: this.#buffer('x_col', 960 * L),
      token_seq_indices: this.#buffer('token_seq_indices', L),
      convNet: Object.freeze(Array.from({ length: 4 }, (_, i) => Object.freeze({
        conv: this.#buffer(`convNet[${i}].conv`, 192 * L),
        meanByCol: i === 0 ? mean : null, varianceByCol: i === 0 ? variance : null,
      }))),
    });
    for (const [name, value] of Object.entries({ dim: 192, L: 1, kernel: 5, padding: 2, stride: 1, contracted: 960, maxCount: 192 }))
      this.#scalars[name] = this.#buffer(name, 1, new Uint32Array([value]));
    this.#scalars.eps = this.#buffer('eps', 1, new Float32Array([MODEL.eps]));
    this.#scalars.slope = this.#buffer('slope', 1, new Float32Array([MODEL.slope]));
  }

  #buildSteps() {
    const b = this.#buffers, s = this.#scalars;
    const add = (name, buffers, elementsPerToken, label = name) => {
      const pipeline = this.#pipelines[name];
      const bindGroup = this.#device.createBindGroup({
        label, layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      this.#steps.push({ pipeline, bindGroup, elementsPerToken, label });
    };
    add('set_x_embeddings', [b.original_x, b.embedding_weights, b.token_seq_indices, s.dim, s.L], 192);
    for (let i = 0; i < 3; ++i) {
      const x = b.convNet[i].conv, w = b.convNetWeights[i];
      add('im2col_prepare_for_1D_conv', [b.x_col, i === 0 ? b.original_x : b.convNet[i - 1].conv,
        s.kernel, s.padding, s.stride, s.dim, s.L], 960, `layer ${i}: im2col`);
      add('matmul', [x, w.conv, b.x_col, s.dim, s.contracted, s.L], 192, `layer ${i}: GEMM`);
      add('add_bias', [x, w.convBias, s.dim, s.L], 192);
      add('get_col_mean', [b.convNet[0].meanByCol, x, s.dim, s.L], 1);
      add('get_col_variance', [b.convNet[0].varianceByCol, b.convNet[0].meanByCol, x, s.dim, s.L], 1);
      add('layer_norm', [x, b.convNet[0].varianceByCol, b.convNet[0].meanByCol,
        w.normGammaBias, w.normBetaBias, s.eps, s.dim, s.L], 192);
      add('leakyReLU', [x, s.slope, s.maxCount], 192);
    }
    add('matmul', [b.convNet[3].conv, b.convNetWeights[3].one_on_one_proj, b.convNet[2].conv,
      s.dim, s.dim, s.L], 192, 'projection: GEMM');
    add('add_bias', [b.convNet[3].conv, b.convNetWeights[3].convBias, s.dim, s.L], 192);
    add('add_residual', [b.convNet[3].conv, b.original_x, s.dim, s.L], 192);
  }

  #assertIdle() {
    if (this.#destroyed) throw new Error('PreConvWebGPU has been destroyed');
    if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
    if (this.#busy) throw new Error('Await the current operation before reusing this workspace');
  }

  async #submit(tokens) {
    const device = this.#device, L = tokens.length;
    await checked(device, async () => {
      device.queue.writeBuffer(this.#buffers.token_seq_indices, 0, tokens);
      device.queue.writeBuffer(this.#scalars.L, 0, new Uint32Array([L]));
      device.queue.writeBuffer(this.#scalars.maxCount, 0, new Uint32Array([192 * L]));
      const encoder = device.createCommandEncoder({ label: 'inference_pre_conv.cu' });
      const pass = encoder.beginComputePass({ label: 'pre-conv (25 dispatches)' });
      for (const { pipeline, bindGroup, elementsPerToken } of this.#steps) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(elementsPerToken * L / WORKGROUP_SIZE));
      }
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    });
    if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
    return Object.freeze({ device, dim: 192, L, ...this.#buffers,
      output: this.#buffers.convNet[3].conv, shape: Object.freeze([192, L]), layout: 'column-major' });
  }

  async runPreConv(tokenIds) {
    this.#assertIdle();
    const tokens = validateTokenIds(tokenIds, this.#maxTokenLength);
    this.#busy = true;
    try { return await this.#submit(tokens); } finally { this.#busy = false; }
  }

  /** Preserve runInference's downstream call order and >1024 early return.
   * Each callback receives (dim, L, context); flow gets (context).
   * Duration must return { compatible_mel_frame_count, ...optional state }.
   * Callbacks must submit their work to context.device and await any CPU data.
   */
  async runInference(tokenIds, stages) {
    this.#assertIdle();
    for (const name of ['run_embedding_transformer_stack', 'infer_text_token_durations', 'run_flow_matching_euler'])
      if (typeof stages?.[name] !== 'function') throw new Error(`Missing downstream stage: ${name}`);
    const tokens = validateTokenIds(tokenIds, this.#maxTokenLength, { matchaProtocol: true });
    this.#busy = true;
    try {
      const preConv = await this.#submit(tokens);
      const context = { ...preConv };
      context.transformer = await stages.run_embedding_transformer_stack(192, tokens.length, context);
      context.duration = await stages.infer_text_token_durations(192, tokens.length, context);
      const count = context.duration?.compatible_mel_frame_count;
      if (!Number.isInteger(count) || count < 0) throw new Error('Duration stage must return a nonnegative integer compatible_mel_frame_count');
      context.compatible_mel_frame_count = count;
      if (count > 1024) return { ...context, skippedFlowMatching: true };
      context.flow = await stages.run_flow_matching_euler(context);
      return { ...context, skippedFlowMatching: false };
    } finally { this.#busy = false; }
  }

  /** cudaMemcpy device-to-host equivalent. Await before another workspace run. */
  async readBuffer(buffer, elementCount) {
    this.#assertIdle();
    if (!this.#owned.includes(buffer) || !Number.isInteger(elementCount) ||
        elementCount < 1 || elementCount * 4 > buffer.size)
      throw new RangeError('Expected an owned buffer and a valid element count');
    this.#busy = true;
    let staging;
    try {
      return await checked(this.#device, async () => {
        staging = this.#device.createBuffer({ label: 'pre-conv readback', size: elementCount * 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = this.#device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, elementCount * 4);
        this.#device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return result;
      });
    } finally { staging?.destroy(); this.#busy = false; }
  }

  destroy() {
    if (this.#busy) throw new Error('Await the current operation before destroying this workspace');
    for (const buffer of this.#owned) buffer.destroy();
    this.#owned.length = 0;
    this.#destroyed = true;
  }
}
