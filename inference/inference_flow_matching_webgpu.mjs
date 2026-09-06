import { TimeEmbeddingWebGPU, extractTimeEmbeddingKernels } from './inference_time_embedding_webgpu.mjs';
import { FlowMatchingTransformerWebGPU, extractFlowTransformerKernels, requestFlowTransformerDevice } from './inference_flow_matching_transformer_webgpu.mjs';
import { FLOW_RESNET_PREFIXES, FLOW_INPUT_DIMS, FLOW_POST_PREFIXES, FLOW_TIMESTEPS,
  flowMatchingTensorSpecs, validateFlowMatchingWeights } from './flow_matching_weights.mjs';
export { parseInferenceWeights, FLOW_TIMESTEPS } from './flow_matching_weights.mjs';
export { requestFlowTransformerDevice as requestInferenceDevice };

const DOT = 'matmul_by_head_implicit_x1_transpose_no_fuse_target';
const FLOW_NAMES = ['concat_duration_repeated_tokens_to_noise', 'mask_compatiblity_cols', 'get_group_mean',
  'get_group_variance', 'group_norm', 'applyMish', 'elementwiseAdd', 'concat_to_double_rows_per_col',
  'attach_splats_conv_transpose_1D', 'add_velocity_step_to_evolving_noise', 'get_final_user_facing_mel'];
const NAMES = [...FLOW_NAMES, 'im2col_prepare_for_1D_conv', 'add_bias', 'matmul', DOT];
const TOKEN = Symbol('FlowMatchingWebGPU');
export function extractFlowMatchingKernels(preSource, flowSource, blasSource) {
  return Object.freeze(Object.fromEntries(NAMES.map(name => {
    const source = FLOW_NAMES.includes(name) ? flowSource : ['matmul', DOT].includes(name) ? blasSource : preSource;
    const matches = [...source.matchAll(new RegExp('(?:^|\\n)const\\s+' + name + '\\s*=\\s*`([^`]*)`;?', 'g'))];
    if (matches.length !== 1 || matches[0][1].includes('${') || matches[0][1].includes('\\'))
      throw new Error(`Expected one plain shader literal: ${name}`);
    return [name, matches[0][1]];
  })));
}
export async function loadFlowMatchingKernels() {
  const [pre, text, flowTransformer, time, flow, blas] = await Promise.all(['inference_pre_conv.js', 'inference_transformer.js',
    'inference_flow_matching_transformer.js', 'inference_time_embedding.js', 'inference_flow_matching.js', 'blas_gemm.js'].map(async name => {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    return response.text();
  }));
  return { flowMatching: extractFlowMatchingKernels(pre, flow, blas),
    flowTransformer: extractFlowTransformerKernels(pre, text, flowTransformer, blas),
    timeEmbedding: extractTimeEmbeddingKernels(time, pre, flow, blas) };
}
export function flowMatchingMemoryPlan(maxMelFrames = 1024) {
  if (!Number.isInteger(maxMelFrames) || maxMelFrames < 4 || maxMelFrames > 1024 || maxMelFrames % 4)
    throw new RangeError('maxMelFrames must be a multiple of four in [4,1024]');
  const F = maxMelFrames;
  const elements = Object.freeze({ evolvingNoiseAndMel: 80 * F, velocity: 80 * F, combined: 512 * F,
    stateA: 256 * F, stateB: 256 * F, columnsAndSplats: 1536 * F,
    skip0: 256 * F, skip1: 128 * F, mean: 8, variance: 8, timeProjection: 256 });
  return Object.freeze({ maxMelFrames, elements, workspaceBytes: Object.values(elements).reduce((a, b) => a + b, 0) * 4,
    weightBytes: flowMatchingTensorSpecs().reduce((n, s) => n + s.shape.reduce((a, b) => a * b, 1) * 4, 0),
    // Existing CUDA decoder workspace already reuses many tensors. Its im2col
    // reservation is too short for stage 5 at full capacity; see the audit.
    cudaAllocatedDecoderBytes: (3522832 + 80 * 1280 + 63) * 4 });
}
export function validateFlowMatchingCapacity(limits, plan) {
  if (limits.maxStorageBuffersPerShaderStage < 10 || limits.maxBindingsPerBindGroup < 10)
    throw new Error('The original flow attention/transpose GEMM needs ten storage bindings');
  if (limits.maxComputeInvocationsPerWorkgroup < 32 || limits.maxComputeWorkgroupSizeX < 32)
    throw new Error('Unsupported flow compute workgroups');
  const largest = Math.max(256 * 1536, ...Object.values(plan.elements)) * 4;
  if (largest > limits.maxBufferSize || largest > limits.maxStorageBufferBindingSize)
    throw new RangeError(`Device cannot hold the ${largest}-byte flow binding`);
  if (Math.ceil(1536 * plan.maxMelFrames / 32) > limits.maxComputeWorkgroupsPerDimension)
    throw new RangeError('Flow im2col dispatch exceeds device limits');
}
export function validateFlowLengths(frames, compatible, maxMelFrames = 1024) {
  if (!Number.isInteger(frames) || frames < 1 || !Number.isInteger(compatible) ||
      compatible !== Math.ceil(frames / 4) * 4 || compatible > maxMelFrames)
    throw new RangeError('Expected positive mel frames with compatible=ceil(frames/4)*4 within flow capacity');
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
/** Complete five-step zero-initialized CUDA Euler loop. Owns the previously
 * translated time and flow-transformer runtimes on the same GPUDevice.
 */
export class FlowMatchingWebGPU {
  #device; #plan; #time; #transformer; #owned = []; #weights = new Map(); #pipelines = {}; #scalars = new Map(); #b = {};
  #busy = false; #destroyed = false; #lost; #lastResult; #generation = 0;
  constructor(token) { if (token !== TOKEN) throw new Error('Use FlowMatchingWebGPU.create()'); }
  static async create({ device, weights, kernels, maxMelFrames = 1024, ffnTileSize = 256, frequencies }) {
    if (!device) throw new TypeError('A GPUDevice is required');
    validateFlowMatchingWeights(weights);
    const plan = flowMatchingMemoryPlan(maxMelFrames); validateFlowMatchingCapacity(device.limits, plan);
    for (const name of NAMES) if (typeof kernels?.flowMatching?.[name] !== 'string') throw new Error(`Missing flow shader: ${name}`);
    const self = new FlowMatchingWebGPU(TOKEN); self.#device = device; self.#plan = plan;
    device.lost.then(info => { self.#lost = info; });
    try {
      self.#time = await TimeEmbeddingWebGPU.create({ device, weights, kernels: kernels.timeEmbedding, frequencies });
      self.#transformer = await FlowMatchingTransformerWebGPU.create({ device, weights, kernels: kernels.flowTransformer, maxFrameLength: maxMelFrames, ffnTileSize });
      await checked(device, async () => {
        for (const name of NAMES) {
          const module = device.createShaderModule({ label: name, code: kernels.flowMatching[name] });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
          if (errors.length) throw new Error(`${name}: ${errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n')}`);
          self.#pipelines[name] = await device.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: 'main' } });
        }
        for (const { name } of flowMatchingTensorSpecs()) {
          const data = weights.get(name); self.#weights.set(name, self.#buffer(name, data.length, data));
        }
        for (const [name, count] of Object.entries(plan.elements)) self.#b[name] = self.#buffer(`flow ${name}`, count);
        await device.queue.onSubmittedWorkDone();
      });
      return self;
    } catch (error) { self.destroy(); throw error; }
  }
  get memoryPlan() { return { decoder: this.#plan, timeEmbedding: this.#time.memoryPlan, transformer: this.#transformer.memoryPlan }; }
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
  #dispatch(pass, name, buffers, count) {
    const pipeline = this.#pipelines[name];
    const group = this.#device.createBindGroup({ label: name, layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    // Original group statistics write one result per invocation; each has size 8.
    pass.dispatchWorkgroups(Math.ceil(count / (['get_group_mean', 'get_group_variance'].includes(name) ? 8 : 32)), 1, 1);
  }
  #submit(label, record) {
    const encoder = this.#device.createCommandEncoder({ label }); record(encoder); this.#device.queue.submit([encoder.finish()]);
  }
  #pass(encoder, label, record) { const pass = encoder.beginComputePass({ label }); record(pass); pass.end(); }
  #gemm(pass, output, weights, input, M, K, L) {
    this.#dispatch(pass, 'matmul', [output, weights, input, this.#scalar(M), this.#scalar(K), this.#scalar(L)], M * L);
  }
  #bias(pass, output, bias, dim, L) { this.#dispatch(pass, 'add_bias', [output, bias, this.#scalar(dim), this.#scalar(L)], dim * L); }
  #mask(pass, x, dim, valid, L) {
    if (valid < L) this.#dispatch(pass, 'mask_compatiblity_cols', [x, this.#scalar(dim), this.#scalar(valid), this.#scalar(L)], dim * (L - valid));
  }
  #conv(pass, output, input, dim, L, weights, bias, stride = 1) {
    const b = this.#b, u = n => this.#scalar(n), outL = Math.floor((L - 1) / stride) + 1;
    this.#dispatch(pass, 'im2col_prepare_for_1D_conv', [b.columnsAndSplats, input, u(3), u(1), u(stride), u(dim), u(L)], dim * 3 * outL);
    this.#gemm(pass, output, weights, b.columnsAndSplats, 256, 3 * dim, outL);
    this.#bias(pass, output, bias, 256, outL);
  }
  #block(pass, output, input, dim, L, valid, prefix) {
    const b = this.#b, u = n => this.#scalar(n), w = suffix => this.#weights.get(`${prefix}.${suffix}`);
    this.#mask(pass, input, dim, valid, L);
    this.#conv(pass, output, input, dim, L, w('0.weight'), w('0.bias'));
    // WGSL assigns all eight statistics, so CUDA's atomic accumulator clears are unnecessary.
    this.#dispatch(pass, 'get_group_mean', [b.mean, output, u(256), u(32), u(L)], 8);
    this.#dispatch(pass, 'get_group_variance', [b.variance, b.mean, output, u(256), u(32), u(L)], 8);
    this.#dispatch(pass, 'group_norm', [output, b.variance, b.mean, w('1.weight'), w('1.bias'), u(256), u(32), u(L)], 256 * L);
    this.#dispatch(pass, 'applyMish', [output, u(256), u(L)], 256 * L);
    this.#mask(pass, output, 256, valid, L);
  }
  async #decode(frames, F, timestep) {
    const time = await this.#time.processTimeEmbedding(timestep), b = this.#b;
    let current = b.combined;
    for (let stage = 0; stage < 7; ++stage) {
      const L = stage > 0 && stage < 5 ? F / 2 : F;
      const valid = stage > 0 && stage < 5 ? Math.ceil(frames / 2) : frames;
      if (stage === 4 || stage === 5) {
        this.#submit(`flow stage ${stage}: concatenate skip`, encoder => this.#pass(encoder, 'concat', pass =>
          this.#dispatch(pass, 'concat_to_double_rows_per_col', [b.combined, current, stage === 4 ? b.skip1 : b.skip0, this.#scalar(256), this.#scalar(L)], 512 * L)));
        current = b.combined;
      }
      const source = current, output = source === b.stateA ? b.stateB : b.stateA;
      this.#submit(`flow stage ${stage}: resnet`, encoder => this.#pass(encoder, 'resnet', pass => {
        if (stage === 6) {
          this.#block(pass, output, source, 256, L, valid, 'decoder.estimator.final_block.block');
          this.#gemm(pass, b.velocity, this.#weights.get('decoder.estimator.final_proj.weight'), output, 80, 256, L);
          this.#bias(pass, b.velocity, this.#weights.get('decoder.estimator.final_proj.bias'), 80, L);
          this.#mask(pass, b.velocity, 80, valid, L);
          return;
        }
        const prefix = FLOW_RESNET_PREFIXES[stage], w = suffix => this.#weights.get(`${prefix}.${suffix}`);
        this.#block(pass, output, source, FLOW_INPUT_DIMS[stage], L, valid, `${prefix}.block1.block`);
        this.#gemm(pass, b.timeProjection, w('mlp.1.weight'), time.output, 256, 1024, 1);
        this.#bias(pass, b.timeProjection, w('mlp.1.bias'), 256, 1);
        this.#bias(pass, output, b.timeProjection, 256, L);
        // Full im2col captures block1 before block2 overwrites this same buffer.
        this.#block(pass, output, output, 256, L, valid, `${prefix}.block2.block`);
        // Original stage source survives both blocks. Columns are dead now and
        // can hold its residual projection, distinct from both source and output.
        this.#gemm(pass, b.columnsAndSplats, w('res_conv.weight'), source, 256, FLOW_INPUT_DIMS[stage], L);
        this.#bias(pass, b.columnsAndSplats, w('res_conv.bias'), 256, L);
        this.#dispatch(pass, 'elementwiseAdd', [output, b.columnsAndSplats, this.#scalar(256), this.#scalar(L)], 256 * L);
      }));
      if (stage === 6) return;
      await this.#transformer.flow_matching_transformer(output, stage, 256, 128, L, valid);
      this.#submit(`flow stage ${stage}: post transformer`, encoder => {
        // CUDA saves the UNMASKED transformer result in its skips.
        if (stage < 2) encoder.copyBufferToBuffer(output, 0, stage === 0 ? b.skip0 : b.skip1, 0, 256 * L * 4);
        if ([0, 1, 4, 5].includes(stage)) this.#pass(encoder, 'post transformer', pass => {
          this.#mask(pass, output, 256, valid, L);
          const prefix = FLOW_POST_PREFIXES[stage], weights = this.#weights.get(`${prefix}.weight`), bias = this.#weights.get(`${prefix}.bias`);
          if (stage === 4) {
            const u = n => this.#scalar(n);
            // Single-head X^T W: [L,256]*[256,1024] -> [L,1024] splats.
            this.#dispatch(pass, DOT, [b.columnsAndSplats, output, weights, u(256), u(256), u(1), u(256), u(L), u(256), u(1024)], L * 1024);
            this.#dispatch(pass, 'attach_splats_conv_transpose_1D', [output, b.columnsAndSplats, u(4), u(2), u(1), u(256), u(L), u(2 * L)], 256 * 2 * L);
            this.#bias(pass, output, bias, 256, 2 * L);
          } else this.#conv(pass, output, output, 256, L, weights, bias, stage === 0 ? 2 : 1);
        });
      });
      current = output;
    }
  }
  #idle() {
    if (this.#destroyed) throw new Error('FlowMatchingWebGPU has been destroyed');
    if (this.#lost) throw new Error(`WebGPU device lost: ${this.#lost.message}`);
    if (this.#busy) throw new Error('Await the current flow matching operation before reusing it');
  }
  #validateBuffer(input, count, needsCopy = false) {
    if (!input || input.size < count * 4 || !(input.usage & GPUBufferUsage.STORAGE) || (needsCopy && !(input.usage & GPUBufferUsage.COPY_SRC)))
      throw new Error('Input must be a sufficiently sized storage GPUBuffer with required copy usage');
    if (this.#owned.includes(input)) throw new Error('Input must be external to the flow workspace');
  }
  #result(output, frames, F, kind) {
    return this.#lastResult = Object.freeze({ device: this.#device, output, kind, generation: this.#generation,
      dim: 80, L: kind === 'mel' ? frames : F, shape: Object.freeze([80, kind === 'mel' ? frames : F]), layout: 'column-major',
      mel_frame_count: frames, compatible_mel_frame_count: F, stepCount: kind === 'mel' ? 5 : 0,
      ...(kind === 'mel' ? { get_final_user_facing_mel: output } : { final_x_down_projected_to_mel: output }) });
  }
  async runBuffer(conditioning, frames, compatible = Math.ceil(frames / 4) * 4) {
    this.#idle(); validateFlowLengths(frames, compatible, this.#plan.maxMelFrames); this.#validateBuffer(conditioning, 80 * frames);
    this.#busy = true; this.#lastResult = undefined; ++this.#generation;
    try {
      return await checked(this.#device, async () => {
        const b = this.#b, u = n => this.#scalar(n);
        this.#submit('flow zero initial noise', encoder => encoder.clearBuffer(b.evolvingNoiseAndMel, 0, 80 * compatible * 4));
        for (let step = 0; step < 5; ++step) {
          this.#submit(`Euler ${step}: condition`, encoder => this.#pass(encoder, 'condition', pass =>
            this.#dispatch(pass, 'concat_duration_repeated_tokens_to_noise', [b.combined, conditioning, b.evolvingNoiseAndMel, u(frames), u(compatible)], 160 * compatible)));
          await this.#decode(frames, compatible, FLOW_TIMESTEPS[step]);
          const dt = Math.fround(FLOW_TIMESTEPS[step + 1] - FLOW_TIMESTEPS[step]);
          this.#submit(`Euler ${step}: update`, encoder => this.#pass(encoder, 'update', pass =>
            this.#dispatch(pass, 'add_velocity_step_to_evolving_noise', [b.evolvingNoiseAndMel, b.velocity, this.#scalar(dt, 'f32'), u(80), u(compatible)], 80 * compatible)));
        }
        // Original final shader is in-place: no later step needs normalized noise.
        this.#submit('flow final mel', encoder => this.#pass(encoder, 'denormalize', pass =>
          this.#dispatch(pass, 'get_final_user_facing_mel', [b.evolvingNoiseAndMel, u(80), u(frames)], 80 * frames)));
        await this.#device.queue.onSubmittedWorkDone(); if (this.#lost) throw new Error('WebGPU device lost');
        return this.#result(b.evolvingNoiseAndMel, frames, compatible, 'mel');
      });
    } finally { this.#busy = false; }
  }
  /** Diagnostic/single-evaluation entry point for the translated decoder.
   * combined is [160,F] (noise first, conditioning second). Source is preserved.
   */
  async decodeBuffer(combined, frames, compatible, timestep) {
    this.#idle(); validateFlowLengths(frames, compatible, this.#plan.maxMelFrames); this.#validateBuffer(combined, 160 * compatible, true);
    this.#busy = true; this.#lastResult = undefined; ++this.#generation;
    try {
      return await checked(this.#device, async () => {
        this.#submit('flow copy decoder input', encoder => encoder.copyBufferToBuffer(combined, 0, this.#b.combined, 0, 160 * compatible * 4));
        await this.#decode(frames, compatible, timestep); await this.#device.queue.onSubmittedWorkDone();
        if (this.#lost) throw new Error('WebGPU device lost'); return this.#result(this.#b.velocity, frames, compatible, 'velocity');
      });
    } finally { this.#busy = false; }
  }
  async run_flow_matching_euler(context) {
    const duration = context?.duration ?? context;
    if (duration?.device !== this.#device) throw new Error('Duration and flow must share the same device');
    return this.runBuffer(duration.output ?? duration.x_mel_dim_repeat_by_duration, duration.mel_frame_count, duration.compatible_mel_frame_count);
  }
  async readOutput(result) {
    this.#idle(); if (!result || result !== this.#lastResult) throw new Error('Expected the current flow result; stale or foreign result');
    this.#busy = true; let staging;
    try {
      return await checked(this.#device, async () => {
        staging = this.#device.createBuffer({ size: 80 * result.L * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        this.#submit('flow readback', encoder => encoder.copyBufferToBuffer(result.output, 0, staging, 0, staging.size));
        await staging.mapAsync(GPUMapMode.READ); const values = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); return values;
      });
    } finally { staging?.destroy(); this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current flow matching operation before destroying it');
    this.#time?.destroy(); this.#transformer?.destroy();
    for (const buffer of this.#owned) buffer.destroy(); this.#owned.length = 0; this.#scalars.clear(); this.#lastResult = undefined; this.#destroyed = true;
  }
}
