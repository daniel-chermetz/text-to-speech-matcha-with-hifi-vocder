import { PreConvWebGPU, extractPreConvKernels, validateTokenIds } from './inference_pre_conv_webgpu.mjs';
import { TransformerWebGPU, extractTransformerKernels } from './inference_transformer_webgpu.mjs';
export { requestTextEncoderDevice } from './inference_transformer_webgpu.mjs';
export { parseTextEncoderWeights } from './transformer_weights.mjs';

export async function loadTextEncoderKernels() {
  const sources = await Promise.all(['inference_pre_conv.js', 'inference_transformer.js', 'blas_gemm.js'].map(async name => {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    return response.text();
  }));
  return { preConv: extractPreConvKernels(sources[0], sources[2]), transformer: extractTransformerKernels(...sources) };
}

const TOKEN = Symbol('TextEncoderWebGPU');
/** Token IDs -> pre-convolution -> six transformer layers; all tensor data stays
 * on one GPUDevice. The pre-conv output is consumed in place by the transformer.
 */
export class TextEncoderWebGPU {
  #pre; #transformer; #capacity; #device; #busy = false; #destroyed = false;
  constructor(token) { if (token !== TOKEN) throw new Error('Use TextEncoderWebGPU.create()'); }
  static async create({ device, weights, kernels, maxTokenLength = 1536, ffnTileSize = 512 }) {
    const self = new TextEncoderWebGPU(TOKEN);
    self.#device = device; self.#capacity = maxTokenLength;
    try {
      // Check the transformer's stricter device requirements first.
      self.#transformer = await TransformerWebGPU.create({ device, weights, kernels: kernels?.transformer, maxTokenLength, ffnTileSize });
      self.#pre = await PreConvWebGPU.create({ device, weights, kernels: kernels?.preConv, maxTokenLength });
      return self;
    } catch (error) { self.destroy(); throw error; }
  }
  get memoryPlan() { return this.#transformer.memoryPlan; }
  #idle() {
    if (this.#destroyed) throw new Error('TextEncoderWebGPU has been destroyed');
    if (this.#busy) throw new Error('Await the current text encoder operation before reusing it');
  }
  async #encode(tokenIds, matchaProtocol) {
    const tokens = validateTokenIds(tokenIds, this.#capacity, { matchaProtocol });
    const pre = await this.#pre.runPreConv(tokens);
    return this.#transformer.run_embedding_transformer_stack(192, pre.L, pre);
  }
  async run(tokenIds, { matchaProtocol = true } = {}) {
    this.#idle(); this.#busy = true;
    try { return await this.#encode(tokenIds, matchaProtocol); } finally { this.#busy = false; }
  }
  /** The untranslated duration and flow stages remain explicit required hooks.
   * context.transformer.output is the six-layer result; context.output is identical.
   */
  async runInference(tokenIds, stages) {
    this.#idle();
    for (const name of ['infer_text_token_durations', 'run_flow_matching_euler'])
      if (typeof stages?.[name] !== 'function') throw new Error(`Missing downstream stage: ${name}`);
    this.#busy = true;
    try {
      const transformer = await this.#encode(tokenIds, true);
      const context = { ...transformer, transformer };
      context.duration = await stages.infer_text_token_durations(192, transformer.L, context);
      const count = context.duration?.compatible_mel_frame_count;
      if (!Number.isInteger(count) || count < 0) throw new Error('Duration must return a nonnegative integer compatible_mel_frame_count');
      context.compatible_mel_frame_count = count;
      if (count > 1024) return { ...context, skippedFlowMatching: true };
      context.flow = await stages.run_flow_matching_euler(context);
      return { ...context, skippedFlowMatching: false };
    } finally { this.#busy = false; }
  }
  async readOutput(result) {
    this.#idle(); this.#busy = true;
    try { return await this.#transformer.readOutput(result); } finally { this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current text encoder operation before destroying it');
    this.#transformer?.destroy(); this.#pre?.destroy(); this.#destroyed = true;
  }
}
