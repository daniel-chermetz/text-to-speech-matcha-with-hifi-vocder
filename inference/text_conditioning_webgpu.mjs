import { TextEncoderWebGPU } from './text_encoder_webgpu.mjs';
import { extractPreConvKernels } from './inference_pre_conv_webgpu.mjs';
import { extractTransformerKernels } from './inference_transformer_webgpu.mjs';
import { DurationWebGPU, extractDurationKernels } from './inference_duration_webgpu.mjs';
export { parseTextConditioningWeights } from './duration_weights.mjs';
export { requestTextEncoderDevice } from './inference_transformer_webgpu.mjs';

export async function loadTextConditioningKernels() {
  const [pre, transformer, duration, blas] = await Promise.all([
    'inference_pre_conv.js', 'inference_transformer.js', 'inference_duration.js', 'blas_gemm.js',
  ].map(async name => {
    const response = await fetch(new URL(name, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`);
    return response.text();
  }));
  return { preConv: extractPreConvKernels(pre, blas), transformer: extractTransformerKernels(pre, transformer, blas),
    duration: extractDurationKernels(pre, duration, blas) };
}

const TOKEN = Symbol('TextConditioningWebGPU');
/** Token IDs -> pre-conv -> transformer -> predicted, duration-expanded mel conditioning.
 * This is the flow matcher's conditioning input, not generated/synthesized mel audio.
 */
export class TextConditioningWebGPU {
  #encoder; #duration; #busy = false; #destroyed = false;
  constructor(token) { if (token !== TOKEN) throw new Error('Use TextConditioningWebGPU.create()'); }
  static async create({ device, weights, kernels, maxTokenLength = 1536, ffnTileSize = 512 }) {
    const self = new TextConditioningWebGPU(TOKEN);
    try {
      self.#encoder = await TextEncoderWebGPU.create({ device, weights, kernels, maxTokenLength, ffnTileSize });
      self.#duration = await DurationWebGPU.create({ device, weights, kernels: kernels?.duration, maxTokenLength });
      return self;
    } catch (error) { self.destroy(); throw error; }
  }
  get memoryPlan() { return { transformer: this.#encoder.memoryPlan, duration: this.#duration.memoryPlan }; }
  #idle() {
    if (this.#destroyed) throw new Error('TextConditioningWebGPU has been destroyed');
    if (this.#busy) throw new Error('Await the current text conditioning operation before reusing it');
  }
  async #run(tokenIds, options) {
    const transformer = await this.#encoder.run(tokenIds, options);
    const duration = await this.#duration.infer_text_token_durations(192, transformer.L, transformer);
    return { ...duration, transformer, duration, skippedFlowMatching: duration.skippedExpansion };
  }
  async run(tokenIds, options) {
    this.#idle(); this.#busy = true;
    try { return await this.#run(tokenIds, options); } finally { this.#busy = false; }
  }
  async runInference(tokenIds, stages) {
    this.#idle();
    if (typeof stages?.run_flow_matching_euler !== 'function') throw new Error('Missing downstream stage: run_flow_matching_euler');
    this.#busy = true;
    try {
      const context = await this.#run(tokenIds);
      if (context.skippedFlowMatching) return context;
      context.flow = await stages.run_flow_matching_euler(context);
      return context;
    } finally { this.#busy = false; }
  }
  async readOutput(result, options) {
    this.#idle(); this.#busy = true;
    try { return await this.#duration.readOutput(result.duration ?? result, options); } finally { this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current text conditioning operation before destroying it');
    this.#duration?.destroy(); this.#encoder?.destroy(); this.#destroyed = true;
  }
}
