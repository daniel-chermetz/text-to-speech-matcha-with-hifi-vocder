import { TextConditioningWebGPU } from './text_conditioning_webgpu.mjs';
import { FlowMatchingWebGPU, extractFlowMatchingKernels } from './inference_flow_matching_webgpu.mjs';
import { extractPreConvKernels } from './inference_pre_conv_webgpu.mjs';
import { extractTransformerKernels, transformerMemoryPlan } from './inference_transformer_webgpu.mjs';
import { extractDurationKernels, durationMemoryPlan } from './inference_duration_webgpu.mjs';
import { extractTimeEmbeddingKernels, timeEmbeddingMemoryPlan } from './inference_time_embedding_webgpu.mjs';
import { extractFlowTransformerKernels, flowTransformerMemoryPlan } from './inference_flow_matching_transformer_webgpu.mjs';
import { flowMatchingMemoryPlan } from './inference_flow_matching_webgpu.mjs';
import { inferenceTensorSpecs } from './flow_matching_weights.mjs';
export { parseInferenceWeights } from './flow_matching_weights.mjs';
export { requestInferenceDevice } from './inference_flow_matching_webgpu.mjs';

export async function loadInferenceKernels() {
  const [pre, text, duration, time, flowTransformer, flow, blas] = await Promise.all(['inference_pre_conv.js', 'inference_transformer.js',
    'inference_duration.js', 'inference_time_embedding.js', 'inference_flow_matching_transformer.js', 'inference_flow_matching.js', 'blas_gemm.js'].map(async name => {
    const response = await fetch(new URL(name, import.meta.url)); if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}`); return response.text();
  }));
  return { preConv: extractPreConvKernels(pre, blas), transformer: extractTransformerKernels(pre, text, blas),
    duration: extractDurationKernels(pre, duration, blas), timeEmbedding: extractTimeEmbeddingKernels(time, pre, flow, blas),
    flowTransformer: extractFlowTransformerKernels(pre, text, flowTransformer, blas), flowMatching: extractFlowMatchingKernels(pre, flow, blas) };
}
const TOKEN = Symbol('InferenceWebGPU');
export function inferenceMemoryPlan({ maxTokenLength = 1536, textFfnTileSize = 512, flowFfnTileSize = 256 } = {}) {
  const workspaceByStage = Object.freeze({ preConv: 1923 * maxTokenLength * 4,
    textTransformer: transformerMemoryPlan(maxTokenLength, textFfnTileSize).workspaceBytes,
    duration: durationMemoryPlan(maxTokenLength).workspaceBytes,
    flowDecoder: flowMatchingMemoryPlan().workspaceBytes,
    flowTransformer: flowTransformerMemoryPlan(1024, flowFfnTileSize).workspaceBytes,
    timeEmbedding: timeEmbeddingMemoryPlan().workspaceBytes });
  return Object.freeze({ workspaceByStage, workspaceBytes: Object.values(workspaceByStage).reduce((a, b) => a + b, 0),
    weightBytes: inferenceTensorSpecs().slice(0, -2).reduce((n, s) => n + s.shape.reduce((a, b) => a * b, 1) * 4, 0),
    // Pre-conv has 9 scalar buffers, time has 4; other scalars are allocated lazily.
    initialScalarBytes: 52 });
}
/** Complete token IDs -> final mel inference. No downstream callback required.
 * Tokenization and vocoding are outside the supplied CUDA inference workflow.
 */
export class InferenceWebGPU {
  #conditioning; #flow; #plan; #busy = false; #destroyed = false; #lastResult;
  constructor(token) { if (token !== TOKEN) throw new Error('Use InferenceWebGPU.create()'); }
  static async create({ device, weights, kernels, maxTokenLength = 1536, textFfnTileSize = 512, flowFfnTileSize = 256, frequencies }) {
    const self = new InferenceWebGPU(TOKEN);
    self.#plan = inferenceMemoryPlan({ maxTokenLength, textFfnTileSize, flowFfnTileSize });
    try {
      self.#conditioning = await TextConditioningWebGPU.create({ device, weights, kernels, maxTokenLength, ffnTileSize: textFfnTileSize });
      self.#flow = await FlowMatchingWebGPU.create({ device, weights, kernels, ffnTileSize: flowFfnTileSize, frequencies });
      return self;
    } catch (error) { self.destroy(); throw error; }
  }
  get memoryPlan() { return { ...this.#plan, conditioning: this.#conditioning.memoryPlan, flow: this.#flow.memoryPlan }; }
  #idle() {
    if (this.#destroyed) throw new Error('InferenceWebGPU has been destroyed');
    if (this.#busy) throw new Error('Await the current inference operation before reusing it');
  }
  async run(tokenIds, options) {
    this.#idle(); this.#busy = true; this.#lastResult = undefined;
    try {
      const conditioning = await this.#conditioning.run(tokenIds, options);
      if (conditioning.skippedFlowMatching) return this.#lastResult = Object.freeze({ conditioning,
        device: conditioning.device, output: null, shape: null, skippedFlowMatching: true,
        mel_frame_count: conditioning.mel_frame_count, compatible_mel_frame_count: conditioning.compatible_mel_frame_count });
      const flow = await this.#flow.run_flow_matching_euler(conditioning);
      return this.#lastResult = Object.freeze({ ...flow, flow, conditioning, skippedFlowMatching: false });
    } finally { this.#busy = false; }
  }
  async runInference(tokenIds, options) { return this.run(tokenIds, options); }
  async readOutput(result) {
    this.#idle(); if (!result || result !== this.#lastResult || result.skippedFlowMatching) throw new Error('No current final mel result; skipped or stale');
    this.#busy = true;
    try { return await this.#flow.readOutput(result.flow); } finally { this.#busy = false; }
  }
  destroy() {
    if (this.#busy) throw new Error('Await the current inference operation before destroying it');
    this.#flow?.destroy(); this.#conditioning?.destroy(); this.#lastResult = undefined; this.#destroyed = true;
  }
}
