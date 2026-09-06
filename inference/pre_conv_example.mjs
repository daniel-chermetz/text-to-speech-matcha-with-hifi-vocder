import { PreConvWebGPU, loadPreConvKernels, parsePreConvWeights } from './inference_pre_conv_webgpu.mjs';

/** Example integration, with weights already exported by port_weights.py.
 * Caller owns/destroys the GPUDevice. This function owns/destroys its workspace.
 * tokenIds are the tokenizer's complete sequence (including interspersed blanks).
 */
export async function preConvToCPU(device, weightsArrayBuffer, tokenIds) {
  const runtime = await PreConvWebGPU.create({
    device,
    kernels: await loadPreConvKernels(),
    weights: parsePreConvWeights(weightsArrayBuffer),
    maxTokenLength: tokenIds.length,
  });
  try {
    const result = await runtime.runPreConv(tokenIds);
    return { shape: result.shape, layout: result.layout,
      values: await runtime.readBuffer(result.output, result.dim * result.L) };
  } finally {
    runtime.destroy();
  }
}
