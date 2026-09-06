import { preConvTensorSpecs, parseWeightsPrefix } from './pre_conv_weights.mjs';
import { transformerTensorSpecs } from './transformer_weights.mjs';
import { durationTensorSpecs } from './duration_weights.mjs';
import { timeEmbeddingTensorSpecs } from './flow_transformer_weights.mjs';
export { timeEmbeddingTensorSpecs };

export const TIME_EMBEDDING_MODEL = Object.freeze({ frequencyCount: 80, inputDim: 160,
  hiddenDim: 1024, outputDim: 1024, timeScale: 1000 });
export function timeEmbeddingPrefixSpecs() {
  return [...preConvTensorSpecs(), ...transformerTensorSpecs(), ...durationTensorSpecs(), ...timeEmbeddingTensorSpecs()];
}
/** Original full MTTSCUDA model, or the exact first 127 records through time MLP. */
export function parseTimeEmbeddingWeights(input) {
  return parseWeightsPrefix(input, timeEmbeddingPrefixSpecs());
}
export function validateTimeEmbeddingWeights(weights) {
  if (!(weights instanceof Map)) throw new TypeError('weights must be a Map of tensor names to Float32Array');
  for (const { name, shape } of timeEmbeddingTensorSpecs()) {
    const values = weights.get(name), count = shape.reduce((a, b) => a * b, 1);
    if (!(values instanceof Float32Array) || values.length !== count || !values.every(Number.isFinite))
      throw new Error(`${name}: expected ${count} finite float32 values in exported layout`);
  }
}
/** CPU initialization from allocate_memory.cu; float operations rounded at each
 * source float boundary. JS and C++ transcendental libraries may differ by ulps.
 * Pass a captured CUDA frequency table to create() if exact table bytes matter.
 */
export function createTimeEmbeddingFrequencies() {
  const f = Math.fround, step = f(f(Math.log(10000)) / 79);
  return Float32Array.from({ length: 80 }, (_, i) => f(Math.exp(f(-i * step))));
}
