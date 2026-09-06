import { preConvTensorSpecs, parseWeightsPrefix } from './pre_conv_weights.mjs';
import { transformerTensorSpecs } from './transformer_weights.mjs';
import { durationTensorSpecs } from './duration_weights.mjs';

export const FLOW_TRANSFORMER_MODEL = Object.freeze({ dim: 256, halfDim: 128, heads: 2,
  headDim: 64, ffnDim: 1024, eps: 1e-5, layers: 6, maxFrameLength: 1280 });
export const FLOW_TRANSFORMER_PREFIXES = Object.freeze([
  'decoder.estimator.down_blocks.0.1.0', 'decoder.estimator.down_blocks.1.1.0',
  'decoder.estimator.mid_blocks.0.1.0', 'decoder.estimator.mid_blocks.1.1.0',
  'decoder.estimator.up_blocks.0.1.0', 'decoder.estimator.up_blocks.1.1.0',
]);

// These four records precede the flow transformer in the existing binary.
// Their metadata is needed for parsing; this module does not run time embedding.
export function timeEmbeddingTensorSpecs() {
  return [1, 2].flatMap(i => [
    { name: `decoder.estimator.time_mlp.linear_${i}.weight`, layout: 2, shape: [1024, i === 1 ? 160 : 1024] },
    { name: `decoder.estimator.time_mlp.linear_${i}.bias`, layout: 3, shape: [1024] },
  ]);
}
export function flowTransformerTensorSpecs() {
  return FLOW_TRANSFORMER_PREFIXES.flatMap(prefix => [
    ['norm1.weight', [256]], ['norm1.bias', [256]],
    ['attn1.to_q.weight', [128, 256]], ['attn1.to_k.weight', [128, 256]], ['attn1.to_v.weight', [128, 256]],
    ['attn1.to_out.0.weight', [256, 128]], ['attn1.to_out.0.bias', [256]],
    ['norm3.weight', [256]], ['norm3.bias', [256]],
    ['ff.net.0.proj.weight', [1024, 256]], ['ff.net.0.proj.bias', [1024]],
    ['ff.net.0.alpha', [1024]], ['ff.net.0.beta', [1024]],
    ['ff.net.2.weight', [256, 1024]], ['ff.net.2.bias', [256]],
  ].map(([suffix, shape]) => ({ name: `${prefix}.${suffix}`, shape, layout: shape.length === 2 ? 2 : 3 })));
}
export function flowTransformerPrefixSpecs() {
  return [...preConvTensorSpecs(), ...transformerTensorSpecs(), ...durationTensorSpecs(),
    ...timeEmbeddingTensorSpecs(), ...flowTransformerTensorSpecs()];
}
/** Full original MTTSCUDA export or exact 217-record prefix; returns that prefix.
 * Snake alpha/beta in this format are ALREADY exponentiated by port_weights.py.
 */
export function parseFlowTransformerWeights(input) {
  return parseWeightsPrefix(input, flowTransformerPrefixSpecs());
}
export function validateFlowTransformerWeights(weights) {
  if (!(weights instanceof Map)) throw new TypeError('weights must be a Map of tensor names to Float32Array');
  for (const { name, shape } of flowTransformerTensorSpecs()) {
    const values = weights.get(name), count = shape.reduce((a, b) => a * b, 1);
    if (!(values instanceof Float32Array) || values.length !== count || !values.every(Number.isFinite))
      throw new Error(`${name}: expected ${count} finite float32 values in exported layout`);
    if ((name.endsWith('.alpha') || name.endsWith('.beta')) && values.some(v => v < 0))
      throw new Error(`${name}: expected nonnegative, already-exponentiated Snake parameters`);
  }
}
