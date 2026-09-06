import { preConvTensorSpecs, parseWeightsPrefix } from './pre_conv_weights.mjs';

export const TRANSFORMER_MODEL = Object.freeze({
  dim: 192, layers: 6, heads: 2, headDim: 96, ropeDim: 48,
  ffnDim: 768, kernel: 3, padding: 1, eps: 1e-4, maxTokenLength: 1536,
});

export function transformerTensorSpecs() {
  const specs = [];
  const add = (name, shape) => specs.push({ name, shape, layout: shape.length === 1 ? 3 : 2 });
  for (let i = 0; i < 6; ++i) {
    for (const p of ['q', 'k', 'v', 'o']) {
      add(`encoder.encoder.attn_layers.${i}.conv_${p}.weight`, [192, 192]);
      add(`encoder.encoder.attn_layers.${i}.conv_${p}.bias`, [192]);
    }
    add(`encoder.encoder.norm_layers_1.${i}.gamma`, [192]);
    add(`encoder.encoder.norm_layers_1.${i}.beta`, [192]);
    add(`encoder.encoder.ffn_layers.${i}.conv_1.weight`, [768, 576]);
    add(`encoder.encoder.ffn_layers.${i}.conv_1.bias`, [768]);
    add(`encoder.encoder.ffn_layers.${i}.conv_2.weight`, [192, 2304]);
    add(`encoder.encoder.ffn_layers.${i}.conv_2.bias`, [192]);
    add(`encoder.encoder.norm_layers_2.${i}.gamma`, [192]);
    add(`encoder.encoder.norm_layers_2.${i}.beta`, [192]);
  }
  return specs;
}

/** Full model or exact 111-record text-encoder prefix; both stages share the Map. */
export function parseTextEncoderWeights(input) {
  return parseWeightsPrefix(input, [...preConvTensorSpecs(), ...transformerTensorSpecs()]);
}

export function validateTransformerWeights(weights) {
  if (!(weights instanceof Map)) throw new TypeError('weights must be a Map of tensor names to Float32Array');
  for (const { name, shape } of transformerTensorSpecs()) {
    const data = weights.get(name), count = shape.reduce((a, b) => a * b, 1);
    if (!(data instanceof Float32Array) || data.length !== count || !data.every(Number.isFinite))
      throw new Error(`${name}: expected ${count} finite float32 values in exported layout`);
  }
}

/** CPU constant-table initialization from allocate_memory.cu, not tensor inference.
 * Each token has a 96-float stride; only the first 48 entries contain 24 cos/sin pairs.
 * Both 96-channel heads rotate coordinates [j,j+24], j=0..23; coordinates 48..95 stay intact.
 */
export function createTransformerRoPE(maxTokenLength) {
  if (!Number.isInteger(maxTokenLength) || maxTokenLength < 1 || maxTokenLength > 1536)
    throw new RangeError('RoPE capacity must be in [1,1536]');
  const f = Math.fround, values = new Float32Array(96 * maxTokenLength);
  for (let t = 0; t < maxTokenLength; ++t) {
    for (let pair = 0; pair < 24; ++pair) {
      const frequency = f(Math.pow(10000, f(-(2 * pair) / 48)));
      const angle = f(f(t) * frequency);
      values[t * 96 + pair * 2] = Math.cos(angle);
      values[t * 96 + pair * 2 + 1] = Math.sin(angle);
    }
  }
  return values;
}
