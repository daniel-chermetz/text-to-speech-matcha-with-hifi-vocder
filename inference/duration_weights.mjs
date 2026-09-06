import { preConvTensorSpecs, parseWeightsPrefix } from './pre_conv_weights.mjs';
import { transformerTensorSpecs } from './transformer_weights.mjs';

export const DURATION_MODEL = Object.freeze({ inputDim: 192, hiddenDim: 256, melDim: 80,
  kernel: 3, padding: 1, eps: 1e-4, maxTokenLength: 1536, maxMelFrames: 1024 });

export function durationTensorSpecs() {
  const specs = [
    { name: 'encoder.proj_m.weight', shape: [80, 192], layout: 2 },
    { name: 'encoder.proj_m.bias', shape: [80], layout: 3 },
  ];
  for (const i of [1, 2]) specs.push(
    { name: `encoder.proj_w.conv_${i}.weight`, shape: [256, i === 1 ? 576 : 768], layout: 2 },
    { name: `encoder.proj_w.conv_${i}.bias`, shape: [256], layout: 3 },
    { name: `encoder.proj_w.norm_${i}.gamma`, shape: [256], layout: 3 },
    { name: `encoder.proj_w.norm_${i}.beta`, shape: [256], layout: 3 },
  );
  specs.push(
    { name: 'encoder.proj_w.proj.weight', shape: [1, 256], layout: 2 },
    { name: 'encoder.proj_w.proj.bias', shape: [1], layout: 3 },
  );
  return specs;
}

/** Full MTTSCUDA model or exact 123-record prefix through duration prediction. */
export function parseTextConditioningWeights(input) {
  return parseWeightsPrefix(input, [...preConvTensorSpecs(), ...transformerTensorSpecs(), ...durationTensorSpecs()]);
}

export function validateDurationWeights(weights) {
  if (!(weights instanceof Map)) throw new TypeError('weights must be a Map of tensor names to Float32Array');
  for (const { name, shape } of durationTensorSpecs()) {
    const values = weights.get(name), count = shape.reduce((a, b) => a * b, 1);
    if (!(values instanceof Float32Array) || values.length !== count || !values.every(Number.isFinite))
      throw new Error(`${name}: expected ${count} finite float32 values in exported layout`);
  }
}
