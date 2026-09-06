import { parseWeightsPrefix } from './pre_conv_weights.mjs';
import { flowTransformerPrefixSpecs } from './flow_transformer_weights.mjs';

export const FLOW_RESNET_PREFIXES = Object.freeze(['decoder.estimator.down_blocks.0.0', 'decoder.estimator.down_blocks.1.0',
  'decoder.estimator.mid_blocks.0.0', 'decoder.estimator.mid_blocks.1.0', 'decoder.estimator.up_blocks.0.0', 'decoder.estimator.up_blocks.1.0']);
export const FLOW_INPUT_DIMS = Object.freeze([160, 256, 256, 256, 512, 512, 256]);
export const FLOW_POST_PREFIXES = Object.freeze(['decoder.estimator.down_blocks.0.2.conv', 'decoder.estimator.down_blocks.1.2',
  null, null, 'decoder.estimator.up_blocks.0.2.conv', 'decoder.estimator.up_blocks.1.2']);
export const FLOW_TIMESTEPS = Object.freeze([0, 0.2, 0.4, 0.6, 0.8, 1].map(Math.fround));
export function flowMatchingTensorSpecs() {
  const result = [];
  const add = (name, shape) => result.push({ name, shape, layout: shape.length === 2 ? 2 : 3 });
  for (const [stage, prefix] of FLOW_RESNET_PREFIXES.entries()) {
    for (const block of [1, 2]) {
      const p = `${prefix}.block${block}.block`;
      add(`${p}.0.weight`, [256, 3 * (block === 1 ? FLOW_INPUT_DIMS[stage] : 256)]);
      add(`${p}.0.bias`, [256]); add(`${p}.1.weight`, [256]); add(`${p}.1.bias`, [256]);
      if (block === 1) { add(`${prefix}.mlp.1.weight`, [256, 1024]); add(`${prefix}.mlp.1.bias`, [256]); }
    }
    add(`${prefix}.res_conv.weight`, [256, FLOW_INPUT_DIMS[stage]]); add(`${prefix}.res_conv.bias`, [256]);
  }
  for (const stage of [0, 1, 4, 5]) {
    add(`${FLOW_POST_PREFIXES[stage]}.weight`, [256, stage === 4 ? 1024 : 768]);
    add(`${FLOW_POST_PREFIXES[stage]}.bias`, [256]);
  }
  add('decoder.estimator.final_block.block.0.weight', [256, 768]);
  add('decoder.estimator.final_block.block.0.bias', [256]);
  add('decoder.estimator.final_block.block.1.weight', [256]);
  add('decoder.estimator.final_block.block.1.bias', [256]);
  add('decoder.estimator.final_proj.weight', [80, 256]); add('decoder.estimator.final_proj.bias', [80]);
  return result;
}
export function inferenceTensorSpecs() {
  return [...flowTransformerPrefixSpecs(), ...flowMatchingTensorSpecs(),
    { name: 'mel_mean', shape: [1], layout: 3 }, { name: 'mel_std', shape: [1], layout: 3 }];
}
/** Exact complete 305-record original model. Earlier prefix readers still work. */
export function parseInferenceWeights(input) {
  const bytes = input instanceof ArrayBuffer ? new Uint8Array(input)
    : ArrayBuffer.isView(input) ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength) : null;
  if (!bytes || bytes.byteLength < 16 || new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(12, true) !== 305)
    throw new Error('Expected the complete 305-record MTTSCUDA model');
  return parseWeightsPrefix(bytes, inferenceTensorSpecs());
}
export function validateFlowMatchingWeights(weights) {
  if (!(weights instanceof Map)) throw new TypeError('weights must be a Map');
  for (const { name, shape } of flowMatchingTensorSpecs()) {
    const data = weights.get(name);
    if (!(data instanceof Float32Array) || data.length !== shape.reduce((a, b) => a * b, 1) || !data.every(Number.isFinite))
      throw new Error(`${name}: expected finite float32 values with exported shape [${shape}]`);
  }
}
