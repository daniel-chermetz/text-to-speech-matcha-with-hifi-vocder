/**
 * HiFi v1 generator weight blob reader.
 *
 * Parses `model/generator_v1_weights.bin` exactly as `weights_load.cu` walks it:
 * a flat little-endian float32 blob with no framing, consumed in a fixed order.
 * The exporter already stores every tensor in the memory order the cuBLAS calls
 * expect, so each tensor is a verbatim slice.
 */

export const GENERATOR = Object.freeze({
  channels: 512,          // h
  melBins: 80,
  upsampleStages: 4,      // N
  mrfStreams: 3,
  mrfIterations: 3,
  upsampleRates: Object.freeze([8, 8, 2, 2]),          // u[]
  resblockKernels: Object.freeze([3, 7, 11]),          // MRF_kernel_by_stream
  resblockDilations: Object.freeze([1, 3, 5]),         // MRF_dilation_by_iteration
  convPreKernel: 7,
  convPostKernel: 7,
  finalChannels: 32,
  samplingRate: 22050,
  hopSize: 256,
  leakyReluSlope: 0.1,
  finalLeakyReluSlope: 0.01,                           // torch default before conv_post
});

/** `u[i] * 2`; the config's upsample_kernel_sizes are exactly twice the rates. */
export const upsampleKernel = stage => GENERATOR.upsampleRates[stage] * 2;

/**
 * The blob's tensor order, matching `weights_load.cu` and the exporter manifest.
 * @returns {{ name: string, count: number }[]} 156 entries, 13,926,017 floats.
 */
export function generatorTensorSpecs() {
  const { channels, melBins, upsampleStages, mrfStreams, mrfIterations,
    resblockKernels, convPreKernel, convPostKernel } = GENERATOR;
  const specs = [
    { name: 'conv_pre.weight', count: melBins * convPreKernel * channels },
    { name: 'conv_pre.bias', count: channels },
  ];

  let current = channels;
  for (let stage = 0; stage < upsampleStages; ++stage) {
    const next = current / 2;
    specs.push({ name: `ups.${stage}.weight`, count: current * next * upsampleKernel(stage) });
    specs.push({ name: `ups.${stage}.bias`, count: next });
    for (let stream = 0; stream < mrfStreams; ++stream) {
      const convCount = next * resblockKernels[stream] * next;
      for (let iteration = 0; iteration < mrfIterations; ++iteration) {
        const base = `resblocks.${stage}.${stream}`;
        specs.push({ name: `${base}.convs1.${iteration}.weight`, count: convCount });
        specs.push({ name: `${base}.convs1.${iteration}.bias`, count: next });
        specs.push({ name: `${base}.convs2.${iteration}.weight`, count: convCount });
        specs.push({ name: `${base}.convs2.${iteration}.bias`, count: next });
      }
    }
    current = next;
  }

  specs.push({ name: 'conv_post.weight', count: current * convPostKernel * 1 });
  specs.push({ name: 'conv_post.bias', count: 1 });
  return specs;
}

/**
 * @param {ArrayBuffer} arrayBuffer The complete `generator_v1_weights.bin`.
 * @returns {Map<string, Float32Array>} Views into a single copy of the blob.
 */
export function parseGeneratorWeights(arrayBuffer) {
  if (!(arrayBuffer instanceof ArrayBuffer)) throw new TypeError('Expected an ArrayBuffer of generator weights');

  const specs = generatorTensorSpecs();
  const expectedFloats = specs.reduce((total, spec) => total + spec.count, 0);
  if (arrayBuffer.byteLength !== expectedFloats * 4)
    throw new Error(`Generator blob is ${arrayBuffer.byteLength} bytes; expected ${expectedFloats * 4}`);

  const all = new Float32Array(arrayBuffer);
  const weights = new Map();
  let offset = 0;
  for (const { name, count } of specs) {
    const tensor = all.subarray(offset, offset + count);
    for (let i = 0; i < count; ++i) {
      if (!Number.isFinite(tensor[i])) throw new Error(`Tensor ${name} holds a non-finite value at ${i}`);
    }
    weights.set(name, tensor);
    offset += count;
  }
  return weights;
}

/**
 * Per-stage tensor geometry, derived the same way `runInference` derives it.
 *
 * `timeSeriesSizePerChannel` is `nextExpandedSize * frames / next_h / 2`, which
 * reduces to `frames * upsampleRates[stage]`.
 */
export function generatorStageGeometry(melFrames) {
  const stages = [];
  let timeSeries = melFrames;
  let currentChannels = GENERATOR.channels;
  for (let stage = 0; stage < GENERATOR.upsampleStages; ++stage) {
    const nextChannels = currentChannels / 2;
    const nextUpsample = upsampleKernel(stage);
    const nextExpandedSize = nextChannels * nextUpsample;
    const frames = timeSeries;
    timeSeries = (nextExpandedSize * frames) / nextChannels / 2;
    stages.push(Object.freeze({
      stage, currentChannels, nextChannels, nextUpsample, nextExpandedSize,
      frames, timeSeries,
      stride: GENERATOR.upsampleRates[stage],
      padding: (nextUpsample - GENERATOR.upsampleRates[stage]) / 2,
    }));
    currentChannels = nextChannels;
  }
  return Object.freeze({ stages: Object.freeze(stages), audioSamples: timeSeries });
}
