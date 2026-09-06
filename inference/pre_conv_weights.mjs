// Host-only reader for the binary format in port_weights.py / load_weights.cu.
export const PRE_CONV_MODEL = Object.freeze({
  dim: 192, vocabularySize: 178, kernel: 5, padding: 2, stride: 1,
  layers: 3, slots: 4, eps: 1e-4, slope: 0,
  maximumInputTokenCount: 1536, maximumTokenLength: 32767,
});

export function preConvTensorSpecs() {
  const specs = [{ name: 'encoder.emb.weight', layout: 1, shape: [178, 192] }];
  for (let i = 0; i < 3; ++i) {
    specs.push(
      { name: `encoder.prenet.conv_layers.${i}.weight`, layout: 2, shape: [192, 960] },
      { name: `encoder.prenet.conv_layers.${i}.bias`, layout: 3, shape: [192] },
      { name: `encoder.prenet.norm_layers.${i}.gamma`, layout: 3, shape: [192] },
      { name: `encoder.prenet.norm_layers.${i}.beta`, layout: 3, shape: [192] },
    );
  }
  specs.push(
    { name: 'encoder.prenet.proj.weight', layout: 2, shape: [192, 192] },
    { name: 'encoder.prenet.proj.bias', layout: 3, shape: [192] },
  );
  return specs;
}

/** Read a full MTTSCUDA v1 file, or a file containing its 15 pre-conv records.
 * Validate the exact pre-conv prefix. Check remaining record framing, but leave
 * validation of downstream model-specific names/shapes to the later loaders.
 * Return independent Float32Arrays in the EXISTING exported memory layouts.
 */
export function parsePreConvWeights(input) {
  return parseWeightsPrefix(input, preConvTensorSpecs());
}

// Shared framing reader; the caller supplies the complete expected prefix.
export function parseWeightsPrefix(input, specs) {
  const bytes = input instanceof ArrayBuffer ? new Uint8Array(input)
    : ArrayBuffer.isView(input) ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
    : null;
  if (!bytes) throw new TypeError('Expected an ArrayBuffer or typed-array view');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const result = new Map();
  const names = new Set();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let offset = 0;
  function need(count) {
    if (!Number.isSafeInteger(count) || count < 0 || offset + count > bytes.length)
      throw new Error(`Truncated or oversized weights record at byte ${offset}`);
  }
  function u64(at) {
    const value = view.getBigUint64(at, true);
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Oversized tensor dimension/count');
    return Number(value);
  }
  need(16);
  if (decoder.decode(bytes.subarray(0, 8)) !== 'MTTSCUDA' || view.getUint32(8, true) !== 1)
    throw new Error('Expected MTTSCUDA weights format version 1');
  const count = view.getUint32(12, true);
  if (count < specs.length || count > Math.floor((bytes.length - 16) / 156))
    throw new Error('Invalid weights tensor count');
  offset = 16;
  for (let index = 0; index < count; ++index) {
    need(152);
    const nameBytes = bytes.subarray(offset, offset + 96);
    const end = nameBytes.indexOf(0);
    if (end < 1) throw new Error('Invalid tensor name');
    const name = decoder.decode(nameBytes.subarray(0, end));
    if (names.has(name)) throw new Error(`Duplicate tensor: ${name}`);
    names.add(name);
    const dtype = view.getUint32(offset + 96, true);
    const layout = view.getUint32(offset + 100, true);
    const rank = view.getUint32(offset + 104, true);
    const reserved = view.getUint32(offset + 108, true);
    const dims = Array.from({ length: 4 }, (_, d) => u64(offset + 112 + d * 8));
    const elements = u64(offset + 144);
    if (dtype !== 1 || ![1, 2, 3].includes(layout) || rank < 1 || rank > 4 || reserved !== 0 ||
        dims.slice(0, rank).some(d => d < 1) || dims.slice(rank).some(d => d !== 0) ||
        dims.slice(0, rank).reduce((a, b) => a * b, 1) !== elements)
      throw new Error(`Invalid tensor header: ${name}`);
    const spec = specs[index];
    if (spec && (name !== spec.name || layout !== spec.layout || rank !== spec.shape.length ||
        spec.shape.some((d, i) => dims[i] !== d)))
      throw new Error(`Expected ${spec.name} with layout ${spec.layout}, shape [${spec.shape}]`);
    offset += 152;
    need(elements * 4);
    if (spec) {
      const values = new Float32Array(elements);
      for (let i = 0; i < elements; ++i) values[i] = view.getFloat32(offset + i * 4, true);
      result.set(name, values);
    }
    offset += elements * 4;
  }
  if (offset !== bytes.length) throw new Error('Weights file contains trailing data');
  return result;
}

export function validatePreConvWeights(weights) {
  if (!(weights instanceof Map)) throw new TypeError('weights must be a Map of tensor names to Float32Array');
  for (const { name, shape } of preConvTensorSpecs()) {
    const values = weights.get(name);
    const count = shape.reduce((a, b) => a * b, 1);
    if (!(values instanceof Float32Array) || values.length !== count)
      throw new Error(`${name}: expected ${count} float32 values in exported layout`);
    if (!values.every(Number.isFinite)) throw new Error(`${name}: non-finite weight`);
  }
}
