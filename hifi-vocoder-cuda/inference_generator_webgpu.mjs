/**
 * HiFi v1 generator on WebGPU.
 *
 * Replicates `inference_generator.cu` stage for stage, driving the unedited WGSL
 * kernels in `inference_generator.js` plus the `matmul` shader from
 * `../inference/blas_gemm.js` in place of the cuBLAS GEMMs.
 *
 * Two differences from the CUDA source, both forced by the kernels themselves:
 *
 * 1. The WGSL kernels are in-place where CUDA wrote to a separate destination
 *    (leaky ReLU, elementwise add, tanh, bias, and the three-way average), so
 *    buffers are shared far more aggressively than the CUDA allocation does.
 *    Eight full-plane buffers cover the whole network; see `#runStage`.
 *
 * 2. Every kernel indexes off `global_invocation_id.x` alone, so one dispatch
 *    reaches at most `maxComputeWorkgroupsPerDimension * 32` threads — 2,097,120
 *    on typical hardware, which a single stage exceeds by ~44x at 1024 mel
 *    frames. Every dispatch is therefore split into tiles along the time axis,
 *    addressed with bind-group buffer offsets rather than shader changes.
 *    Convolution tiles are widened by the kernel halo so the shader's own
 *    zero-padding still lands only at the true sequence edges.
 */

import { GENERATOR, generatorStageGeometry } from './generator_weights.mjs';

export { GENERATOR, parseGeneratorWeights, generatorStageGeometry } from './generator_weights.mjs';

const KERNEL_NAMES = Object.freeze([
  'vocoder_concatAndOverlayTConv1D', 'vocoder_adaptTensorToRightSideConvMul',
  'vocoder_leakyReLU', 'vocoder_elementwiseAdd', 'vocoder_elementwiseAddAverageThreeTensors',
  'vocoder_apply_tanh', 'vocoder_add_bias',
]);

const WORKGROUP_SIZE = 32;      // From the unchanged WGSL, not CUDA threadsPerBlock.
const COLUMN_ALIGNMENT = 64;    // Keeps every binding byte offset a multiple of 256.
const CONSTRUCTION_TOKEN = Symbol('GeneratorWebGPU construction');

/** Extract the existing plain template literals without executing their JS. */
export function extractGeneratorKernels(generatorJavaScript, blasJavaScript) {
  function extract(source, name) {
    const pattern = new RegExp('(?:^|\\n)const\\s+' + name + '\\s*=\\s*`([^`]*)`;?', 'g');
    const matches = [...source.matchAll(pattern)];
    if (matches.length !== 1 || matches[0][1].includes('${') || matches[0][1].includes('\\'))
      throw new Error(`Expected one plain, unescaped shader literal: ${name}`);
    return matches[0][1];
  }
  return Object.freeze(Object.fromEntries([
    ...KERNEL_NAMES.map(name => [name, extract(generatorJavaScript, name)]),
    ['matmul', extract(blasJavaScript, 'matmul')],
  ]));
}

export async function loadGeneratorKernels({
  generatorURL = new URL('./inference_generator.js', import.meta.url),
  blasURL = new URL('../inference/blas_gemm.js', import.meta.url),
} = {}) {
  const sources = await Promise.all([generatorURL, blasURL].map(async url => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
    return response.text();
  }));
  return extractGeneratorKernels(...sources);
}

// --- planning --------------------------------------------------------------

const alignDown = (value, multiple) => value - (value % multiple);
const ceilDiv = (value, divisor) => Math.ceil(value / divisor);

/**
 * Widest convolution tile one dispatch can cover.
 *
 * The view handed to the shader spans the tile plus a halo on each side, and its
 * start is rounded down for binding alignment, so the tile is shrunk by a
 * further alignment step to keep the widened view inside the thread budget.
 */
function convolutionTilePlan(inputChannels, kernel, dilation, maxThreads) {
  const halo = Math.floor(kernel / 2) * dilation;
  const viewColumns = Math.floor(maxThreads / (inputChannels * kernel));
  const tileWidth = alignDown(viewColumns - 2 * halo - COLUMN_ALIGNMENT, COLUMN_ALIGNMENT);
  if (tileWidth < COLUMN_ALIGNMENT)
    throw new RangeError(`Device cannot dispatch a ${inputChannels}-channel kernel-${kernel} tile`);
  return { halo, tileWidth, viewFloats: viewColumns * inputChannels * kernel };
}

/** Allocation sizes, in floats, for a given mel-frame capacity. */
export function generatorMemoryPlan(maxMelFrames, maxThreads = 65535 * WORKGROUP_SIZE) {
  const { stages, audioSamples } = generatorStageGeometry(maxMelFrames);
  let plane = 0, expanded = 0, view = 0;
  for (const stage of stages) {
    plane = Math.max(plane, stage.nextChannels * stage.timeSeries);
    expanded = Math.max(expanded, stage.nextExpandedSize * stage.frames);
    for (const kernel of GENERATOR.resblockKernels) {
      for (const dilation of GENERATOR.resblockDilations) {
        view = Math.max(view, convolutionTilePlan(stage.nextChannels, kernel, dilation, maxThreads).viewFloats);
      }
    }
  }
  view = Math.max(view,
    convolutionTilePlan(GENERATOR.finalChannels, GENERATOR.convPostKernel, 1, maxThreads).viewFloats);

  const elements = Object.freeze({
    melIm2col: GENERATOR.convPreKernel * GENERATOR.melBins * maxMelFrames,
    melPostConv: GENERATOR.channels * maxMelFrames,
    scratchpad: Math.max(expanded, view),
    overlay: plane, reluScratch: plane, midConv: plane, convA: plane, convB: plane,
    streamResult0: plane, streamResult1: plane, streamResult2: plane,
    audio: audioSamples,
  });
  const workspaceBytes = Object.values(elements).reduce((total, count) => total + count, 0) * 4;
  return Object.freeze({
    maxMelFrames, audioSamples, elements, workspaceBytes,
    largestBufferBytes: Math.max(...Object.values(elements)) * 4,
  });
}

/** A device with the limits this generator needs. Also satisfies the Matcha runtime. */
export async function requestGeneratorDevice({ maxMelFrames = 1024, ...adapterOptions } = {}) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter(adapterOptions);
  if (!adapter) throw new Error('No WebGPU adapter is available');
  if (adapter.limits.maxStorageBuffersPerShaderStage < 10)
    throw new Error('Device must support ten compute storage bindings');

  const plan = generatorMemoryPlan(maxMelFrames,
    adapter.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE);
  const needed = plan.largestBufferBytes;
  if (Number(adapter.limits.maxBufferSize) < needed ||
      Number(adapter.limits.maxStorageBufferBindingSize) < needed)
    throw new Error(`Device cannot allocate the ${(needed / 1048576).toFixed(0)} MiB buffer the vocoder needs`);

  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: 10,
      maxBufferSize: Math.max(needed, 268435456),
      maxStorageBufferBindingSize: Math.max(needed, 134217728),
    },
  });
}

/** Opens compute passes lazily so buffer copies can be interleaved. */
class Recorder {
  #encoder; #pass = null;
  constructor(device, label) { this.#encoder = device.createCommandEncoder({ label }); }
  get pass() { return this.#pass ??= this.#encoder.beginComputePass(); }
  get encoder() {
    if (this.#pass) { this.#pass.end(); this.#pass = null; }
    return this.#encoder;
  }
  copy(source, destination, floats) {
    this.encoder.copyBufferToBuffer(source, 0, destination, 0, floats * 4);
  }
  finish() { return this.encoder.finish(); }
}

// --- runtime ---------------------------------------------------------------

export class GeneratorWebGPU {
  #device; #pipelines = new Map(); #weights = new Map();
  #buffers = new Map(); #scalars = new Map(); #plan;
  #maxThreads; #busy = false; #destroyed = false;

  constructor(token) { if (token !== CONSTRUCTION_TOKEN) throw new Error('Use GeneratorWebGPU.create()'); }

  static async create({ device, weights, kernels, maxMelFrames = 1024 }) {
    if (!device) throw new TypeError('A GPUDevice is required');
    if (!Number.isInteger(maxMelFrames) || maxMelFrames < 1)
      throw new RangeError('maxMelFrames must be a positive integer');

    const self = new GeneratorWebGPU(CONSTRUCTION_TOKEN);
    self.#device = device;
    self.#maxThreads = device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
    self.#plan = generatorMemoryPlan(maxMelFrames, self.#maxThreads);

    try {
      await Promise.all([...KERNEL_NAMES, 'matmul'].map(async name => {
        const code = kernels?.[name];
        if (typeof code !== 'string') throw new Error(`Missing shader source: ${name}`);
        self.#pipelines.set(name, await device.createComputePipelineAsync({
          label: name, layout: 'auto',
          compute: { module: device.createShaderModule({ code, label: name }), entryPoint: 'main' },
        }));
      }));

      for (const [name, count] of Object.entries(self.#plan.elements)) {
        self.#buffers.set(name, device.createBuffer({
          label: name, size: count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }));
      }
      for (const [name, tensor] of weights) {
        const buffer = device.createBuffer({
          label: `weight:${name}`, size: tensor.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, tensor);
        self.#weights.set(name, buffer);
      }
      return self;
    } catch (error) { self.destroy(); throw error; }
  }

  get memoryPlan() { return this.#plan; }

  // --- dispatch helpers ----------------------------------------------------

  #scalar(kind, value) {
    const key = `${kind}:${value}`;
    let buffer = this.#scalars.get(key);
    if (!buffer) {
      buffer = this.#device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.#device.queue.writeBuffer(buffer, 0,
        kind === 'u32' ? new Uint32Array([value]) : new Float32Array([value]));
      this.#scalars.set(key, buffer);
    }
    return buffer;
  }

  #u32 = value => this.#scalar('u32', value);
  #f32 = value => this.#scalar('f32', value);

  /** Entries are `[buffer, offsetFloats, sizeFloats]` or a bare GPUBuffer. */
  #dispatch(recorder, name, entries, threads) {
    if (threads <= 0) return;
    if (threads > this.#maxThreads)
      throw new RangeError(`${name} needs ${threads} threads; this device allows ${this.#maxThreads} per dispatch`);
    const pipeline = this.#pipelines.get(name);
    const pass = recorder.pass;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((entry, binding) => {
        const [buffer, offset = 0, size] = Array.isArray(entry) ? entry : [entry];
        const resource = { buffer };
        if (offset) resource.offset = offset * 4;
        if (size !== undefined) resource.size = size * 4;
        return { binding, resource };
      }),
    }));
    pass.dispatchWorkgroups(ceilDiv(threads, WORKGROUP_SIZE));
  }

  /** Elementwise kernel over `count` values, split into dispatchable tiles. */
  #elementwise(recorder, name, buffers, count, leading = [], granularity = COLUMN_ALIGNMENT) {
    const step = alignDown(this.#maxThreads, granularity) || granularity;
    for (let start = 0; start < count; start += step) {
      const length = Math.min(step, count - start);
      this.#dispatch(recorder, name, [
        ...buffers.map(buffer => [buffer, start, length]),
        ...leading, this.#u32(length),
      ], length);
    }
  }

  #leakyReLU(recorder, buffer, slope, count) {
    this.#elementwise(recorder, 'vocoder_leakyReLU', [buffer], count, [this.#f32(slope)]);
  }

  #add(recorder, target, addend, count) {
    this.#elementwise(recorder, 'vocoder_elementwiseAdd', [target, addend], count);
  }

  #addBias(recorder, buffer, bias, channels, count) {
    this.#elementwise(recorder, 'vocoder_add_bias', [buffer], count, [bias, this.#u32(channels)],
      Math.max(COLUMN_ALIGNMENT, channels));
  }

  /** `y = x1 @ x2`, column-major, tiled over the columns of `x2`. */
  #matmul(recorder, y, x1, x2, rows, contracting, columns, { yColumn = 0, x2Column = 0 } = {}) {
    const step = alignDown(Math.floor(this.#maxThreads / rows), COLUMN_ALIGNMENT) || COLUMN_ALIGNMENT;
    for (let start = 0; start < columns; start += step) {
      const width = Math.min(step, columns - start);
      this.#dispatch(recorder, 'matmul', [
        [y, (yColumn + start) * rows, width * rows],
        x1,
        [x2, (x2Column + start) * contracting, width * contracting],
        this.#u32(rows), this.#u32(contracting), this.#u32(width),
      ], rows * width);
    }
  }

  /**
   * One 1-D convolution: im2col and GEMM fused per tile, so the scratchpad only
   * ever holds a single tile of the expanded tensor instead of the whole thing.
   */
  #convolve(recorder, { output, input, weight, bias, inputChannels, outputChannels, length, kernel, dilation }) {
    const scratchpad = this.#buffers.get('scratchpad');
    const { halo, tileWidth } = convolutionTilePlan(inputChannels, kernel, dilation, this.#maxThreads);
    const contracting = inputChannels * kernel;

    for (let start = 0; start < length; start += tileWidth) {
      const width = Math.min(tileWidth, length - start);
      // Widen the view by the halo so the shader zero-pads only at the real edges.
      const viewStart = alignDown(Math.max(0, start - halo), COLUMN_ALIGNMENT);
      const viewWidth = Math.min(length, start + width + halo) - viewStart;

      this.#dispatch(recorder, 'vocoder_adaptTensorToRightSideConvMul', [
        [scratchpad, 0, viewWidth * contracting],
        [input, viewStart * inputChannels, viewWidth * inputChannels],
        this.#u32(inputChannels), this.#u32(viewWidth), this.#u32(kernel), this.#u32(dilation),
      ], inputChannels * viewWidth * kernel);

      this.#matmul(recorder, output, weight, scratchpad, outputChannels, contracting, width,
        { yColumn: start, x2Column: start - viewStart });
    }
    this.#addBias(recorder, output, bias, outputChannels, outputChannels * length);
  }

  /**
   * Overlap-add of the transposed convolution, tiled over output columns.
   *
   * Interior tiles shift the input view one frame left and add one stride to the
   * padding, so the shader's leading-frame special case never fires mid-sequence.
   */
  #overlayTransposed(recorder, o) {
    const step = alignDown(Math.floor(this.#maxThreads / o.channels), COLUMN_ALIGNMENT) || COLUMN_ALIGNMENT;
    for (let start = 0; start < o.timeSeries; start += step) {
      const width = Math.min(step, o.timeSeries - start);
      const frameOffset = start === 0 ? 0 : start / o.stride - 1;
      const padding = start === 0 ? o.padding : o.padding + o.stride;
      this.#dispatch(recorder, 'vocoder_concatAndOverlayTConv1D', [
        [o.post, start * o.channels, width * o.channels],
        [o.pre, frameOffset * o.preFrameSize, (o.preFrameCount - frameOffset) * o.preFrameSize],
        o.bias,
        this.#u32(o.channels), this.#u32(width), this.#u32(o.preFrameCount - frameOffset),
        this.#u32(o.preFrameSize), this.#u32(o.preChannelSize), this.#u32(o.stride), this.#u32(padding),
      ], o.channels * width);
    }
  }

  // --- network -------------------------------------------------------------

  #runStage(recorder, geometry, source) {
    const { stage, currentChannels, nextChannels, nextUpsample, nextExpandedSize,
      frames, timeSeries, stride, padding } = geometry;
    const plane = nextChannels * timeSeries;
    const overlay = this.#buffers.get('overlay');
    const reluScratch = this.#buffers.get('reluScratch');
    const midConv = this.#buffers.get('midConv');
    const results = [0, 1, 2].map(index => this.#buffers.get(`streamResult${index}`));

    // CUDA wrote this leaky ReLU to its own buffer; the source is dead either way.
    this.#leakyReLU(recorder, source, GENERATOR.leakyReluSlope, currentChannels * frames);

    this.#matmul(recorder, this.#buffers.get('scratchpad'),
      this.#weights.get(`ups.${stage}.weight`), source, nextExpandedSize, currentChannels, frames);
    this.#overlayTransposed(recorder, {
      post: overlay, pre: this.#buffers.get('scratchpad'), bias: this.#weights.get(`ups.${stage}.bias`),
      channels: nextChannels, timeSeries, preFrameCount: frames,
      preFrameSize: nextExpandedSize, preChannelSize: nextUpsample, stride, padding,
    });

    for (let stream = 0; stream < GENERATOR.mrfStreams; ++stream) {
      const kernel = GENERATOR.resblockKernels[stream];
      // convs2 output per iteration; the last must survive until the MRF average.
      const output = [this.#buffers.get('convA'), this.#buffers.get('convB'), results[stream]];

      for (let iteration = 0; iteration < GENERATOR.mrfIterations; ++iteration) {
        if (iteration === 1) this.#add(recorder, output[0], overlay, plane);
        if (iteration > 1) this.#add(recorder, output[iteration - 1], output[iteration - 2], plane);

        // The in-place leaky ReLU would destroy a residual that later reads need.
        const residual = iteration === 0 ? overlay : output[iteration - 1];
        recorder.copy(residual, reluScratch, plane);
        this.#leakyReLU(recorder, reluScratch, GENERATOR.leakyReluSlope, plane);

        const base = `resblocks.${stage}.${stream}`;
        this.#convolve(recorder, {
          output: midConv, input: reluScratch,
          weight: this.#weights.get(`${base}.convs1.${iteration}.weight`),
          bias: this.#weights.get(`${base}.convs1.${iteration}.bias`),
          inputChannels: nextChannels, outputChannels: nextChannels,
          length: timeSeries, kernel, dilation: GENERATOR.resblockDilations[iteration],
        });
        this.#leakyReLU(recorder, midConv, GENERATOR.leakyReluSlope, plane);
        this.#convolve(recorder, {
          output: output[iteration], input: midConv,
          weight: this.#weights.get(`${base}.convs2.${iteration}.weight`),
          bias: this.#weights.get(`${base}.convs2.${iteration}.bias`),
          inputChannels: nextChannels, outputChannels: nextChannels,
          length: timeSeries, kernel, dilation: 1,
        });
      }
      this.#add(recorder, output[2], output[1], plane);
    }

    // MRF_final_sum lands in streamResult0, which becomes the next stage's input.
    this.#elementwise(recorder, 'vocoder_elementwiseAddAverageThreeTensors', results, plane);
    return results[0];
  }

  #checkReady(mel, melFrames) {
    if (this.#destroyed) throw new Error('GeneratorWebGPU has been destroyed');
    if (this.#busy) throw new Error('Await the current generator run before reusing it');
    if (!Number.isInteger(melFrames) || melFrames < 1 || melFrames > this.#plan.maxMelFrames)
      throw new RangeError(`melFrames must be in [1, ${this.#plan.maxMelFrames}]`);
    if (mel.length < melFrames * GENERATOR.melBins) throw new RangeError('Mel is shorter than melFrames implies');
  }

  /** Records the whole generator into `recorder` and returns the sample count. */
  #encode(recorder, mel, melFrames) {
    const geometry = generatorStageGeometry(melFrames);
    const samples = geometry.audioSamples;
    this.#uploadMel(mel, melFrames);

    let source = this.#buffers.get('melPostConv');
    this.#matmul(recorder, source, this.#weights.get('conv_pre.weight'),
      this.#buffers.get('melIm2col'), GENERATOR.channels,
      GENERATOR.convPreKernel * GENERATOR.melBins, melFrames);
    this.#addBias(recorder, source, this.#weights.get('conv_pre.bias'),
      GENERATOR.channels, GENERATOR.channels * melFrames);

    for (const stage of geometry.stages) source = this.#runStage(recorder, stage, source);

    const audio = this.#buffers.get('audio');
    this.#leakyReLU(recorder, source, GENERATOR.finalLeakyReluSlope, GENERATOR.finalChannels * samples);
    this.#convolve(recorder, {
      output: audio, input: source,
      weight: this.#weights.get('conv_post.weight'), bias: this.#weights.get('conv_post.bias'),
      inputChannels: GENERATOR.finalChannels, outputChannels: 1,
      length: samples, kernel: GENERATOR.convPostKernel, dilation: 1,
    });
    this.#elementwise(recorder, 'vocoder_apply_tanh', [audio], samples);
    return samples;
  }

  /**
   * @param {Float32Array} mel Denormalized mel, column-major [80, melFrames].
   * @returns {Promise<Float32Array>} `melFrames * 256` samples in [-1, 1].
   */
  async run(mel, melFrames) {
    this.#checkReady(mel, melFrames);
    this.#busy = true;
    let staging;
    try {
      const recorder = new Recorder(this.#device, 'hifi');
      const samples = this.#encode(recorder, mel, melFrames);

      staging = this.#device.createBuffer({
        size: samples * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      recorder.encoder.copyBufferToBuffer(this.#buffers.get('audio'), 0, staging, 0, samples * 4);
      this.#device.queue.submit([recorder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return result;
    } finally {
      staging?.destroy();
      this.#busy = false;
    }
  }

  /**
   * Render straight into a caller-owned buffer, with no host readback.
   *
   * Lets a batch keep many utterances on the GPU and map once, instead of
   * paying a device-to-host copy per utterance.
   *
   * @returns {number} Samples written at `offsetSamples`.
   */
  renderInto(mel, melFrames, target, offsetSamples = 0) {
    this.#checkReady(mel, melFrames);
    const recorder = new Recorder(this.#device, 'hifi-batched');
    const samples = this.#encode(recorder, mel, melFrames);
    if ((offsetSamples + samples) * 4 > target.size)
      throw new RangeError('Target buffer cannot hold this utterance at that offset');
    recorder.encoder.copyBufferToBuffer(
      this.#buffers.get('audio'), 0, target, offsetSamples * 4, samples * 4);
    this.#device.queue.submit([recorder.finish()]);
    return samples;
  }

  /** Builds conv_pre's right-hand side: column-major [7 * 80, frames]. */
  #uploadMel(mel, melFrames) {
    const { convPreKernel: taps, melBins } = GENERATOR;
    const rows = taps * melBins;
    const staged = new Float32Array(rows * melFrames);
    for (let frame = 0; frame < melFrames; ++frame) {
      for (let tap = 0; tap < taps; ++tap) {
        const source = frame + tap - (taps >> 1);
        if (source < 0 || source >= melFrames) continue;
        for (let bin = 0; bin < melBins; ++bin) {
          staged[frame * rows + bin * taps + tap] = mel[source * melBins + bin];
        }
      }
    }
    this.#device.queue.writeBuffer(this.#buffers.get('melIm2col'), 0, staged);
  }

  destroy() {
    if (this.#busy) throw new Error('Await the current generator run before destroying it');
    for (const collection of [this.#buffers, this.#weights, this.#scalars]) {
      for (const buffer of collection.values()) buffer.destroy();
      collection.clear();
    }
    this.#destroyed = true;
  }
}
