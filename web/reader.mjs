/**
 * Batched reader: paste text, hear it read.
 *
 * Sentences are synthesized one at a time, but their audio stays on the GPU in
 * an accumulator sized to hold a whole integer number of maximum-length
 * inferences, so the device-to-host copy happens roughly once every two minutes
 * of speech instead of once per sentence.
 *
 * Models and every GPU buffer are allocated on the first run and kept for the
 * lifetime of the page, so a second paste reuses all of it.
 */

import {
  InferenceWebGPU, loadInferenceKernels, parseInferenceWeights,
} from '../inference/inference_webgpu.mjs';
import {
  GeneratorWebGPU, loadGeneratorKernels, parseGeneratorWeights, requestGeneratorDevice,
} from '../hifi-vocoder-cuda/inference_generator_webgpu.mjs';
import { phonemizeForMatcha } from './matcha_phonemize.mjs';
import { phonemesToTokenIds } from './matcha_tokenizer.mjs';
import { splitSentences, splitTokenSequence, splitToCapacity } from './text_segmentation.mjs';
import { encodeWav } from './wav.mjs';

const MAX_TOKENS = 768;
const MAX_MEL_FRAMES = 1024;
const SAMPLE_RATE = 22050;
const HOP_SIZE = 256;
const GAP_SAMPLES = Math.round(0.2 * SAMPLE_RATE);   // 200 ms between sentences

/**
 * How many times a part may be halved when the duration predictor overshoots.
 *
 * Measured on this checkpoint, speech runs about 2.7 mel frames per token, so
 * the 1024-frame limit binds at roughly 380 tokens — well below the 768-token
 * capacity. A 768-token part therefore needs two halvings, not one, and setting
 * this to 1 silently drops whole sentences in the 700-768 token range.
 */
const MAX_DURATION_SPLITS = 4;

const WEIGHTS_URL = new URL('../matcha_ljspeech_cuda.bin', import.meta.url);
const VOCODER_WEIGHTS_URL = new URL('../hifi-vocoder-cuda/model/generator_v1_weights.bin', import.meta.url);

/** Samples one maximum-length inference produces. */
const INFERENCE_SAMPLES = MAX_MEL_FRAMES * HOP_SIZE;   // 262,144 = 11.89 s
/** Whole inferences whose total is closest to two minutes of audio. */
const BATCH_INFERENCES = Math.max(1, Math.round(SAMPLE_RATE * 120 / INFERENCE_SAMPLES));  // 10
const ACCUMULATOR_SAMPLES = BATCH_INFERENCES * INFERENCE_SAMPLES;                          // 118.9 s

const ui = {
  text: document.querySelector('#text'),
  convert: document.querySelector('#convert'),
  cancel: document.querySelector('#cancel'),
  download: document.querySelector('#download'),
  charCount: document.querySelector('#char-count'),
  progressWrap: document.querySelector('#progress-wrap'),
  progressFill: document.querySelector('#progress-fill'),
  progressText: document.querySelector('#progress-text'),
  progressPercent: document.querySelector('#progress-percent'),
  status: document.querySelector('#status'),
  audio: document.querySelector('#audio'),
  summary: document.querySelector('#summary'),
  modelPanel: document.querySelector('#model-loading'),
  modelStatus: document.querySelector('#model-status'),
  modelFill: document.querySelector('#model-fill'),
  modelSpinner: document.querySelector('#model-spinner'),
};

let runtime = null;     // Created once, never released.
let lastWav = null;
let cancelled = false;

const setStatus = (message, kind = 'info') => {
  ui.status.textContent = message;
  ui.status.dataset.kind = kind;
};

function setProgress(fraction, label) {
  ui.progressWrap.classList.remove('hidden');
  const percent = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  ui.progressFill.style.width = `${percent}%`;
  ui.progressPercent.textContent = `${percent}%`;
  if (label) ui.progressText.textContent = label;
}

function showSummary(entries) {
  ui.summary.replaceChildren(...entries.map(([value, label]) => {
    const card = document.createElement('div');
    card.className = 'stat';
    const valueNode = document.createElement('div');
    valueNode.className = 'stat-value';
    valueNode.textContent = value;
    const labelNode = document.createElement('div');
    labelNode.className = 'stat-label';
    labelNode.textContent = label;
    card.append(valueNode, labelNode);
    return card;
  }));
}

/** Yields to the browser so progress paints during a long batch. */
const breathe = () => new Promise(resolve => setTimeout(resolve, 0));

// --- GPU-side audio accumulation ------------------------------------------

/**
 * Collects rendered utterances in one GPU buffer and maps it only when full.
 * Silence is host-side: it never needs to make the trip.
 */
class AudioBatch {
  #device; #generator; #buffer; #staging; #capacity;
  #timeline = []; #pending = []; #used = 0; #flushes = 0;

  constructor(device, generator, capacity) {
    this.#device = device;
    this.#generator = generator;
    this.#capacity = capacity;
    this.#buffer = device.createBuffer({
      label: 'audio accumulator', size: capacity * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.#staging = device.createBuffer({
      label: 'audio readback', size: capacity * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  reset() { this.#timeline = []; this.#pending = []; this.#used = 0; this.#flushes = 0; }
  get flushes() { return this.#flushes; }

  addSilence(samples) {
    if (samples > 0) this.#timeline.push({ data: new Float32Array(samples), length: samples });
  }

  async append(mel, melFrames) {
    const samples = melFrames * HOP_SIZE;
    if (this.#used + samples > this.#capacity) await this.flush();
    const entry = { offset: this.#used, length: samples, data: null };
    this.#timeline.push(entry);
    this.#pending.push(entry);
    this.#generator.renderInto(mel, melFrames, this.#buffer, this.#used);
    this.#used += samples;
  }

  async flush() {
    if (this.#pending.length === 0) return;
    const bytes = this.#used * 4;
    const encoder = this.#device.createCommandEncoder({ label: 'audio flush' });
    encoder.copyBufferToBuffer(this.#buffer, 0, this.#staging, 0, bytes);
    this.#device.queue.submit([encoder.finish()]);

    await this.#staging.mapAsync(GPUMapMode.READ, 0, bytes);
    const view = new Float32Array(this.#staging.getMappedRange(0, bytes));
    for (const entry of this.#pending) {
      entry.data = view.slice(entry.offset, entry.offset + entry.length);
    }
    this.#staging.unmap();
    this.#pending = [];
    this.#used = 0;
    this.#flushes += 1;
  }

  result() {
    const total = this.#timeline.reduce((sum, entry) => sum + entry.length, 0);
    const output = new Float32Array(total);
    let offset = 0;
    for (const entry of this.#timeline) {
      if (entry.data) output.set(entry.data, offset);
      offset += entry.length;
    }
    return output;
  }
}

// --- one-time model loading ------------------------------------------------

async function fetchWeights(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body) return response.arrayBuffer();

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
  return buffer.buffer;
}

async function ensureRuntime() {
  if (runtime) return runtime;
  if (!navigator.gpu) throw new Error('This browser does not expose WebGPU. Try Chrome or Edge 113+, or Safari 26+.');

  ui.modelPanel.classList.remove('hidden', 'loaded');
  const stage = (label, fraction) => {
    ui.modelStatus.textContent = label;
    ui.modelFill.style.width = `${Math.round(fraction * 100)}%`;
  };

  try {
    stage('Requesting a WebGPU device…', 0.02);
    const device = await requestGeneratorDevice({ maxMelFrames: MAX_MEL_FRAMES });

    // Two downloads share one bar: the acoustic model is 57% of the bytes.
    const acoustic = await fetchWeights(WEIGHTS_URL, (received, total) =>
      stage('Downloading acoustic model…', 0.05 + 0.45 * (total ? received / total : 0)));
    const vocoder = await fetchWeights(VOCODER_WEIGHTS_URL, (received, total) =>
      stage('Downloading vocoder…', 0.50 + 0.30 * (total ? received / total : 0)));

    stage('Parsing weights…', 0.82);
    const weights = parseInferenceWeights(acoustic);
    const generatorWeights = parseGeneratorWeights(vocoder);

    stage('Compiling shaders…', 0.88);
    const [kernels, generatorKernels] = await Promise.all([
      loadInferenceKernels(), loadGeneratorKernels(),
    ]);

    stage('Allocating GPU memory…', 0.94);
    const inference = await InferenceWebGPU.create({ device, weights, kernels, maxTokenLength: MAX_TOKENS });
    const generator = await GeneratorWebGPU.create({
      device, weights: generatorWeights, kernels: generatorKernels, maxMelFrames: MAX_MEL_FRAMES,
    });
    const batch = new AudioBatch(device, generator, ACCUMULATOR_SAMPLES);

    stage('Ready', 1);
    ui.modelSpinner.classList.add('hidden');
    ui.modelPanel.classList.add('loaded');
    setTimeout(() => ui.modelPanel.classList.add('hidden'), 600);

    runtime = { device, inference, generator, batch };
    return runtime;
  } catch (error) {
    ui.modelPanel.classList.add('hidden');
    throw error;
  }
}

// --- synthesis -------------------------------------------------------------

/**
 * Render one sequence, splitting or skipping when the duration predictor
 * overshoots the model's 1024-frame limit.
 *
 * An overshooting part is halved and retried; a part still overshooting after
 * `MAX_DURATION_SPLITS` halvings is skipped, with 200 ms of silence standing in.
 */
async function renderPart(part, depth, context) {
  const { inference, batch, stats } = context;
  if (part.length < 3) return;

  const result = await inference.runInference(Array.from(part));
  if (result.skippedFlowMatching) {
    const halves = depth < MAX_DURATION_SPLITS ? splitTokenSequence(part) : [part];
    if (halves.length === 2) {
      stats.frameSplits += 1;
      for (const half of halves) await renderPart(half, depth + 1, context);
      return;
    }
    stats.skipped += 1;
    batch.addSilence(GAP_SAMPLES);
    return;
  }

  const mel = await inference.readOutput(result);
  stats.parts += 1;
  stats.frames += result.mel_frame_count;
  await batch.append(mel, result.mel_frame_count);
}

async function convert() {
  cancelled = false;
  ui.convert.disabled = true;
  ui.cancel.disabled = false;
  ui.download.classList.add('hidden');
  ui.audio.classList.add('hidden');
  ui.summary.replaceChildren();

  try {
    const sentences = splitSentences(ui.text.value);
    if (sentences.length === 0) { setStatus('Enter some text first.', 'warn'); return; }

    setStatus(`Preparing ${sentences.length} sentence${sentences.length === 1 ? '' : 's'}…`);
    setProgress(0, 'Loading models…');

    const { inference, batch } = await ensureRuntime();
    batch.reset();

    const stats = { parts: 0, frames: 0, lengthSplits: 0, frameSplits: 0, skipped: 0 };
    const context = { inference, batch, stats };
    const started = performance.now();

    for (const [index, sentence] of sentences.entries()) {
      if (cancelled) break;
      setProgress(index / sentences.length, `Sentence ${index + 1} of ${sentences.length}`);
      await breathe();

      const { phonemes } = await phonemizeForMatcha(sentence);
      if (!phonemes) continue;
      const { tokenIds } = phonemesToTokenIds(phonemes);
      if (tokenIds.length < 3) continue;

      const parts = splitToCapacity(tokenIds, MAX_TOKENS);
      stats.lengthSplits += parts.length - 1;
      for (const part of parts) {
        if (cancelled) break;
        await renderPart(part, 0, context);
      }
      if (index < sentences.length - 1) batch.addSilence(GAP_SAMPLES);
    }

    setProgress(1, 'Reading back audio');
    await batch.flush();
    const audio = batch.result();
    if (audio.length === 0) { setStatus('Nothing could be synthesized from that text.', 'warn'); return; }

    if (lastWav) URL.revokeObjectURL(lastWav);
    const blob = encodeWav(audio, SAMPLE_RATE);
    lastWav = URL.createObjectURL(blob);
    ui.audio.src = lastWav;
    ui.audio.classList.remove('hidden');
    ui.download.classList.remove('hidden');

    const seconds = audio.length / SAMPLE_RATE;
    const elapsed = (performance.now() - started) / 1000;
    setStatus(
      cancelled
        ? `Stopped early. ${seconds.toFixed(1)} s of audio from ${stats.parts} segment${stats.parts === 1 ? '' : 's'}.`
        : `Done. ${seconds.toFixed(1)} s of audio in ${elapsed.toFixed(1)} s (${(seconds / elapsed).toFixed(2)}× real time).`,
      cancelled ? 'warn' : 'ok',
    );

    showSummary([
      [String(sentences.length), 'Sentences'],
      [String(stats.parts), 'Segments'],
      [`${seconds.toFixed(1)}s`, 'Audio'],
      [String(stats.lengthSplits), 'Length splits'],
      [String(stats.frameSplits), 'Duration splits'],
      [String(stats.skipped), 'Skipped'],
      [String(batch.flushes), 'GPU readbacks'],
    ]);
  } catch (error) {
    console.error(error);
    setStatus(`Failed: ${error.message}`, 'error');
  } finally {
    ui.convert.disabled = false;
    ui.cancel.disabled = true;
    ui.progressWrap.classList.add('hidden');
  }
}

// --- wiring ----------------------------------------------------------------

const updateCharCount = () => {
  const characters = ui.text.value.trim().length;
  ui.charCount.textContent = characters ? `${characters.toLocaleString()} characters` : '';
};

ui.convert.addEventListener('click', convert);
ui.cancel.addEventListener('click', () => { cancelled = true; ui.cancel.disabled = true; });
ui.text.addEventListener('input', updateCharCount);
ui.download.addEventListener('click', () => {
  if (!lastWav) return;
  const link = document.createElement('a');
  link.href = lastWav;
  link.download = 'matcha_reading.wav';
  link.click();
});

updateCharCount();
if (!navigator.gpu) setStatus('WebGPU is unavailable in this browser. Try Chrome or Edge 113+, or Safari 26+.', 'error');
