/**
 * Browser entry point: sentence -> phonemes -> token IDs -> WebGPU mel.
 *
 * The vocoder stage is not included; this page stops at Matcha's denormalized
 * mel output and displays it. The exported JSON matches the
 * `matcha-tts-mel-v1` format that `run_tts_pipeline.py --matcha-json` reads, so
 * the existing HiFi script can vocode a downloaded result.
 */

import {
  InferenceWebGPU,
  loadInferenceKernels,
  parseInferenceWeights,
} from '../inference/inference_webgpu.mjs';
import {
  GeneratorWebGPU,
  loadGeneratorKernels,
  parseGeneratorWeights,
  requestGeneratorDevice,
} from '../hifi-vocoder-cuda/inference_generator_webgpu.mjs';
import { phonemizeForMatcha } from './matcha_phonemize.mjs';
import { phonemesToTokenIds, checkMatchaProtocol, sequenceToText } from './matcha_tokenizer.mjs';
import { encodeWav } from './wav.mjs';

/**
 * Token capacity. Sequences are blank-interspersed to `2 * phonemes + 1`, so the
 * largest accepted sequence is 767 tokens (383 phonemes). Allocating 768 instead
 * of the runtime default of 1536 saves 13.86 MiB of workspace. The flow decoder
 * keeps its 1024-frame allocation, which is the model's own cutoff.
 */
const MAX_TOKENS = 768;

// Resolved against this module, so the page works wherever the HTML lives.
const WEIGHTS_URL = new URL('../matcha_ljspeech_cuda.bin', import.meta.url);
const VOCODER_WEIGHTS_URL = new URL('../hifi-vocoder-cuda/model/generator_v1_weights.bin', import.meta.url);
const MEL_CHANNELS = 80;
const SAMPLE_RATE = 22050;
/** The vocoder emits 256 samples per mel frame, so this is the audio capacity. */
const MAX_MEL_FRAMES = 1024;

const ui = {
  text: document.querySelector('#text'),
  speak: document.querySelector('#speak'),
  status: document.querySelector('#status'),
  details: document.querySelector('#details'),
  canvas: document.querySelector('#mel'),
  download: document.querySelector('#download'),
  downloadWav: document.querySelector('#download-wav'),
  audio: document.querySelector('#audio'),
  tokenCount: document.querySelector('#token-count'),
};

let runtime = null;      // { device, inference, generator }
let lastMel = null;      // { values, frames, text, phonemes }
let lastWav = null;      // { blob, url }

const setStatus = (message, kind = 'info') => {
  ui.status.textContent = message;
  ui.status.dataset.kind = kind;
};

const appendDetail = (label, value) => {
  const row = document.createElement('div');
  row.className = 'detail';
  row.innerHTML = `<span class="label"></span><span class="value"></span>`;
  row.querySelector('.label').textContent = label;
  row.querySelector('.value').textContent = value;
  ui.details.append(row);
};

// --- live token counter ----------------------------------------------------

let counterToken = 0;
async function updateTokenCount() {
  const token = ++counterToken;
  const text = ui.text.value.trim();
  if (!text) { ui.tokenCount.textContent = ''; return; }
  try {
    const { phonemes } = await phonemizeForMatcha(text);
    if (token !== counterToken) return; // A newer keystroke superseded this run.
    const { tokenIds } = phonemesToTokenIds(phonemes);
    const over = tokenIds.length > MAX_TOKENS;
    ui.tokenCount.textContent = `${tokenIds.length} / ${MAX_TOKENS} tokens`;
    ui.tokenCount.classList.toggle('over', over);
  } catch {
    if (token === counterToken) ui.tokenCount.textContent = '';
  }
}

let counterTimer;
const scheduleTokenCount = () => {
  clearTimeout(counterTimer);
  counterTimer = setTimeout(updateTokenCount, 300);
};

// --- model loading ---------------------------------------------------------

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

  setStatus('Requesting a WebGPU device…');
  // The vocoder needs the larger buffer limits; they also cover the Matcha runtime.
  const device = await requestGeneratorDevice({ maxMelFrames: MAX_MEL_FRAMES });

  try {
    const progress = label => (received, total) => {
      const suffix = total ? ` (${Math.round((received / total) * 100)}%)` : '';
      setStatus(`Downloading ${label} weights${suffix}…`);
    };

    const acousticBinary = await fetchWeights(WEIGHTS_URL, progress('acoustic model'));
    const vocoderBinary = await fetchWeights(VOCODER_WEIGHTS_URL, progress('vocoder'));

    setStatus('Parsing weights…');
    const weights = parseInferenceWeights(acousticBinary);
    const generatorWeights = parseGeneratorWeights(vocoderBinary);

    setStatus('Compiling shaders…');
    const [kernels, generatorKernels] = await Promise.all([
      loadInferenceKernels(), loadGeneratorKernels(),
    ]);

    setStatus('Uploading weights to the GPU…');
    const inference = await InferenceWebGPU.create({ device, weights, kernels, maxTokenLength: MAX_TOKENS });
    const generator = await GeneratorWebGPU.create({
      device, weights: generatorWeights, kernels: generatorKernels, maxMelFrames: MAX_MEL_FRAMES,
    });

    runtime = { device, inference, generator };
    return runtime;
  } catch (error) {
    device.destroy();
    throw error;
  }
}

// --- mel rendering ---------------------------------------------------------

/** Perceptually ordered ramp; values are already denormalized log-mel. */
function colormap(t) {
  const stops = [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [252, 194, 71], [252, 253, 191],
  ];
  const scaled = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const f = scaled - index;
  const a = stops[index], b = stops[index + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** `values` is column-major `[80, frames]`: flat index = frame * 80 + melBin. */
function renderMel(values, frames) {
  const canvas = ui.canvas;
  canvas.width = frames;
  canvas.height = MEL_CHANNELS;
  canvas.style.display = 'block';

  let min = Infinity, max = -Infinity;
  for (const value of values) { if (value < min) min = value; if (value > max) max = value; }
  const range = max - min || 1;

  const context = canvas.getContext('2d');
  const image = context.createImageData(frames, MEL_CHANNELS);
  for (let frame = 0; frame < frames; ++frame) {
    for (let bin = 0; bin < MEL_CHANNELS; ++bin) {
      // Draw low frequencies at the bottom.
      const row = MEL_CHANNELS - 1 - bin;
      const [r, g, b] = colormap((values[frame * MEL_CHANNELS + bin] - min) / range);
      const pixel = (row * frames + frame) * 4;
      image.data[pixel] = r; image.data[pixel + 1] = g; image.data[pixel + 2] = b; image.data[pixel + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  return { min, max };
}

// --- mel export ------------------------------------------------------------

function downloadMel() {
  if (!lastMel) return;
  const { values, frames, text, phonemes } = lastMel;
  // `matcha-tts-mel-v1` is channels-first: mel[channel][frame].
  const mel = Array.from({ length: MEL_CHANNELS }, (_, channel) => {
    const row = new Array(frames);
    for (let frame = 0; frame < frames; ++frame) row[frame] = values[frame * MEL_CHANNELS + channel];
    return row;
  });
  const payload = {
    format: 'matcha-tts-mel-v1',
    mel_layout: 'channels_first',
    denormalized: true,
    mel_channels: MEL_CHANNELS,
    mel_frames: frames,
    source: 'matcha-tts-webgpu',
    text,
    phonemes,
    mel,
  };
  saveBlob(new Blob([JSON.stringify(payload)], { type: 'application/json' }), 'matcha_mel.json');
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadWav() {
  if (lastWav) saveBlob(lastWav.blob, 'matcha_tts.wav');
}

// --- synthesis -------------------------------------------------------------

async function synthesize() {
  ui.speak.disabled = true;
  ui.details.replaceChildren();
  ui.download.hidden = true;
  ui.downloadWav.hidden = true;
  ui.audio.hidden = true;
  ui.canvas.style.display = 'none';

  try {
    const text = ui.text.value.trim();
    if (!text) { setStatus('Enter a sentence first.', 'warn'); return; }

    setStatus('Converting text to phonemes…');
    const started = performance.now();
    const { phonemes } = await phonemizeForMatcha(text, { onProgress: setStatus });
    if (!phonemes) { setStatus('That text produced no pronounceable phonemes.', 'warn'); return; }

    const { tokenIds, sequence, unknown } = phonemesToTokenIds(phonemes);

    if (tokenIds.length > MAX_TOKENS) {
      setStatus(
        `Too long: ${tokenIds.length} tokens, but this page allocates ${MAX_TOKENS}. ` +
        `Shorten the text to about ${Math.floor((MAX_TOKENS - 1) / 2)} phonemes or fewer.`,
        'error',
      );
      appendDetail('Phonemes', phonemes);
      appendDetail('Tokens required', `${tokenIds.length} (limit ${MAX_TOKENS})`);
      return;
    }

    const protocolError = checkMatchaProtocol(tokenIds, MAX_TOKENS);
    if (protocolError) { setStatus(`Token sequence rejected: ${protocolError}`, 'error'); return; }

    const { inference, generator } = await ensureRuntime();

    setStatus('Running the acoustic model…');
    const acousticStarted = performance.now();
    const result = await inference.runInference(tokenIds);

    if (result.skippedFlowMatching) {
      setStatus(
        `The predicted duration is ${result.compatible_mel_frame_count} frames, past the model's ` +
        '1024-frame cutoff, so flow matching was skipped. Try shorter text.',
        'error',
      );
      return;
    }

    const values = await inference.readOutput(result);
    const frames = result.mel_frame_count;
    const acousticElapsed = performance.now() - acousticStarted;

    const { min, max } = renderMel(values, frames);
    lastMel = { values, frames, text, phonemes };
    ui.download.hidden = false;

    setStatus('Running the vocoder…');
    const vocoderStarted = performance.now();
    const audio = await generator.run(values, frames);
    const vocoderElapsed = performance.now() - vocoderStarted;

    if (lastWav) URL.revokeObjectURL(lastWav.url);
    const blob = encodeWav(audio, SAMPLE_RATE);
    const url = URL.createObjectURL(blob);
    lastWav = { blob, url };
    ui.audio.src = url;
    ui.audio.hidden = false;
    ui.downloadWav.hidden = false;

    const seconds = audio.length / SAMPLE_RATE;
    const elapsed = performance.now() - started;
    setStatus(`Done. ${frames} mel frames, ${audio.length.toLocaleString()} samples, ${seconds.toFixed(2)} s of audio.`, 'ok');
    appendDetail('Phonemes', phonemes);
    appendDetail('Symbols', `${sequence.length} (${tokenIds.length} tokens after blank padding)`);
    appendDetail('Mel shape', `80 × ${frames}`);
    appendDetail('Mel range', `${min.toFixed(3)} … ${max.toFixed(3)}`);
    appendDetail('Acoustic model', `${acousticElapsed.toFixed(0)} ms`);
    appendDetail('Vocoder', `${vocoderElapsed.toFixed(0)} ms (${(seconds * 1000 / vocoderElapsed).toFixed(2)}× real time)`);
    appendDetail('Total time', `${elapsed.toFixed(0)} ms`);
    if (unknown.length) {
      appendDetail('Dropped symbols', `${unknown.join(' ')} — not in Matcha's 178-symbol table`);
    }
    // Round-tripping the IDs is a cheap check that the mapping stayed reversible.
    console.debug('token IDs', tokenIds, 'round trip', sequenceToText(sequence));
  } catch (error) {
    console.error(error);
    setStatus(`Failed: ${error.message}`, 'error');
  } finally {
    ui.speak.disabled = false;
  }
}

ui.speak.addEventListener('click', synthesize);
ui.download.addEventListener('click', downloadMel);
ui.downloadWav.addEventListener('click', downloadWav);
ui.text.addEventListener('input', scheduleTokenCount);
ui.text.addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) synthesize();
});

setStatus(
  navigator.gpu
    ? 'Ready. The first run downloads 122 MiB of weights and compiles shaders.'
    : 'WebGPU is unavailable in this browser. Try Chrome or Edge 113+, or Safari 26+.',
  navigator.gpu ? 'info' : 'error',
);
updateTokenCount();
