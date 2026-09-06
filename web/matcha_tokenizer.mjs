/**
 * Matcha-TTS tokenizer, ported to JavaScript.
 *
 * Mirrors `Matcha-TTS-main/matcha/text/symbols.py` and
 * `Matcha-TTS-main/matcha/text/__init__.py` exactly, plus the `intersperse`
 * blank padding applied by `matcha/models/matcha_tts.py` before inference.
 *
 * The CUDA path performs this work in `matcha_tokenize.py`; this module is the
 * browser equivalent and must agree with it token for token.
 */

// --- symbols.py ------------------------------------------------------------

const PAD = '_';
const PUNCTUATION = ';:,.!?¡¿—…"«»“” ';
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
// eslint-disable-next-line no-irregular-whitespace
const LETTERS_IPA =
  "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

/** The 178-entry symbol list, in Python's order. */
export const SYMBOLS = Object.freeze([PAD, ...PUNCTUATION, ...LETTERS, ...LETTERS_IPA]);

/**
 * Symbol -> ID.
 *
 * The list holds 178 entries but only 177 distinct symbols: the apostrophe
 * appears at both index 174 and index 176. Python builds this with
 * `{s: i for i, s in enumerate(symbols)}`, so the later index wins and
 * `"'"` resolves to 176. `Map` has the same last-write-wins behaviour, so the
 * mapping matches without a special case.
 */
export const SYMBOL_TO_ID = new Map(SYMBOLS.map((symbol, index) => [symbol, index]));

/** Vocabulary size the model was built for. */
export const VOCABULARY_SIZE = SYMBOLS.length; // 178

/** Blank ID inserted by `intersperse`. */
export const BLANK_ID = 0;

// --- __init__.py -----------------------------------------------------------

/**
 * `cleaned_text_to_sequence`: map already-phonemized text to symbol IDs.
 *
 * Python raises `KeyError` on an unknown symbol. eSpeak builds differ slightly,
 * so a stray symbol should not destroy the whole request; unknown characters are
 * dropped and reported instead, and the caller decides how to surface them.
 *
 * @param {string} cleanedText
 * @returns {{ sequence: number[], unknown: string[] }}
 */
export function cleanedTextToSequence(cleanedText) {
  const sequence = [];
  const unknown = [];
  // Iterate by code point, matching Python's `for symbol in clean_text`.
  for (const symbol of cleanedText) {
    const id = SYMBOL_TO_ID.get(symbol);
    if (id === undefined) unknown.push(symbol);
    else sequence.push(id);
  }
  return { sequence, unknown };
}

/** Inverse mapping, for debugging. Uses the first occurrence, like Python's `_id_to_symbol`. */
export function sequenceToText(sequence) {
  return Array.from(sequence, id => SYMBOLS[id] ?? '').join('');
}

// --- matcha_tts.py ---------------------------------------------------------

/**
 * `intersperse(lst, item)`: `[item] * (len(lst) * 2 + 1)` with `result[1::2] = lst`.
 *
 * Produces a blank before the first symbol, between every pair, and after the
 * last, so the result length is always odd.
 */
export function intersperse(sequence, item = BLANK_ID) {
  const result = new Array(sequence.length * 2 + 1).fill(item);
  for (let i = 0; i < sequence.length; ++i) result[i * 2 + 1] = sequence[i];
  return result;
}

/**
 * Full phonemes -> model-ready token IDs.
 *
 * @param {string} phonemes Output of the phonemizer + cleaner chain.
 * @returns {{ tokenIds: number[], sequence: number[], unknown: string[] }}
 */
export function phonemesToTokenIds(phonemes) {
  const { sequence, unknown } = cleanedTextToSequence(phonemes);
  return { tokenIds: intersperse(sequence, BLANK_ID), sequence, unknown };
}

/**
 * Validate a token sequence against the Matcha protocol enforced by
 * `validateTokenIds(..., { matchaProtocol: true })` in the WebGPU runtime.
 *
 * @returns {string|null} An error message, or null when the sequence is valid.
 */
export function checkMatchaProtocol(tokenIds, capacity) {
  if (tokenIds.length < 1) return 'The sequence is empty.';
  if (tokenIds.length > capacity) return `The sequence has ${tokenIds.length} tokens; the limit is ${capacity}.`;
  if (tokenIds.length % 2 !== 1) return 'The sequence length must be odd.';
  for (let i = 0; i < tokenIds.length; ++i) {
    const value = tokenIds[i];
    if (!Number.isInteger(value) || value < 0 || value >= VOCABULARY_SIZE)
      return `Token ${i} is not an integer in [0, ${VOCABULARY_SIZE - 1}].`;
    if (i % 2 === 0 && value !== BLANK_ID) return `Position ${i} must hold the blank ID 0.`;
  }
  return null;
}
