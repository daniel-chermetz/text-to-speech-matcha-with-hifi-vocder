/**
 * Browser port of Matcha-TTS `english_cleaners2`
 * (`Matcha-TTS-main/matcha/text/cleaners.py`).
 *
 * Python chain:
 *   convert_to_ascii -> lowercase -> expand_abbreviations
 *   -> espeak-ng (en-us, preserve_punctuation, with_stress, remove-flags, strip)
 *   -> remove_brackets -> collapse_whitespace
 *
 * The npm `phonemizer` package exposes no `preserve_punctuation` option: it
 * drops punctuation and returns the text split into chunks at those marks. This
 * module therefore splits the text at clause punctuation itself, phonemizes each
 * clause, and re-inserts the original marks — which is where Python's
 * punctuation would have ended up.
 *
 * Remaining differences come from the eSpeak NG build (Python uses 1.52, the npm
 * package bundles its own) and are accepted for this project.
 */

// --- ASCII folding (unidecode subset) --------------------------------------

/**
 * Characters unidecode rewrites that Unicode NFD decomposition does not handle.
 * Values verified against the installed `Unidecode` package.
 */
const ASCII_FOLD = new Map(Object.entries({
  'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE', 'ø': 'o', 'Ø': 'O',
  'ß': 'ss', 'þ': 'th', 'Þ': 'TH', 'ð': 'd', 'Ð': 'D',
  'ł': 'l', 'Ł': 'L', 'đ': 'd', 'Đ': 'D', 'ħ': 'h', 'ı': 'i',
  '\u2018': "'", '\u2019': "'", '\u201A': ',', '\u201B': "'",
  '\u201C': '"', '\u201D': '"', '\u201E': '"', '\u2032': "'", '\u2033': '"',
  '\u2013': '-', '\u2014': '--', '\u2015': '--', '\u2212': '-',
  '\u2026': '...', '\u00AB': '<<', '\u00BB': '>>',
  '\u00A3': 'PS', '\u20AC': 'EUR', '\u00A5': 'Y=', '\u00A2': 'c',
  '\u00A9': '(c)', '\u00AE': '(r)', '\u2122': '(tm)',
  '\u00BD': ' 1/2', '\u00BC': ' 1/4', '\u00BE': ' 3/4',
  '\u00B0': 'deg', '\u2116': 'No. ', '\u2022': '*', '\u00D7': 'x', '\u00F7': '/',
  '\u00A0': ' ', '\u202F': ' ', '\u2009': ' ', '\u200B': '',
}));

/**
 * `convert_to_ascii`. NFD decomposition plus combining-mark removal covers the
 * accented Latin letters; the table above covers the rest. This is a subset of
 * full unidecode, so scripts such as Cyrillic or CJK are dropped rather than
 * transliterated.
 */
export function convertToAscii(text) {
  let folded = '';
  for (const character of text) folded += ASCII_FOLD.get(character) ?? character;
  return folded
    .normalize('NFD')
    .replace(/\p{M}+/gu, '')
    .replace(/[^\x00-\x7F]/gu, '');
}

// --- remaining cleaners ----------------------------------------------------

/** `_abbreviations`, in cleaners.py order. */
const ABBREVIATIONS = [
  ['mrs', 'misess'], ['mr', 'mister'], ['dr', 'doctor'], ['st', 'saint'],
  ['co', 'company'], ['jr', 'junior'], ['maj', 'major'], ['gen', 'general'],
  ['drs', 'doctors'], ['rev', 'reverend'], ['lt', 'lieutenant'], ['hon', 'honorable'],
  ['sgt', 'sergeant'], ['capt', 'captain'], ['esq', 'esquire'], ['ltd', 'limited'],
  ['col', 'colonel'], ['ft', 'fort'],
].map(([abbreviation, expansion]) => [new RegExp(`\\b${abbreviation}\\.`, 'gi'), expansion]);

export function expandAbbreviations(text) {
  let expanded = text;
  for (const [pattern, replacement] of ABBREVIATIONS) expanded = expanded.replace(pattern, replacement);
  return expanded;
}

/** `remove_brackets`, plus the `(en)` language-switch flags `remove-flags` deletes. */
export function removeBrackets(text) {
  return text.replace(/\([a-z]{1,3}\)/gi, '').replace(/[[\](){}]/g, '');
}

/** `collapse_whitespace`. */
export function collapseWhitespace(text) {
  return text.replace(/\s+/g, ' ');
}

// --- clause segmentation ---------------------------------------------------

/** Punctuation espeak preserves that also exists in Matcha's symbol table. */
const CLAUSE_PUNCTUATION = ';:,.!?';

const isDigit = character => character >= '0' && character <= '9';

/**
 * True when `text[index]` ends a clause rather than sitting inside a number.
 * Guards decimals ("3.5"), thousands separators ("1,000") and times ("3:30"),
 * which espeak reads as numbers rather than as separate clauses.
 */
function isClauseBoundary(text, index) {
  if (!CLAUSE_PUNCTUATION.includes(text[index])) return false;
  return !(isDigit(text[index - 1] ?? '') && isDigit(text[index + 1] ?? ''));
}

/**
 * Split cleaned text into alternating spoken runs and punctuation runs.
 * @returns {{ kind: 'text'|'punctuation', value: string }[]}
 */
export function splitIntoClauses(text) {
  const segments = [];
  let start = 0;
  for (let i = 0; i <= text.length; ++i) {
    const boundary = i < text.length && isClauseBoundary(text, i);
    if (!boundary && i < text.length) continue;
    if (i > start) segments.push({ kind: 'text', value: text.slice(start, i) });
    if (i < text.length) {
      let end = i;
      while (end < text.length && isClauseBoundary(text, end)) ++end;
      segments.push({ kind: 'punctuation', value: text.slice(i, end) });
      i = end - 1;
      start = end;
    }
  }
  return segments;
}

// --- phonemizer loading ----------------------------------------------------

/**
 * Candidate locations for the `phonemizer` ESM bundle, tried in order.
 * The bundle inlines its eSpeak NG WASM, so no extra asset fetch is required.
 */
const PHONEMIZER_SOURCES = [
  // 'phonemizer', // Honours an import map when the page provides one.
  // '../node_modules/phonemizer/dist/phonemizer.js',
  // '../../node_modules/phonemizer/dist/phonemizer.js',
  'https://cdn.jsdelivr.net/npm/phonemizer@1.2.1/dist/phonemizer.js',
];

let phonemizePromise;

/** Resolve `phonemize` once, falling back through the candidate sources. */
export function loadPhonemize({ onProgress } = {}) {
  phonemizePromise ??= (async () => {
    const failures = [];
    for (const source of PHONEMIZER_SOURCES) {
      try {
        onProgress?.(`Loading phonemizer from ${source}…`);
        const module = await import(/* @vite-ignore */ source);
        if (typeof module.phonemize !== 'function') throw new Error('module has no phonemize export');
        return module.phonemize;
      } catch (error) {
        failures.push(`${source}: ${error.message}`);
      }
    }
    throw new Error(`Could not load the phonemizer package.\n${failures.join('\n')}`);
  })();
  return phonemizePromise;
}

// --- public entry point ----------------------------------------------------

/**
 * Run the full `english_cleaners2` chain and return espeak phonemes.
 *
 * @param {string} text Raw user sentence.
 * @param {{ language?: string, onProgress?: (message: string) => void }} [options]
 * @returns {Promise<{ phonemes: string, ascii: string }>}
 */
export async function phonemizeForMatcha(text, { language = 'en-us', onProgress } = {}) {
  const phonemize = await loadPhonemize({ onProgress });

  const ascii = expandAbbreviations(convertToAscii(text).toLowerCase());
  const segments = splitIntoClauses(ascii);

  let phonemes = '';
  for (const segment of segments) {
    if (segment.kind === 'punctuation') {
      phonemes += segment.value;
      continue;
    }
    // espeak strips surrounding whitespace, so restore the gaps the text had.
    const leading = /^\s/.test(segment.value) ? ' ' : '';
    const trailing = /\s$/.test(segment.value) ? ' ' : '';
    const spoken = segment.value.trim();
    if (!spoken) {
      phonemes += ' ';
      continue;
    }
    const chunks = await phonemize(spoken, language);
    phonemes += leading + (Array.isArray(chunks) ? chunks.join(' ') : String(chunks)) + trailing;
  }

  return { phonemes: collapseWhitespace(removeBrackets(phonemes)).trim(), ascii };
}
