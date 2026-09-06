/**
 * Splitting helpers for reading long text.
 *
 * Matcha sequences are blank-interspersed: `[0, s0, 0, s1, 0, …, 0]`, always odd
 * with blank ID 0 at every even index. Splitting therefore has to land on a
 * blank so both halves stay valid standalone sequences.
 */

import { SYMBOL_TO_ID } from './matcha_tokenizer.mjs';

/** Matcha's `SPACE_ID`: the last entry of `_punctuation`. */
export const SPACE_ID = SYMBOL_TO_ID.get(' ');

/**
 * Abbreviations whose trailing period is not a sentence end. Covers the list
 * `cleaners.py` expands, plus the common ones a reader is likely to meet.
 */
const ABBREVIATIONS = new Set([
  'mr', 'mrs', 'ms', 'dr', 'drs', 'st', 'co', 'jr', 'sr', 'maj', 'gen', 'rev',
  'lt', 'hon', 'sgt', 'capt', 'esq', 'ltd', 'col', 'ft', 'prof', 'inc', 'dept',
  'vs', 'etc', 'eg', 'ie', 'fig', 'al', 'approx', 'jan', 'feb', 'mar', 'apr',
  'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
]);

/** A single initial ("J.") or a known abbreviation is not a sentence break. */
function endsWithAbbreviation(sentence) {
  const match = /(?:^|[\s(])([A-Za-z]{1,6})\.$/.exec(sentence);
  return Boolean(match) && (match[1].length === 1 || ABBREVIATIONS.has(match[1].toLowerCase()));
}

/**
 * Break text into sentences. `Intl.Segmenter` handles quotes and most edge
 * cases; the fallback is only for older engines. Either way, segments ending in
 * an abbreviation are rejoined so a pause is not inserted mid-phrase.
 */
export function splitSentences(text) {
  const paragraphs = text.split(/\n\s*\n/);
  const segmenter = typeof Intl !== 'undefined' && Intl.Segmenter
    ? new Intl.Segmenter('en', { granularity: 'sentence' })
    : null;

  const sentences = [];
  for (const paragraph of paragraphs) {
    const normalized = paragraph.replace(/\s+/g, ' ').trim();
    if (!normalized) continue;
    const pieces = segmenter
      ? [...segmenter.segment(normalized)].map(entry => entry.segment)
      : normalized.split(/(?<=[.!?…])\s+/);

    let carried = '';
    for (const piece of pieces) {
      const sentence = piece.trim();
      if (!sentence) continue;
      const combined = carried ? `${carried} ${sentence}` : sentence;
      if (endsWithAbbreviation(combined)) { carried = combined; continue; }
      sentences.push(combined);
      carried = '';
    }
    if (carried) sentences.push(carried);
  }
  return sentences;
}

/**
 * Cut one sequence into two valid sequences.
 *
 * Prefers a word boundary, dropping the space symbol itself; both halves then
 * already start and end on a blank, so no padding has to be added. Otherwise it
 * cuts at the nearest blank, which both halves share.
 */
export function splitTokenSequence(tokenIds) {
  const length = tokenIds.length;
  if (length < 5) return [tokenIds];
  const middle = length >> 1;

  let best = -1;
  for (let index = 1; index < length; index += 2) {
    if (tokenIds[index] !== SPACE_ID) continue;
    if (best < 0 || Math.abs(index - middle) < Math.abs(best - middle)) best = index;
  }
  if (best > 1 && best < length - 2) return [tokenIds.slice(0, best), tokenIds.slice(best + 1)];

  const cut = middle % 2 === 0 ? middle : middle + 1;
  if (cut < 2 || cut > length - 3) return [tokenIds];
  return [tokenIds.slice(0, cut + 1), tokenIds.slice(cut)];
}

/** Split repeatedly until every part fits `capacity`. */
export function splitToCapacity(tokenIds, capacity) {
  if (tokenIds.length <= capacity) return [tokenIds];
  const halves = splitTokenSequence(tokenIds);
  if (halves.length === 1) return halves;
  return halves.flatMap(half => splitToCapacity(half, capacity));
}
