import { doubleMetaphone } from "double-metaphone";
import type { NameRecord } from "./types";

// ==================================================
// Index Store
// ==================================================

export const recordsById   = new Map<string, NameRecord>();

export const exactFirst    = new Map<string, string[]>();
export const exactLast     = new Map<string, string[]>();

export const phoneticFirst = new Map<string, string[]>();
export const phoneticLast  = new Map<string, string[]>();

export const initialFirst  = new Map<string, string[]>();
export const initialLast   = new Map<string, string[]>();

/**
 * Tracks how many dataset records share a given last-word (surname position).
 * OCR-noise variants appear 1-3 times; genuine person clusters appear 8-15+.
 * Used during disambiguation to avoid cross-cluster false positives.
 */
export const surnameFreq = new Map<string, number>();

export function clearAll(): void {
  for (const m of [
    recordsById, exactFirst, exactLast,
    phoneticFirst, phoneticLast,
    initialFirst, initialLast,
    surnameFreq,
  ]) m.clear();
}

// ==================================================
// Helpers
// ==================================================

export function pushIndex(map: Map<string, string[]>, key: string, id: string): void {
  if (!key) return;
  const arr = map.get(key);
  if (arr) arr.push(id);
  else map.set(key, [id]);
}

export function metaphoneKey(word: string): string {
  const [primary] = doubleMetaphone(word);
  return primary || word;
}

/** Index a single record into all lookup maps. */
export function indexRecord(rec: NameRecord): void {
  recordsById.set(rec.id, rec);

  pushIndex(exactFirst,    rec.firstWord,                rec.id);
  pushIndex(exactLast,     rec.lastWord,                 rec.id);
  pushIndex(phoneticFirst, metaphoneKey(rec.firstWord),  rec.id);
  pushIndex(phoneticLast,  metaphoneKey(rec.lastWord),   rec.id);
  pushIndex(initialFirst,  rec.firstWord[0],             rec.id);
  pushIndex(initialLast,   rec.lastWord[0],              rec.id);

  surnameFreq.set(rec.lastWord, (surnameFreq.get(rec.lastWord) ?? 0) + 1);
}