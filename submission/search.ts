/**
 * submission/search.ts
 *
 * Public API: setup / search / cleanup
 *
 * Strategy
 * --------
 * 1. setup() parses the CSV and builds in-memory lookup maps:
 *    exact token, phonetic (double-metaphone), and first-initial.
 * 2. search() collects candidates from all maps then re-ranks with a
 *    blended trigram-Jaccard + LCS score across four surname/given-name
 *    pairings (handles reversed names, initials, OCR noise, typos).
 * 3. A surname-cluster disambiguation pass drops high-frequency parallel
 *    clusters that sneak in via phonetic indexing.
 */

import { PARAMS, LARGE_DATASET_CUTOFF } from "./lib/params";
import { normalize, splitNorm } from "./lib/normalize";
import { clearTrigramCache } from "./lib/similarity";
import {
  recordsById, exactFirst, exactLast,
  phoneticFirst, phoneticLast,
  initialFirst, initialLast,
  surnameFreq, clearAll, indexRecord, metaphoneKey,
} from "./lib/store";
import { scoreMatch } from "./lib/score";
import type { NameRecord } from "./lib/types";

let THRESHOLD = PARAMS.THRESHOLD_SMALL;

// ==================================================
// Setup
// ==================================================

export async function setup(datasetPath: string): Promise<void> {
  clearAll();

  const lines = (await Bun.file(datasetPath).text()).split(/\r?\n/);
  let count = 0;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const comma = line.indexOf(",");
    if (comma === -1) continue;

    const id = line.slice(0, comma).trim();
    let rawName = line.slice(comma + 1).trim();
    if (rawName.startsWith('"') && rawName.endsWith('"'))
      rawName = rawName.slice(1, -1).replace(/""/g, '"');

    const norm = normalize(rawName);
    const { initials, words } = splitNorm(norm);
    if (!words.length) continue;

    indexRecord({
      id, rawName, normName: norm, initials, words,
      firstWord: words[0],
      lastWord:  words[words.length - 1],
      wordCount: words.length,
    });
    count++;
  }

  THRESHOLD = count > LARGE_DATASET_CUTOFF
    ? PARAMS.THRESHOLD_LARGE
    : PARAMS.THRESHOLD_SMALL;
}

// ==================================================
// Search
// ==================================================

export async function search(query: string): Promise<string[]> {
  clearTrigramCache();

  const norm = normalize(query);
  const { initials, words } = splitNorm(norm);
  if (!words.length) return [];

  const q: NameRecord = {
    id: "", rawName: query, normName: norm, initials, words,
    firstWord: words[0],
    lastWord:  words[words.length - 1],
    wordCount: words.length,
  };

  // --- Candidate retrieval ---
  const candidates = new Set<string>();
  const add = (arr?: string[]) => arr?.forEach(id => candidates.add(id));

  add(exactLast.get(q.lastWord));
  add(exactFirst.get(q.firstWord));
  add(exactLast.get(q.firstWord));   // reversed-name queries
  add(exactFirst.get(q.lastWord));   // reversed-name queries
  add(phoneticLast.get(metaphoneKey(q.lastWord)));
  add(phoneticFirst.get(metaphoneKey(q.firstWord)));
  for (const i of initials) {
    add(initialFirst.get(i));
    add(initialLast.get(i));
  }

  // --- Scoring pass ---
  const results = [...candidates].filter(
    id => scoreMatch(q, recordsById.get(id)!, THRESHOLD) >= THRESHOLD
  );

  // --- Surname-cluster disambiguation ---
  //
  // Phonetic indexing can pull in a different person cluster that merely
  // sounds like the query surname (e.g. Dawsen vs Dawson). If any result
  // already has an exact surname match, drop results whose surname is both
  // different AND high-frequency (>=4) -- those belong to a separate cluster.
  // OCR-noise variants appear only 1-3 times; genuine clusters appear 8-15+.
  //
  // qSurname uses whichever query token appears more often as a dataset
  // last-word, correctly handling reversed input like "Brown Gavin".
  if (results.length === 0) return results;

  const qLastCount  = exactLast.get(q.lastWord)?.length  ?? 0;
  const qFirstCount = exactLast.get(q.firstWord)?.length ?? 0;
  const qSurname    = qFirstCount > qLastCount ? q.firstWord : q.lastWord;

  const exactSurnameHit = results.some(id => {
    const r = recordsById.get(id)!;
    return r.lastWord === qSurname || r.firstWord === qSurname;
  });

  if (!exactSurnameHit) return results;

  return results.filter(id => {
    const r = recordsById.get(id)!;
    if (r.lastWord === qSurname || r.firstWord === qSurname) return true;
    return (surnameFreq.get(r.lastWord) ?? 0) < 4;
  });
}

// ==================================================
// Cleanup
// ==================================================

export async function cleanup(): Promise<void> {
  clearAll();
}

export default search;