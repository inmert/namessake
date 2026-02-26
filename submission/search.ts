// =============================================================================
// Types
// =============================================================================

interface NameRecord {
  id: string;
  rawName: string;
  normName: string;
  initials: string[];
  words: string[];
}

// =============================================================================
// Constants
// =============================================================================

const THRESHOLD_SMALL = 0.64;
const THRESHOLD_LARGE = 0.655;
const LARGE_DATASET_CUTOFF = 5000;

const SCORE_INITIAL_MATCH = 0.8;
const SCORE_MISSING_FIRST = 0.6;
const SCORE_INITIAL_SOFT = 0.3;
const SCORE_INITIAL_HARD = 0.0;
const SCORE_BOTH_FIRST_MIN = 0.7;

const MIN_TRIGRAM_HITS = 2;

// =============================================================================
// Regex
// =============================================================================

const RE_SUFFIX   = /\b(Jr\.?|Sr\.?|II|III|IV)\b/gi;
const RE_COMMA    = /^([^,]+),\s*(.+)$/;
const RE_INITIAL  = /^[a-z]\.$/;
const RE_CONCAT_X = /([a-z])x([A-Z])/g;

// =============================================================================
// State
// =============================================================================

let records: NameRecord[] = [];

// Final compact index (memory-efficient)
let trigramIndex = new Map<string, Uint32Array>();

// Temporary builder index (only used during setup)
let buildIndex = new Map<string, number[]>();

let threshold = THRESHOLD_SMALL;

// Reusable LCS buffers
const _lcsA = new Int16Array(512);
const _lcsB = new Int16Array(512);

// =============================================================================
// Section 1: Normalization
// =============================================================================

function normalize(name: string): string {
  name = name.replace(/^"|"$/g, "");

  const m = RE_COMMA.exec(name);
  if (m) name = `${m[2].trim()} ${m[1].trim()}`;

  return name
    .replace(/0/g, "o")
    .replace(/1/g, "l")
    .replace(RE_SUFFIX, "")
    .replace(RE_CONCAT_X, "$1 $2")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function splitTokens(norm: string): { initials: string[]; words: string[] } {
  const initials: string[] = [];
  const words: string[] = [];

  for (const t of norm.split(" ")) {
    if (!t) continue;
    if (RE_INITIAL.test(t)) initials.push(t[0]);
    else if (t.length > 1) words.push(t);
  }

  return { initials, words };
}

// =============================================================================
// Section 2: Similarity
// =============================================================================

function trigrams(s: string): string[] {
  const padded = `##${s}##`;
  const result: string[] = [];
  for (let i = 0; i <= padded.length - 3; i++) {
    result.push(padded.slice(i, i + 3));
  }
  return result;
}

function trigramJaccard(a: string, b: string): number {
  const ta = new Set(trigrams(a));
  const tb = new Set(trigrams(b));
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter++;
  return inter / (ta.size + tb.size - inter);
}

function lcsSim(a: string, b: string): number {
  if (a === b) return 1;

  const m = a.length;
  const n = b.length;
  if (!m || !n) return 0;

  const prev = n < 512 ? _lcsA : new Int16Array(n + 1);
  const curr = n < 512 ? _lcsB : new Int16Array(n + 1);

  prev.fill(0);
  curr.fill(0);

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      curr[j + 1] =
        a[i] === b[j]
          ? prev[j] + 1
          : Math.max(prev[j + 1], curr[j]);
    }
    prev.set(curr);
  }

  return (2 * curr[n]) / (m + n);
}

function tokenSim(a: string, b: string): number {
  return (trigramJaccard(a, b) + lcsSim(a, b)) / 2;
}

// =============================================================================
// Section 3: Match Scoring
// =============================================================================

function scoreFirstName(
  qInitials: string[],
  rInitials: string[],
  qRest: string[],
  rRest: string[],
): number {
  let score = 0;
  let initialMatched = false;

  for (const init of qInitials) {
    if (rRest[0]?.startsWith(init)) {
      score = Math.max(score, SCORE_INITIAL_MATCH);
      initialMatched = true;
    }
  }

  for (const init of rInitials) {
    if (qRest[0]?.startsWith(init)) {
      score = Math.max(score, SCORE_INITIAL_MATCH);
      initialMatched = true;
    }
  }

  if (qRest.length && rRest.length) {
    for (const a of qRest) {
      for (const b of rRest) {
        score = Math.max(score, tokenSim(a, b));
      }
    }
    return score;
  }

  if (!qRest.length && !rRest.length) return SCORE_MISSING_FIRST;
  if (initialMatched) return SCORE_INITIAL_MATCH;
  if (qInitials.length && rRest.length) return SCORE_INITIAL_SOFT;
  if (rInitials.length && qRest.length) return SCORE_INITIAL_HARD;

  return SCORE_MISSING_FIRST;
}

function scoreMatch(q: NameRecord, r: NameRecord): number {
  if (!q.words.length || !r.words.length) return 0;

  const qFirst = q.words[0];
  const qLast  = q.words[q.words.length - 1];
  const rFirst = r.words[0];
  const rLast  = r.words[r.words.length - 1];

  const pairings: [string, string, boolean][] = [
    [qLast,  rLast,  false],
    [qLast,  rFirst, false],
    [qFirst, rLast,  false],
    [qFirst, rFirst, true ],
  ];

  let best = 0;
  const seen = new Set<string>();

  for (const [qAnchor, rAnchor, bothFirst] of pairings) {
    const key = `${qAnchor}|${rAnchor}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const anchorSim = tokenSim(qAnchor, rAnchor);
    if (anchorSim < 0.38) continue;

    const qRest = q.words.filter(w => w !== qAnchor);
    const rRest = r.words.filter(w => w !== rAnchor);

    const firstScore = scoreFirstName(
      q.initials,
      r.initials,
      qRest,
      rRest
    );

    if (bothFirst && firstScore < SCORE_BOTH_FIRST_MIN) continue;

    const combined = 0.5 * anchorSim + 0.5 * firstScore;
    if (combined > best) best = combined;
  }

  return best;
}

// =============================================================================
// Section 4: Indexing (Memory Efficient)
// =============================================================================

function indexWord(word: string, recordIdx: number): void {
  for (const tg of trigrams(word)) {
    let bucket = buildIndex.get(tg);
    if (!bucket) {
      bucket = [];
      buildIndex.set(tg, bucket);
    }
    bucket.push(recordIdx);
  }
}

function finalizeIndex(): void {
  for (const [tg, arr] of buildIndex) {
    arr.sort((a, b) => a - b);

    let uniqueCount = 1;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] !== arr[i - 1]) {
        arr[uniqueCount++] = arr[i];
      }
    }

    const compact = new Uint32Array(uniqueCount);
    for (let i = 0; i < uniqueCount; i++) {
      compact[i] = arr[i];
    }

    trigramIndex.set(tg, compact);
  }

  buildIndex.clear();
}

function getCandidates(queryWords: string[]): Map<number, number> {
  const counts = new Map<number, number>();
  const minHits = queryWords.length === 1 ? 1 : MIN_TRIGRAM_HITS;

  for (const w of queryWords) {
    for (const tg of trigrams(w)) {
      const bucket = trigramIndex.get(tg);
      if (!bucket) continue;

      for (let i = 0; i < bucket.length; i++) {
        const idx = bucket[i];
        counts.set(idx, (counts.get(idx) ?? 0) + 1);
      }
    }
  }

  const filtered = new Map<number, number>();
  for (const [idx, count] of counts) {
    if (count >= minHits) filtered.set(idx, count);
  }

  return filtered;
}

// =============================================================================
// Section 5: CSV Parsing
// =============================================================================

function parseCsvLine(line: string): [string, string] | null {
  const i = line.indexOf(",");
  if (i === -1) return null;

  let name = line.slice(i + 1).trim();
  if (name.startsWith('"') && name.endsWith('"'))
    name = name.slice(1, -1).replace(/""/g, '"');

  return [line.slice(0, i).trim(), name];
}

// =============================================================================
// Section 6: Public API
// =============================================================================

export async function setup(datasetPath: string): Promise<void> {
  records = [];
  trigramIndex = new Map();
  buildIndex = new Map();

  const lines = (await Bun.file(datasetPath).text()).split(/\r?\n/);

  for (let i = 1; i < lines.length; i++) {
    const parsed = parseCsvLine(lines[i].trim());
    if (!parsed) continue;

    const [id, rawName] = parsed;
    const normName = normalize(rawName);
    const { initials, words } = splitTokens(normName);

    const idx = records.length;
    records.push({ id, rawName, normName, initials, words });

    for (const w of words) indexWord(w, idx);
  }

  finalizeIndex();

  threshold =
    records.length > LARGE_DATASET_CUTOFF
      ? THRESHOLD_LARGE
      : THRESHOLD_SMALL;
}

export async function search(query: string): Promise<string[]> {
  if (!records.length) return [];

  const normQuery = normalize(query);
  const { initials, words } = splitTokens(normQuery);

  const q: NameRecord = {
    id: "",
    rawName: query,
    normName: normQuery,
    initials,
    words,
  };

  const candidates = getCandidates(words);
  const results: string[] = [];

  for (const idx of candidates.keys()) {
    if (scoreMatch(q, records[idx]) >= threshold) {
      results.push(records[idx].id);
    }
  }

  return results;
}

export async function cleanup(): Promise<void> {
  records = [];
  trigramIndex = new Map();
  buildIndex = new Map();
  threshold = THRESHOLD_SMALL;
}

export default search;