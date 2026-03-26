import { doubleMetaphone } from "double-metaphone";

// ==================================================
// Types
// ==================================================

export interface NameRecord {
  id:        string;
  rawName:   string;
  normName:  string;
  initials:  string[];
  words:     string[];
  firstWord: string;
  lastWord:  string;
  wordCount: number;
}

// ==================================================
// Params
// ==================================================

export const PARAMS = {
  SURNAME_FLOOR:   0.5,
  BOTH_FIRST_MIN:  0.7,
  THRESHOLD_SMALL: 0.665,
  THRESHOLD_LARGE: 0.655,
};

export const LARGE_DATASET_CUTOFF = 5_000;

export const INITIAL_MATCH = 0.8;
export const MISSING_FIRST = 0.6;
export const INITIAL_SOFT  = 0.37;
export const INITIAL_HARD  = 0.0;

// ==================================================
// Normalization
// ==================================================

const SUFFIX_RE      = /\b(Jr\.?|Sr\.?|II|III|IV)\b/gi;
const COMMA_RE       = /^([^,]+),\s*(.+)$/;
const INITIAL_RE     = /^[a-z]\.$/;
const JUNK_PREFIX_RE = /^[a-z][A-Z]/;
const CONCAT_X_RE    = /([a-z])x([A-Z])/g;

/** Normalise a raw name string into a clean lowercase token string. */
export function normalize(name: string): string {
  name = name.replace(/^"|"$/g, "");

  const m = COMMA_RE.exec(name);
  if (m) name = `${m[2].trim()} ${m[1].trim()}`;

  return name
    .replace(/0/g, "o")
    .replace(/1/g, "l")
    .replace(SUFFIX_RE, "")
    .replace(CONCAT_X_RE, "$1 $2")
    .split(/\s+/)
    .filter(Boolean)
    .map(t => (t.length >= 2 && JUNK_PREFIX_RE.test(t) ? t.slice(1) : t))
    .join(" ")
    .toLowerCase()
    .trim()
    .replace(/\s+/g, " ");
}

/** Split a normalised string into standalone initials and full word tokens. */
export function splitNorm(norm: string): { initials: string[]; words: string[] } {
  const initials: string[] = [];
  const words:    string[] = [];

  for (const t of norm.split(" ")) {
    if (!t) continue;
    if (INITIAL_RE.test(t)) initials.push(t[0]);
    else if (t.length > 1)  words.push(t);
  }

  return { initials, words };
}

// ==================================================
// Similarity
// ==================================================

const trigramCache = new Map<string, Set<string>>();

export function clearTrigramCache(): void {
  trigramCache.clear();
}

function trigrams(s: string): Set<string> {
  const cached = trigramCache.get(s);
  if (cached) return cached;

  const padded = `##${s}##`;
  const set = new Set<string>();
  for (let i = 0; i <= padded.length - 3; i++)
    set.add(padded.slice(i, i + 3));

  trigramCache.set(s, set);
  return set;
}

function trigramJaccard(a: string, b: string): number {
  const ta = trigrams(a);
  const tb = trigrams(b);

  let inter = 0;
  const [small, large] = ta.size < tb.size ? [ta, tb] : [tb, ta];
  for (const t of small)
    if (large.has(t)) inter++;

  return inter / (ta.size + tb.size - inter);
}

function seqSim(a: string, b: string): number {
  if (a === b) return 1;
  const m = a.length, n = b.length;
  if (!m || !n || Math.abs(m - n) > Math.max(m, n) * 0.7) return 0;

  const dp = new Int16Array(n + 1);
  for (let i = 0; i < m; i++) {
    let prev = 0;
    for (let j = 0; j < n; j++) {
      const tmp = dp[j + 1];
      dp[j + 1] = a[i] === b[j] ? prev + 1 : Math.max(dp[j + 1], dp[j]);
      prev = tmp;
    }
  }

  return (2 * dp[n]) / (m + n);
}

/** Blended token similarity: average of trigram Jaccard and LCS ratio. */
export const tokenSim = (a: string, b: string): number =>
  (trigramJaccard(a, b) + seqSim(a, b)) / 2;

// ==================================================
// Store
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

function pushIndex(map: Map<string, string[]>, key: string, id: string): void {
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

// ==================================================
// Scoring
// ==================================================

/**
 * Score how well query record q matches dataset record r.
 *
 * Strategy: try four surname/given-name pairings to handle reversed-name input,
 * then blend surname similarity with the best first-name evidence found.
 * Returns a value in [0, 1]; caller compares against THRESHOLD.
 */
export function scoreMatch(q: NameRecord, r: NameRecord, threshold: number): number {
  if (!q.words.length || !r.words.length) return 0;

  // (qSurname, rSurname, requireBothFirstHigh)
  const pairings: [string, string, boolean][] = [
    [q.lastWord,  r.lastWord,  false],
    [q.firstWord, r.firstWord, true ],
    [q.lastWord,  r.firstWord, false],
    [q.firstWord, r.lastWord,  false],
  ];

  let best = 0;
  const seen = new Set<string>();

  for (const [qLN, rLN, bothFirst] of pairings) {
    const key = qLN + "|" + rLN;
    if (seen.has(key)) continue;
    seen.add(key);

    const lastSim = tokenSim(qLN, rLN);
    if (lastSim < PARAMS.SURNAME_FLOOR) continue;

    const qfw = q.words.filter(w => w !== qLN);
    const rfw = r.words.filter(w => w !== rLN);

    let firstScore     = 0;
    let initialMatched = false;

    for (const init of q.initials) {
      if (rfw[0]?.startsWith(init)) { firstScore = INITIAL_MATCH; initialMatched = true; break; }
    }

    if (!initialMatched) {
      for (const init of r.initials) {
        if (qfw[0]?.startsWith(init)) { firstScore = INITIAL_MATCH; initialMatched = true; break; }
      }
    }

    if (!initialMatched && qfw.length && rfw.length) {
      let compared = 0;
      outer: for (const a of qfw) {
        for (const b of rfw) {
          if (compared++ > 3) break outer;
          firstScore = Math.max(firstScore, tokenSim(a, b));
          if (firstScore > 0.8) break outer;
        }
      }
    } else if (!qfw.length && !rfw.length) {
      const initMatch = q.initials.some(qi => r.initials.some(ri => qi === ri));
      firstScore = initMatch ? INITIAL_MATCH : 0;
    } else if (!initialMatched) {
      firstScore =
        q.initials.length && rfw.length ? INITIAL_SOFT  :
        r.initials.length && qfw.length ? INITIAL_HARD  :
        MISSING_FIRST;
    }

    if (bothFirst && firstScore < PARAMS.BOTH_FIRST_MIN) continue;
    if (firstScore < 0.40) continue;

    const combined = 0.5 * lastSim + 0.5 * firstScore;
    if (combined > best) best = combined;
    if (best > threshold + 0.15) break; // early-exit: clearly above bar
  }

  return best;
}