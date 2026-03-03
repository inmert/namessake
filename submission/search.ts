// submission/search.ts
// ES phonetic + fuzzy retrieval + strict local precision re-rank

// ==================================================
// Tunable Parameters
// ==================================================

export const PARAMS = {
  SURNAME_FLOOR: 0.5,
  BOTH_FIRST_MIN: 0.7,
  THRESHOLD_SMALL: 0.64,
  THRESHOLD_LARGE: 0.655,
};

const INITIAL_MATCH = 0.8;
const MISSING_FIRST = 0.6;
const INITIAL_SOFT  = 0.30;
const INITIAL_HARD  = 0.0;

const LARGE_DATASET_CUTOFF = 5000;
const ES_CANDIDATE_SIZE    = 350; // slightly higher for recall safety

// ==================================================
// Types
// ==================================================

interface NameRecord {
  id: string;
  rawName: string;
  normName: string;
  initials: string[];
  words: string[];
  firstWord: string;
  lastWord: string;
  wordCount: number;
}

interface ESDoc {
  record_id: string;
  rawName: string;
  normName: string;
  initials: string[];
  words: string[];
  firstWord: string;
  lastWord: string;
  wordCount: number;
}

// ==================================================
// ES Connection
// ==================================================

let ES_BASE  = (process.env.ES_URL ?? "http://localhost:9200").replace(/\/$/, "");
let ES_INDEX = "names_search";
let THRESHOLD = PARAMS.THRESHOLD_SMALL;

async function esRequest(method: string, path: string, body?: unknown) {
  const res = await fetch(`${ES_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`ES ${method} ${path} → ${res.status}: ${text}`);
  return text ? JSON.parse(text) : {};
}

// ==================================================
// Normalization
// ==================================================

const SUFFIX_RE      = /\b(Jr\.?|Sr\.?|II|III|IV)\b/gi;
const COMMA_RE       = /^([^,]+),\s*(.+)$/;
const INITIAL_RE     = /^[a-z]\.$/;
const JUNK_PREFIX_RE = /^[a-z][A-Z]/;
const CONCAT_X_RE    = /([a-z])x([A-Z])/g;

function normalize(name: string): string {
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

function splitNorm(norm: string) {
  const initials: string[] = [];
  const words: string[] = [];
  for (const t of norm.split(" ")) {
    if (!t) continue;
    if (INITIAL_RE.test(t)) initials.push(t[0]);
    else if (t.length > 1) words.push(t);
  }
  return { initials, words };
}

// ==================================================
// Similarity
// ==================================================

let trigramCache = new Map<string, Set<string>>();

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

function trigramJaccard(a: string, b: string) {
  const ta = trigrams(a);
  const tb = trigrams(b);
  let inter = 0;
  const [small, large] = ta.size < tb.size ? [ta, tb] : [tb, ta];
  for (const t of small) if (large.has(t)) inter++;
  return inter / (ta.size + tb.size - inter);
}

function seqSim(a: string, b: string) {
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

const tokenSim = (a: string, b: string) =>
  (trigramJaccard(a, b) + seqSim(a, b)) / 2;

// ==================================================
// Scoring (unchanged logic)
// ==================================================

function scoreMatch(q: NameRecord, r: NameRecord): number {
  if (!q.words.length || !r.words.length) return 0;

  const pairings: [string, string, boolean][] = [
    [q.lastWord,  r.lastWord,  false],
    [q.firstWord, r.firstWord, true],
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

    let firstScore    = 0;
    let initialMatched = false;

    for (const init of q.initials)
      if (rfw[0]?.startsWith(init)) {
        firstScore     = INITIAL_MATCH;
        initialMatched = true;
        break;
      }

    if (!initialMatched)
      for (const init of r.initials)
        if (qfw[0]?.startsWith(init)) {
          firstScore     = INITIAL_MATCH;
          initialMatched = true;
          break;
        }

    if (!initialMatched && qfw.length && rfw.length) {
      let compared = 0;
      for (const a of qfw) {
        for (const b of rfw) {
          if (compared++ > 3) break;
          firstScore = Math.max(firstScore, tokenSim(a, b));
          if (firstScore > 0.8) break;
        }
        if (firstScore > 0.8) break;
      }
    } else if (!qfw.length && !rfw.length) {
      if (q.initials.length > 0 && r.initials.length > 0) {
        const initMatch = q.initials.some(qi =>
          r.initials.some(ri => qi === ri)
        );
        firstScore = initMatch ? INITIAL_MATCH : 0;
      } else {
        firstScore = MISSING_FIRST;
      }
    } else {
      firstScore = initialMatched
        ? INITIAL_MATCH
        : q.initials.length && rfw.length
        ? INITIAL_SOFT
        : r.initials.length && qfw.length
        ? INITIAL_HARD
        : MISSING_FIRST;
    }

    if (bothFirst && firstScore < PARAMS.BOTH_FIRST_MIN) continue;
    if (firstScore < 0.45) continue;

    const combined = 0.5 * lastSim + 0.5 * firstScore;
    if (combined > best) best = combined;
    if (best > THRESHOLD + 0.15) break;
  }

  return best;
}

// ==================================================
// Setup
// ==================================================

export async function setup(datasetPath: string): Promise<void> {
  ES_INDEX = `names_search_${Date.now()}`;

  await esRequest("PUT", `/${ES_INDEX}`, {
    settings: {
      analysis: {
        filter: {
          my_metaphone: {
            type: "phonetic",
            encoder: "double_metaphone",
            replace: true
          }
        },
        analyzer: {
          phonetic_analyzer: {
            tokenizer: "standard",
            filter: ["lowercase", "asciifolding", "my_metaphone"]
          }
        }
      },
      index: {
        number_of_shards: 1,
        number_of_replicas: 0,
        refresh_interval: "-1"
      }
    },
    mappings: {
      properties: {
        record_id: { type: "keyword" },
        rawName:   { type: "keyword" },
        normName:  { type: "keyword" },
        initials:  { type: "keyword" },
        words:     { type: "keyword" },
        firstWord: { type: "keyword" },
        lastWord:  { type: "keyword" },
        wordCount: { type: "integer" },

        first_name: {
          type: "text",
          fields: {
            phonetic: { type: "text", analyzer: "phonetic_analyzer" },
            keyword:  { type: "keyword" }
          }
        },
        last_name: {
          type: "text",
          fields: {
            phonetic: { type: "text", analyzer: "phonetic_analyzer" },
            keyword:  { type: "keyword" }
          }
        }
      }
    }
  });

  const lines = (await Bun.file(datasetPath).text()).split(/\r?\n/);
  let count = 0;
  let batch: string[] = [];

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

    const doc = {
      record_id: id,
      rawName,
      normName: norm,
      initials,
      words,
      firstWord: words[0],
      lastWord: words[words.length - 1],
      wordCount: words.length,
      first_name: words[0],
      last_name: words[words.length - 1],
    };

    batch.push(JSON.stringify({ index: { _index: ES_INDEX, _id: id } }));
    batch.push(JSON.stringify(doc));
    count++;

    if (batch.length >= 1000) {
      await fetch(`${ES_BASE}/_bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/x-ndjson" },
        body: batch.join("\n") + "\n",
      });
      batch = [];
    }
  }

  if (batch.length) {
    await fetch(`${ES_BASE}/_bulk`, {
      method: "POST",
      headers: { "Content-Type": "application/x-ndjson" },
      body: batch.join("\n") + "\n",
    });
  }

  await esRequest("POST", `/${ES_INDEX}/_refresh`);

  THRESHOLD = count > LARGE_DATASET_CUTOFF
    ? PARAMS.THRESHOLD_LARGE
    : PARAMS.THRESHOLD_SMALL;
}

// ==================================================
// Search
// ==================================================

export async function search(query: string): Promise<string[]> {
  trigramCache.clear();

  const norm = normalize(query);
  const { initials, words } = splitNorm(norm);
  if (!words.length) return [];

  const q: NameRecord = {
    id: "",
    rawName: query,
    normName: norm,
    initials,
    words,
    firstWord: words[0],
    lastWord: words[words.length - 1],
    wordCount: words.length,
  };

  const should: any[] = [
    { match: { last_name:  { query: q.lastWord,  fuzziness: "AUTO", boost: 3 } } },
    { match: { first_name: { query: q.firstWord, fuzziness: "AUTO", boost: 2 } } },
    { match: { last_name:  { query: q.firstWord, fuzziness: "AUTO", boost: 1.5 } } },
    { match: { first_name: { query: q.lastWord,  fuzziness: "AUTO", boost: 1 } } },

    { match: { "last_name.phonetic":  { query: q.lastWord,  boost: 2 } } },
    { match: { "first_name.phonetic": { query: q.firstWord, boost: 1.5 } } },
    { match: { "last_name.phonetic":  { query: q.firstWord, boost: 1 } } },
    { match: { "first_name.phonetic": { query: q.lastWord,  boost: 0.8 } } },
  ];

  for (const i of initials) {
    should.push({ prefix: { "first_name.keyword": { value: i, boost: 2 } } });
    should.push({ prefix: { "last_name.keyword":  { value: i, boost: 1 } } });
  }

  const resp = await esRequest("POST", `/${ES_INDEX}/_search`, {
    query: { bool: { should, minimum_should_match: 1 } },
    size: ES_CANDIDATE_SIZE,
  });

  const results: string[] = [];

  for (const hit of resp.hits.hits as Array<{ _source: ESDoc }>) {
    const doc = hit._source;

    const rec: NameRecord = {
      id: doc.record_id,
      rawName: doc.rawName,
      normName: doc.normName,
      initials: doc.initials,
      words: doc.words,
      firstWord: doc.firstWord,
      lastWord: doc.lastWord,
      wordCount: doc.wordCount,
    };

    if (scoreMatch(q, rec) >= THRESHOLD)
      results.push(rec.id);
  }

  return results;
}

export async function cleanup(): Promise<void> {
  try { await esRequest("DELETE", `/${ES_INDEX}`); }
  catch {}
}

export default search;