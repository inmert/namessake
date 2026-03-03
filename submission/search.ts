// submission/search.ts
// Architecture: Elasticsearch for candidate retrieval at scale,
//               existing scoreMatch() for precision re-ranking.
//
// Uses Bun-native fetch() to talk to ES directly — no @elastic/elasticsearch
// client transport layer, which is incompatible with Bun's HTTP runtime.

// ==================================================
// Tunable Parameters (unchanged)
// ==================================================

export const PARAMS = {
  SURNAME_FLOOR: 0.5,
  BOTH_FIRST_MIN: 0.7,
  THRESHOLD_SMALL: 0.64,
  THRESHOLD_LARGE: 0.655,
};

// ==================================================
// Types & Globals
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

// Stored in ES — superset of NameRecord with extra search fields
interface ESDoc {
  record_id: string;
  rawName: string;
  normName: string;
  initials: string[];
  words: string[];
  firstWord: string;
  lastWord: string;
  wordCount: number;
  dm_codes: string[];
  // Text fields ES actually searches on
  first_name: string;
  last_name: string;
}

// ── Elasticsearch connection ──────────────────────────────────────────────
// All ES calls go through esRequest() using Bun's native fetch.
// No @elastic/* client is needed — avoids Undici/Bun incompatibility.

let ES_BASE  = (process.env.ES_URL ?? "http://localhost:9200").replace(/\/$/, "");
let ES_INDEX = "names_search";
let THRESHOLD = PARAMS.THRESHOLD_SMALL;
const LARGE_DATASET_CUTOFF = 5000;

// Per-query memoisation cache for trigram sets (same role as before)
let trigramCache = new Map<string, Set<string>>();

// How many candidates ES returns before re-ranking filter
const ES_CANDIDATE_SIZE = 300;

// ── Thin fetch wrapper ────────────────────────────────────────────────────

async function esRequest(
  method: string,
  path: string,
  body?: unknown,
): Promise<any> {
  const url  = `${ES_BASE}${path}`;
  const init: RequestInit = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) init.body = JSON.stringify(body);
  const res = await fetch(url, init);
  const text = await res.text();
  if (!res.ok) throw new Error(`ES ${method} ${path} → ${res.status}: ${text}`);
  return text ? JSON.parse(text) : {};
}

const INITIAL_MATCH  = 0.8;
const MISSING_FIRST  = 0.6;
const INITIAL_SOFT   = 0.30;
const INITIAL_HARD   = 0.0;

// ==================================================
// Double Metaphone  (Lawrence Philips / faithful port)
// ==================================================

function doubleMetaphone(word: string): string[] {
  const VOWELS = "AEIOUY";
  word = word.toUpperCase();

  if (/^(GN|KN|PN|AE|WR)/.test(word)) word = word.slice(1);

  const len  = word.length;
  if (!len) return [];

  const get = (i: number) => (i >= 0 && i < len ? word[i] : "");
  const sub = (i: number, n: number) => word.slice(i, i + n);

  const slavoGermanic = () =>
    word.includes("W") || word.includes("K") ||
    word.includes("CZ") || word.includes("WITZ");

  let p = "", s = "";
  const add = (a: string, b?: string) => { p += a; s += b ?? a; };

  let i = 0;

  if (VOWELS.includes(word[0])) { add("A"); i = 1; }

  while (i < len) {
    const c = get(i);

    if (VOWELS.includes(c) && i > 0) { i++; continue; }

    switch (c) {
      case "B":
        add("P");
        i += get(i + 1) === "B" ? 2 : 1;
        break;

      case "C":
        if (i > 1 && !VOWELS.includes(get(i-2)) && sub(i-1,3) === "ACH" &&
            get(i+2) !== "I" && (get(i+2) !== "E" || sub(i-2,6) === "BACHER" || sub(i-2,6) === "MACHER")) {
          add("K"); i += 2;
        } else if (i === 0 && sub(0,6) === "CAESAR") {
          add("S"); i += 2;
        } else if (sub(i,4) === "CHIA") {
          add("K"); i += 2;
        } else if (sub(i,2) === "CH") {
          if (i > 0 && sub(i,4) === "CHAE") { add("K","X"); i += 2; }
          else if (i === 0 && (sub(i+1,5) === "HARAC" || sub(i+1,5) === "HARIS" ||
              ["HOR","HYM","HIA","HEM"].includes(sub(i+1,3))) && sub(0,5) !== "CHORE") {
            add("K"); i += 2;
          } else if (["VAN ","VON "].includes(sub(0,4)) || sub(0,3) === "SCH" ||
              ["ORCHES","ARCHIT","ORCHID"].includes(sub(i-2,6)) ||
              "TS".includes(get(i+2)) ||
              ("AOUE".includes(get(i-1)) && "LRNMBHFVW ".includes(get(i+2)))) {
            add("K"); i += 2;
          } else if (i > 0) {
            add(sub(0,2) === "MC" ? "K" : "X", "K"); i += 2;
          } else { add("X"); i += 2; }
        } else if (sub(i,2) === "CZ" && sub(i-2,4) !== "WICZ") {
          add("S","X"); i += 2;
        } else if (sub(i+1,3) === "CIA") {
          add("X"); i += 3;
        } else if (sub(i,2) === "CC" && !(i === 1 && get(0) === "M")) {
          if ("IEH".includes(get(i+2))) { add(sub(i+2,2) === "HU" ? "K" : "KS"); i += 3; }
          else { add("K"); i += 2; }
        } else if (["CK","CG","CQ"].includes(sub(i,2))) {
          add("K"); i += 2;
        } else if (["CI","CE","CY"].includes(sub(i,2))) {
          add(["CIO","CIE","CIA"].includes(sub(i,3)) ? "S" : "S",
              ["CIO","CIE","CIA"].includes(sub(i,3)) ? "X" : "S");
          i += 2;
        } else {
          add("K");
          i += [" C"," Q"," G"].includes(sub(i+1,2)) ? 3 : "CKQ".includes(get(i+1)) ? 2 : 1;
        }
        break;

      case "D":
        if (sub(i,2) === "DG") {
          if ("IEY".includes(get(i+2))) { add("J"); i += 3; }
          else { add("TK"); i += 2; }
        } else if (["DT","DD"].includes(sub(i,2))) {
          add("T"); i += 2;
        } else { add("T"); i += 1; }
        break;

      case "F":
        add("F"); i += get(i+1) === "F" ? 2 : 1;
        break;

      case "G":
        if (get(i+1) === "H") {
          if (i > 0 && !VOWELS.includes(get(i-1))) { add("K"); i += 2; }
          else if (i === 0) {
            add("IEY".includes(get(i+2)) ? "J" : "K"); i += 2;
          } else if ((i > 1 && "BHD".includes(get(i-2))) ||
                     (i > 2 && "BHD".includes(get(i-3))) ||
                     (i > 3 && "BH".includes(get(i-4)))) {
            i += 2;
          } else {
            if (i > 2 && get(i-1) === "U" && "CGLOQRU".includes(get(i-3))) add("F");
            else if (i > 0 && get(i-1) !== "I") add("K");
            i += 2;
          }
        } else if (get(i+1) === "N") {
          if (i === 1 && VOWELS.includes(get(0)) && !slavoGermanic()) { add("KN","N"); }
          else { add(!(sub(i+2,2) === "EY" || get(i+1) === "Y" || slavoGermanic()) ? "N" : "KN",
                     "KN"); }
          i += 2;
        } else if (sub(i+1,2) === "LI" && !slavoGermanic()) {
          add("KL","L"); i += 2;
        } else if (i === 0 && (get(i+1) === "Y" ||
            ["ES","EP","EB","EL","EY","IB","IL","IN","IE","EI","ER"].includes(sub(i+1,2)))) {
          add("K","J"); i += 2;
        } else if ((sub(i+1,2) === "ER" || get(i+1) === "Y") &&
            !["DANGER","RANGER","MANGER"].includes(sub(0,6)) &&
            !"EI".includes(get(i-1)) && !["RGY","OGY"].includes(sub(i-1,3))) {
          add("K","J"); i += 2;
        } else if ("EIY".includes(get(i+1)) || sub(i-1,4) === "AGGI") {
          if (["VAN ","VON "].includes(sub(0,4)) || sub(0,3) === "SCH" || sub(i+1,2) === "ET") add("K");
          else add(sub(i+1,4) === "IER " ? "J" : "J", "K");
          i += 2;
        } else { add("K"); i += get(i+1) === "G" ? 2 : 1; }
        break;

      case "H":
        if ((i === 0 || VOWELS.includes(get(i-1))) && VOWELS.includes(get(i+1))) {
          add("H"); i += 2;
        } else i += 1;
        break;

      case "J":
        if (sub(i,4) === "JOSE" || sub(0,4) === "SAN ") {
          add((i === 0 && get(i+4) === " ") || sub(0,4) === "SAN " ? "H" : "J", "H"); i++;
        } else if (i === 0 && sub(0,4) !== "JOSE") {
          add("J","A"); i += get(i+1) === "J" ? 2 : 1;
        } else {
          if (VOWELS.includes(get(i-1)) && !slavoGermanic() && "AO".includes(get(i+1))) add("J","H");
          else if (i === len-1) add("J"," ");
          else if (!"LTKSNMBZ".includes(get(i+1)) && !"SKL".includes(get(i-1))) add("J");
          i += get(i+1) === "J" ? 2 : 1;
        }
        break;

      case "K":
        add("K"); i += get(i+1) === "K" ? 2 : 1;
        break;

      case "L":
        if (get(i+1) === "L") {
          if ((i === len-3 && ["ILLO","ILLA","ALLE"].includes(sub(i-1,4))) ||
              ((["AS","OS"].includes(sub(len-2,2)) || "AO".includes(get(len-1))) && sub(i-1,4) === "ALLE")) {
            add("L"," ");
          } else add("L");
          i += 2;
        } else { add("L"); i++; }
        break;

      case "M":
        if ((sub(i-1,3) === "UMB" && (i+1 === len-1 || sub(i+2,2) === "ER")) || get(i+1) === "M") i += 2;
        else i++;
        add("M");
        break;

      case "N":
        add("N"); i += get(i+1) === "N" ? 2 : 1;
        break;

      case "P":
        if (get(i+1) === "H") { add("F"); i += 2; }
        else { add("P"); i += "PB".includes(get(i+1)) ? 2 : 1; }
        break;

      case "Q":
        add("K"); i += get(i+1) === "Q" ? 2 : 1;
        break;

      case "R":
        if (i === len-1 && !slavoGermanic() && sub(i-2,2) === "IE" &&
            !["ME","MA"].includes(sub(i-4,2))) add(" ","R");
        else add("R");
        i += get(i+1) === "R" ? 2 : 1;
        break;

      case "S":
        if (["ISL","YSL"].includes(sub(i-1,3))) { i++; break; }
        if (i === 0 && sub(0,5) === "SUGAR") { add("X","S"); i++; break; }
        if (sub(i,2) === "SH") { add("X"); i += 2; break; }
        if (["SIO","SIA"].includes(sub(i,3))) { add("S", slavoGermanic() ? "S" : "X"); i += 3; break; }
        if ((i === 0 && "MNLW".includes(get(i+1))) || get(i+1) === "Z") {
          add("S","X"); i += get(i+1) === "Z" ? 2 : 1; break;
        }
        if (sub(i,2) === "SC") {
          if (get(i+2) === "H") {
            if (["OO","ER","EN","UY","ED","EM"].includes(sub(i+3,2))) {
              add(["ER","EN"].includes(sub(i+3,2)) ? "X" : "SK",
                  ["ER","EN"].includes(sub(i+3,2)) ? "SK" : "SK");
            } else {
              add(i === 0 && !VOWELS.includes(get(3)) && get(3) !== "W" ? "X" : "X", "S");
            }
            i += 3;
          } else if ("IEY".includes(get(i+2))) { add("S"); i += 3; }
          else { add("SK"); i += 3; }
          break;
        }
        if (i === len-1 && ["AI","OI"].includes(sub(i-2,2))) add(" ","S");
        else add("S");
        i += "SZ".includes(get(i+1)) ? 2 : 1;
        break;

      case "T":
        if (sub(i,4) === "TION") { add("X"); i += 3; break; }
        if (["TIA","TCH"].includes(sub(i,3))) { add("X"); i += 3; break; }
        if (sub(i,2) === "TH" || sub(i,3) === "TTH") {
          add(["OM","AM"].includes(sub(i+2,2)) || ["VAN ","VON "].includes(sub(0,4)) ||
              sub(0,3) === "SCH" ? "T" : "0", "T");
          i += 2; break;
        }
        add("T"); i += "TD".includes(get(i+1)) ? 2 : 1;
        break;

      case "V":
        add("F"); i += get(i+1) === "V" ? 2 : 1;
        break;

      case "W":
        if (sub(i,2) === "WR") { add("R"); i += 2; break; }
        if (i === 0 && (VOWELS.includes(get(i+1)) || sub(i,2) === "WH")) {
          add(VOWELS.includes(get(i+1)) ? "A" : "A", VOWELS.includes(get(i+1)) ? "F" : "A");
          i++; break;
        }
        if ((i === len-1 && VOWELS.includes(get(i-1))) ||
            ["EWSKI","EWSKY","OWSKI","OWSKY"].includes(sub(i-1,5)) ||
            sub(0,3) === "SCH") { add(" ","F"); i++; break; }
        if (["WICZ","WITZ"].includes(sub(i,4))) { add("TS","FX"); i += 4; break; }
        i++;
        break;

      case "X":
        if (!(i === len-1 && (["IAU","EAU"].includes(sub(i-3,3)) || ["AU","OU"].includes(sub(i-2,2)))))
          add("KS");
        i += "CX".includes(get(i+1)) ? 2 : 1;
        break;

      case "Z":
        if (get(i+1) === "H") { add("J"); i += 2; break; }
        if (["ZO","ZI","ZA"].includes(sub(i+1,2)) || (slavoGermanic() && i > 0 && get(i-1) !== "T")) {
          add("S","TS");
        } else add("S");
        i += get(i+1) === "Z" ? 2 : 1;
        break;

      default:
        i++;
    }
  }

  const codes = [p.slice(0,4), s.slice(0,4)].filter(Boolean);
  return codes[0] === codes[1] ? [codes[0]] : codes.filter((v,idx,arr) => arr.indexOf(v) === idx);
}

// ==================================================
// Regex constants
// ==================================================

const SUFFIX_RE      = /\b(Jr\.?|Sr\.?|II|III|IV)\b/gi;
const COMMA_RE       = /^([^,]+),\s*(.+)$/;
const INITIAL_RE     = /^[a-z]\.$/;
const JUNK_PREFIX_RE = /^[a-z][A-Z]/;
const CONCAT_X_RE    = /([a-z])x([A-Z])/g;

// ==================================================
// Normalization (unchanged)
// ==================================================

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
// Similarity (unchanged)
// ==================================================

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
  const [small, large] = ta.size <= tb.size ? [ta, tb] : [tb, ta];

  let inter = 0;
  for (const t of small) if (large.has(t)) inter++;

  return inter / (ta.size + tb.size - inter);
}

function seqSim(a: string, b: string) {
  if (a === b) return 1;

  const m = a.length, n = b.length;
  if (!m || !n) return 0;
  if (Math.abs(m - n) > Math.max(m, n) * 0.7) return 0;

  const prev = new Int16Array(n + 1);
  const curr = new Int16Array(n + 1);

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++)
      curr[j + 1] =
        a[i] === b[j] ? prev[j] + 1 : Math.max(prev[j + 1], curr[j]);
    prev.set(curr);
  }

  return (2 * curr[n]) / (m + n);
}

const tokenSim = (a: string, b: string) =>
  (trigramJaccard(a, b) + seqSim(a, b)) / 2;

// ==================================================
// Core Scoring (unchanged from original)
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
        const initMatch = q.initials.some(qi => r.initials.some(ri => qi === ri));
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
// CSV Parsing (unchanged)
// ==================================================

function parseCsvLine(line: string): [string, string] | null {
  const i = line.indexOf(",");
  if (i === -1) return null;

  let name = line.slice(i + 1).trim();
  if (name.startsWith('"') && name.endsWith('"'))
    name = name.slice(1, -1).replace(/""/g, '"');

  return [line.slice(0, i).trim(), name];
}

// ==================================================
// Setup — parse CSV → bulk-index into Elasticsearch
// ==================================================

export async function setup(datasetPath: string): Promise<void> {
  ES_BASE  = (process.env.ES_URL ?? "http://localhost:9200").replace(/\/$/, "");
  ES_INDEX = `names_search_${Date.now()}`;

  // ── Create index with custom analyzer + mappings ──────────────────────────
  await esRequest("PUT", `/${ES_INDEX}`, {
    settings: {
      analysis: {
        analyzer: {
          name_analyzer: {
            type: "custom",
            tokenizer: "standard",
            filter: ["lowercase", "asciifolding"],
          },
        },
      },
      index: {
        number_of_shards:   1,
        number_of_replicas: 0,
        refresh_interval:   "-1",   // disabled during bulk load
      },
    },
    mappings: {
      properties: {
        record_id:  { type: "keyword" },
        rawName:    { type: "keyword" },
        normName:   { type: "keyword" },
        initials:   { type: "keyword" },
        words:      { type: "keyword" },
        firstWord:  { type: "keyword" },
        lastWord:   { type: "keyword" },
        wordCount:  { type: "integer" },
        dm_codes:   { type: "keyword" },
        first_name: {
          type:     "text",
          analyzer: "name_analyzer",
          fields: { keyword: { type: "keyword" } },
        },
        last_name: {
          type:     "text",
          analyzer: "name_analyzer",
          fields: { keyword: { type: "keyword" } },
        },
      },
    },
  });

  // ── Parse CSV ─────────────────────────────────────────────────────────────
  const lines = (await Bun.file(datasetPath).text()).split(/\r?\n/);
  const bulkOps: object[] = [];
  let count = 0;

  for (let li = 1; li < lines.length; li++) {
    const parsed = parseCsvLine(lines[li].trim());
    if (!parsed) continue;

    const [id, rawName] = parsed;
    const normName      = normalize(rawName);
    const { initials, words } = splitNorm(normName);
    if (!words.length) continue;

    const dmSet = new Set<string>();
    for (const w of words)
      for (const code of doubleMetaphone(w)) dmSet.add(code);

    const doc: ESDoc = {
      record_id: id,
      rawName,
      normName,
      initials,
      words,
      firstWord: words[0],
      lastWord:  words[words.length - 1],
      wordCount: words.length,
      dm_codes:  [...dmSet],
      first_name: words[0],
      last_name:  words[words.length - 1],
    };

    bulkOps.push({ index: { _index: ES_INDEX, _id: id } });
    bulkOps.push(doc);
    count++;
  }

  // ── Bulk index via NDJSON ─────────────────────────────────────────────────
  // ES bulk endpoint expects newline-delimited JSON (not a JSON array).
  const BATCH_DOCS = 500;
  for (let b = 0; b < bulkOps.length; b += BATCH_DOCS * 2) {
    const slice = bulkOps.slice(b, b + BATCH_DOCS * 2);
    const ndjson = slice.map(o => JSON.stringify(o)).join("\n") + "\n";

    const res = await fetch(`${ES_BASE}/_bulk`, {
      method:  "POST",
      headers: { "Content-Type": "application/x-ndjson" },
      body:    ndjson,
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(`Bulk index failed ${res.status}: ${t}`);
    }
    const resp = await res.json() as { errors: boolean; items: any[] };
    if (resp.errors) {
      const errs = resp.items.filter(i => i.index?.error);
      if (errs.length) console.error("Bulk index errors:", errs.slice(0, 3));
    }
  }

  // Re-enable refresh and force a refresh so docs are searchable immediately
  await esRequest("PUT", `/${ES_INDEX}/_settings`, {
    index: { refresh_interval: "1s" },
  });
  await esRequest("POST", `/${ES_INDEX}/_refresh`);

  THRESHOLD = count > LARGE_DATASET_CUTOFF
    ? PARAMS.THRESHOLD_LARGE
    : PARAMS.THRESHOLD_SMALL;
}

// ==================================================
// Search — ES candidate retrieval → scoreMatch filter
// ==================================================

export async function search(query: string): Promise<string[]> {
  trigramCache.clear();

  const norm = normalize(query);
  const { initials, words } = splitNorm(norm);
  if (!words.length) return [];

  const q: NameRecord = {
    id:        "",
    rawName:   query,
    normName:  norm,
    initials,
    words,
    firstWord: words[0],
    lastWord:  words[words.length - 1],
    wordCount: words.length,
  };

  const qFirst = q.firstWord;
  const qLast  = q.lastWord;

  // Collect DM codes for the full query
  const qDmCodes: string[] = [];
  for (const w of words)
    for (const code of doubleMetaphone(w))
      if (!qDmCodes.includes(code)) qDmCodes.push(code);

  // ── Build ES bool/should query ────────────────────────────────────────────
  const shouldClauses: object[] = [
    { match: { last_name:  { query: qLast,  fuzziness: "AUTO", boost: 3   } } },
    { match: { first_name: { query: qFirst, fuzziness: "AUTO", boost: 2   } } },
    { match: { last_name:  { query: qFirst, fuzziness: "AUTO", boost: 1.5 } } },
    { match: { first_name: { query: qLast,  fuzziness: "AUTO", boost: 1   } } },
    ...(qDmCodes.length ? [{ terms: { dm_codes: qDmCodes, boost: 1 } }] : []),
  ];

  for (const init of initials) {
    shouldClauses.push({ prefix: { "first_name.keyword": { value: init, boost: 2 } } });
    shouldClauses.push({ prefix: { "last_name.keyword":  { value: init, boost: 1 } } });
  }

  // ── Execute ES query via fetch ────────────────────────────────────────────
  const resp = await esRequest("POST", `/${ES_INDEX}/_search`, {
    query: {
      bool: {
        should:               shouldClauses,
        minimum_should_match: 1,
      },
    },
    size:    ES_CANDIDATE_SIZE,
    _source: true,
  });

  // ── Re-rank candidates through scoreMatch ─────────────────────────────────
  const results: string[] = [];

  for (const hit of resp.hits.hits as Array<{ _source: ESDoc }>) {
    const doc = hit._source;

    const rec: NameRecord = {
      id:        doc.record_id,
      rawName:   doc.rawName,
      normName:  doc.normName,
      initials:  doc.initials,
      words:     doc.words,
      firstWord: doc.firstWord,
      lastWord:  doc.lastWord,
      wordCount: doc.wordCount,
    };

    if (scoreMatch(q, rec) >= THRESHOLD) {
      results.push(rec.id);
    }
  }

  return results;
}

// ==================================================
// Cleanup — delete index (no client to close)
// ==================================================

export async function cleanup(): Promise<void> {
  try {
    await esRequest("DELETE", `/${ES_INDEX}`);
  } catch {
    // ignore if already gone
  }
}

export default search;