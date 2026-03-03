// submission/search.ts

// ==================================================
// Tunable Parameters (Simple + Stable)
// ==================================================

export const PARAMS = {
  SURNAME_FLOOR: 0.5,        // was 0.38
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

let records: NameRecord[] = [];
const trigramIndex = new Map<string, Set<number>>();
const dmIndex     = new Map<string, Set<number>>();   // DM code → record indices
let trigramCache = new Map<string, Set<string>>();

// ==================================================
// Double Metaphone  (Lawrence Philips / faithful port)
// Returns up to two 4-char phonetic codes.
// ==================================================

function doubleMetaphone(word: string): string[] {
  const VOWELS = "AEIOUY";
  word = word.toUpperCase();

  // Strip silent leading pairs
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

  // Initial vowel → "A"
  if (VOWELS.includes(word[0])) { add("A"); i = 1; }

  while (i < len) {
    const c = get(i);

    // Skip standalone vowels after position 0
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

  // Return deduplicated, non-empty codes (max 4 chars each)
  const codes = [p.slice(0,4), s.slice(0,4)].filter(Boolean);
  return codes[0] === codes[1] ? [codes[0]] : codes.filter((v,idx,arr) => arr.indexOf(v) === idx);
}

let THRESHOLD = PARAMS.THRESHOLD_SMALL;
const LARGE_DATASET_CUTOFF = 5000;

const INITIAL_MATCH = 0.8;
const MISSING_FIRST = 0.6;
const INITIAL_SOFT = 0.30;
const INITIAL_HARD = 0.0;

const SUFFIX_RE = /\b(Jr\.?|Sr\.?|II|III|IV)\b/gi;
const COMMA_RE = /^([^,]+),\s*(.+)$/;
const INITIAL_RE = /^[a-z]\.$/;
const JUNK_PREFIX_RE = /^[a-z][A-Z]/;
const CONCAT_X_RE = /([a-z])x([A-Z])/g;

// ==================================================
// Normalization
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
// Similarity
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
// Core Scoring (Simplified + Stable)
// ==================================================

function scoreMatch(q: NameRecord, r: NameRecord): number {
  if (!q.words.length || !r.words.length) return 0;

  const pairings: [string, string, boolean][] = [
    [q.lastWord, r.lastWord, false],
    [q.firstWord, r.firstWord, true],
    [q.lastWord, r.firstWord, false],
    [q.firstWord, r.lastWord, false],
  ];

  let best = 0;
  const seen = new Set<string>();

  for (const [qLN, rLN, bothFirst] of pairings) {
    const key = qLN + "|" + rLN;
    if (seen.has(key)) continue;
    seen.add(key);

    const lastSim = tokenSim(qLN, rLN);

    // 🔒 Stronger surname gate
    if (lastSim < PARAMS.SURNAME_FLOOR) continue;

    const qfw = q.words.filter(w => w !== qLN);
    const rfw = r.words.filter(w => w !== rLN);

    let firstScore = 0;
    let initialMatched = false;

    for (const init of q.initials)
      if (rfw[0]?.startsWith(init)) {
        firstScore = INITIAL_MATCH;
        initialMatched = true;
        break;
      }

    if (!initialMatched)
      for (const init of r.initials)
        if (qfw[0]?.startsWith(init)) {
          firstScore = INITIAL_MATCH;
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
      // Both sides have no first-name words beyond the anchor.
      // When BOTH sides carry initials, compare them directly so that
      // "A. Moen" does not match "D. Moen" (different people, different initials).
      // If only one side has initials (or neither does), fall back to MISSING_FIRST.
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

    if (bothFirst && firstScore < PARAMS.BOTH_FIRST_MIN)
      continue;

    // Require a minimum first-name score so that an exact or near-exact surname
    // cannot drag an otherwise weak first-name match over the threshold.
    // e.g. "Tina Botsford" must not match query "Rosina Botsford" (sim≈0.44),
    //      "Marion Lowe" must not match query "Marlne Lowe" (sim≈0.45),
    //      "Sandrine Lowe" must not match (sim≈0.35).
    // Genuine typo pairs (felix/fehx=0.48, marlne/marlene=0.73, etc.) all clear 0.45.
    if (firstScore < 0.45) continue;

    // 50/50 weighting restored
    const combined = 0.5 * lastSim + 0.5 * firstScore;

    if (combined > best) best = combined;
    if (best > THRESHOLD + 0.15) break;
  }

  return best;
}

// ==================================================
// CSV Parsing
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
// Setup / Search / Cleanup
// ==================================================

export async function setup(datasetPath: string) {
  records = [];
  trigramIndex.clear();

  const lines = (await Bun.file(datasetPath).text()).split(/\r?\n/);
  records = new Array(lines.length - 1);

  let count = 0;

  for (let i = 1; i < lines.length; i++) {
    const parsed = parseCsvLine(lines[i].trim());
    if (!parsed) continue;

    const [id, rawName] = parsed;
    const normName = normalize(rawName);
    const { initials, words } = splitNorm(normName);
    if (!words.length) continue;

    const rec: NameRecord = {
      id,
      rawName,
      normName,
      initials,
      words,
      firstWord: words[0],
      lastWord: words[words.length - 1],
      wordCount: words.length,
    };

    records[count] = rec;

    for (const w of words) {
      // Trigram index
      for (const tg of trigrams(w)) {
        let set = trigramIndex.get(tg);
        if (!set) trigramIndex.set(tg, (set = new Set()));
        set.add(count);
      }
      // Double Metaphone index
      for (const code of doubleMetaphone(w)) {
        let set = dmIndex.get(code);
        if (!set) dmIndex.set(code, (set = new Set()));
        set.add(count);
      }
    }

    count++;
  }

  records.length = count;

  THRESHOLD = count > LARGE_DATASET_CUTOFF
    ? PARAMS.THRESHOLD_LARGE
    : PARAMS.THRESHOLD_SMALL;
}

export async function search(query: string) {
  if (!records.length) return [];

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

  const candidateScores = new Map<number, number>();
  const queryTrigrams = new Set<string>();

  for (const w of words)
    for (const tg of trigrams(w))
      queryTrigrams.add(tg);

  for (const tg of queryTrigrams) {
    const indices = trigramIndex.get(tg);
    if (!indices) continue;
    for (const idx of indices)
      candidateScores.set(idx, (candidateScores.get(idx) || 0) + 1);
  }

  // DM pass: add phonetic candidates not already found by trigrams.
  // These are scored by the same threshold — DM only widens the candidate pool.
  const dmCandidates = new Set<number>();
  for (const w of words) {
    for (const code of doubleMetaphone(w)) {
      const indices = dmIndex.get(code);
      if (!indices) continue;
      for (const idx of indices) {
        if (!candidateScores.has(idx)) dmCandidates.add(idx);
      }
    }
  }

  const minMatches = Math.max(2, Math.floor(words.length * 1.2));
  const results: string[] = [];

  for (const [idx, score] of candidateScores) {
    if (score < minMatches) continue;
    if (scoreMatch(q, records[idx]) >= THRESHOLD)
      results.push(records[idx].id);
  }

  // Score DM-only candidates (no minMatches gate — DM is already selective)
  for (const idx of dmCandidates) {
    if (scoreMatch(q, records[idx]) >= THRESHOLD)
      results.push(records[idx].id);
  }

  return results;
}

export async function cleanup() {
  records = [];
  trigramIndex.clear();
  dmIndex.clear();
  trigramCache.clear();
}

export default search;