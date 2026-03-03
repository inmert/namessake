import { PARAMS, INITIAL_MATCH, MISSING_FIRST, INITIAL_SOFT, INITIAL_HARD } from "./params";
import { tokenSim } from "./similarity";
import type { NameRecord } from "./types";

// ==================================================
// Match Scoring
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

    // Query initial matches first token of record given-name
    for (const init of q.initials) {
      if (rfw[0]?.startsWith(init)) { firstScore = INITIAL_MATCH; initialMatched = true; break; }
    }

    // Record initial matches first token of query given-name
    if (!initialMatched) {
      for (const init of r.initials) {
        if (qfw[0]?.startsWith(init)) { firstScore = INITIAL_MATCH; initialMatched = true; break; }
      }
    }

    // Full token comparison
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
      // Both sides are initials-only
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