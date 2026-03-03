// ==================================================
// String Similarity
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