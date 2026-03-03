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

  // "Last, First" → "First Last"
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