#!/usr/bin/env python3
"""
Parameter optimizer for submission/search.ts
Ports the full TypeScript matching logic to Python, then grid-searches
over the tunable parameters to maximize the scoring formula.

Usage (from your project root):
    python3 optimizer.py                          # uses default paths
    python3 optimizer.py --dataset large          # small | large
    python3 optimizer.py --dataset large --top 20
    python3 optimizer.py --dataset large --analysis-only   # just show stuck cases

Paths it expects (relative to CWD, matching the project layout):
    data/datasets/<dataset>/names.csv
    tests/public_<dataset>.json
"""

import csv
import json
import re
import math
import time
import argparse
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",        default="large",  choices=["small", "large"])
parser.add_argument("--top",            default=15,       type=int, help="Top N results to print")
parser.add_argument("--analysis-only",  action="store_true", help="Print case analysis and exit")
parser.add_argument("--csv",            default=None,     help="Override path to names.csv")
parser.add_argument("--suite",          default=None,     help="Override path to test JSON")
ARGS = parser.parse_args()

DATASET      = ARGS.dataset
CSV_PATH     = Path(ARGS.csv)   if ARGS.csv   else Path(f"data/datasets/{DATASET}/names.csv")
SUITE_PATH   = Path(ARGS.suite) if ARGS.suite else Path(f"tests/public_{DATASET}.json")
LARGE_CUTOFF = 5000

# ============================================================
# Matching logic (faithful port of the TypeScript)
# ============================================================

INITIAL_RE    = re.compile(r"^[a-z]\.$")
JUNK_PFX_RE   = re.compile(r"^[a-z][A-Z]")
SUFFIX_RE     = re.compile(r"\b(Jr\.?|Sr\.?|II|III|IV)\b", re.IGNORECASE)
COMMA_RE      = re.compile(r"^([^,]+),\s*(.+)$")
CONCAT_X_RE   = re.compile(r"([a-z])x([A-Z])")

_trigram_cache: dict[str, frozenset[str]] = {}

def trigrams(s: str) -> frozenset[str]:
    if s in _trigram_cache:
        return _trigram_cache[s]
    padded = f"##{s}##"
    t = frozenset(padded[i:i+3] for i in range(len(padded) - 2))
    _trigram_cache[s] = t
    return t

def trigram_jaccard(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    inter = len(ta & tb)
    return inter / (len(ta) + len(tb) - inter)

def seq_sim(a: str, b: str) -> float:
    if a == b: return 1.0
    m, n = len(a), len(b)
    if not m or not n: return 0.0
    if abs(m - n) > max(m, n) * 0.7: return 0.0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(m):
        for j in range(n):
            curr[j+1] = prev[j]+1 if a[i]==b[j] else max(prev[j+1], curr[j])
        prev = curr[:]
    return (2 * curr[n]) / (m + n)

def token_sim(a: str, b: str) -> float:
    return (trigram_jaccard(a, b) + seq_sim(a, b)) / 2

def normalize(name: str) -> str:
    name = name.strip('"')
    m = COMMA_RE.match(name)
    if m:
        name = f"{m.group(2).strip()} {m.group(1).strip()}"
    name = name.replace("0", "o").replace("1", "l")
    name = SUFFIX_RE.sub("", name)
    name = CONCAT_X_RE.sub(r"\1 \2", name)
    tokens = name.split()
    fixed = []
    for t in tokens:
        if len(t) >= 2 and t[0].islower() and t[1].isupper():
            t = t[1:]
        fixed.append(t)
    return " ".join(fixed).lower().strip()

def split_norm(norm: str):
    initials, words = [], []
    for t in norm.split():
        if not t: continue
        if INITIAL_RE.match(t):
            initials.append(t[0])
        elif len(t) > 1:
            words.append(t)
    return initials, words


@dataclass
class NameRecord:
    id: str
    raw_name: str
    norm_name: str
    initials: list[str]
    words: list[str]
    first_word: str
    last_word: str


def score_match(q: NameRecord, r: NameRecord, params: dict) -> float:
    SURNAME_FLOOR  = params["SURNAME_FLOOR"]
    BOTH_FIRST_MIN = params["BOTH_FIRST_MIN"]
    MIN_FIRST      = params["MIN_FIRST_SCORE"]
    INITIAL_MATCH  = params["INITIAL_MATCH"]
    MISSING_FIRST  = params["MISSING_FIRST"]
    INITIAL_SOFT   = params["INITIAL_SOFT"]

    if not q.words or not r.words:
        return 0.0

    pairings = [
        (q.last_word, r.last_word, False),
        (q.first_word, r.first_word, True),
        (q.last_word, r.first_word, False),
        (q.first_word, r.last_word, False),
    ]

    best = 0.0
    seen: set[str] = set()

    for q_ln, r_ln, both_first in pairings:
        key = f"{q_ln}|{r_ln}"
        if key in seen:
            continue
        seen.add(key)

        last_sim = token_sim(q_ln, r_ln)
        if last_sim < SURNAME_FLOOR:
            continue

        qfw = [w for w in q.words if w != q_ln]
        rfw = [w for w in r.words if w != r_ln]

        first_score = 0.0
        initial_matched = False

        for init in q.initials:
            if rfw and rfw[0].startswith(init):
                first_score = INITIAL_MATCH
                initial_matched = True
                break

        if not initial_matched:
            for init in r.initials:
                if qfw and qfw[0].startswith(init):
                    first_score = INITIAL_MATCH
                    initial_matched = True
                    break

        if not initial_matched and qfw and rfw:
            compared = 0
            for a in qfw:
                for b in rfw:
                    if compared > 3: break
                    first_score = max(first_score, token_sim(a, b))
                    compared += 1
                    if first_score > 0.8: break
                if first_score > 0.8: break
        elif not qfw and not rfw:
            # Both anchor-only: compare initials directly if both have them
            if q.initials and r.initials:
                init_match = any(qi == ri for qi in q.initials for ri in r.initials)
                first_score = INITIAL_MATCH if init_match else 0.0
            else:
                first_score = MISSING_FIRST
        else:
            if initial_matched:
                first_score = INITIAL_MATCH
            elif q.initials and rfw:
                first_score = INITIAL_SOFT
            elif r.initials and qfw:
                first_score = 0.0
            else:
                first_score = MISSING_FIRST

        if both_first and first_score < BOTH_FIRST_MIN:
            continue

        if first_score < MIN_FIRST:
            continue

        combined = 0.5 * last_sim + 0.5 * first_score
        if combined > best:
            best = combined

    return best


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(csv_path: Path) -> tuple[list[NameRecord], dict[str, NameRecord]]:
    records: list[NameRecord] = []
    by_id: dict[str, NameRecord] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2: continue
            rid, raw = row[0].strip(), row[1].strip()
            norm = normalize(raw)
            inits, words = split_norm(norm)
            if not words: continue
            rec = NameRecord(
                id=rid, raw_name=raw, norm_name=norm,
                initials=inits, words=words,
                first_word=words[0], last_word=words[-1],
            )
            records.append(rec)
            by_id[rid] = rec

    return records, by_id


# ============================================================
# Scorer
# ============================================================

def evaluate(records: list[NameRecord], cases: list[dict], params: dict) -> dict:
    threshold = params["THRESHOLD"]
    INITIAL_MATCH = params["INITIAL_MATCH"]

    total_expected = 0
    total_found    = 0
    total_listed_fp = 0
    total_extras   = 0

    case_results = []

    for case in cases:
        query          = case["query"]
        expected_ids   = set(case["expectedIds"])
        false_pos_ids  = set(case["falsePositiveIds"])

        norm = normalize(query)
        inits, words = split_norm(norm)
        if not words:
            case_results.append({"id": case["id"], "found": 0, "expected": len(expected_ids),
                                  "listed_fp": 0, "extras": 0})
            total_expected += len(expected_ids)
            continue

        q = NameRecord(id="", raw_name=query, norm_name=norm,
                       initials=inits, words=words,
                       first_word=words[0], last_word=words[-1])

        returned_ids = []
        for rec in records:
            if score_match(q, rec, params) >= threshold:
                returned_ids.append(rec.id)

        returned_set = set(returned_ids)
        hits         = len(returned_set & expected_ids)
        listed_fp    = len(returned_set & false_pos_ids)
        extras       = len(returned_set - expected_ids - false_pos_ids)

        total_expected  += len(expected_ids)
        total_found     += hits
        total_listed_fp += listed_fp
        total_extras    += extras

        case_results.append({
            "id": case["id"], "query": query,
            "found": hits, "expected": len(expected_ids),
            "returned": sorted(returned_ids),
            "listed_fp": listed_fp, "extras": extras,
            "missing": sorted(expected_ids - returned_set),
            "extra_ids": sorted(returned_set - expected_ids - false_pos_ids),
            "fp_ids": sorted(returned_set & false_pos_ids),
        })

    recall_pct   = (total_found / total_expected * 100) if total_expected else 0
    penalty      = total_listed_fp * 0.02 + total_extras * 0.05
    raw_score    = recall_pct - penalty
    final_score  = max(0.0, min(100.0, raw_score))

    return {
        "score": final_score,
        "recall": recall_pct,
        "listed_fp": total_listed_fp,
        "extras": total_extras,
        "cases": case_results,
    }


# ============================================================
# Analysis: per-case "can params fix this?" summary
# ============================================================

def print_analysis(records: list[NameRecord], cases: list[dict], current_params: dict):
    print("\n" + "="*80)
    print("CASE-BY-CASE ANALYSIS")
    print("="*80)

    for case in cases:
        query        = case["query"]
        expected_ids = set(case["expectedIds"])
        fp_ids       = set(case["falsePositiveIds"])

        norm = normalize(query)
        inits, words = split_norm(norm)
        if not words: continue

        q = NameRecord(id="", raw_name=query, norm_name=norm,
                       initials=inits, words=words,
                       first_word=words[0], last_word=words[-1])

        # Score every record for this query
        scored = []
        for rec in records:
            s = score_match(q, rec, current_params)
            if s > 0.4:  # only care about near-matches
                scored.append((s, rec))

        scored.sort(reverse=True, key=lambda x: x[0])
        threshold = current_params["THRESHOLD"]

        # Categorize
        true_hits    = [(s, r) for s, r in scored if r.id in expected_ids and s >= threshold]
        missing      = [(s, r) for s, r in scored if r.id in expected_ids and s < threshold]
        # Also check expected IDs with score below 0.4
        scored_ids   = {r.id for _, r in scored}
        missing_low  = [eid for eid in expected_ids if eid not in scored_ids]

        returned_set = {r.id for s, r in scored if s >= threshold}
        extras       = [(s, r) for s, r in scored if r.id not in expected_ids
                        and r.id not in fp_ids and s >= threshold]
        listed_fp    = [(s, r) for s, r in scored if r.id in fp_ids and s >= threshold]

        has_issues = missing or missing_low or extras or listed_fp
        status = "✗ FAIL" if has_issues else "✓ PASS"
        print(f"\n{status}  [{case['id']}]  \"{query}\"")

        if missing:
            print(f"  MISSING (score below threshold {threshold}):")
            for s, r in missing:
                print(f"    [{r.id}] {r.raw_name!r}  score={s:.4f}  (need +{threshold-s:.4f})")

        if missing_low:
            print(f"  MISSING (not in top candidates, score likely <0.4):")
            for eid in missing_low[:5]:
                print(f"    [{eid}] (not scored)")

        if extras:
            print(f"  UNWANTED EXTRAS (unscored, score above threshold):")
            for s, r in extras:
                # Explain why it matched — which surname pair drove it
                best_surname_sim = max(
                    token_sim(q.last_word, r.last_word),
                    token_sim(q.first_word, r.last_word),
                    token_sim(q.last_word, r.first_word),
                    token_sim(q.first_word, r.first_word),
                )
                print(f"    [{r.id}] {r.raw_name!r}  score={s:.4f}  "
                      f"best_surn_sim={best_surname_sim:.4f}")

        if listed_fp:
            print(f"  LISTED FALSE POSITIVES (penalized):")
            for s, r in listed_fp:
                print(f"    [{r.id}] {r.raw_name!r}  score={s:.4f}")

    print()


# ============================================================
# Grid search
# ============================================================

# Build the search grid — adjust ranges here to explore more aggressively
GRID = {
    "THRESHOLD":      [0.640, 0.645, 0.650, 0.655, 0.660, 0.665, 0.670, 0.675, 0.680],
    "SURNAME_FLOOR":  [0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60],
    "BOTH_FIRST_MIN": [0.65, 0.70, 0.75],
    "MIN_FIRST_SCORE":[0.42, 0.45, 0.47, 0.50],
    "INITIAL_MATCH":  [0.80],         # stable, no reason to vary
    "MISSING_FIRST":  [0.55, 0.60],
    "INITIAL_SOFT":   [0.28, 0.30],
}

def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"

def _render_bar(frac: float, width: int = 30) -> str:
    filled = int(frac * width)
    partial = "▌" if (frac * width - filled) >= 0.5 else ""
    bar = "█" * filled + partial
    return f"[{bar:<{width}}]"

def run_grid_search(records: list[NameRecord], cases: list[dict]):
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"\nGrid search: {total:,} combinations across {len(keys)} parameters")
    print("  " + "  ".join(f"{k}({len(GRID[k])})" for k in keys))
    print()

    results      = []
    best_score   = 0.0
    t0           = time.time()
    update_every = max(1, total // 200)   # redraw ~200 times

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        result = evaluate(records, cases, params)
        entry  = (result["score"], result["recall"], -result["extras"],
                  -result["listed_fp"], params, result)
        results.append(entry)

        if result["score"] > best_score:
            best_score = result["score"]

        if (i + 1) % update_every == 0 or i == total - 1:
            done    = i + 1
            elapsed = time.time() - t0
            speed   = done / elapsed
            eta     = (total - done) / speed if speed else 0
            frac    = done / total
            w       = len(str(total))

            line = (f"\r  {_render_bar(frac)} {frac*100:5.1f}%"
                    f"  {done:>{w}}/{total}"
                    f"  best={best_score:.4f}"
                    f"  {speed:,.0f} c/s"
                    f"  elapsed={_fmt_duration(elapsed)}"
                    f"  eta={_fmt_duration(eta)}   ")
            print(line, end="", flush=True)

    print(f"\n  Done — {total:,} combinations in {_fmt_duration(time.time() - t0)}\n")

    results.sort(reverse=True)
    return results


# ============================================================
# Main
# ============================================================

def main():
    print(f"\nLoading dataset: {CSV_PATH}")
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run from your project root.")
        return
    if not SUITE_PATH.exists():
        print(f"ERROR: {SUITE_PATH} not found.")
        return

    records, by_id = load_dataset(CSV_PATH)
    print(f"  {len(records):,} records loaded")

    with open(SUITE_PATH) as f:
        suite = json.load(f)
    cases = suite["cases"]
    print(f"  {len(cases)} test cases loaded from {SUITE_PATH.name}")

    # Current baseline params (matching search.ts after last fix)
    current_params = {
        "THRESHOLD":       0.655,
        "SURNAME_FLOOR":   0.50,
        "BOTH_FIRST_MIN":  0.70,
        "MIN_FIRST_SCORE": 0.45,
        "INITIAL_MATCH":   0.80,
        "MISSING_FIRST":   0.60,
        "INITIAL_SOFT":    0.30,
    }

    baseline = evaluate(records, cases, current_params)
    print(f"\nBaseline score: {baseline['score']:.4f}  "
          f"recall={baseline['recall']:.2f}%  "
          f"listed_fp={baseline['listed_fp']}  extras={baseline['extras']}")

    if ARGS.analysis_only:
        print_analysis(records, cases, current_params)
        return

    # Analysis first so you understand what you're optimizing against
    print_analysis(records, cases, current_params)

    # Grid search
    top_results = run_grid_search(records, cases)

    print(f"\n{'='*80}")
    print(f"TOP {ARGS.top} PARAMETER SETS")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Rank':>4}  {'Score':>7}  {'Recall':>7}  {'FP':>4}  {'Ext':>4}  "
          f"{'THR':>6}  {'SURN':>6}  {'BFM':>5}  {'MFS':>5}  {'MF':>5}")
    print("-"*80)

    seen_scores: set[float] = set()
    printed = 0

    for rank, (score, recall, neg_extras, neg_fp, params, result) in enumerate(top_results):
        if printed >= ARGS.top: break

        print(f"  {printed+1:>3}  {score:>7.4f}  {recall:>6.2f}%  "
              f"{result['listed_fp']:>4}  {result['extras']:>4}  "
              f"{params['THRESHOLD']:>6.3f}  {params['SURNAME_FLOOR']:>6.3f}  "
              f"{params['BOTH_FIRST_MIN']:>5.2f}  {params['MIN_FIRST_SCORE']:>5.2f}  "
              f"{params['MISSING_FIRST']:>5.2f}")
        printed += 1

    # Print detailed breakdown for the best result
    best_params = top_results[0][4]
    best_result = top_results[0][5]

    print(f"\n{'='*80}")
    print("BEST PARAMS DETAIL")
    print(f"{'='*80}")
    print(f"  Score:    {top_results[0][0]:.4f}")
    print(f"  Recall:   {best_result['recall']:.2f}%")
    print(f"  Listed FP:{best_result['listed_fp']}")
    print(f"  Extras:   {best_result['extras']}")
    print()
    for k, v in best_params.items():
        baseline_v = current_params[k]
        change = " ← changed" if abs(v - baseline_v) > 0.001 else ""
        print(f"  {k:<20} {v}{change}")

    # Show which cases pass/fail with best params
    print(f"\nPer-case with best params:")
    for c in best_result["cases"]:
        passed = c["found"] == c["expected"] and c["listed_fp"] == 0 and c["extras"] == 0
        status = "✓" if passed else "✗"
        issues = []
        if c["found"] < c["expected"]:
            issues.append(f"miss={c['expected']-c['found']}")
        if c["listed_fp"] > 0:
            issues.append(f"FP={c['listed_fp']}")
        if c["extras"] > 0:
            issues.append(f"ext={c['extras']}")
        print(f"  {status} [{c['id']}]  {c.get('query',''):<30}  {', '.join(issues) or 'ok'}")

    # Show diff vs baseline
    print(f"\nBaseline → Best:")
    print(f"  Score:  {baseline['score']:.4f} → {top_results[0][0]:.4f}  "
          f"({top_results[0][0] - baseline['score']:+.4f})")
    print(f"  Extras: {baseline['extras']} → {best_result['extras']}")
    print(f"  FP:     {baseline['listed_fp']} → {best_result['listed_fp']}")

    print(f"\nTo apply best params, update PARAMS in search.ts:")
    print(f"  SURNAME_FLOOR:  {best_params['SURNAME_FLOOR']}")
    print(f"  BOTH_FIRST_MIN: {best_params['BOTH_FIRST_MIN']}")
    print(f"  THRESHOLD_LARGE:{best_params['THRESHOLD']}")
    print(f"  MIN_FIRST_SCORE:{best_params['MIN_FIRST_SCORE']}")

    print()


if __name__ == "__main__":
    main()