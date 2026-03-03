// =============================================================================
// Improved Rust Name Search Backend
// =============================================================================

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::RwLock;

// =============================================================================
// Constants
// =============================================================================

const THRESHOLD_SMALL: f64 = 0.639;
const THRESHOLD_LARGE: f64 = 0.667;
const LARGE_DATASET_CUTOFF: usize = 5_000;

const SCORE_INITIAL_MATCH: f64 = 0.8;
const SCORE_MISSING_FIRST: f64 = 0.523;
const SCORE_INITIAL_SOFT: f64 = 0.3;
const SCORE_INITIAL_HARD: f64 = 0.0;
const ANCHOR_SIM_MIN: f64 = 0.417;
const SCORE_BOTH_FIRST_MIN: f64 = 0.685;
const MIN_TRIGRAM_HITS: usize = 2;

/// Minimum trigram Jaccard before we allow LCS to contribute to token_sim.
///
/// Prevents LCS from rescuing names that share only scattered common characters
/// as a subsequence but whose trigram profiles are nearly disjoint.
const TOKEN_SIM_TRIGRAM_FLOOR: f64 = 0.193;

// =============================================================================
// Core Types
// =============================================================================

#[derive(Clone)]
struct NameRecord {
    id: String,
    initials: Vec<char>,
    words: Vec<String>,
}

struct SearchState {
    records: Vec<NameRecord>,
    trigram_index: HashMap<u32, Vec<u32>>,
    threshold: f64,
}

impl SearchState {
    fn empty() -> Self {
        Self {
            records: Vec::new(),
            trigram_index: HashMap::new(),
            threshold: THRESHOLD_SMALL,
        }
    }
}

type AppState = Arc<RwLock<SearchState>>;

// =============================================================================
// HTTP types
// =============================================================================

#[derive(Deserialize)]
struct SetupRequest {
    dataset_path: String,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
}

#[derive(Serialize)]
struct SearchResponse {
    ids: Vec<String>,
}

// =============================================================================
// Normalisation
// =============================================================================

fn normalize(raw: &str) -> String {
    raw.trim_matches('"')
        .replace(',', " ")
        .replace('0', "o")
        .replace('1', "l")
        .to_lowercase()
        .split_whitespace()
        .filter(|w| !matches!(*w, "jr" | "jr." | "sr" | "sr." | "ii" | "iii" | "iv"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn split_tokens(norm: &str) -> (Vec<char>, Vec<String>) {
    let mut initials = Vec::new();
    let mut words = Vec::new();

    for t in norm.split_whitespace() {
        let mut chars = t.chars();
        match (chars.next(), chars.next(), chars.next()) {
            (Some(c), Some('.'), None) => initials.push(c),
            _ if t.len() > 1 => words.push(t.to_string()),
            _ => {}
        }
    }

    (initials, words)
}

// =============================================================================
// Trigram utilities — packed u32, no heap allocation per trigram
// =============================================================================

#[inline]
fn pack_trigram(a: u8, b: u8, c: u8) -> u32 {
    ((a as u32) << 16) | ((b as u32) << 8) | (c as u32)
}

fn get_trigrams(s: &str) -> Vec<u32> {
    let mut padded = Vec::with_capacity(s.len() + 4);
    padded.extend_from_slice(b"##");
    padded.extend_from_slice(s.as_bytes());
    padded.extend_from_slice(b"##");

    if padded.len() < 3 {
        return Vec::new();
    }

    let mut v: Vec<u32> = padded.windows(3).map(|w| pack_trigram(w[0], w[1], w[2])).collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[inline]
fn trigram_jaccard(a: &str, b: &str) -> f64 {
    let ta = get_trigrams(a);
    let tb = get_trigrams(b);
    let mut i = 0;
    let mut j = 0;
    let mut inter: usize = 0;

    while i < ta.len() && j < tb.len() {
        match ta[i].cmp(&tb[j]) {
            std::cmp::Ordering::Equal   => { inter += 1; i += 1; j += 1; }
            std::cmp::Ordering::Less    => { i += 1; }
            std::cmp::Ordering::Greater => { j += 1; }
        }
    }

    let union = ta.len() + tb.len() - inter;
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

// =============================================================================
// LCS similarity — two-row DP with swap, no clone
// =============================================================================

#[inline]
fn lcs_sim(a: &str, b: &str) -> f64 {
    if a == b { return 1.0; }
    let a_ch: Vec<char> = a.chars().collect();
    let b_ch: Vec<char> = b.chars().collect();
    let (m, n) = (a_ch.len(), b_ch.len());
    if m == 0 || n == 0 { return 0.0; }

    let mut prev = vec![0i32; n + 1];
    let mut curr = vec![0i32; n + 1];

    for i in 0..m {
        for x in curr.iter_mut() { *x = 0; }
        for j in 0..n {
            curr[j + 1] = if a_ch[i] == b_ch[j] {
                prev[j] + 1
            } else {
                prev[j + 1].max(curr[j])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    (2 * prev[n]) as f64 / (m + n) as f64
}

/// Combined token similarity.
///
/// When trigram Jaccard is below TOKEN_SIM_TRIGRAM_FLOOR we return it directly,
/// skipping LCS.  This stops LCS from inflating the score for names that are
/// genuinely different but happen to share a few characters as a subsequence.
#[inline]
fn token_sim(a: &str, b: &str) -> f64 {
    let tj = trigram_jaccard(a, b);
    if tj < TOKEN_SIM_TRIGRAM_FLOOR { return tj; }
    (tj + lcs_sim(a, b)) / 2.0
}

// =============================================================================
// Core scoring
// =============================================================================

fn score_match(q: &NameRecord, r: &NameRecord) -> f64 {
    if q.words.is_empty() || r.words.is_empty() { return 0.0; }

    let q_first = q.words[0].as_str();
    let q_last  = q.words[q.words.len() - 1].as_str();
    let r_first = r.words[0].as_str();
    let r_last  = r.words[r.words.len() - 1].as_str();

    let pairings = [
        (q_last,  r_last,  false),
        (q_last,  r_first, false),
        (q_first, r_last,  false),
        (q_first, r_first, true),
    ];

    let mut best: f64 = 0.0;

    for &(q_anchor, r_anchor, both_first) in &pairings {
        let anchor_sim = token_sim(q_anchor, r_anchor);
        if anchor_sim < ANCHOR_SIM_MIN { continue; }

        let q_rest: Vec<&str> = q.words.iter().map(|w| w.as_str()).filter(|&w| w != q_anchor).collect();
        let r_rest: Vec<&str> = r.words.iter().map(|w| w.as_str()).filter(|&w| w != r_anchor).collect();

        let first_score: f64 = match (q_rest.is_empty(), r_rest.is_empty()) {

            // ── Both sides have leftover words ────────────────────────────────
            (false, false) => {
                let mut s = 0.0f64;
                for &a in &q_rest { for &b in &r_rest { s = s.max(token_sim(a, b)); } }
                s
            }

            // ── Query has initials, record has leftover words ─────────────────
            (true, false) if !q.initials.is_empty() => {
                let matched = q.initials.iter().any(|&qi| r_rest.iter().any(|rw| rw.starts_with(qi)));
                if matched { SCORE_INITIAL_MATCH } else { SCORE_INITIAL_HARD }
            }

            // ── Record has initials, query has leftover words ─────────────────
            // Average over every q_rest word so unmatched tokens penalise the score.
            (false, true) if !r.initials.is_empty() => {
                let sum: f64 = q_rest.iter().map(|qw| {
                    if r.initials.iter().any(|&ri| qw.starts_with(ri)) {
                        SCORE_INITIAL_MATCH
                    } else {
                        0.0
                    }
                }).sum();
                sum / q_rest.len() as f64
            }

            // ── Both sides are single-token-with-initial (no leftover words) ──
            // Compare initial sets: any overlap = match, else reject.
            (true, true) if !q.initials.is_empty() => {
                if r.initials.is_empty() {
                    SCORE_INITIAL_SOFT
                } else {
                    let matched = q.initials.iter().any(|&qi| r.initials.contains(&qi));
                    if matched { SCORE_INITIAL_MATCH } else { SCORE_INITIAL_HARD }
                }
            }

            _ => SCORE_MISSING_FIRST,
        };

        if both_first && q.words.len() > 1 && r.words.len() > 1
            && first_score < SCORE_BOTH_FIRST_MIN
        {
            continue;
        }

        best = best.max(0.5 * anchor_sim + 0.5 * first_score);
    }

    best
}

// =============================================================================
// Indexing & candidate retrieval
// =============================================================================

fn load_and_index(path: &str) -> std::io::Result<SearchState> {
    let content = std::fs::read_to_string(path)?;
    let mut records: Vec<NameRecord> = Vec::new();
    let mut build_index: HashMap<u32, Vec<u32>> = HashMap::new();

    for line in content.lines().skip(1) {
        let Some(comma) = line.find(',') else { continue };
        let id  = line[..comma].trim().to_string();
        let raw = line[comma + 1..].trim();
        let norm = normalize(raw);
        let (initials, words) = split_tokens(&norm);
        let idx = records.len() as u32;

        for w in &words {
            for tg in get_trigrams(w) {
                build_index.entry(tg).or_default().push(idx);
            }
        }

        records.push(NameRecord { id, initials, words });
    }

    for bucket in build_index.values_mut() {
        bucket.sort_unstable();
        bucket.dedup();
    }

    let threshold = if records.len() > LARGE_DATASET_CUTOFF {
        THRESHOLD_LARGE
    } else {
        THRESHOLD_SMALL
    };

    Ok(SearchState { records, trigram_index: build_index, threshold })
}

/// Candidate retrieval using a global trigram hit count.
///
/// All query words contribute to a shared counter per record.
/// A record passes if it accumulates at least `min_hits` trigram hits total.
/// For single-word queries, 1 hit suffices; for multi-word queries, 2 hits
/// are required to avoid scanning the entire dataset for common tokens.
///
/// The bitset-per-word approach was tested but reverted: it did not reduce
/// unexpected extras (the extras come from borderline scores in score_match,
/// not from the candidate filter) and caused recall regression by dropping
/// valid candidates whose fuzzy tokens share few trigrams with the query word.
/// The score_match function is the right place to handle precision; the
/// index filter's only job is to cheaply eliminate obviously irrelevant records.
fn get_candidates(words: &[String], state: &SearchState) -> Vec<usize> {
    let min_hits = if words.len() == 1 { 1 } else { MIN_TRIGRAM_HITS };
    let mut counts: HashMap<u32, usize> = HashMap::new();

    for w in words {
        for tg in get_trigrams(w) {
            if let Some(bucket) = state.trigram_index.get(&tg) {
                for &idx in bucket {
                    *counts.entry(idx).or_insert(0) += 1;
                }
            }
        }
    }

    counts.into_iter()
        .filter(|(_, c)| *c >= min_hits)
        .map(|(i, _)| i as usize)
        .collect()
}

// =============================================================================
// HTTP handlers
// =============================================================================

async fn health() -> StatusCode { StatusCode::OK }

async fn setup_handler(
    State(state): State<AppState>,
    Json(req): Json<SetupRequest>,
) -> StatusCode {
    match load_and_index(&req.dataset_path) {
        Ok(new_state) => {
            *state.write().await = new_state;
            StatusCode::OK
        }
        Err(e) => {
            eprintln!("setup error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

async fn search_handler(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let state = state.read().await;
    let norm = normalize(&req.query);
    let (initials, words) = split_tokens(&norm);
    let q = NameRecord { id: String::new(), initials, words };

    let ids = get_candidates(&q.words, &state)
        .into_iter()
        .filter(|&i| score_match(&q, &state.records[i]) >= state.threshold)
        .map(|i| state.records[i].id.clone())
        .collect();

    Json(SearchResponse { ids })
}

async fn cleanup_handler(State(state): State<AppState>) -> StatusCode {
    *state.write().await = SearchState::empty();
    StatusCode::OK
}

// =============================================================================
// Entry point
// =============================================================================

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("NAMESAKE_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(7070);

    let state: AppState = Arc::new(RwLock::new(SearchState::empty()));

    let app = Router::new()
        .route("/health",  get(health))
        .route("/setup",   post(setup_handler))
        .route("/search",  post(search_handler))
        .route("/cleanup", post(cleanup_handler))
        .with_state(state);

    let addr = format!("127.0.0.1:{port}");
    println!("Listening on {addr}");

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}