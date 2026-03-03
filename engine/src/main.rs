use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::LazyLock;

use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use regex::Regex;
use rphonetic::{DoubleMetaphone, Encoder};

// ==================================================
// Constants
// ==================================================

const PORT: u16 = 7878;
const LARGE_CUTOFF: usize = 5000;
const THRESHOLD_SMALL: f64 = 0.64;
const THRESHOLD_LARGE: f64 = 0.655;
const SURNAME_FLOOR: f64 = 0.50;

// ==================================================
// Regexes (compiled once)
// ==================================================

static SUFFIX_RE:     LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\b(Jr\.?|Sr\.?|II|III|IV)\b").unwrap());
static COMMA_RE:      LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([^,]+),\s*(.+)$").unwrap());
static CONCAT_X_RE:   LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([a-z])x([A-Z])").unwrap());
static INITIAL_RE:    LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z]\.$").unwrap());
static JUNK_PREFIX_RE:LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z][A-Z]").unwrap());

// ==================================================
// Types
// ==================================================

#[derive(Clone)]
struct Name {
    words:      Vec<String>,
    initials:   Vec<char>,
    first:      String,
    last:       String,
}

#[derive(Default)]
struct Index {
    records:    AHashMap<String, Name>,
    ex_first:   AHashMap<String, Vec<String>>,
    ex_last:    AHashMap<String, Vec<String>>,
    ph_first:   AHashMap<String, Vec<String>>,
    ph_last:    AHashMap<String, Vec<String>>,
    init_first: AHashMap<char, Vec<String>>,
    init_last:  AHashMap<char, Vec<String>>,
    threshold:  f64,
    dm:         DoubleMetaphone,
}

// ==================================================
// Normalization
// ==================================================

fn normalize(raw: &str) -> String {
    let s = raw.trim_matches('"');
    let s = if let Some(c) = COMMA_RE.captures(s) {
        format!("{} {}", c[2].trim(), c[1].trim())
    } else {
        s.to_string()
    };
    let s = s.replace('0', "o").replace('1', "l");
    let s = SUFFIX_RE.replace_all(&s, "");
    let s = CONCAT_X_RE.replace_all(&s, "$1 $2");
    s.split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|t| if t.len() >= 2 && JUNK_PREFIX_RE.is_match(t) { t[1..].to_string() } else { t.to_string() })
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn parse(norm: &str) -> (Vec<char>, Vec<String>) {
    let mut initials = Vec::new();
    let mut words    = Vec::new();
    for tok in norm.split_whitespace() {
        if INITIAL_RE.is_match(tok) {
            if let Some(c) = tok.chars().next() { initials.push(c); }
        } else if tok.len() > 1 {
            words.push(tok.to_string());
        }
    }
    (initials, words)
}

// ==================================================
// Similarity
// ==================================================

fn trigrams(s: &str) -> AHashSet<[u8; 3]> {
    let p = format!("##{s}##");
    let b = p.as_bytes();
    (0..b.len().saturating_sub(2)).map(|i| [b[i], b[i+1], b[i+2]]).collect()
}

fn jaccard(a: &str, b: &str) -> f64 {
    let ta = trigrams(a);
    let tb = trigrams(b);
    if ta.is_empty() && tb.is_empty() { return 1.0; }
    let inter = ta.intersection(&tb).count();
    inter as f64 / (ta.len() + tb.len() - inter) as f64
}

fn lcs_sim(a: &str, b: &str) -> f64 {
    if a == b { return 1.0; }
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    let (m, n) = (ab.len(), bb.len());
    if m == 0 || n == 0 { return 0.0; }
    if (m as isize - n as isize).unsigned_abs() as f64 > m.max(n) as f64 * 0.7 { return 0.0; }
    let mut dp = vec![0i16; n + 1];
    for i in 0..m {
        let mut prev = 0i16;
        for j in 0..n {
            let tmp = dp[j+1];
            dp[j+1] = if ab[i] == bb[j] { prev + 1 } else { dp[j+1].max(dp[j]) };
            prev = tmp;
        }
    }
    (2 * dp[n] as usize) as f64 / (m + n) as f64
}

#[inline]
fn sim(a: &str, b: &str) -> f64 { (jaccard(a, b) + lcs_sim(a, b)) / 2.0 }

// ==================================================
// Scoring
// ==================================================

fn score_first(q: &Name, r: &Name, qf: &str, rf: &str) -> f64 {
    let qfw: Vec<&str> = q.words.iter().filter(|w| w.as_str() != qf).map(|s| s.as_str()).collect();
    let rfw: Vec<&str> = r.words.iter().filter(|w| w.as_str() != rf).map(|s| s.as_str()).collect();

    // Initial vs first-word match
    let init_hit = q.initials.iter().any(|&c| rfw.first().map(|w| w.starts_with(c)).unwrap_or(false))
        || r.initials.iter().any(|&c| qfw.first().map(|w| w.starts_with(c)).unwrap_or(false));
    if init_hit { return 0.80; }

    // Both have words — compare them
    if !qfw.is_empty() && !rfw.is_empty() {
        let mut best = 0.0f64;
        'outer: for &a in &qfw {
            for &b in &rfw {
                best = best.max(sim(a, b));
                if best > 0.8 { break 'outer; }
            }
        }
        return best;
    }

    // Both have only initials
    if qfw.is_empty() && rfw.is_empty() {
        if !q.initials.is_empty() && !r.initials.is_empty() {
            return if q.initials.iter().any(|c| r.initials.contains(c)) { 0.80 } else { 0.0 };
        }
        return 0.60; // one or both names missing first name entirely
    }

    // One side has words, other has only initials
    if !q.initials.is_empty() && !rfw.is_empty() { return 0.30; }
    if !r.initials.is_empty() && !qfw.is_empty() { return 0.0; }
    0.60 // one side has no first-name info
}

fn score(q: &Name, r: &Name, threshold: f64) -> f64 {
    if q.words.is_empty() || r.words.is_empty() { return 0.0; }

    // Try normal orientation and name-swapped orientation
    let pairs: [(&str, &str, bool); 3] = [
        (&q.last,  &r.last,  false),
        (&q.last,  &r.first, false),
        (&q.first, &r.last,  false),
    ];

    let mut best = 0.0f64;
    for (ql, rl, _) in &pairs {
        let last_sim = sim(ql, rl);
        if last_sim < SURNAME_FLOOR { continue; }
        let first_sim = score_first(q, r, ql, rl);
        if first_sim < 0.45 { continue; }
        best = best.max(0.5 * last_sim + 0.5 * first_sim);
        if best > threshold + 0.15 { break; }
    }
    best
}

// ==================================================
// Index
// ==================================================

impl Index {
    fn add_str(map: &mut AHashMap<String, Vec<String>>, key: &str, id: &str) {
        if !key.is_empty() { map.entry(key.into()).or_default().push(id.into()); }
    }
    fn add_char(map: &mut AHashMap<char, Vec<String>>, key: char, id: &str) {
        map.entry(key).or_default().push(id.into());
    }

    fn load(&mut self, path: &str) -> Result<(), String> {
        self.records.clear();
        self.ex_first.clear(); self.ex_last.clear();
        self.ph_first.clear(); self.ph_last.clear();
        self.init_first.clear(); self.init_last.clear();

        let mut rdr = csv::Reader::from_path(path).map_err(|e| e.to_string())?;
        let mut count = 0usize;

        for rec in rdr.records() {
            let rec = rec.map_err(|e| e.to_string())?;
            let id  = match rec.get(0) { Some(s) => s.trim().to_string(), None => continue };
            let raw = match rec.get(1) { Some(s) => s.trim().to_string(), None => continue };

            let norm = normalize(&raw);
            let (initials, words) = parse(&norm);
            if words.is_empty() { continue; }

            let first = words[0].clone();
            let last  = words[words.len()-1].clone();

            Self::add_str(&mut self.ex_first,   &first, &id);
            Self::add_str(&mut self.ex_last,    &last,  &id);
            Self::add_str(&mut self.ph_first,   &self.dm.encode(&first), &id);
            Self::add_str(&mut self.ph_last,    &self.dm.encode(&last),  &id);
            if let Some(c) = first.chars().next() { Self::add_char(&mut self.init_first, c, &id); }
            if let Some(c) = last.chars().next()  { Self::add_char(&mut self.init_last,  c, &id); }

            self.records.insert(id, Name { words, initials, first, last });
            count += 1;
        }

        self.threshold = if count > LARGE_CUTOFF { THRESHOLD_LARGE } else { THRESHOLD_SMALL };
        eprintln!("[engine] {count} records, threshold={}", self.threshold);
        Ok(())
    }

    fn search(&self, query: &str) -> Vec<String> {
        let norm = normalize(query);
        let (initials, words) = parse(&norm);
        if words.is_empty() { return vec![]; }

        let first = words[0].clone();
        let last  = words[words.len()-1].clone();
        let q = Name { words, initials, first, last };

        let mut cands: AHashSet<String> = AHashSet::new();
        macro_rules! add {
            ($map:expr, $key:expr) => {
                if let Some(ids) = $map.get($key as &str) { cands.extend(ids.iter().cloned()); }
            };
        }

        add!(self.ex_last,  &q.last);  add!(self.ex_first, &q.first);
        add!(self.ex_last,  &q.first); add!(self.ex_first, &q.last);
        add!(self.ph_first, &self.dm.encode(&q.first));
        add!(self.ph_last,  &self.dm.encode(&q.last));
        for &c in &q.initials {
            if let Some(ids) = self.init_first.get(&c) { cands.extend(ids.iter().cloned()); }
            if let Some(ids) = self.init_last.get(&c)  { cands.extend(ids.iter().cloned()); }
        }

        let t = self.threshold;
        cands.into_iter().collect::<Vec<_>>().into_par_iter()
            .filter(|id| self.records.get(id).map(|r| score(&q, r, t) >= t).unwrap_or(false))
            .collect()
    }
}

// ==================================================
// TCP server
// ==================================================

fn handle(stream: TcpStream) {
    let reader = BufReader::new(&stream);
    let mut writer = BufWriter::new(&stream);
    let mut idx = Index::default();

    if writeln!(writer, "READY").is_err() || writer.flush().is_err() { return; }

    for line in reader.lines() {
        let msg = match line { Ok(l) => l.trim().to_string(), Err(_) => break };
        if msg.is_empty() { continue; }

        let reply = if let Some(path) = msg.strip_prefix("LOAD ") {
            match idx.load(path.trim()) {
                Ok(())  => "LOADED".to_string(),
                Err(e)  => { eprintln!("[engine] {e}"); format!("ERROR {e}") }
            }
        } else {
            let ids = idx.search(&msg);
            serde_json::to_string(&ids).unwrap_or_else(|_| "[]".into())
        };

        if writeln!(writer, "{reply}").is_err() || writer.flush().is_err() { break; }
    }
}

fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind(format!("127.0.0.1:{PORT}"))?;
    eprintln!("[engine] listening on 127.0.0.1:{PORT}");
    for stream in listener.incoming() {
        match stream { Ok(s) => handle(s), Err(e) => eprintln!("[engine] {e}") }
    }
    Ok(())
}

