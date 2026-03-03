// ==================================================
// Tunable Parameters
// ==================================================

export const PARAMS = {
  SURNAME_FLOOR:    0.5,
  BOTH_FIRST_MIN:   0.7,
  THRESHOLD_SMALL:  0.665,
  THRESHOLD_LARGE:  0.655,
};

export const LARGE_DATASET_CUTOFF = 5_000;

// First-name scoring constants
export const INITIAL_MATCH   = 0.8;
export const MISSING_FIRST   = 0.6;
export const INITIAL_SOFT    = 0.37;
export const INITIAL_HARD    = 0.0;