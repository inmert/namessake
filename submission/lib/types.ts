// ==================================================
// Types
// ==================================================

export interface NameRecord {
  id:        string;
  rawName:   string;
  normName:  string;
  initials:  string[];
  words:     string[];
  firstWord: string;
  lastWord:  string;
  wordCount: number;
}