import type { WordFrequency } from '../types';

// Standard English Stopwords
export const DEFAULT_STOPWORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "arent",
  "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
  "can", "can't", "cant", "cannot", "could", "couldn't", "couldnt", "did", "didn't", "didnt", "do", "does", "doesn't", "doesnt", "doing",
  "don't", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "hadnt", "has", "hasn't", "hasnt",
  "have", "haven't", "havent", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
  "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
  "into", "is", "isn't", "isnt", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
  "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
  "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
  "shouldn't", "shouldnt", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
  "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
  "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "wasnt", "we",
  "we'd", "we'll", "we're", "we've", "were", "weren't", "werent", "what", "what's", "when", "when's",
  "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "wont",
  "would", "wouldn't", "wouldnt", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
  "yourselves", "just", "also", "will", "say", "said", "get", "got", "go", "going", "went", "make",
  "made", "take", "taken", "come", "came", "see", "saw", "seen", "one", "two", "three", "thing",
  "things", "way", "ways", "much", "many", "lot", "lots", "good", "well", "really", "sure"
]);

// Spoken Classroom Fillers
export const LECTURE_FILLERS = new Set([
  "um", "uh", "like", "actually", "basically", "literally", "sort", "kind", "right",
  "yeah", "okay", "ok", "alright", "know", "mean", "guess", "gonna", "wanna", "gotta",
  "y'know", "so", "anyway", "hey", "guys", "listen", "look", "see", "maybe", "probably"
]);

/**
 * Splits a body of text into raw word tokens, discarding punctuation but
 * keeping intra-word apostrophes and hyphens ("don't", "well-known").
 */
export function tokenize(text: string): string[] {
  return text.replace(/[^\w\s'-]/g, ' ').split(/\s+/).filter(Boolean);
}

/**
 * Words that end in "s" but are not plurals. The "-us"/"-is"/"-ss" endings are
 * handled by rule below (campus, analysis, process), so this list only needs the
 * cases those rules miss — mostly academic subject names, which matter here
 * because they are exactly the words a lecture repeats.
 */
const NON_PLURAL_S = new Set([
  'series', 'species', 'physics', 'mathematics', 'maths', 'statistics', 'economics', 'ethics',
  'politics', 'mechanics', 'dynamics', 'thermodynamics', 'genetics', 'optics', 'acoustics',
  'linguistics', 'semantics', 'robotics', 'graphics', 'logistics', 'aesthetics', 'heuristics',
  'bias', 'atlas', 'canvas', 'gas', 'lens', 'news', 'means', 'plus', 'minus', 'always',
  'perhaps', 'towards', 'unless', 'yes',
]);

/** Plurals of "-ie" words, which take a plain "-s" rather than the "-y"/"-ies" swap. */
const IE_PLURALS = new Set([
  'movies', 'calories', 'cookies', 'series', 'genies', 'zombies', 'rookies', 'newbies', 'selfies',
]);

/**
 * Normalizes a word: lowers case, strips punctuation, basic suffix stemming.
 *
 * The plural rules deliberately stay conservative. "-es" is only stripped after
 * a sibilant (boxes, matches, processes), because the blanket rule mangles
 * ordinary words: "series" -> "seri", "species" -> "speci".
 */
export function normalizeWord(rawWord: string): string {
  const word = rawWord.toLowerCase().trim().replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '');

  if (word.length <= 3 || NON_PLURAL_S.has(word)) return word;

  // consonant + "ies" -> "y": theories -> theory, energies -> energy.
  // IE_PLURALS are the ones whose singular really ends in "-ie", not "-y".
  if (word.endsWith('ies') && word.length > 4 && !/[aeiou]ies$/.test(word)) {
    return IE_PLURALS.has(word) ? word.slice(0, -1) : word.slice(0, -3) + 'y';
  }
  // sibilant + "es": classes -> class, boxes -> box, matches -> match
  if (/(?:ss|sh|ch|x|z)es$/.test(word)) {
    return word.slice(0, -2);
  }
  // Plain plural "-s", skipping the endings that are almost never plural markers.
  if (word.endsWith('s') && !/(?:ss|us|is)$/.test(word)) {
    return word.slice(0, -1);
  }

  return word;
}

/**
 * Picks the nicer of two surface spellings for the same normalized word.
 * Short all-caps forms are treated as acronyms and win outright (GPU, DNA).
 */
function preferredSurface(current: string, candidate: string): string {
  const currentIsAcronym = current.length <= 5 && current === current.toUpperCase();
  const candidateIsAcronym = candidate.length <= 5 && candidate === candidate.toUpperCase();
  if (candidateIsAcronym && !currentIsAcronym) return candidate;
  return current;
}

/** Formats a raw token for display: Title case, or upper case for short tokens. */
function toDisplayForm(raw: string): string {
  const trimmed = raw.replace(/^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$/g, '');
  if (trimmed.length < 3) return trimmed.toUpperCase();
  if (trimmed === trimmed.toUpperCase() && trimmed.length <= 5) return trimmed; // keep acronyms
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1).toLowerCase();
}

export interface LexiconFilterOptions {
  minWordLength: number;
  minRepetitions: number;
  maxWords: number;
  includeFillers: boolean;
  bannedWords: Set<string>;
}

export interface LexiconSnapshot {
  frequencies: WordFrequency[];
  /** Every spoken token, including stopwords — this is the words-per-minute basis. */
  totalWords: number;
  /** Tokens that survived stopword/filler/ban filtering. */
  contentWords: number;
  /** Distinct surviving forms. */
  uniqueWords: number;
}

interface WordEntry {
  count: number;
  display: string;
  firstSeen: number;
  lastSeen: number;
}

const EMPTY_SNAPSHOT: LexiconSnapshot = {
  frequencies: [],
  totalWords: 0,
  contentWords: 0,
  uniqueWords: 0,
};

/**
 * Running tally of every word spoken in a session.
 *
 * Text is folded in once, as it arrives, and each word keeps the real wall-clock
 * time it was actually spoken — which is what drives the "recently said" glow in
 * the visualizers. Re-deriving the tally from the full transcript on every speech
 * event would be O(total words) per event and would reset every timestamp to now,
 * so a two-hour lecture would both crawl and render every word as permanently fresh.
 *
 * Filtering happens at snapshot time rather than ingest time, so changing the
 * stopword/ban/length settings re-filters instantly without re-reading the transcript.
 */
export class LexiconIndex {
  private entries = new Map<string, WordEntry>();
  private tokenCount = 0;

  /** Folds a chunk of speech into the tally. `timestamp` is when it was spoken. */
  public ingest(text: string, timestamp: number): void {
    for (const raw of tokenize(text)) {
      const normalized = normalizeWord(raw);
      if (!normalized) continue;

      this.tokenCount += 1;

      const existing = this.entries.get(normalized);
      if (existing) {
        existing.count += 1;
        existing.lastSeen = timestamp;
        existing.display = preferredSurface(existing.display, toDisplayForm(raw));
      } else {
        this.entries.set(normalized, {
          count: 1,
          display: toDisplayForm(raw),
          firstSeen: timestamp,
          lastSeen: timestamp,
        });
      }
    }
  }

  public reset(): void {
    this.entries.clear();
    this.tokenCount = 0;
  }

  public get size(): number {
    return this.entries.size;
  }

  /** Merges another index into this one, keeping the later `lastSeen` per word. */
  public merge(other: LexiconIndex): void {
    this.tokenCount += other.tokenCount;
    for (const [key, incoming] of other.entries) {
      const existing = this.entries.get(key);
      if (existing) {
        existing.count += incoming.count;
        existing.lastSeen = Math.max(existing.lastSeen, incoming.lastSeen);
        existing.firstSeen = Math.min(existing.firstSeen, incoming.firstSeen);
        existing.display = preferredSurface(existing.display, incoming.display);
      } else {
        this.entries.set(key, { ...incoming });
      }
    }
  }

  /** Non-destructive union of this index with another, for snapshot purposes. */
  private combinedEntries(overlay: LexiconIndex): Map<string, WordEntry> {
    const merged = new Map(this.entries);
    for (const [key, incoming] of overlay.entries) {
      const existing = merged.get(key);
      merged.set(
        key,
        existing
          ? {
              ...existing,
              count: existing.count + incoming.count,
              lastSeen: Math.max(existing.lastSeen, incoming.lastSeen),
            }
          : { ...incoming }
      );
    }
    return merged;
  }

  /**
   * Applies the current filters and returns the ranked word list.
   *
   * `overlay` folds in a second index without committing it — used for the
   * in-flight interim phrase, which is still being revised by the recognizer
   * and must not pollute the permanent tally.
   */
  public snapshot(options: LexiconFilterOptions, overlay?: LexiconIndex): LexiconSnapshot {
    const entries = overlay && overlay.size > 0 ? this.combinedEntries(overlay) : this.entries;
    const totalWords = this.tokenCount + (overlay?.tokenCount ?? 0);
    const kept: Array<{ entry: WordEntry }> = [];
    let contentWords = 0;

    for (const [normalized, entry] of entries) {
      if (normalized.length < options.minWordLength) continue;
      if (DEFAULT_STOPWORDS.has(normalized)) continue;
      if (!options.includeFillers && LECTURE_FILLERS.has(normalized)) continue;
      if (options.bannedWords.has(normalized)) continue;
      if (/^\d+$/.test(normalized)) continue;

      contentWords += entry.count;
      kept.push({ entry });
    }

    const frequencies: WordFrequency[] = kept
      .filter(({ entry }) => entry.count >= options.minRepetitions)
      .sort((a, b) => b.entry.count - a.entry.count)
      .slice(0, options.maxWords)
      .map(({ entry }) => ({
        text: entry.display,
        count: entry.count,
        lastSeen: entry.lastSeen,
        initialSeen: entry.firstSeen,
        percentage: contentWords > 0 ? (entry.count / contentWords) * 100 : 0,
        category: (entry.count >= 6 ? 'core' : entry.count >= 3 ? 'frequent' : 'emerging') as
          | 'core'
          | 'frequent'
          | 'emerging',
      }));

    return {
      frequencies,
      totalWords,
      contentWords,
      uniqueWords: kept.length,
    };
  }
}

export const EMPTY_LEXICON_SNAPSHOT = EMPTY_SNAPSHOT;

/**
 * One-shot convenience wrapper: tally a whole body of text and snapshot it.
 * Live sessions should hold a LexiconIndex instead so word timestamps survive.
 */
export function processTranscriptText(
  fullText: string,
  options: LexiconFilterOptions,
  timestamp: number = Date.now()
): LexiconSnapshot {
  const index = new LexiconIndex();
  index.ingest(fullText, timestamp);
  return index.snapshot(options);
}
