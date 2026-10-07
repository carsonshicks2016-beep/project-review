import type { TranscriptItem } from '../types';

export interface ParsedCue {
  /** Offset from the start of the recording, in seconds. */
  startSeconds: number;
  text: string;
}

const CUE_TIMING = /(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}/;
/** Leading clock stamp used by this app's own .txt export, e.g. "[10:30:45 AM] ". */
const LEADING_STAMP = /^\[\s*\d{1,2}:\d{2}(:\d{2})?(\s*[AaPp]\.?[Mm]\.?)?\s*\]\s*/;

function parseTimecode(raw: string): number {
  const parts = raw.trim().replace(',', '.').split(':').map(Number);
  if (parts.some((n) => Number.isNaN(n))) return 0;
  // hh:mm:ss.mmm or mm:ss.mmm
  return parts.reduce((acc, part) => acc * 60 + part, 0);
}

/**
 * Parses SubRip (.srt) and WebVTT (.vtt) subtitle files into ordered cues.
 *
 * The cue timings and sequence numbers must not survive into the transcript —
 * they are not spoken words, and counting them pollutes the word diagram with
 * numeric noise.
 */
export function parseSubtitles(content: string): ParsedCue[] {
  const blocks = content.replace(/\r\n?/g, '\n').split(/\n{2,}/);
  const cues: ParsedCue[] = [];

  for (const block of blocks) {
    const lines = block.split('\n').map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) continue;

    const timingIndex = lines.findIndex((line) => CUE_TIMING.test(line));
    if (timingIndex === -1) continue;

    const startSeconds = parseTimecode(lines[timingIndex].split('-->')[0]);
    // Everything after the timing line is the caption body; anything before it is
    // a cue number or identifier.
    const text = lines
      .slice(timingIndex + 1)
      .join(' ')
      // Strip WebVTT inline markup such as <v Speaker> and <00:00:01.000>.
      .replace(/<[^>]*>/g, '')
      .trim();

    if (text) cues.push({ startSeconds, text });
  }

  return cues;
}

/** Parses a plain-text transcript, one utterance per non-blank line. */
export function parsePlainText(content: string): ParsedCue[] {
  return content
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((line) => line.replace(LEADING_STAMP, '').trim())
    .filter(Boolean)
    .map((text, index) => ({ startSeconds: index * 3, text }));
}

/**
 * Turns an imported transcript file into transcript items.
 *
 * Timestamps are anchored to `baseTime` and offset by each cue's real position in
 * the recording, so session duration and words-per-minute reflect the lecture
 * rather than how long the import took.
 */
export function parseTranscriptFile(
  fileName: string,
  content: string,
  baseTime: number = Date.now()
): TranscriptItem[] {
  const isSubtitle = /\.(srt|vtt)$/i.test(fileName);
  const cues = isSubtitle ? parseSubtitles(content) : [];
  const resolved = cues.length > 0 ? cues : parsePlainText(content);

  return resolved.map((cue, index) => ({
    id: `import-${baseTime}-${index}`,
    text: cue.text,
    timestamp: baseTime + Math.round(cue.startSeconds * 1000),
    isFinal: true,
  }));
}
