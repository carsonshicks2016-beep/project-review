export interface WordFrequency {
  text: string;
  count: number;
  lastSeen: number; // timestamp
  initialSeen: number;
  percentage: number;
  category?: 'core' | 'frequent' | 'emerging';
}

export interface TranscriptItem {
  id: string;
  text: string;
  timestamp: number;
  isFinal: boolean;
}

export type VisualizerMode =
  | 'kinetic'
  | 'editorial'
  | 'celestial'
  | 'holosphere'
  | 'voronoi'
  | 'matrix'
  | 'sunburst'
  | 'honeycomb';

export type ColorTheme = 'cyber' | 'editorial' | 'bioluminescent' | 'sunset' | 'monochrome';

export type FontFamily = 'inter' | 'playfair' | 'jetbrains' | 'space' | 'cinzel';

export interface VisualizerSettings {
  mode: VisualizerMode;
  theme: ColorTheme;
  fontFamily: FontFamily;
  minWordLength: number;
  minRepetitions: number;
  maxWords: number;
  includeFillers: boolean;
  enableRotation: boolean;
  gravityStrength: number;
  micSensitivity: number;
}

export interface LectureSessionStats {
  durationSeconds: number;
  totalWords: number;
  uniqueWords: number;
  wordsPerMinute: number;
  topKeyword: string;
  repetitionRatio: number;
}
