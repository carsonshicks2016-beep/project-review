/**
 * AudioFrame — emitted by the native Swift audio helper (ScreenCaptureKit + vDSP)
 * over a local WebSocket at ~60 Hz. Drives the *reactive* layer of the visuals.
 *
 * Owner: Agent B (helper/). Consumers: Agent D (renderer), Agent C (gen priors).
 */
export interface AudioFrame {
  /** Monotonic timestamp in ms since helper start. */
  t: number;
  /** Overall loudness, normalized 0..1 (smoothed RMS). */
  rms: number;
  /** Log-spaced spectral bands, each 0..1. Length === AUDIO_FRAME_BANDS. */
  bands: number[];
  /** Convenience aggregates of `bands`, each 0..1. */
  bass: number;
  mid: number;
  treble: number;
  /** Spectral centroid ("brightness"), normalized 0..1. */
  centroid: number;
  /** True on a detected onset/transient this frame. */
  onset: boolean;
  /** Confidence in the current beat estimate, 0..1. Tempo is approximate. */
  beatConfidence: number;
}

export const AUDIO_FRAME_BANDS = 8;

/** Default local WebSocket the helper serves and the renderer connects to. */
export const AUDIO_WS_URL = "ws://127.0.0.1:17653";
