// AudioFrame — JSON payload emitted over the WebSocket at ~60 Hz.
//
// The shape MUST stay in lockstep with src/contracts/audioFrame.ts:
//   { t, rms, bands:[8], bass, mid, treble, centroid, onset, beatConfidence }
// Codable synthesizes keys from the property names, which match the contract
// 1:1, so the on-wire JSON keys are exactly the names below.

import Foundation

/// Number of log-spaced spectral bands. Mirrors AUDIO_FRAME_BANDS in audioFrame.ts.
let kAudioFrameBands = 8

struct AudioFrame: Codable {
    /// Monotonic timestamp in ms since helper start.
    var t: Int
    /// Overall loudness, normalized 0..1 (smoothed RMS).
    var rms: Double
    /// Log-spaced spectral bands, each 0..1. count == kAudioFrameBands.
    var bands: [Double]
    /// Convenience aggregates of `bands`, each 0..1.
    var bass: Double
    var mid: Double
    var treble: Double
    /// Spectral centroid ("brightness"), normalized 0..1.
    var centroid: Double
    /// True on a detected onset/transient this frame.
    var onset: Bool
    /// Confidence in the current beat estimate, 0..1. Tempo is approximate.
    var beatConfidence: Double

    /// A silent / zeroed frame (used before any audio arrives).
    static func silent(t: Int) -> AudioFrame {
        AudioFrame(
            t: t,
            rms: 0,
            bands: Array(repeating: 0, count: kAudioFrameBands),
            bass: 0, mid: 0, treble: 0,
            centroid: 0,
            onset: false,
            beatConfidence: 0
        )
    }
}
