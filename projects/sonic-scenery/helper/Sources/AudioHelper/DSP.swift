// DSP — turns a window of mono PCM into an AudioFrame using Accelerate/vDSP.
//
// Pipeline per analysis window:
//   1. Window the samples (Hann) and run a real FFT (vDSP_fft_zrip).
//   2. Magnitude spectrum -> 8 log-spaced energy bands (0..1).
//   3. RMS (time domain) -> smoothed loudness.
//   4. Spectral centroid (energy-weighted mean bin) -> brightness 0..1.
//   5. Spectral flux (positive change vs previous magnitude spectrum) -> onset.
//   6. A lightweight autocorrelation-of-flux gives an *approximate* beat
//      confidence. This is intentionally NOT a locked BPM — see PLAN.md risk #2.
//
// All outputs are normalized into 0..1 with running peak / EMA smoothing so the
// visuals get stable, well-behaved signals regardless of source loudness.

import Foundation
import Accelerate

final class DSP {
    // FFT configuration.
    private let fftSize: Int
    private let log2n: vDSP_Length
    private let fftSetup: FFTSetup
    private let halfSize: Int
    private let sampleRate: Double

    // Reusable buffers.
    private var window: [Float]
    private var realp: [Float]
    private var imagp: [Float]
    private var magnitudes: [Float]
    private var windowed: [Float]

    // Onset / flux state.
    private var prevMagnitudes: [Float]
    private var fluxHistory: [Float] = []      // recent spectral-flux values
    private let fluxHistoryMax = 172           // ~ a few seconds of frames
    private var lastOnsetTime: Double = -1

    // Smoothing / normalization state.
    private var rmsSmoothed: Double = 0
    private var rmsPeak: Double = 1e-4
    private var bandPeaks: [Double]
    private var fluxPeak: Float = 1e-4

    // Log band edges (bin indices), inclusive lower .. exclusive upper.
    private let bandBinRanges: [(Int, Int)]

    init(fftSize: Int = 2048, sampleRate: Double) {
        self.fftSize = fftSize
        self.sampleRate = sampleRate
        self.log2n = vDSP_Length(log2(Double(fftSize)))
        self.halfSize = fftSize / 2
        self.fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!

        self.window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

        self.realp = [Float](repeating: 0, count: halfSize)
        self.imagp = [Float](repeating: 0, count: halfSize)
        self.magnitudes = [Float](repeating: 0, count: halfSize)
        self.windowed = [Float](repeating: 0, count: fftSize)
        self.prevMagnitudes = [Float](repeating: 0, count: halfSize)
        self.bandPeaks = [Double](repeating: 1e-4, count: kAudioFrameBands)

        // Log-spaced band edges between ~40 Hz and Nyquist.
        let nyquist = sampleRate / 2
        let minHz = 40.0
        let maxHz = nyquist
        var ranges: [(Int, Int)] = []
        let binHz = sampleRate / Double(fftSize)
        for b in 0..<kAudioFrameBands {
            let f0 = minHz * pow(maxHz / minHz, Double(b) / Double(kAudioFrameBands))
            let f1 = minHz * pow(maxHz / minHz, Double(b + 1) / Double(kAudioFrameBands))
            var lo = Int((f0 / binHz).rounded(.down))
            var hi = Int((f1 / binHz).rounded(.down))
            lo = max(1, min(lo, halfSize - 1))           // skip DC bin
            hi = max(lo + 1, min(hi, halfSize))
            ranges.append((lo, hi))
        }
        self.bandBinRanges = ranges
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Analyze one window of mono samples. `samples.count` should be >= fftSize;
    /// only the first fftSize are used. Returns a fully populated AudioFrame
    /// (timestamp is filled by the caller via `t`).
    func analyze(samples: [Float], t: Int) -> AudioFrame {
        guard samples.count >= fftSize else {
            return AudioFrame.silent(t: t)
        }

        // --- Time-domain RMS (before windowing) ---
        var meanSquare: Float = 0
        samples.withUnsafeBufferPointer { ptr in
            vDSP_measqv(ptr.baseAddress!, 1, &meanSquare, vDSP_Length(fftSize))
        }
        let rmsRaw = Double(sqrt(meanSquare))

        // --- Window + FFT ---
        vDSP_vmul(samples, 1, window, 1, &windowed, 1, vDSP_Length(fftSize))

        windowed.withUnsafeBufferPointer { wptr in
            wptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { complexPtr in
                realp.withUnsafeMutableBufferPointer { rp in
                    imagp.withUnsafeMutableBufferPointer { ip in
                        var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                        // Pack real input into split-complex form.
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(halfSize))
                        // In-place real forward FFT.
                        vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                        // Magnitudes (|z|), not squared — gentler dynamics for bands.
                        vDSP_zvabs(&split, 1, &magnitudes, 1, vDSP_Length(halfSize))
                    }
                }
            }
        }
        // vDSP_fft_zrip is scaled by 2; normalize roughly so values are sane.
        var scale = Float(1.0 / Double(fftSize))
        vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(halfSize))

        // --- 8 log bands ---
        var bands = [Double](repeating: 0, count: kAudioFrameBands)
        for (i, range) in bandBinRanges.enumerated() {
            var sum: Float = 0
            magnitudes.withUnsafeBufferPointer { ptr in
                vDSP_sve(ptr.baseAddress! + range.0, 1, &sum, vDSP_Length(range.1 - range.0))
            }
            let count = Double(range.1 - range.0)
            let avg = count > 0 ? Double(sum) / count : 0
            // Adaptive per-band peak normalization with slow decay.
            bandPeaks[i] = max(bandPeaks[i] * 0.999, avg, 1e-4)
            bands[i] = clamp01(avg / bandPeaks[i])
        }

        // --- bass / mid / treble aggregates ---
        // Bands 0-1 = bass, 2-4 = mid, 5-7 = treble (log-spaced).
        let bass = avg(of: bands[0...1])
        let mid = avg(of: bands[2...4])
        let treble = avg(of: bands[5...7])

        // --- Spectral centroid (brightness) ---
        var weightedSum: Double = 0
        var magSum: Double = 0
        for bin in 1..<halfSize {
            let m = Double(magnitudes[bin])
            weightedSum += Double(bin) * m
            magSum += m
        }
        let centroidBin = magSum > 1e-9 ? weightedSum / magSum : 0
        let centroid = clamp01(centroidBin / Double(halfSize)) // 0..1 across spectrum

        // --- RMS smoothing + normalization ---
        rmsPeak = max(rmsPeak * 0.9995, rmsRaw, 1e-4)
        let rmsNorm = clamp01(rmsRaw / rmsPeak)
        // Asymmetric EMA: fast attack, slow release -> punchy but stable.
        let attack = 0.5, release = 0.08
        let coeff = rmsNorm > rmsSmoothed ? attack : release
        rmsSmoothed = rmsSmoothed + coeff * (rmsNorm - rmsSmoothed)

        // --- Spectral flux -> onset ---
        var flux: Float = 0
        for bin in 0..<halfSize {
            let diff = magnitudes[bin] - prevMagnitudes[bin]
            if diff > 0 { flux += diff }   // half-wave rectified
        }
        prevMagnitudes = magnitudes

        fluxPeak = max(fluxPeak * 0.999, flux, 1e-4)
        let fluxNorm = flux / fluxPeak

        fluxHistory.append(flux)
        if fluxHistory.count > fluxHistoryMax { fluxHistory.removeFirst() }

        // Onset: flux exceeds an adaptive threshold (mean + k*std of recent flux),
        // with a short refractory window to avoid double-triggering.
        let nowSec = Double(t) / 1000.0
        var onset = false
        if fluxHistory.count >= 8 {
            let (mean, std) = meanStd(fluxHistory)
            let threshold = mean + 1.5 * std
            let refractoryOK = (nowSec - lastOnsetTime) > 0.08
            if Float(flux) > threshold && fluxNorm > 0.25 && refractoryOK {
                onset = true
                lastOnsetTime = nowSec
            }
        }

        // --- Approximate beat confidence (NOT a locked BPM) ---
        let beatConfidence = estimateBeatConfidence()

        return AudioFrame(
            t: t,
            rms: rmsSmoothed,
            bands: bands,
            bass: bass,
            mid: mid,
            treble: treble,
            centroid: centroid,
            onset: onset,
            beatConfidence: beatConfidence
        )
    }

    // MARK: - Beat confidence

    /// Autocorrelation of the recent spectral-flux envelope. A strong, narrow
    /// peak in the musically-plausible tempo range implies a steady pulse; we
    /// report the relative strength of that peak as a 0..1 confidence. Tempo
    /// itself is deliberately not reported — capture-based BPM is unreliable.
    private func estimateBeatConfidence() -> Double {
        let n = fluxHistory.count
        guard n >= 64 else { return 0 }

        // Frame rate of the flux envelope (windows per second).
        let framesPerSec = sampleRate / Double(DSP.hopSize)
        // Lags corresponding to ~60..180 BPM.
        let minLag = max(2, Int(framesPerSec * 60.0 / 180.0))
        let maxLag = min(n - 1, Int(framesPerSec * 60.0 / 60.0))
        guard maxLag > minLag else { return 0 }

        // Mean-remove the envelope.
        let (mean, _) = meanStd(fluxHistory)
        let env = fluxHistory.map { Double($0) - Double(mean) }

        var zeroLag: Double = 0
        for v in env { zeroLag += v * v }
        guard zeroLag > 1e-9 else { return 0 }

        var bestPeak: Double = 0
        for lag in minLag...maxLag {
            var acc: Double = 0
            for i in lag..<n { acc += env[i] * env[i - lag] }
            let norm = acc / zeroLag
            if norm > bestPeak { bestPeak = norm }
        }
        return clamp01(bestPeak)
    }

    /// Hop size between analysis windows (set by Capture). Used only to convert
    /// the flux-history length into seconds for tempo math.
    static var hopSize: Int = 1024
}

// MARK: - small helpers

private func clamp01(_ x: Double) -> Double { min(1, max(0, x)) }

private func avg(of slice: ArraySlice<Double>) -> Double {
    slice.isEmpty ? 0 : slice.reduce(0, +) / Double(slice.count)
}

private func meanStd(_ xs: [Float]) -> (Float, Float) {
    guard !xs.isEmpty else { return (0, 0) }
    let n = Float(xs.count)
    let mean = xs.reduce(0, +) / n
    var variance: Float = 0
    for x in xs { let d = x - mean; variance += d * d }
    return (mean, (variance / n).squareRoot())
}
