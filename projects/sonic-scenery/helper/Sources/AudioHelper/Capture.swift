// Capture — system-audio capture via ScreenCaptureKit (SCStream, capturesAudio).
//
// On macOS 13+ this captures the system audio mix with no virtual driver — it
// requires only the one-time Screen Recording permission grant. We:
//   1. Query SCShareableContent (this both lists displays and, on first run,
//      triggers the Screen Recording permission prompt). A failure here almost
//      always means permission was denied — we surface an actionable error.
//   2. Configure an SCStream with capturesAudio = true and attach an audio
//      stream output.
//   3. On each audio CMSampleBuffer, copy mono PCM into a ring buffer.
//   4. A ~60 Hz timer pulls the most recent window, runs the DSP, and hands the
//      resulting AudioFrame to the sink (the WebSocket broadcaster).
//
// We deliberately decouple capture callback rate from emit rate: ScreenCaptureKit
// delivers audio in device-sized chunks, but the contract wants a steady ~60 Hz.

import Foundation
import ScreenCaptureKit
import AVFoundation
import OSLog

enum CaptureError: Error, CustomStringConvertible {
    case permissionDenied(underlying: Error?)
    case noDisplay
    case streamSetup(Error)

    var description: String {
        switch self {
        case .permissionDenied(let e):
            return """
            Screen Recording permission is required to capture system audio.
            Grant it under: System Settings ▸ Privacy & Security ▸ Screen Recording,
            enable this binary (or your terminal), then re-run.
            \(e.map { "Underlying error: \($0)" } ?? "")
            """
        case .noDisplay:
            return "No display available to attach the audio capture stream to."
        case .streamSetup(let e):
            return "Failed to start the ScreenCaptureKit stream: \(e)"
        }
    }
}

final class Capture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let fftSize = 2048
    private let hopSize = 1024            // emit DSP analysis over a sliding window
    private let emitHz = 60.0
    private let sampleRate: Double = 48_000

    private var stream: SCStream?
    private var dsp: DSP!
    private let startTime = DispatchTime.now()
    private var emitTimer: DispatchSourceTimer?
    private let emitQueue = DispatchQueue(label: "audio.emit")        // ~60 Hz DSP + broadcast
    private let captureQueue = DispatchQueue(label: "audio.capture")  // SCK sample delivery

    // Ring buffer of recent mono samples (lock-protected; written from the SCK
    // callback, read from the emit timer).
    private var ring: [Float]
    private var ringWrite = 0
    private var ringFilled = 0
    private let ringCapacity: Int
    private let ringLock = NSLock()

    /// Called ~60 Hz with each analyzed frame.
    var onFrame: ((AudioFrame) -> Void)?

    override init() {
        // ~1s of headroom so the analysis window is always satisfiable.
        self.ringCapacity = 65536
        self.ring = [Float](repeating: 0, count: ringCapacity)
        super.init()
        DSP.hopSize = hopSize
        self.dsp = DSP(fftSize: fftSize, sampleRate: sampleRate)
    }

    /// Begin capture. Throws CaptureError (e.g. permission denied) on failure.
    func start() async throws {
        let content: SCShareableContent
        do {
            // This call triggers / checks the Screen Recording permission.
            content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: false)
        } catch {
            throw CaptureError.permissionDenied(underlying: error)
        }

        guard let display = content.displays.first else {
            throw CaptureError.noDisplay
        }

        // Capture the whole display; we only care about the audio track, but
        // SCStream still needs a video content filter.
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = Int(sampleRate)
        config.channelCount = 2
        config.excludesCurrentProcessAudio = true   // don't capture our own output
        // Minimal video config — we discard video frames.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.queueDepth = 6

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        self.stream = stream

        do {
            try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: captureQueue)
            // Adding a screen output is required even though we ignore the frames.
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
            try await stream.startCapture()
        } catch {
            throw CaptureError.streamSetup(error)
        }

        startEmitTimer()
        FileHandle.standardError.write(Data("[capture] system audio capture started\n".utf8))
    }

    func stop() async {
        emitTimer?.cancel()
        emitTimer = nil
        if let stream = stream {
            try? await stream.stopCapture()
        }
        stream = nil
    }

    // MARK: - Emit timer (~60 Hz)

    private func startEmitTimer() {
        let timer = DispatchSource.makeTimerSource(queue: emitQueue)
        let interval = 1.0 / emitHz
        timer.schedule(deadline: .now() + interval, repeating: interval)
        timer.setEventHandler { [weak self] in self?.emitFrame() }
        timer.resume()
        emitTimer = timer
    }

    private func emitFrame() {
        let t = elapsedMs()
        let window = latestWindow()
        let frame: AudioFrame
        if let window = window {
            frame = dsp.analyze(samples: window, t: t)
        } else {
            frame = AudioFrame.silent(t: t)
        }
        onFrame?(frame)
    }

    private func elapsedMs() -> Int {
        let now = DispatchTime.now()
        let ns = now.uptimeNanoseconds - startTime.uptimeNanoseconds
        return Int(ns / 1_000_000)
    }

    /// Copy the most recent `fftSize` samples out of the ring (chronological).
    private func latestWindow() -> [Float]? {
        ringLock.lock()
        defer { ringLock.unlock() }
        guard ringFilled >= fftSize else { return nil }
        var out = [Float](repeating: 0, count: fftSize)
        // The newest sample is at (ringWrite - 1); copy the trailing fftSize.
        let start = (ringWrite - fftSize + ringCapacity) % ringCapacity
        for i in 0..<fftSize {
            out[i] = ring[(start + i) % ringCapacity]
        }
        return out
    }

    private func appendSamples(_ samples: UnsafeBufferPointer<Float>) {
        ringLock.lock()
        for s in samples {
            ring[ringWrite] = s
            ringWrite = (ringWrite + 1) % ringCapacity
        }
        ringFilled = min(ringCapacity, ringFilled + samples.count)
        ringLock.unlock()
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio else { return }   // discard video frames
        guard sampleBuffer.isValid else { return }
        handleAudio(sampleBuffer)
    }

    private func handleAudio(_ sampleBuffer: CMSampleBuffer) {
        // Pull the PCM into an AVAudioPCMBuffer-friendly form. ScreenCaptureKit
        // delivers Float32; we downmix to mono by averaging channels.
        guard let formatDesc = sampleBuffer.formatDescription,
              let asbd = formatDesc.audioStreamBasicDescription else { return }

        let channels = Int(asbd.mChannelsPerFrame)
        let frames = Int(sampleBuffer.numSamples)
        guard frames > 0, channels > 0 else { return }

        // Copy the audio data into a contiguous block.
        guard let blockBuffer = sampleBuffer.dataBuffer else { return }
        var lengthAtOffset = 0
        var totalLength = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer, atOffset: 0, lengthAtOffsetOut: &lengthAtOffset,
            totalLengthOut: &totalLength, dataPointerOut: &dataPointer)
        guard status == kCMBlockBufferNoErr, let dataPointer = dataPointer else { return }

        let isInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) == 0
        let floatPtr = dataPointer.withMemoryRebound(to: Float.self, capacity: totalLength / 4) { $0 }

        var mono = [Float](repeating: 0, count: frames)
        if channels == 1 {
            for i in 0..<frames { mono[i] = floatPtr[i] }
        } else if isInterleaved {
            for i in 0..<frames {
                var sum: Float = 0
                for c in 0..<channels { sum += floatPtr[i * channels + c] }
                mono[i] = sum / Float(channels)
            }
        } else {
            // Non-interleaved: channels laid out as contiguous planes.
            for i in 0..<frames {
                var sum: Float = 0
                for c in 0..<channels { sum += floatPtr[c * frames + i] }
                mono[i] = sum / Float(channels)
            }
        }

        mono.withUnsafeBufferPointer { appendSamples($0) }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write(Data("[capture] stream stopped: \(error)\n".utf8))
    }
}
