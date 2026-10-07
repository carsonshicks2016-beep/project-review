// Agent B — Swift audio-capture helper.  Brief: docs/tasks/agent-B-audio-helper.md
//
// Wires together:
//   - Capture (ScreenCaptureKit system-audio capture; one-time Screen Recording
//     permission grant required on macOS 13+; no virtual driver).
//   - DSP (Accelerate/vDSP FFT -> rms, 8 log bands, bass/mid/treble, centroid,
//     spectral-flux onset, approximate beatConfidence).
//   - WSServer (minimal RFC 6455 WebSocket on 127.0.0.1:17653, broadcasting
//     AudioFrame JSON at ~60 Hz).
//
// AudioFrame JSON shape matches src/contracts/audioFrame.ts exactly:
//   { t, rms, bands:[8], bass, mid, treble, centroid, onset, beatConfidence }

import Foundation

// AUDIO_WS_URL = ws://127.0.0.1:17653  (see src/contracts/audioFrame.ts)
let kWebSocketPort: UInt16 = 17653

func logErr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// JSON encoder shared across frames. The contract uses plain numbers; we keep
// keys in their natural (declared) order — JSON key order is not significant.
let encoder: JSONEncoder = {
    let e = JSONEncoder()
    e.outputFormatting = []
    return e
}()

// Entry point. `main.swift` permits top-level code, so we kick off an async
// task and then park the main thread on the RunLoop to keep timers and
// Network.framework callbacks serviced for the lifetime of the process.
func runHelper() async {
    logErr("[helper] Sonic Scenery audio helper starting…")

    // 1. Start the WebSocket server first so clients can connect immediately.
    let server = WSServer(port: kWebSocketPort)
    do {
        try server.start()
    } catch {
        logErr("[helper] FATAL: could not start WebSocket server on port \(kWebSocketPort): \(error)")
        exit(1)
    }

    // 2. Start system-audio capture. Permission failures exit non-zero with
    //    an actionable message.
    let capture = Capture()
    capture.onFrame = { frame in
        if let data = try? encoder.encode(frame),
           let json = String(data: data, encoding: .utf8) {
            server.broadcast(text: json)
        }
    }

    do {
        try await capture.start()
    } catch let error as CaptureError {
        logErr("[helper] FATAL: \(error.description)")
        exit(2)
    } catch {
        logErr("[helper] FATAL: capture failed to start: \(error)")
        exit(2)
    }

    logErr("[helper] running. Broadcasting AudioFrame JSON at ~60 Hz on ws://127.0.0.1:\(kWebSocketPort)")
}

// Launch the async startup, then keep the process alive. Capture + emit and the
// WebSocket server all run on their own dispatch queues; the main RunLoop just
// needs to stay resident.
Task { await runHelper() }
RunLoop.main.run()
