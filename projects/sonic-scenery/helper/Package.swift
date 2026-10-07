// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AudioHelper",
    platforms: [.macOS(.v13)], // ScreenCaptureKit system-audio capture
    targets: [
        .executableTarget(
            name: "AudioHelper",
            path: "Sources/AudioHelper"
        )
    ]
)
