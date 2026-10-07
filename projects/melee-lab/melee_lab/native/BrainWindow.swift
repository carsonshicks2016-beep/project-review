import Cocoa
import WebKit

// A small native host for the local, read-only training visualization.
final class BrainWindow: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var timer: Timer?
    var ready = false
    var placedRun = ""
    let statePath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] :
        Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("visualizer-state.json").path
    let htmlPath = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] :
        Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("melee_lab/web/brain.html").path

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1000, height: 320),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Melee Brain · Live training & lineage"
        window.minSize = NSSize(width: 420, height: 180)
        window.backgroundColor = NSColor(calibratedRed: 0.035, green: 0.052, blue: 0.074, alpha: 1)
        window.appearance = NSAppearance(named: .darkAqua)
        window.delegate = self
        window.isReleasedWhenClosed = false
        web = WKWebView(frame: .zero)
        web.navigationDelegate = self
        web.setValue(false, forKey: "drawsBackground")
        window.contentView = web
        let url = URL(fileURLWithPath: htmlPath)
        web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        update()
        window.orderFrontRegardless()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in self?.update() }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        ready = true
        update()
    }

    func update() {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: statePath)),
              let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = state["id"] as? String else { return }
        if placedRun != id, let screen = NSScreen.screens.first {
            let visible = screen.visibleFrame
            var rect = NSRect(x: visible.midX-500, y: visible.midY-170, width: 1000, height: 340)
            if let box = state["geometry"] as? [String: Double],
               let x = box["x"], let y = box["y"], let w = box["width"], let h = box["height"] {
                rect = NSRect(x: screen.frame.minX+x, y: screen.frame.maxY-y-h, width: w, height: h)
            }
            rect.size.width = min(rect.width, visible.width)
            rect.size.height = min(rect.height, visible.height)
            rect.origin.x = max(visible.minX, min(rect.minX, visible.maxX-rect.width))
            rect.origin.y = max(visible.minY, min(rect.minY, visible.maxY-rect.height))
            window.setFrame(rect, display: true)
            placedRun = id
            window.orderFrontRegardless()
        }
        if ready, let json = String(data: data, encoding: .utf8) {
            web.evaluateJavaScript("window.renderBrain(\(json));", completionHandler: nil)
        }
    }

    func windowWillClose(_ notification: Notification) {
        timer?.invalidate()
        NSApp.terminate(nil)
    }

    // No external navigation from this local view.
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        decisionHandler(navigationAction.request.url?.isFileURL == true ? .allow : .cancel)
    }
}

let application = NSApplication.shared
let delegate = BrainWindow()
application.delegate = delegate
application.run()
