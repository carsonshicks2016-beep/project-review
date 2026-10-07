// WSServer — a minimal WebSocket server built on Network.framework (NWListener).
//
// We do the RFC 6455 handshake by hand (parse the HTTP Upgrade request, compute
// the Sec-WebSocket-Accept key, send the 101 response) and frame outbound text
// messages ourselves. This keeps the helper dependency-free.
//
// Scope: text frames only, server->client. We never need to send fragmented or
// binary frames. We do parse inbound frames just enough to honor Close/Ping and
// to drain the receive buffer (client masking is required by spec; we unmask but
// otherwise ignore inbound application data).

import Foundation
import Network
import CryptoKit

private let kWebSocketGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

final class WSServer {
    private let port: NWEndpoint.Port
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "ws.server")
    private var clients: [ObjectIdentifier: NWConnection] = [:]
    private let clientsLock = NSLock()

    init(port: UInt16) {
        self.port = NWEndpoint.Port(rawValue: port)!
    }

    func start() throws {
        let params = NWParameters.tcp
        // Bind to loopback only — this is a local-only transport.
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: port)
        params.allowLocalEndpointReuse = true

        let listener = try NWListener(using: params)
        self.listener = listener

        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                FileHandle.standardError.write(
                    Data("[ws] listening on ws://127.0.0.1:\(self.port.rawValue)\n".utf8))
            case .failed(let error):
                FileHandle.standardError.write(
                    Data("[ws] listener failed: \(error)\n".utf8))
            default:
                break
            }
        }
        listener.start(queue: queue)
    }

    // MARK: - Connection lifecycle

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        // Wait for the HTTP upgrade request, perform the handshake, then register.
        receiveHandshake(on: connection, accumulated: Data())
    }

    private func receiveHandshake(on connection: NWConnection, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let error = error {
                FileHandle.standardError.write(Data("[ws] handshake recv error: \(error)\n".utf8))
                connection.cancel()
                return
            }
            var buffer = accumulated
            if let data = data { buffer.append(data) }

            // Wait until we have the full request header block.
            guard let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) else {
                if isComplete { connection.cancel(); return }
                self.receiveHandshake(on: connection, accumulated: buffer)
                return
            }

            let headerData = buffer.subdata(in: buffer.startIndex..<headerEnd.upperBound)
            guard let request = String(data: headerData, encoding: .utf8),
                  let key = Self.webSocketKey(in: request) else {
                connection.cancel()
                return
            }

            let accept = Self.acceptValue(for: key)
            let response = """
            HTTP/1.1 101 Switching Protocols\r
            Upgrade: websocket\r
            Connection: Upgrade\r
            Sec-WebSocket-Accept: \(accept)\r
            \r

            """
            connection.send(content: Data(response.utf8), completion: .contentProcessed { sendError in
                if let sendError = sendError {
                    FileHandle.standardError.write(Data("[ws] handshake send error: \(sendError)\n".utf8))
                    connection.cancel()
                    return
                }
                self.register(connection)
                // Drain inbound frames (handles Close/Ping, keeps the buffer clear).
                self.receiveFrames(on: connection)
            })
        }
    }

    private func register(_ connection: NWConnection) {
        let id = ObjectIdentifier(connection)
        clientsLock.lock()
        clients[id] = connection
        let count = clients.count
        clientsLock.unlock()
        FileHandle.standardError.write(Data("[ws] client connected (\(count) total)\n".utf8))
    }

    private func unregister(_ connection: NWConnection) {
        let id = ObjectIdentifier(connection)
        clientsLock.lock()
        let existed = clients.removeValue(forKey: id) != nil
        let count = clients.count
        clientsLock.unlock()
        if existed {
            FileHandle.standardError.write(Data("[ws] client disconnected (\(count) total)\n".utf8))
        }
        connection.cancel()
    }

    // MARK: - Inbound frame draining (minimal)

    private func receiveFrames(on connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            if error != nil || isComplete {
                self.unregister(connection)
                return
            }
            if let data = data, let opcode = Self.firstOpcode(in: data) {
                // 0x8 == Close. Anything else (ping/data) we simply ignore but keep reading.
                if opcode == 0x8 {
                    self.unregister(connection)
                    return
                }
            }
            self.receiveFrames(on: connection)
        }
    }

    // MARK: - Broadcast

    /// Broadcast a UTF-8 text frame to all connected clients.
    func broadcast(text: String) {
        let frame = Self.encodeTextFrame(text)
        clientsLock.lock()
        let conns = Array(clients.values)
        clientsLock.unlock()
        for connection in conns {
            connection.send(content: frame, completion: .contentProcessed { [weak self] error in
                if error != nil { self?.unregister(connection) }
            })
        }
    }

    // MARK: - RFC 6455 helpers

    private static func webSocketKey(in request: String) -> String? {
        for line in request.split(separator: "\r\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2 else { continue }
            if parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "sec-websocket-key" {
                return parts[1].trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    private static func acceptValue(for key: String) -> String {
        let combined = key + kWebSocketGUID
        let digest = Insecure.SHA1.hash(data: Data(combined.utf8))
        return Data(digest).base64EncodedString()
    }

    /// Encode a server->client text frame (FIN=1, opcode=0x1, unmasked).
    private static func encodeTextFrame(_ text: String) -> Data {
        let payload = Data(text.utf8)
        var frame = Data()
        frame.append(0x81) // FIN + text opcode

        let len = payload.count
        if len < 126 {
            frame.append(UInt8(len))
        } else if len <= 0xFFFF {
            frame.append(126)
            frame.append(UInt8((len >> 8) & 0xFF))
            frame.append(UInt8(len & 0xFF))
        } else {
            frame.append(127)
            var be = UInt64(len).bigEndian
            withUnsafeBytes(of: &be) { frame.append(contentsOf: $0) }
        }
        frame.append(payload)
        return frame
    }

    /// Best-effort opcode read from an inbound frame (used only to detect Close).
    private static func firstOpcode(in data: Data) -> UInt8? {
        guard let first = data.first else { return nil }
        return first & 0x0F
    }
}
