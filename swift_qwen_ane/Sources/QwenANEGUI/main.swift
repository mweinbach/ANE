import AppKit
import Charts
import Foundation
import SwiftUI
import UniformTypeIdentifiers

struct ChatMessage: Identifiable, Equatable {
    enum Role: String {
        case user
        case assistant
    }

    var id = UUID()
    let role: Role
    var content: String
    var imagePaths: [String] = []
    var reasoning: String? = nil
    var isStreaming: Bool = false
    var createdAt = Date()
}

struct PerfMetrics {
    let promptTokens: Int
    let generatedTokens: Int
    let prefillTokens: Int
    let decodeTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let totalSeconds: Double
    let prefillTPS: Double
    let decodeTPS: Double
    let endToEndTPS: Double
}

struct PowerMetrics {
    let socWatts: Double?
    let aneWatts: Double?
    let cpuWatts: Double?
    let gpuWatts: Double?
    let sampleCount: Int
    let warning: String?
}

struct PowerSeriesPoint: Identifiable {
    let id: Int
    let tSec: Double
    let socWatts: Double?
    let aneWatts: Double?
    let cpuWatts: Double?
    let gpuWatts: Double?
}

struct BackendReadyInfo {
    let prefillDevice: String
    let aneMode: String
    let aneLayers: Int
    let aneSpatial: Int
    let aneHiddenTile: Int
    let bridgeCompiles: Int
    let visionProcessorReady: Bool
    let visionProcessorStatus: String
    let visionProcessorError: String?
}

struct BackendResponse {
    let assistant: String
    let rawAssistant: String
    let reasoning: String?
    let answer: String
    let perf: PerfMetrics
    let power: PowerMetrics?
    let powerSeries: [PowerSeriesPoint]
    let bridgeCompiles: Int
    let aneKernelsCompiled: Int
    let multimodalMode: String?
}

struct BackendLaunchConfig {
    let modelID: String
    let prefillDevice: String
    let aneMode: String
    let aneLayers: Int
    let aneSpatial: Int
    let aneHiddenTile: Int
    let aneShapePolicy: String
    let aneSramTargetMB: Double
    let aneTileMultiple: Int
    let aneMinHiddenTile: Int
    let dtype: String
    let powermetricsSampleRateMs: Int
    let powermetricsSamplers: String
    let requestSudoAccess: Bool
}

struct GenerateConfig {
    let maxNewTokens: Int
    let temperature: Double
    let topP: Double
    let topK: Int
    let contextWindow: Int
    let powermetrics: Bool
    let stream: Bool
    let reasoningEnabled: Bool
}

struct AutoTuneConfig {
    let warmup: Int
    let iters: Int
    let peakTFLOPS: Double
    let seed: Int
    let spatialValues: [Int]
    let tileHiddenValues: [Int]
}

struct AutoTuneCandidate: Identifiable {
    let id: String
    let spatial: Int
    let tileHidden: Int
    let evalMs: Double
    let utilPct: Double
}

struct AutoTuneResult {
    let moduleName: String
    let dim: Int
    let hidden: Int
    let mode: String
    let warmup: Int
    let iters: Int
    let bestSpatial: Int
    let bestTileHidden: Int
    let bestEvalMs: Double
    let bestUtilPct: Double
    let recommendedSpatial: Int
    let recommendedHiddenTile: Int
    let bridgeCompiles: Int
    let candidates: [AutoTuneCandidate]
}

struct ModelCatalogEntry: Identifiable, Hashable {
    let id: String
    let title: String
    let repoID: String
    let localPath: String?

    var isLocalAvailable: Bool {
        if let localPath {
            return !localPath.isEmpty
        }
        return false
    }
}

enum BackendError: LocalizedError {
    case backendScriptNotFound
    case processNotRunning
    case backendNotReady
    case malformedEvent(String)
    case requestFailed(String)

    var errorDescription: String? {
        switch self {
        case .backendScriptNotFound:
            return "Could not locate qwen_ane/gui_backend.py from current working directory or executable path"
        case .processNotRunning:
            return "Backend process is not running"
        case .backendNotReady:
            return "Backend not ready yet"
        case .malformedEvent(let line):
            return "Malformed backend event: \(line)"
        case .requestFailed(let msg):
            return msg
        }
    }
}

final class BackendClient {
    var onStatus: ((String) -> Void)?
    var onReady: ((BackendReadyInfo) -> Void)?

    private let lock = NSLock()
    private var process: Process?
    private var stdinHandle: FileHandle?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var askpassScriptURL: URL?
    private var stdoutBuffer = Data()
    private var stderrBuffer = Data()
    private var pending: [String: CheckedContinuation<BackendResponse, Error>] = [:]
    private var pendingAutoTune: [String: CheckedContinuation<AutoTuneResult, Error>] = [:]
    private var chunkHandlers: [String: (String, String, String?, String, Int) -> Void] = [:]
    private var isReady = false

    deinit {
        stop()
    }

    func start(config: BackendLaunchConfig) throws {
        stop()

        guard let backendScript = resolveBackendScriptPath() else {
            throw BackendError.backendScriptNotFound
        }

        let proc = Process()
        let backendArgs = [
            "--model-id", config.modelID,
            "--prefill-device", config.prefillDevice,
            "--ane-mode", config.aneMode,
            "--ane-layers", "\(config.aneLayers)",
            "--ane-spatial", "\(config.aneSpatial)",
            "--ane-hidden-tile", "\(config.aneHiddenTile)",
            "--ane-shape-policy", config.aneShapePolicy,
            "--ane-sram-target-mb", String(format: "%.1f", config.aneSramTargetMB),
            "--ane-tile-multiple", "\(config.aneTileMultiple)",
            "--ane-min-hidden-tile", "\(config.aneMinHiddenTile)",
            "--dtype", config.dtype,
            "--powermetrics-sample-rate-ms", "\(config.powermetricsSampleRateMs)",
            "--powermetrics-samplers", config.powermetricsSamplers,
        ]

        if config.requestSudoAccess {
            let askpass = try ensureAskpassScript()
            emitStatus("Requesting administrator access for powermetrics")
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/sudo")
            proc.arguments = ["-A", "/usr/bin/env", "python3", backendScript.path] + backendArgs
            var env = ProcessInfo.processInfo.environment
            env["SUDO_ASKPASS"] = askpass.path
            proc.environment = env
        } else {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            proc.arguments = ["python3", backendScript.path] + backendArgs
        }

        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = stderr

        proc.terminationHandler = { [weak self] process in
            self?.handleTermination(status: process.terminationStatus)
        }

        try proc.run()

        lock.lock()
        process = proc
        stdinHandle = stdin.fileHandleForWriting
        stdoutPipe = stdout
        stderrPipe = stderr
        stdoutBuffer.removeAll(keepingCapacity: true)
        stderrBuffer.removeAll(keepingCapacity: true)
        isReady = false
        lock.unlock()

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { return }
            self?.consumeStdout(data)
        }

        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { return }
            self?.consumeStderr(data)
        }

        emitStatus("Backend starting (pid \(proc.processIdentifier))")
    }

    func stop() {
        let proc: Process?
        let input: FileHandle?
        let pendingContinuations: [CheckedContinuation<BackendResponse, Error>]
        let pendingAutoTuneContinuations: [CheckedContinuation<AutoTuneResult, Error>]
        let stdout: Pipe?
        let stderr: Pipe?

        lock.lock()
        proc = process
        input = stdinHandle
        pendingContinuations = Array(pending.values)
        pendingAutoTuneContinuations = Array(pendingAutoTune.values)
        pending.removeAll()
        pendingAutoTune.removeAll()
        chunkHandlers.removeAll()
        process = nil
        stdinHandle = nil
        stdout = stdoutPipe
        stderr = stderrPipe
        stdoutPipe = nil
        stderrPipe = nil
        isReady = false
        lock.unlock()

        stdout?.fileHandleForReading.readabilityHandler = nil
        stderr?.fileHandleForReading.readabilityHandler = nil

        for continuation in pendingContinuations {
            continuation.resume(throwing: BackendError.processNotRunning)
        }
        for continuation in pendingAutoTuneContinuations {
            continuation.resume(throwing: BackendError.processNotRunning)
        }

        if let input {
            let shutdownReq: [String: Any] = ["type": "shutdown", "id": UUID().uuidString]
            _ = try? writeJSON(shutdownReq, to: input)
            try? input.close()
        }

        if let proc, proc.isRunning {
            proc.terminate()
        }

        cleanupAskpassScript()
        emitStatus("Backend stopped")
    }

    func generate(
        messages: [[String: Any]],
        config: GenerateConfig,
        onChunk: ((String, String, String?, String, Int) -> Void)? = nil
    ) async throws -> BackendResponse {
        let requestID = UUID().uuidString

        let request: [String: Any] = [
            "id": requestID,
            "type": "generate",
            "messages": messages,
            "max_new_tokens": config.maxNewTokens,
            "temperature": config.temperature,
            "top_p": config.topP,
            "top_k": config.topK,
            "context_window": config.contextWindow,
            "powermetrics": config.powermetrics,
            "stream": config.stream,
            "reasoning_enabled": config.reasoningEnabled,
        ]

        return try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            guard process != nil, let input = stdinHandle else {
                lock.unlock()
                continuation.resume(throwing: BackendError.processNotRunning)
                return
            }
            guard isReady else {
                lock.unlock()
                continuation.resume(throwing: BackendError.backendNotReady)
                return
            }
            pending[requestID] = continuation
            if let onChunk {
                chunkHandlers[requestID] = onChunk
            }
            lock.unlock()

            do {
                try writeJSON(request, to: input)
            } catch {
                lock.lock()
                let c = pending.removeValue(forKey: requestID)
                _ = chunkHandlers.removeValue(forKey: requestID)
                lock.unlock()
                c?.resume(throwing: error)
            }
        }
    }

    func autotune(config: AutoTuneConfig) async throws -> AutoTuneResult {
        let requestID = UUID().uuidString
        let request: [String: Any] = [
            "id": requestID,
            "type": "autotune",
            "warmup": config.warmup,
            "iters": config.iters,
            "peak_tflops": config.peakTFLOPS,
            "seed": config.seed,
            "spatial_values": config.spatialValues,
            "tile_hidden_values": config.tileHiddenValues,
        ]

        return try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            guard process != nil, let input = stdinHandle else {
                lock.unlock()
                continuation.resume(throwing: BackendError.processNotRunning)
                return
            }
            guard isReady else {
                lock.unlock()
                continuation.resume(throwing: BackendError.backendNotReady)
                return
            }
            pendingAutoTune[requestID] = continuation
            lock.unlock()

            do {
                try writeJSON(request, to: input)
            } catch {
                lock.lock()
                let c = pendingAutoTune.removeValue(forKey: requestID)
                lock.unlock()
                c?.resume(throwing: error)
            }
        }
    }

    private func handleTermination(status: Int32) {
        let continuations: [CheckedContinuation<BackendResponse, Error>]
        let autoTuneContinuations: [CheckedContinuation<AutoTuneResult, Error>]
        lock.lock()
        continuations = Array(pending.values)
        autoTuneContinuations = Array(pendingAutoTune.values)
        pending.removeAll()
        pendingAutoTune.removeAll()
        chunkHandlers.removeAll()
        process = nil
        stdinHandle = nil
        stdoutPipe = nil
        stderrPipe = nil
        isReady = false
        lock.unlock()

        for c in continuations {
            c.resume(throwing: BackendError.processNotRunning)
        }
        for c in autoTuneContinuations {
            c.resume(throwing: BackendError.processNotRunning)
        }
        cleanupAskpassScript()
        emitStatus("Backend exited with status \(status)")
    }

    private func emitStatus(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onStatus?(text)
        }
    }

    private func consumeStdout(_ data: Data) {
        lock.lock()
        stdoutBuffer.append(data)
        let lines = drainLines(from: &stdoutBuffer)
        lock.unlock()

        for line in lines {
            handleStdoutLine(line)
        }
    }

    private func consumeStderr(_ data: Data) {
        lock.lock()
        stderrBuffer.append(data)
        let lines = drainLines(from: &stderrBuffer)
        lock.unlock()

        for line in lines where !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if parseEventLine(line) != nil {
                handleStdoutLine(line)
                continue
            }
            emitStatus(line)
        }
    }

    private func drainLines(from buffer: inout Data) -> [String] {
        var lines: [String] = []
        let newline = Data([0x0A])
        while let range = buffer.range(of: newline) {
            let lineData = buffer.subdata(in: 0..<range.lowerBound)
            buffer.removeSubrange(0..<range.upperBound)
            if let line = String(data: lineData, encoding: .utf8) {
                lines.append(line)
            }
        }
        return lines
    }

    private func handleStdoutLine(_ line: String) {
        guard let event = parseEventLine(line),
              let eventType = event["event"] as? String
        else {
            emitStatus("[backend-json] \(line)")
            return
        }

        switch eventType {
        case "ready":
            let info = BackendReadyInfo(
                prefillDevice: asString(event["prefill_device"]) ?? "unknown",
                aneMode: asString(event["ane_mode"]) ?? "unknown",
                aneLayers: asInt(event["ane_layers"]) ?? 0,
                aneSpatial: asInt(event["ane_spatial"]) ?? 0,
                aneHiddenTile: asInt(event["ane_hidden_tile"]) ?? 0,
                bridgeCompiles: asInt(event["bridge_compiles"]) ?? 0,
                visionProcessorReady: asBool(event["vision_processor_ready"]) ?? false,
                visionProcessorStatus: asString(event["vision_processor_status"]) ?? "unknown",
                visionProcessorError: asString(event["vision_processor_error"])
            )
            lock.lock()
            isReady = true
            lock.unlock()
            DispatchQueue.main.async { [weak self] in
                self?.onReady?(info)
            }
            emitStatus(
                "Backend ready (mode=\(info.aneMode), prefill=\(info.prefillDevice), vision=\(info.visionProcessorStatus))"
            )

        case "response_start":
            if let reqID = asString(event["id"]) {
                emitStatus("Streaming response (\(reqID.prefix(6)))")
            }

        case "response_chunk":
            guard let reqID = asString(event["id"]) else {
                emitStatus("response_chunk missing id")
                return
            }
            let delta = asString(event["delta"]) ?? ""
            let text = asString(event["text"]) ?? ""
            let reasoning = asString(event["reasoning"])
            let answer = asString(event["answer"]) ?? ""
            let generated = asInt(event["generated_tokens"]) ?? 0
            emitChunk(
                id: reqID,
                delta: delta,
                text: text,
                reasoning: reasoning,
                answer: answer,
                generatedTokens: generated
            )

        case "response":
            guard let reqID = asString(event["id"]) else {
                emitStatus("response missing id")
                return
            }
            guard let perfObj = event["perf"] as? [String: Any] else {
                resumeRequest(id: reqID, with: .failure(BackendError.malformedEvent(line)))
                return
            }

            let perf = PerfMetrics(
                promptTokens: asInt(perfObj["prompt_tokens"]) ?? 0,
                generatedTokens: asInt(perfObj["generated_tokens"]) ?? 0,
                prefillTokens: asInt(perfObj["prefill_tokens"]) ?? 0,
                decodeTokens: asInt(perfObj["decode_tokens"]) ?? 0,
                prefillSeconds: asDouble(perfObj["prefill_seconds"]) ?? 0,
                decodeSeconds: asDouble(perfObj["decode_seconds"]) ?? 0,
                totalSeconds: asDouble(perfObj["total_seconds"]) ?? 0,
                prefillTPS: asDouble(perfObj["prefill_tps"]) ?? 0,
                decodeTPS: asDouble(perfObj["decode_tps"]) ?? 0,
                endToEndTPS: asDouble(perfObj["e2e_tps"]) ?? 0
            )

            let power: PowerMetrics?
            if let p = event["power"] as? [String: Any] {
                power = PowerMetrics(
                    socWatts: asDouble(p["soc_watts"]),
                    aneWatts: asDouble(p["ane_watts"]),
                    cpuWatts: asDouble(p["cpu_watts"]),
                    gpuWatts: asDouble(p["gpu_watts"]),
                    sampleCount: asInt(p["sample_count"]) ?? 0,
                    warning: asString(p["warning"])
                )
            } else {
                power = nil
            }

            let powerSeries = parsePowerSeries(event["power_series"])
            let answer = asString(event["answer"]) ?? asString(event["assistant"]) ?? ""
            let payload = BackendResponse(
                assistant: asString(event["assistant"]) ?? answer,
                rawAssistant: asString(event["raw_assistant"]) ?? asString(event["assistant"]) ?? "",
                reasoning: asString(event["reasoning"]),
                answer: answer,
                perf: perf,
                power: power,
                powerSeries: powerSeries,
                bridgeCompiles: asInt(event["bridge_compiles"]) ?? 0,
                aneKernelsCompiled: asInt(event["ane_kernels_compiled"]) ?? 0,
                multimodalMode: asString(event["multimodal_mode"])
            )
            resumeRequest(id: reqID, with: .success(payload))

        case "autotune_result":
            guard let reqID = asString(event["id"]) else {
                emitStatus("autotune_result missing id")
                return
            }
            guard let shape = event["shape"] as? [String: Any],
                  let best = event["best"] as? [String: Any],
                  let recommended = event["recommended"] as? [String: Any]
            else {
                resumeAutoTune(id: reqID, with: .failure(BackendError.malformedEvent(line)))
                return
            }

            let moduleName = asString(shape["module"]) ?? "unknown"
            let dim = asInt(shape["dim"]) ?? 0
            let hidden = asInt(shape["hidden"]) ?? 0
            let mode = asString(event["mode"]) ?? "unknown"
            let warmup = asInt(event["warmup"]) ?? 0
            let iters = asInt(event["iters"]) ?? 0
            let bestSpatial = asInt(best["spatial"]) ?? 0
            let bestTileHidden = asInt(best["tile_hidden"]) ?? 0
            let bestEvalMs = asDouble(best["eval_ms"]) ?? 0
            let bestUtilPct = asDouble(best["util_pct"]) ?? 0
            let recSpatial = asInt(recommended["ane_spatial"]) ?? bestSpatial
            let recTile = asInt(recommended["ane_hidden_tile"]) ?? bestTileHidden
            let bridgeCompiles = asInt(event["bridge_compiles"]) ?? 0

            let candidateRows = (event["candidates"] as? [[String: Any]] ?? []).enumerated().map { idx, row in
                AutoTuneCandidate(
                    id: "\(idx)-\(asInt(row["spatial"]) ?? 0)-\(asInt(row["tile_hidden"]) ?? 0)",
                    spatial: asInt(row["spatial"]) ?? 0,
                    tileHidden: asInt(row["tile_hidden"]) ?? 0,
                    evalMs: asDouble(row["eval_ms"]) ?? 0,
                    utilPct: asDouble(row["util_pct"]) ?? 0
                )
            }

            let payload = AutoTuneResult(
                moduleName: moduleName,
                dim: dim,
                hidden: hidden,
                mode: mode,
                warmup: warmup,
                iters: iters,
                bestSpatial: bestSpatial,
                bestTileHidden: bestTileHidden,
                bestEvalMs: bestEvalMs,
                bestUtilPct: bestUtilPct,
                recommendedSpatial: recSpatial,
                recommendedHiddenTile: recTile,
                bridgeCompiles: bridgeCompiles,
                candidates: candidateRows
            )
            resumeAutoTune(id: reqID, with: .success(payload))

        case "error":
            let msg = asString(event["error"]) ?? "Unknown backend error"
            if let reqID = asString(event["id"]) {
                let handledGenerate = resumeRequest(id: reqID, with: .failure(BackendError.requestFailed(msg)))
                if !handledGenerate {
                    let handledTune = resumeAutoTune(id: reqID, with: .failure(BackendError.requestFailed(msg)))
                    if !handledTune {
                        emitStatus("backend error for unknown request id: \(reqID) \(msg)")
                    }
                }
            } else {
                emitStatus("backend error: \(msg)")
            }

        case "shutdown":
            emitStatus("backend acknowledged shutdown")

        default:
            emitStatus("backend event: \(eventType)")
        }
    }

    private func parseEventLine(_ line: String) -> [String: Any]? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let event = parseEventJSON(trimmed) {
            return event
        }

        guard let start = trimmed.firstIndex(of: "{"),
              let end = trimmed.lastIndex(of: "}"),
              start <= end
        else {
            return nil
        }

        let candidate = String(trimmed[start...end])
        return parseEventJSON(candidate)
    }

    private func parseEventJSON(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8),
              let raw = try? JSONSerialization.jsonObject(with: data),
              let event = raw as? [String: Any],
              event["event"] != nil
        else {
            return nil
        }
        return event
    }

    @discardableResult
    private func resumeRequest(id: String, with result: Result<BackendResponse, Error>) -> Bool {
        lock.lock()
        let continuation = pending.removeValue(forKey: id)
        _ = chunkHandlers.removeValue(forKey: id)
        lock.unlock()

        guard let continuation else {
            return false
        }

        switch result {
        case .success(let response):
            continuation.resume(returning: response)
        case .failure(let error):
            continuation.resume(throwing: error)
        }
        return true
    }

    @discardableResult
    private func resumeAutoTune(id: String, with result: Result<AutoTuneResult, Error>) -> Bool {
        lock.lock()
        let continuation = pendingAutoTune.removeValue(forKey: id)
        lock.unlock()

        guard let continuation else {
            return false
        }

        switch result {
        case .success(let response):
            continuation.resume(returning: response)
        case .failure(let error):
            continuation.resume(throwing: error)
        }
        return true
    }

    private func emitChunk(
        id: String,
        delta: String,
        text: String,
        reasoning: String?,
        answer: String,
        generatedTokens: Int
    ) {
        let handler: ((String, String, String?, String, Int) -> Void)?
        lock.lock()
        handler = chunkHandlers[id]
        lock.unlock()

        guard let handler else { return }
        DispatchQueue.main.async {
            handler(delta, text, reasoning, answer, generatedTokens)
        }
    }

    private func writeJSON(_ payload: [String: Any], to handle: FileHandle) throws {
        let body = try JSONSerialization.data(withJSONObject: payload)
        var line = Data()
        line.append(body)
        line.append(0x0A)
        try handle.write(contentsOf: line)
    }

    private func asString(_ value: Any?) -> String? {
        if let v = value as? String {
            return v
        }
        if let v = value as? NSNumber {
            return v.stringValue
        }
        return nil
    }

    private func asInt(_ value: Any?) -> Int? {
        if let v = value as? Int {
            return v
        }
        if let v = value as? NSNumber {
            return v.intValue
        }
        if let v = value as? String {
            return Int(v)
        }
        return nil
    }

    private func asDouble(_ value: Any?) -> Double? {
        if let v = value as? Double {
            return v
        }
        if let v = value as? NSNumber {
            return v.doubleValue
        }
        if let v = value as? String {
            return Double(v)
        }
        return nil
    }

    private func asBool(_ value: Any?) -> Bool? {
        if let v = value as? Bool {
            return v
        }
        if let v = value as? NSNumber {
            return v.boolValue
        }
        if let v = value as? String {
            if v == "1" || v.lowercased() == "true" {
                return true
            }
            if v == "0" || v.lowercased() == "false" {
                return false
            }
        }
        return nil
    }

    private func parsePowerSeries(_ value: Any?) -> [PowerSeriesPoint] {
        guard let rows = value as? [[String: Any]] else { return [] }
        return rows.enumerated().map { idx, row in
            PowerSeriesPoint(
                id: asInt(row["index"]) ?? idx,
                tSec: asDouble(row["t_sec"]) ?? Double(idx),
                socWatts: asDouble(row["soc_watts"]),
                aneWatts: asDouble(row["ane_watts"]),
                cpuWatts: asDouble(row["cpu_watts"]),
                gpuWatts: asDouble(row["gpu_watts"])
            )
        }
    }

    private func resolveBackendScriptPath() -> URL? {
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment

        if let explicitScript = env["QWEN_ANE_BACKEND_SCRIPT"], !explicitScript.isEmpty {
            let url = URL(fileURLWithPath: explicitScript).resolvingSymlinksInPath()
            if fm.fileExists(atPath: url.path) {
                return url
            }
        }

        if let explicitRoot = env["QWEN_ANE_REPO_ROOT"], !explicitRoot.isEmpty {
            let script = URL(fileURLWithPath: explicitRoot)
                .resolvingSymlinksInPath()
                .appendingPathComponent("qwen_ane/gui_backend.py")
            if fm.fileExists(atPath: script.path) {
                return script
            }
        }

        var candidates: [URL] = []

        candidates.append(URL(fileURLWithPath: fm.currentDirectoryPath))

        let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        candidates.append(executableURL.deletingLastPathComponent())

        for base in candidates {
            for root in ascend(from: base, levels: 8) {
                let script = root.appendingPathComponent("qwen_ane/gui_backend.py")
                if fm.fileExists(atPath: script.path) {
                    return script
                }
            }
        }

        return nil
    }

    private func ascend(from start: URL, levels: Int) -> [URL] {
        var urls: [URL] = [start]
        var current = start
        if levels < 1 { return urls }
        for _ in 0..<levels {
            current = current.deletingLastPathComponent()
            urls.append(current)
        }
        return urls
    }

    private func ensureAskpassScript() throws -> URL {
        if let existing = askpassScriptURL,
           FileManager.default.fileExists(atPath: existing.path)
        {
            return existing
        }

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwen_ane_askpass_\(UUID().uuidString).sh")
        let script = """
        #!/bin/zsh
        /usr/bin/osascript <<'APPLESCRIPT'
        tell application "System Events"
            activate
            try
                set d to display dialog "Qwen ANE GUI needs administrator access to capture powermetrics (SoC/ANE watts)." with title "Qwen ANE GUI" default answer "" with hidden answer buttons {"Cancel", "OK"} default button "OK"
                return text returned of d
            on error number -128
                error number 1
            end try
        end tell
        APPLESCRIPT
        """
        try script.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
        askpassScriptURL = url
        return url
    }

    private func cleanupAskpassScript() {
        guard let url = askpassScriptURL else { return }
        try? FileManager.default.removeItem(at: url)
        askpassScriptURL = nil
    }
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText = ""
    @Published var pendingImagePaths: [String] = []
    @Published var systemPrompt = ""

    @Published var modelCatalog: [ModelCatalogEntry] = []
    @Published var selectedCatalogModelRepoID = "mlx-community/Qwen3.5-4B-mxfp4"
    @Published var modelCatalogStatus = "Catalog not loaded"
    @Published var isDownloadingModel = false
    @Published var isProbingModel = false
    @Published var modelProbeStatus = "No probe run yet"

    @Published var modelID: String = ChatViewModel.defaultModelID()
    @Published var prefillDevice = "mps"
    @Published var aneMode = "mlp_tiled"
    @Published var aneLayers = 12
    @Published var aneSpatial = 32
    @Published var aneHiddenTile = 2048
    @Published var aneShapePolicy = "auto"
    @Published var aneSramTargetMB = 30.0
    @Published var aneTileMultiple = 256
    @Published var aneMinHiddenTile = 512
    @Published var dtype = "fp16"

    @Published var contextWindow = 0
    @Published var maxNewTokens = 256
    @Published var temperature = 0.7
    @Published var topP = 0.9
    @Published var topK = 0
    @Published var streamResponses = true
    @Published var reasoningEnabled = false
    @Published var showReasoningPanels = true
    @Published var powermetricsEnabled = true
    @Published var autoTuneOnStartup = false

    @Published var powermetricsSampleRateMs = 500
    @Published var powermetricsSamplers = "cpu_power,gpu_power,ane_power"

    @Published var backendStatus = "Not started"
    @Published var backendReady = false
    @Published var isGenerating = false
    @Published var isAutoTuning = false

    @Published var latestPerf: PerfMetrics?
    @Published var latestPower: PowerMetrics?
    @Published var latestPowerSeries: [PowerSeriesPoint] = []
    @Published var bridgeCompiles = 0
    @Published var aneKernelsCompiled = 0
    @Published var visionProcessorReady = false
    @Published var visionProcessorStatus = "unknown"
    @Published var visionProcessorError = ""
    @Published var latestMultimodalMode = "text"
    @Published var autoTuneStatus = "Not tuned"
    @Published var lastAutoTune: AutoTuneResult?
    @Published var lastError = ""
    @Published var backendUsesSudo = false

    private let backend = BackendClient()
    private var startupAutoTuneDidRun = false

    init() {
        backend.onStatus = { [weak self] status in
            Task { @MainActor in
                self?.backendStatus = status
            }
        }

        backend.onReady = { [weak self] info in
            Task { @MainActor in
                guard let self else { return }
                self.backendReady = true
                self.bridgeCompiles = info.bridgeCompiles
                self.visionProcessorReady = info.visionProcessorReady
                self.visionProcessorStatus = info.visionProcessorStatus
                self.visionProcessorError = info.visionProcessorError ?? ""
                self.lastError = ""
                if self.autoTuneOnStartup, !self.startupAutoTuneDidRun, !self.isAutoTuning {
                    self.startupAutoTuneDidRun = true
                    self.runAutoTune(applyAndRestart: true)
                }
            }
        }

        refreshModelCatalog(applyBestAvailable: true)
        Task {
            await startBackend()
        }
    }

    deinit {
        backend.stop()
    }

    func startBackend() async {
        isGenerating = false
        backendReady = false
        latestPerf = nil
        latestPower = nil
        latestPowerSeries = []
        visionProcessorReady = false
        visionProcessorStatus = "unknown"
        visionProcessorError = ""
        latestMultimodalMode = "text"
        lastError = ""

        let config = BackendLaunchConfig(
            modelID: modelID,
            prefillDevice: prefillDevice,
            aneMode: aneMode,
            aneLayers: max(1, aneLayers),
            aneSpatial: max(1, aneSpatial),
            aneHiddenTile: max(1, aneHiddenTile),
            aneShapePolicy: aneShapePolicy,
            aneSramTargetMB: max(1.0, aneSramTargetMB),
            aneTileMultiple: max(1, aneTileMultiple),
            aneMinHiddenTile: max(1, aneMinHiddenTile),
            dtype: dtype,
            powermetricsSampleRateMs: max(10, powermetricsSampleRateMs),
            powermetricsSamplers: powermetricsSamplers.isEmpty ? "cpu_power,gpu_power,ane_power" : powermetricsSamplers,
            requestSudoAccess: powermetricsEnabled
        )
        backendUsesSudo = config.requestSudoAccess

        do {
            try backend.start(config: config)
        } catch {
            backendStatus = "Failed to start backend"
            lastError = error.localizedDescription
        }
    }

    func stopBackend() {
        backend.stop()
        backendReady = false
        isGenerating = false
        isAutoTuning = false
        visionProcessorReady = false
        visionProcessorStatus = "unknown"
        visionProcessorError = ""
        latestMultimodalMode = "text"
    }

    func clearChat() {
        messages.removeAll()
        pendingImagePaths.removeAll()
    }

    func refreshModelCatalog(applyBestAvailable: Bool = false) {
        let entries = Self.curatedModels.map { curated in
            ModelCatalogEntry(
                id: curated.repoID,
                title: curated.title,
                repoID: curated.repoID,
                localPath: Self.findCachedSnapshotPath(for: curated.repoID)
            )
        }
        modelCatalog = entries
        let localCount = entries.filter { $0.isLocalAvailable }.count
        modelCatalogStatus = "Local models: \(localCount)/\(entries.count)"
        if applyBestAvailable {
            useBestAvailableCatalogModel()
        }
    }

    func useSelectedCatalogModel() {
        guard let entry = modelCatalog.first(where: { $0.repoID == selectedCatalogModelRepoID }) else {
            modelCatalogStatus = "Selected model not found in catalog"
            return
        }
        if let path = entry.localPath {
            modelID = path
            modelCatalogStatus = "Using local snapshot for \(entry.repoID)"
        } else {
            modelID = entry.repoID
            modelCatalogStatus = "Using remote repo ID for \(entry.repoID)"
        }
    }

    func useBestAvailableCatalogModel() {
        guard !modelCatalog.isEmpty else {
            modelCatalogStatus = "Catalog is empty"
            return
        }
        if let bestLocal = modelCatalog.first(where: { $0.isLocalAvailable }), let path = bestLocal.localPath {
            selectedCatalogModelRepoID = bestLocal.repoID
            modelID = path
            modelCatalogStatus = "Best available local model: \(bestLocal.repoID)"
            return
        }
        if let selected = modelCatalog.first(where: { $0.repoID == selectedCatalogModelRepoID }) {
            modelID = selected.repoID
            modelCatalogStatus = "No local snapshots found; using remote repo ID"
            return
        }
        if let first = modelCatalog.first {
            selectedCatalogModelRepoID = first.repoID
            modelID = first.repoID
            modelCatalogStatus = "No local snapshots found; defaulted to \(first.repoID)"
        }
    }

    func downloadSelectedCatalogModel() {
        guard !isDownloadingModel else { return }
        guard !isGenerating else {
            lastError = "Wait for generation to finish before downloading a model"
            return
        }
        guard !isAutoTuning else {
            lastError = "Wait for auto-tune to finish before downloading a model"
            return
        }
        let repoID = selectedCatalogModelRepoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repoID.isEmpty else {
            lastError = "Select a catalog model first"
            return
        }

        isDownloadingModel = true
        modelCatalogStatus = "Downloading \(repoID)…"
        lastError = ""

        Task.detached(priority: .userInitiated) {
            do {
                let snapshotPath = try Self.snapshotDownload(repoID: repoID)
                await MainActor.run {
                    self.isDownloadingModel = false
                    self.refreshModelCatalog(applyBestAvailable: false)
                    self.selectedCatalogModelRepoID = repoID
                    self.modelID = snapshotPath
                    self.modelCatalogStatus = "Downloaded \(repoID) to local snapshot"
                }
            } catch {
                await MainActor.run {
                    self.isDownloadingModel = false
                    self.modelCatalogStatus = "Download failed"
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    func probeSelectedCatalogModel() {
        guard !isProbingModel else { return }
        guard !isGenerating else {
            lastError = "Wait for generation to finish before probing a model"
            return
        }
        guard !isAutoTuning else {
            lastError = "Wait for auto-tune to finish before probing a model"
            return
        }
        guard !isDownloadingModel else {
            lastError = "Wait for model download to finish before probing"
            return
        }

        let modelRef = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelRef.isEmpty else {
            lastError = "Model ID / Path is empty"
            return
        }

        isProbingModel = true
        lastError = ""
        modelProbeStatus = "Probing selected model with backend warm-start…"

        let probePrefillDevice = prefillDevice
        let probeAneMode = aneMode
        let probeAneSpatial = max(1, aneSpatial)
        let probeAneHiddenTile = max(1, aneHiddenTile)
        let probeAneShapePolicy = aneShapePolicy
        let probeSramTarget = max(1.0, aneSramTargetMB)
        let probeTileMultiple = max(1, aneTileMultiple)
        let probeMinHiddenTile = max(1, aneMinHiddenTile)
        let probeDType = dtype
        let probePowermetricsRate = max(10, powermetricsSampleRateMs)
        let probePowermetricsSamplers = powermetricsSamplers.isEmpty ? "cpu_power,gpu_power,ane_power" : powermetricsSamplers

        Task.detached(priority: .userInitiated) {
            do {
                let output = try Self.runBackendWarmStartProbe(
                    modelID: modelRef,
                    prefillDevice: probePrefillDevice,
                    aneMode: probeAneMode,
                    aneSpatial: probeAneSpatial,
                    aneHiddenTile: probeAneHiddenTile,
                    aneShapePolicy: probeAneShapePolicy,
                    aneSramTargetMB: probeSramTarget,
                    aneTileMultiple: probeTileMultiple,
                    aneMinHiddenTile: probeMinHiddenTile,
                    dtype: probeDType,
                    powermetricsSampleRateMs: probePowermetricsRate,
                    powermetricsSamplers: probePowermetricsSamplers
                )
                await MainActor.run {
                    self.isProbingModel = false
                    self.modelProbeStatus = output
                }
            } catch {
                await MainActor.run {
                    self.isProbingModel = false
                    self.modelProbeStatus = "Probe failed"
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    func runAutoTune(applyAndRestart: Bool = true) {
        guard backendReady else {
            lastError = "Backend is not ready"
            return
        }
        guard !isGenerating else {
            lastError = "Wait for the current generation to finish before autotune"
            return
        }
        guard !isAutoTuning else { return }

        isAutoTuning = true
        lastError = ""
        autoTuneStatus = "Benchmarking ANE decode kernels…"

        let tileCandidates: [Int]
        if aneMode == "mlp_tiled" {
            tileCandidates = [1024, 1536, 2048, 2560]
        } else {
            tileCandidates = [max(1, aneHiddenTile)]
        }

        let tuneConfig = AutoTuneConfig(
            warmup: 5,
            iters: 80,
            peakTFLOPS: 15.8,
            seed: 7,
            spatialValues: [16, 24, 32, 40, 48, 64],
            tileHiddenValues: tileCandidates
        )

        Task {
            do {
                let result = try await backend.autotune(config: tuneConfig)
                lastAutoTune = result
                autoTuneStatus = String(
                    format: "Best: spatial=%d tile=%d eval=%.3fms util=%.2f%%",
                    result.bestSpatial,
                    result.bestTileHidden,
                    result.bestEvalMs,
                    result.bestUtilPct
                )

                guard applyAndRestart else {
                    isAutoTuning = false
                    return
                }

                aneSpatial = max(1, result.recommendedSpatial)
                if aneMode == "mlp_tiled" {
                    aneHiddenTile = max(1, result.recommendedHiddenTile)
                }
                autoTuneStatus = "Applied tuned shape; restarting backend…"
                isAutoTuning = false
                await startBackend()
            } catch {
                lastError = "Autotune failed: \(error.localizedDescription)"
                autoTuneStatus = "Autotune failed"
                isAutoTuning = false
            }
        }
    }

    func sendCurrentInput() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let images = pendingImagePaths
        guard !text.isEmpty || !images.isEmpty else { return }
        guard !isGenerating else { return }

        inputText = ""
        pendingImagePaths.removeAll()
        messages.append(ChatMessage(role: .user, content: text, imagePaths: images))
        generateAssistantReply()
    }

    func selectImagesForCurrentInput() {
        guard !isGenerating else { return }

        let panel = NSOpenPanel()
        panel.title = "Select Images"
        panel.prompt = "Attach"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image]

        if panel.runModal() == .OK {
            let selected = panel.urls.map { $0.resolvingSymlinksInPath().path }
            appendPendingImagePaths(selected)
        }
    }

    func handleDroppedItemProviders(_ providers: [NSItemProvider]) -> Bool {
        let typeID = UTType.fileURL.identifier
        let matches = providers.filter { $0.hasItemConformingToTypeIdentifier(typeID) }
        guard !matches.isEmpty else { return false }

        for provider in matches {
            provider.loadItem(forTypeIdentifier: typeID, options: nil) { [weak self] item, _error in
                guard let self else { return }
                guard let url = Self.decodeDroppedFileURL(item),
                      Self.isImageURL(url)
                else {
                    return
                }
                let path = url.resolvingSymlinksInPath().path
                Task { @MainActor in
                    self.appendPendingImagePaths([path])
                }
            }
        }
        return true
    }

    func removePendingImage(path: String) {
        pendingImagePaths.removeAll { $0 == path }
    }

    private func appendPendingImagePaths(_ paths: [String]) {
        guard !paths.isEmpty else { return }
        var seen = Set(pendingImagePaths)
        for path in paths where !seen.contains(path) {
            pendingImagePaths.append(path)
            seen.insert(path)
        }
    }

    nonisolated private static func decodeDroppedFileURL(_ item: NSSecureCoding?) -> URL? {
        if let url = item as? URL {
            return url
        }
        if let data = item as? Data {
            if let url = URL(dataRepresentation: data, relativeTo: nil) {
                return url
            }
            if let text = String(data: data, encoding: .utf8),
               let url = URL(string: text.trimmingCharacters(in: .whitespacesAndNewlines))
            {
                return url
            }
        }
        if let text = item as? String {
            return URL(string: text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    nonisolated private static func isImageURL(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        let keys: Set<URLResourceKey> = [.contentTypeKey]
        if let values = try? url.resourceValues(forKeys: keys),
           let contentType = values.contentType
        {
            return contentType.conforms(to: .image)
        }
        let ext = url.pathExtension.lowercased()
        return ["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic"].contains(ext)
    }

    private func generateAssistantReply() {
        guard backendReady else {
            lastError = "Backend is not ready"
            return
        }

        let reqMessages = buildRequestMessages()
        let genConfig = GenerateConfig(
            maxNewTokens: max(1, maxNewTokens),
            temperature: max(0.0, temperature),
            topP: min(max(topP, 0.0), 1.0),
            topK: max(0, topK),
            contextWindow: max(0, contextWindow),
            powermetrics: powermetricsEnabled,
            stream: streamResponses,
            reasoningEnabled: reasoningEnabled
        )

        isGenerating = true
        lastError = ""
        let streamMessageID = UUID()
        if streamResponses {
            messages.append(
                ChatMessage(
                    id: streamMessageID,
                    role: .assistant,
                    content: "",
                    isStreaming: true
                )
            )
        }

        Task {
            do {
                let response = try await backend.generate(
                    messages: reqMessages,
                    config: genConfig,
                    onChunk: streamResponses
                        ? { [weak self] _delta, _text, reasoning, answer, _generatedTokens in
                            self?.updateStreamingMessage(
                                id: streamMessageID,
                                answer: answer,
                                reasoning: reasoning,
                                isFinal: false
                            )
                        }
                        : nil
                )
                if streamResponses {
                    updateStreamingMessage(
                        id: streamMessageID,
                        answer: response.answer,
                        reasoning: response.reasoning,
                        isFinal: true
                    )
                } else {
                    messages.append(
                        ChatMessage(
                            role: .assistant,
                            content: response.answer,
                            reasoning: response.reasoning
                        )
                    )
                }
                latestPerf = response.perf
                latestPower = response.power
                latestPowerSeries = response.powerSeries
                bridgeCompiles = response.bridgeCompiles
                aneKernelsCompiled = response.aneKernelsCompiled
                latestMultimodalMode = response.multimodalMode ?? "text"
            } catch {
                if streamResponses {
                    removeStreamingMessageIfEmpty(id: streamMessageID)
                }
                lastError = error.localizedDescription
            }
            isGenerating = false
        }
    }

    private func updateStreamingMessage(
        id: UUID,
        answer: String,
        reasoning: String?,
        isFinal: Bool
    ) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else {
            messages.append(
                ChatMessage(
                    id: id,
                    role: .assistant,
                    content: answer,
                    reasoning: reasoning,
                    isStreaming: !isFinal
                )
            )
            return
        }
        messages[idx].content = answer
        messages[idx].reasoning = reasoning
        messages[idx].isStreaming = !isFinal
    }

    private func removeStreamingMessageIfEmpty(id: UUID) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        if messages[idx].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.remove(at: idx)
        } else {
            messages[idx].isStreaming = false
        }
    }

    private func buildRequestMessages() -> [[String: Any]] {
        var payload: [[String: Any]] = []
        let prompt = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !prompt.isEmpty {
            payload.append(["role": "system", "content": prompt])
        }
        for msg in messages {
            var item: [String: Any] = ["role": msg.role.rawValue]
            if msg.role == .user, !msg.imagePaths.isEmpty {
                var contentItems: [[String: Any]] = []
                if !msg.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    contentItems.append(["type": "text", "text": msg.content])
                }
                for path in msg.imagePaths {
                    contentItems.append(
                        [
                            "type": "image",
                            "image": URL(fileURLWithPath: path).absoluteString,
                        ]
                    )
                }
                item["content"] = contentItems
            } else {
                item["content"] = msg.content
            }
            if msg.role == .assistant, let reasoning = msg.reasoning, !reasoning.isEmpty {
                item["reasoning_content"] = reasoning
            }
            payload.append(item)
        }
        return payload
    }

    private static func defaultModelID() -> String {
        let local = "/Users/mweinbach/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/manual"
        if FileManager.default.fileExists(atPath: local) {
            return local
        }
        return "Qwen/Qwen3.5-4B"
    }

    private struct CuratedModel {
        let repoID: String
        let title: String
    }

    private static let curatedModels: [CuratedModel] = [
        CuratedModel(repoID: "mlx-community/Qwen3.5-4B-mxfp4", title: "Qwen3.5 4B MXFP4"),
        CuratedModel(repoID: "mlx-community/Qwen3.5-2B-6bit", title: "Qwen3.5 2B 6-bit"),
        CuratedModel(repoID: "Qwen/Qwen3.5-4B", title: "Qwen3.5 4B (baseline)"),
    ]

    nonisolated private static func findCachedSnapshotPath(for repoID: String) -> String? {
        let fm = FileManager.default
        let modelKey = "models--" + repoID.replacingOccurrences(of: "/", with: "--")
        let snapshots = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent(modelKey)
            .appendingPathComponent("snapshots")

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: snapshots.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }

        let manual = snapshots.appendingPathComponent("manual")
        if isUsableModelSnapshot(manual) {
            return manual.path
        }

        guard let children = try? fm.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        let dirs = children.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
        }
        let sorted = dirs.sorted { lhs, rhs in
            let ld = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rd = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return ld > rd
        }
        for candidate in sorted where isUsableModelSnapshot(candidate) {
            return candidate.path
        }
        return nil
    }

    nonisolated private static func isUsableModelSnapshot(_ url: URL) -> Bool {
        let fm = FileManager.default
        let config = url.appendingPathComponent("config.json")
        let tokenizer = url.appendingPathComponent("tokenizer_config.json")
        return fm.fileExists(atPath: config.path) && fm.fileExists(atPath: tokenizer.path)
    }

    nonisolated private static func snapshotDownload(repoID: String) throws -> String {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let script = """
        import json, sys
        from huggingface_hub import snapshot_download

        repo_id = sys.argv[1]
        path = snapshot_download(repo_id=repo_id, resume_download=True)
        print(json.dumps({"path": path}))
        """
        proc.arguments = ["python3", "-c", script, repoID]

        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr

        try proc.run()
        proc.waitUntilExit()

        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        let outText = String(data: outData, encoding: .utf8) ?? ""
        let errText = String(data: errData, encoding: .utf8) ?? ""

        guard proc.terminationStatus == 0 else {
            let detail = errText.trimmingCharacters(in: .whitespacesAndNewlines)
            throw BackendError.requestFailed("Model download failed for \(repoID): \(detail.isEmpty ? "unknown error" : detail)")
        }

        let lines = outText.split(separator: "\n").map(String.init)
        guard let last = lines.last,
              let data = last.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let path = obj["path"] as? String,
              !path.isEmpty
        else {
            throw BackendError.requestFailed("Model download succeeded but no snapshot path was returned")
        }
        return path
    }

    nonisolated private static func runBackendWarmStartProbe(
        modelID: String,
        prefillDevice: String,
        aneMode: String,
        aneSpatial: Int,
        aneHiddenTile: Int,
        aneShapePolicy: String,
        aneSramTargetMB: Double,
        aneTileMultiple: Int,
        aneMinHiddenTile: Int,
        dtype: String,
        powermetricsSampleRateMs: Int,
        powermetricsSamplers: String
    ) throws -> String {
        let script = """
        import json
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        model_id = sys.argv[1]
        prefill_device = sys.argv[2]
        ane_mode = sys.argv[3]
        ane_spatial = sys.argv[4]
        ane_hidden_tile = sys.argv[5]
        ane_shape_policy = sys.argv[6]
        ane_sram_target_mb = sys.argv[7]
        ane_tile_multiple = sys.argv[8]
        ane_min_hidden_tile = sys.argv[9]
        dtype = sys.argv[10]
        powermetrics_rate = sys.argv[11]
        powermetrics_samplers = sys.argv[12]

        def resolve_backend():
            env_script = os.environ.get("QWEN_ANE_BACKEND_SCRIPT", "").strip()
            if env_script and Path(env_script).exists():
                return str(Path(env_script).resolve())
            cwd = Path.cwd().resolve()
            for base in [cwd, *cwd.parents]:
                candidate = base / "qwen_ane" / "gui_backend.py"
                if candidate.exists():
                    return str(candidate)
            return None

        backend = resolve_backend()
        if backend is None:
            print(json.dumps({"ok": False, "error": "Could not locate qwen_ane/gui_backend.py"}))
            raise SystemExit(1)

        cmd = [
            sys.executable,
            backend,
            "--model-id", model_id,
            "--prefill-device", prefill_device,
            "--ane-mode", ane_mode,
            "--ane-layers", "1",
            "--ane-spatial", ane_spatial,
            "--ane-hidden-tile", ane_hidden_tile,
            "--ane-shape-policy", ane_shape_policy,
            "--ane-sram-target-mb", ane_sram_target_mb,
            "--ane-tile-multiple", ane_tile_multiple,
            "--ane-min-hidden-tile", ane_min_hidden_tile,
            "--dtype", dtype,
            "--powermetrics-sample-rate-ms", powermetrics_rate,
            "--powermetrics-samplers", powermetrics_samplers,
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        deadline = time.time() + 180.0
        ready = None
        last_line = ""
        while time.time() < deadline:
            line = proc.stdout.readline() if proc.stdout is not None else ""
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            line = line.strip()
            if not line:
                continue
            last_line = line
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event") == "ready":
                ready = event
                break
            if event.get("event") == "error":
                break

        if ready is not None and proc.stdin is not None:
            try:
                proc.stdin.write(json.dumps({"id": "probe-shutdown", "type": "shutdown"}) + "\\n")
                proc.stdin.flush()
            except Exception:
                pass

        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

        if ready is None:
            detail = last_line if last_line else f"backend exited rc={proc.returncode}"
            print(json.dumps({"ok": False, "error": f"No ready event. Last: {detail}"}))
            raise SystemExit(1)

        print(
            json.dumps(
                {
                    "ok": True,
                    "prefill_device": ready.get("prefill_device"),
                    "ane_mode": ready.get("ane_mode"),
                    "vision_ready": ready.get("vision_processor_ready"),
                    "vision_status": ready.get("vision_processor_status"),
                    "vision_error": ready.get("vision_processor_error"),
                }
            )
        )
        """

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [
            "python3",
            "-c",
            script,
            modelID,
            prefillDevice,
            aneMode,
            "\(aneSpatial)",
            "\(aneHiddenTile)",
            aneShapePolicy,
            String(format: "%.1f", aneSramTargetMB),
            "\(aneTileMultiple)",
            "\(aneMinHiddenTile)",
            dtype,
            "\(powermetricsSampleRateMs)",
            powermetricsSamplers,
        ]

        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr

        try proc.run()
        proc.waitUntilExit()

        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        let outText = String(data: outData, encoding: .utf8) ?? ""
        let errText = String(data: errData, encoding: .utf8) ?? ""
        let lines = outText.split(separator: "\n").map(String.init)

        guard let last = lines.last,
              let data = last.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            let msg = errText.trimmingCharacters(in: .whitespacesAndNewlines)
            throw BackendError.requestFailed("Probe failed with malformed output: \(msg.isEmpty ? outText : msg)")
        }

        let ok = (obj["ok"] as? Bool) ?? false
        if !ok {
            let detail = (obj["error"] as? String) ?? "unknown error"
            throw BackendError.requestFailed(detail)
        }

        let prefill = (obj["prefill_device"] as? String) ?? "unknown"
        let mode = (obj["ane_mode"] as? String) ?? "unknown"
        let visionReady = (obj["vision_ready"] as? Bool) ?? false
        let visionStatus = (obj["vision_status"] as? String) ?? "unknown"
        let visionError = (obj["vision_error"] as? String) ?? ""

        if visionError.isEmpty {
            return "Probe OK: prefill=\(prefill) mode=\(mode) vision=\(visionReady) (\(visionStatus))"
        }
        return "Probe OK: prefill=\(prefill) mode=\(mode) vision=\(visionReady) (\(visionStatus)); \(visionError)"
    }
}

struct ContentView: View {
    @ObservedObject var model: ChatViewModel
    @FocusState private var composerFocused: Bool
    @State private var isImageDropTargeted = false

    var body: some View {
        HSplitView {
            chatPane
            sidePane
                .frame(minWidth: 360, maxWidth: 420)
        }
        .frame(minWidth: 1180, minHeight: 760)
    }

    private var chatPane: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()
            transcript
            Divider()
            composer
        }
    }

    private var headerBar: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(model.backendReady ? Color.green : Color.orange)
                .frame(width: 10, height: 10)
            Text(model.backendStatus)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Spacer()

            Button("Restart Backend") {
                Task { await model.startBackend() }
            }
            .buttonStyle(.bordered)
            .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel)

            Button("Stop") {
                model.stopBackend()
            }
            .buttonStyle(.bordered)
            .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel)

            Button("Clear Chat") {
                model.clearChat()
            }
            .buttonStyle(.bordered)
        }
        .padding(12)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(model.messages) { message in
                        MessageBubble(
                            message: message,
                            showReasoning: model.showReasoningPanels
                        )
                            .id(message.id)
                    }

                    if model.isGenerating {
                        HStack {
                            ProgressView()
                            Text("Generating…")
                                .foregroundStyle(.secondary)
                                .font(.subheadline)
                        }
                        .padding(.horizontal, 12)
                    }
                    if model.isAutoTuning {
                        HStack {
                            ProgressView()
                            Text("Auto-tuning…")
                                .foregroundStyle(.secondary)
                                .font(.subheadline)
                        }
                        .padding(.horizontal, 12)
                    }
                }
                .padding(16)
            }
            .onChange(of: model.messages.count) { _, _ in
                if let last = model.messages.last {
                    withAnimation(.easeOut(duration: 0.18)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.lastError.isEmpty {
                Text(model.lastError)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .padding(.horizontal, 12)
            }

            if !model.pendingImagePaths.isEmpty {
                AttachmentStrip(
                    paths: model.pendingImagePaths,
                    removable: true,
                    onRemove: { path in
                        model.removePendingImage(path: path)
                    }
                )
                .padding(.horizontal, 12)
            }

            HStack(spacing: 8) {
                Button("Add Images") {
                    model.selectImagesForCurrentInput()
                    focusComposer()
                }
                .buttonStyle(.bordered)
                .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel)

                if !model.pendingImagePaths.isEmpty {
                    Text("\(model.pendingImagePaths.count) image(s) attached")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 12)

            HStack(alignment: .bottom, spacing: 8) {
                TextEditor(text: $model.inputText)
                    .font(.system(size: 14, weight: .regular, design: .monospaced))
                    .frame(minHeight: 80, maxHeight: 120)
                    .focused($composerFocused)
                    .padding(6)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.25), lineWidth: 1)
                    )

                Button(model.isGenerating ? "Running" : "Send") {
                    model.sendCurrentInput()
                    focusComposer()
                }
                .buttonStyle(.borderedProminent)
                .disabled(
                    model.isGenerating
                        || model.isAutoTuning
                        || model.isDownloadingModel
                        || !model.backendReady
                        || (
                            model.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                && model.pendingImagePaths.isEmpty
                        )
                )
            }
            .padding(12)
            .onAppear {
                focusComposer()
            }
            .onChange(of: model.isGenerating) { _, isGenerating in
                if !isGenerating {
                    focusComposer()
                }
            }
        }
        .onDrop(
            of: [UTType.fileURL.identifier],
            isTargeted: $isImageDropTargeted
        ) { providers in
            model.handleDroppedItemProviders(providers)
        }
        .overlay {
            if isImageDropTargeted {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
                    .overlay(
                        Text("Drop images to attach")
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.regularMaterial)
                            .clipShape(Capsule())
                    )
            }
        }
    }

    private var sidePane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                GroupBox("Sampling") {
                    VStack(spacing: 10) {
                        numberRow(title: "Context Window", value: $model.contextWindow)
                        numberRow(title: "Max New Tokens", value: $model.maxNewTokens)
                        numberRow(title: "Top-K", value: $model.topK)
                        doubleRow(title: "Top-P", value: $model.topP)
                        doubleRow(title: "Temperature", value: $model.temperature)
                        Toggle("Stream Responses", isOn: $model.streamResponses)
                            .toggleStyle(.switch)
                        Toggle("Enable Reasoning", isOn: $model.reasoningEnabled)
                            .toggleStyle(.switch)
                        Toggle("Show Reasoning Panels", isOn: $model.showReasoningPanels)
                            .toggleStyle(.switch)
                        Toggle("Capture Powermetrics", isOn: $model.powermetricsEnabled)
                            .toggleStyle(.switch)
                        Toggle("Auto-Tune On Startup", isOn: $model.autoTuneOnStartup)
                            .toggleStyle(.switch)
                        if model.powermetricsEnabled != model.backendUsesSudo {
                            Text("Restart backend to apply powermetrics privilege change.")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }
                    .padding(.vertical, 4)
                }

                GroupBox("Model Catalog") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Preset")
                            Spacer(minLength: 10)
                            Picker("Preset", selection: $model.selectedCatalogModelRepoID) {
                                ForEach(model.modelCatalog, id: \.repoID) { entry in
                                    Text(modelCatalogLabel(for: entry))
                                        .tag(entry.repoID)
                                }
                            }
                            .labelsHidden()
                            .pickerStyle(.menu)
                            .frame(width: 200)
                        }

                        if let selected = model.modelCatalog.first(where: { $0.repoID == model.selectedCatalogModelRepoID }) {
                            statsRow("Repo", selected.repoID)
                            statsRow("Local", selected.localPath ?? "not cached")
                        } else {
                            statsRow("Repo", model.selectedCatalogModelRepoID)
                            statsRow("Local", "not cached")
                        }

                        HStack(spacing: 8) {
                            Button("Refresh") {
                                model.refreshModelCatalog(applyBestAvailable: false)
                            }
                            .buttonStyle(.bordered)
                            .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel || model.isProbingModel)

                            Button("Use Selected") {
                                model.useSelectedCatalogModel()
                            }
                            .buttonStyle(.bordered)
                            .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel || model.isProbingModel)

                            Button("Best Available") {
                                model.useBestAvailableCatalogModel()
                            }
                            .buttonStyle(.bordered)
                            .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel || model.isProbingModel)
                        }

                        Button(model.isDownloadingModel ? "Downloading…" : "Download Selected") {
                            model.downloadSelectedCatalogModel()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel || model.isProbingModel)

                        Button(model.isProbingModel ? "Probing…" : "Probe Selected") {
                            model.probeSelectedCatalogModel()
                        }
                        .buttonStyle(.bordered)
                        .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel || model.isProbingModel)

                        Text(model.modelCatalogStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)

                        Text(model.modelProbeStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.vertical, 4)
                }

                GroupBox("Model Runtime") {
                    VStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Model ID / Path")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextField("Qwen/Qwen3.5-4B", text: $model.modelID)
                                .textFieldStyle(.roundedBorder)
                        }
                        pickerRow(title: "Prefill Device", selection: $model.prefillDevice, values: ["mps", "cpu"])
                        pickerRow(title: "ANE Mode", selection: $model.aneMode, values: ["mlp_tiled", "mlp_fused", "linear"])
                        pickerRow(title: "DType", selection: $model.dtype, values: ["fp16", "bf16"])
                        pickerRow(title: "Shape Policy", selection: $model.aneShapePolicy, values: ["auto", "manual"])
                        statsRow(
                            "Vision Processor",
                            model.visionProcessorReady
                                ? "ready (\(model.visionProcessorStatus))"
                                : "not ready (\(model.visionProcessorStatus))"
                        )
                        if !model.visionProcessorError.isEmpty {
                            Text(model.visionProcessorError)
                                .font(.caption2)
                                .foregroundStyle(.orange)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        numberRow(title: "ANE Layers", value: $model.aneLayers)
                        numberRow(title: "ANE Spatial", value: $model.aneSpatial)
                        numberRow(title: "ANE Hidden Tile", value: $model.aneHiddenTile)
                        doubleRow(title: "ANE SRAM Target MB", value: $model.aneSramTargetMB)
                        numberRow(title: "ANE Tile Multiple", value: $model.aneTileMultiple)
                        numberRow(title: "ANE Min Hidden Tile", value: $model.aneMinHiddenTile)
                        Button(model.isAutoTuning ? "Auto-Tuning…" : "Auto-Tune ANE") {
                            model.runAutoTune(applyAndRestart: true)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.backendReady || model.isGenerating || model.isAutoTuning)
                        Text(model.autoTuneStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        numberRow(title: "Powermetrics Rate (ms)", value: $model.powermetricsSampleRateMs)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Powermetrics Samplers")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextField("cpu_power,gpu_power,ane_power", text: $model.powermetricsSamplers)
                                .textFieldStyle(.roundedBorder)
                        }
                    }
                    .padding(.vertical, 4)
                }

                GroupBox("Stats") {
                    VStack(alignment: .leading, spacing: 6) {
                        if let tune = model.lastAutoTune {
                            statsRow("Tune Shape", "\(tune.dim):\(tune.hidden)")
                            statsRow("Tune Module", tune.moduleName)
                            statsRow("Tune Best", "s\(tune.bestSpatial)/t\(tune.bestTileHidden)")
                            statsRow("Tune Eval ms", fmt(tune.bestEvalMs))
                            statsRow("Tune Util %", fmt(tune.bestUtilPct))
                        }
                        if let perf = model.latestPerf {
                            statsRow("MM Mode", model.latestMultimodalMode)
                            statsRow("Prompt Tokens", "\(perf.promptTokens)")
                            statsRow("Generated", "\(perf.generatedTokens)")
                            statsRow("Prefill tok/s", fmt(perf.prefillTPS))
                            statsRow("Decode tok/s", fmt(perf.decodeTPS))
                            statsRow("E2E tok/s", fmt(perf.endToEndTPS))
                            statsRow("Prefill sec", fmt(perf.prefillSeconds))
                            statsRow("Decode sec", fmt(perf.decodeSeconds))
                            statsRow("Total sec", fmt(perf.totalSeconds))
                        } else {
                            Text("No generation metrics yet")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }

                GroupBox("Power") {
                    VStack(alignment: .leading, spacing: 6) {
                        if let power = model.latestPower {
                            statsRow("SoC W", fmt(power.socWatts))
                            statsRow("ANE W", fmt(power.aneWatts))
                            statsRow("CPU W", fmt(power.cpuWatts))
                            statsRow("GPU W", fmt(power.gpuWatts))
                            statsRow("Samples", "\(power.sampleCount)")
                            if let warning = power.warning, !warning.isEmpty {
                                Text(warning)
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            if !model.latestPowerSeries.isEmpty {
                                powerChart(series: model.latestPowerSeries)
                                    .frame(height: 140)
                                    .padding(.top, 4)
                            }
                        } else {
                            Text(model.powermetricsEnabled ? "No powermetrics sample yet" : "Powermetrics disabled")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }

                GroupBox("Kernel Cache") {
                    VStack(alignment: .leading, spacing: 6) {
                        statsRow("Bridge Compiles", "\(model.bridgeCompiles)")
                        statsRow("ANE Kernels Compiled", "\(model.aneKernelsCompiled)")
                    }
                    .padding(.vertical, 4)
                }

                GroupBox("System Prompt") {
                    TextEditor(text: $model.systemPrompt)
                        .font(.system(size: 13, weight: .regular, design: .monospaced))
                        .frame(minHeight: 80, maxHeight: 160)
                        .padding(6)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.gray.opacity(0.25), lineWidth: 1)
                        )
                }
            }
            .padding(12)
        }
    }

    private func numberRow(title: String, value: Binding<Int>) -> some View {
        HStack {
            Text(title)
            Spacer(minLength: 10)
            TextField("0", value: value, format: .number)
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.roundedBorder)
                .frame(width: 110)
        }
    }

    private func doubleRow(title: String, value: Binding<Double>) -> some View {
        HStack {
            Text(title)
            Spacer(minLength: 10)
            TextField("0", value: value, format: .number.precision(.fractionLength(3)))
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.roundedBorder)
                .frame(width: 110)
        }
    }

    private func pickerRow(title: String, selection: Binding<String>, values: [String]) -> some View {
        HStack {
            Text(title)
            Spacer(minLength: 10)
            Picker(title, selection: selection) {
                ForEach(values, id: \.self) { value in
                    Text(value).tag(value)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 140)
        }
    }

    private func statsRow(_ name: String, _ value: String) -> some View {
        HStack {
            Text(name)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
        }
    }

    private func modelCatalogLabel(for entry: ModelCatalogEntry) -> String {
        entry.isLocalAvailable ? "\(entry.title) (local)" : "\(entry.title) (remote)"
    }

    private func powerChart(series: [PowerSeriesPoint]) -> some View {
        Chart {
            ForEach(series) { point in
                if let ane = point.aneWatts {
                    LineMark(
                        x: .value("Time", point.tSec),
                        y: .value("ANE W", ane)
                    )
                    .foregroundStyle(.orange)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                }
                if let soc = point.socWatts {
                    LineMark(
                        x: .value("Time", point.tSec),
                        y: .value("SoC W", soc)
                    )
                    .foregroundStyle(.blue.opacity(0.8))
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                }
            }
        }
        .chartLegend(position: .bottom, spacing: 14)
        .chartXAxisLabel("sec")
        .chartYAxisLabel("watts")
    }

    private func fmt(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.3f", value)
    }

    private func focusComposer() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) {
            composerFocused = true
        }
    }
}

struct MessageBubble: View {
    let message: ChatMessage
    let showReasoning: Bool

    var body: some View {
        HStack {
            if message.role == .assistant {
                bubble
                Spacer(minLength: 40)
            } else {
                Spacer(minLength: 40)
                bubble
            }
        }
    }

    private var bubble: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(headerText)
                .font(.caption)
                .foregroundStyle(.secondary)
            if showReasoning, message.role == .assistant, let reasoning = trimmedReasoning {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Reasoning")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(reasoning)
                        .textSelection(.enabled)
                        .font(.system(size: 12, weight: .regular, design: .monospaced))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(8)
                .background(Color.orange.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            if !message.imagePaths.isEmpty {
                AttachmentStrip(paths: message.imagePaths)
            }

            if !trimmedContent.isEmpty {
                Text(trimmedContent)
                    .textSelection(.enabled)
                    .font(.system(size: 14, weight: .regular, design: .default))
                    .fixedSize(horizontal: false, vertical: true)
            } else if message.isStreaming {
                Text("Thinking…")
                    .font(.system(size: 13, weight: .regular, design: .default))
                    .foregroundStyle(.secondary)
                    .italic()
            }
        }
        .padding(10)
        .frame(maxWidth: 680, alignment: .leading)
        .background(message.role == .assistant ? Color.gray.opacity(0.12) : Color.blue.opacity(0.14))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var headerText: String {
        if message.role == .assistant {
            return message.isStreaming ? "Assistant (streaming)" : "Assistant"
        }
        return "You"
    }

    private var trimmedContent: String {
        message.content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var trimmedReasoning: String? {
        guard let reasoning = message.reasoning?.trimmingCharacters(in: .whitespacesAndNewlines),
              !reasoning.isEmpty
        else {
            return nil
        }
        return reasoning
    }
}

struct AttachmentStrip: View {
    let paths: [String]
    var removable: Bool = false
    var onRemove: ((String) -> Void)? = nil

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(paths, id: \.self) { path in
                    HStack(spacing: 6) {
                        AttachmentThumbnail(path: path)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(URL(fileURLWithPath: path).lastPathComponent)
                                .font(.caption)
                                .lineLimit(1)
                            Text(path)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        if removable, let onRemove {
                            Button {
                                onRemove(path)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(6)
                    .background(Color.gray.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .frame(minHeight: 56, maxHeight: 72)
    }
}

struct AttachmentThumbnail: View {
    let path: String

    var body: some View {
        Group {
            if let image = NSImage(contentsOfFile: path) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.gray.opacity(0.2))
                    Image(systemName: "photo")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(width: 42, height: 42)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
        )
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
    }
}

@main
struct QwenANEGUIApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var viewModel = ChatViewModel()

    var body: some Scene {
        WindowGroup("Qwen ANE GUI") {
            ContentView(model: viewModel)
        }
    }
}
