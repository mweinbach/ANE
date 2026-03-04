import AppKit
import Charts
import Foundation
import SwiftUI
import UniformTypeIdentifiers

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

