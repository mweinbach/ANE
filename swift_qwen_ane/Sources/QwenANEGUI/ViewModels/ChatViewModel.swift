import AppKit
import Charts
import Foundation
import SwiftUI
import UniformTypeIdentifiers

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

