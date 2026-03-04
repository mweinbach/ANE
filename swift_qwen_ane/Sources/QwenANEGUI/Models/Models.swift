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
    let kvCacheRequested: String
    let kvCacheResolved: String
    let modelQuantMode: String?
    let runtimeModelID: String?
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
    let kvCacheDtype: String?
    let modelQuantMode: String?
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
    let kvCacheDtype: String
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
