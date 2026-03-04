import CAneBridge
import Darwin
import Foundation

enum AppError: Error, CustomStringConvertible {
    case usage(String)
    case runtime(String)

    var description: String {
        switch self {
        case .usage(let msg): return msg
        case .runtime(let msg): return msg
        }
    }
}

struct TurnPerfMetrics {
    let generatedTokens: Int
    let prefillTokens: Int
    let decodeTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let totalSeconds: Double

    var prefillTPS: Double {
        guard prefillSeconds > 0 else { return 0 }
        return Double(prefillTokens) / prefillSeconds
    }

    var decodeTPS: Double {
        guard decodeSeconds > 0 else { return 0 }
        return Double(decodeTokens) / decodeSeconds
    }

    var endToEndTPS: Double {
        guard totalSeconds > 0 else { return 0 }
        return Double(generatedTokens) / totalSeconds
    }
}

struct PowerMetrics {
    var socWatts: Double?
    var aneWatts: Double?
    var cpuWatts: Double?
    var gpuWatts: Double?
    var sampleCount: Int = 0
    var warning: String?
}

final class PowerMetricsSession {
    private let enabled: Bool
    private let sampleRateMs: Int
    private let samplers: String
    private var process: Process?
    private var outputURL: URL?
    private var warning: String?

    init(enabled: Bool, sampleRateMs: Int, samplers: String) {
        self.enabled = enabled
        self.sampleRateMs = sampleRateMs
        self.samplers = samplers
    }

    func start() {
        guard enabled else { return }
        guard geteuid() == 0 else {
            warning = "powermetrics requires sudo/root; skipping power capture"
            return
        }

        let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("powermetrics_\(UUID().uuidString).txt")
        outputURL = tmpURL

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/powermetrics")
        proc.arguments = [
            "--samplers", samplers,
            "--sample-rate", "\(sampleRateMs)",
            "--sample-count", "-1",
            "--output-file", tmpURL.path,
        ]
        let errPipe = Pipe()
        proc.standardError = errPipe
        proc.standardOutput = FileHandle.nullDevice

        do {
            try proc.run()
            process = proc
            Thread.sleep(forTimeInterval: 0.08)
            if !proc.isRunning {
                let err = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                warning = err?.isEmpty == false ? err : "powermetrics exited immediately"
                process = nil
                cleanup()
            }
        } catch {
            warning = "failed to start powermetrics: \(error)"
            process = nil
            cleanup()
        }
    }

    func stop() -> PowerMetrics? {
        guard enabled else { return nil }

        if let proc = process {
            proc.interrupt()
            if proc.isRunning {
                proc.terminate()
            }
        }

        var text = ""
        if let url = outputURL {
            text = (try? String(contentsOf: url)) ?? ""
        }
        cleanup()

        var parsed = parsePowermetrics(text: text)
        parsed.warning = mergeWarning(parsed.warning, warning)
        return parsed
    }

    private func cleanup() {
        if let url = outputURL {
            try? FileManager.default.removeItem(at: url)
        }
        outputURL = nil
        process = nil
    }
}

func mergeWarning(_ a: String?, _ b: String?) -> String? {
    if let a, let b { return "\(a); \(b)" }
    return a ?? b
}

func avgOrNil(_ values: [Double]) -> Double? {
    guard !values.isEmpty else { return nil }
    return values.reduce(0, +) / Double(values.count)
}

func parsePowermetrics(text: String) -> PowerMetrics {
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return PowerMetrics()
    }
    let pattern = #"(?i)\b([a-z0-9 _/\\-]*power[a-z0-9 _/\\-]*)\b[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mw|w)\b"#
    let regex = try? NSRegularExpression(pattern: pattern)

    var aneVals: [Double] = []
    var cpuVals: [Double] = []
    var gpuVals: [Double] = []
    var socVals: [Double] = []

    for line in text.split(separator: "\n") {
        let lineStr = String(line)
        guard let regex else { continue }
        guard let m = regex.firstMatch(
            in: lineStr,
            range: NSRange(location: 0, length: lineStr.utf16.count)
        ) else { continue }

        guard
            let labelRange = Range(m.range(at: 1), in: lineStr),
            let valRange = Range(m.range(at: 2), in: lineStr),
            let unitRange = Range(m.range(at: 3), in: lineStr)
        else { continue }

        let label = lineStr[labelRange].lowercased()
        let val = Double(lineStr[valRange]) ?? 0
        let unit = lineStr[unitRange].lowercased()
        let watts = (unit == "mw") ? val / 1000.0 : val

        if label.contains("ane") {
            aneVals.append(watts)
        } else if label.contains("cpu") {
            cpuVals.append(watts)
        } else if label.contains("gpu") {
            gpuVals.append(watts)
        } else if label.contains("package") || label.contains("combined") || label.contains("soc")
            || label.contains("total")
        {
            socVals.append(watts)
        }
    }

    let ane = avgOrNil(aneVals)
    let cpu = avgOrNil(cpuVals)
    let gpu = avgOrNil(gpuVals)
    var soc = avgOrNil(socVals)
    if soc == nil {
        let parts = [ane, cpu, gpu].compactMap { $0 }
        if !parts.isEmpty { soc = parts.reduce(0, +) }
    }

    return PowerMetrics(
        socWatts: soc,
        aneWatts: ane,
        cpuWatts: cpu,
        gpuWatts: gpu,
        sampleCount: max(aneVals.count, cpuVals.count, gpuVals.count, socVals.count),
        warning: nil
    )
}

func printTurnMetrics(_ perf: TurnPerfMetrics, power: PowerMetrics?) {
    print(
        "[perf] "
            + "generated_tokens=\(perf.generatedTokens) "
            + "prefill=\(String(format: "%.3f", perf.prefillSeconds))s (\(String(format: "%.2f", perf.prefillTPS)) tok/s) "
            + "decode=\(String(format: "%.3f", perf.decodeSeconds))s (\(String(format: "%.2f", perf.decodeTPS)) tok/s) "
            + "e2e=\(String(format: "%.3f", perf.totalSeconds))s (\(String(format: "%.2f", perf.endToEndTPS)) tok/s)"
    )
    guard let power else { return }
    if let warning = power.warning {
        print("[power] warning: \(warning)")
    }
    var chunks: [String] = []
    if let soc = power.socWatts { chunks.append("SoC=\(String(format: "%.2f", soc))W") }
    if let ane = power.aneWatts { chunks.append("ANE=\(String(format: "%.2f", ane))W") }
    if let cpu = power.cpuWatts { chunks.append("CPU=\(String(format: "%.2f", cpu))W") }
    if let gpu = power.gpuWatts { chunks.append("GPU=\(String(format: "%.2f", gpu))W") }
    if !chunks.isEmpty {
        var line = "[power] " + chunks.joined(separator: " ")
        if power.sampleCount > 0 { line += " samples=\(power.sampleCount)" }
        print(line)
    }
}

func milHeader() -> String {
    return
        "program(1.3)\n"
        + "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        + "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        + "{\"coremltools-version\", \"9.0\"}})]\n"
}

func commonConvConsts() -> [String] {
    return [
        #"        string pt = const()[name = string("pt"), val = string("valid")];"# + "\n",
        #"        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];"# + "\n",
        #"        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];"# + "\n",
        #"        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];"# + "\n",
        #"        int32 gr = const()[name = string("gr"), val = int32(1)];"# + "\n",
    ]
}

func milLinearConv(inFeatures: Int, outFeatures: Int, spatial: Int) -> String {
    var lines = [milHeader(), "{\n",
                 "    func main<ios18>(tensor<fp16, [1, \(inFeatures), 1, \(spatial)]> x) {\n"]
    lines.append(contentsOf: commonConvConsts())
    lines.append(
        "        tensor<fp16, [\(outFeatures), \(inFeatures), 1, 1]> W = const()[name = string(\"W\"), "
            + "val = tensor<fp16, [\(outFeatures), \(inFeatures), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
    )
    lines.append(
        "        tensor<fp16, [1, \(outFeatures), 1, \(spatial)]> out = conv(dilations = dl, groups = gr, "
            + "pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = string(\"conv\")];\n"
    )
    lines.append("    } -> (out);\n")
    lines.append("}\n")
    return lines.joined()
}

func milSwiGLUFused(dim: Int, hidden: Int, spatial: Int) -> String {
    var lines = [milHeader(), "{\n",
                 "    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(spatial)]> x) {\n"]
    lines.append(contentsOf: commonConvConsts())
    lines.append(
        "        tensor<fp16, [\(hidden), \(dim), 1, 1]> W1 = const()[name = string(\"W1\"), "
            + "val = tensor<fp16, [\(hidden), \(dim), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w1.bin\"), offset = uint64(64)))];\n"
    )
    lines.append(
        "        tensor<fp16, [\(hidden), \(dim), 1, 1]> W3 = const()[name = string(\"W3\"), "
            + "val = tensor<fp16, [\(hidden), \(dim), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w3.bin\"), offset = uint64(64)))];\n"
    )
    lines.append(
        "        tensor<fp16, [\(dim), \(hidden), 1, 1]> W2 = const()[name = string(\"W2\"), "
            + "val = tensor<fp16, [\(dim), \(hidden), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w2.bin\"), offset = uint64(64)))];\n"
    )
    lines.append(
        "        tensor<fp16, [1, \(hidden), 1, \(spatial)]> h1 = conv(dilations = dl, groups = gr, "
            + "pad = pd, pad_type = pt, strides = st, weight = W1, x = x)[name = string(\"c1\")];\n"
    )
    lines.append(
        "        tensor<fp16, [1, \(hidden), 1, \(spatial)]> h3 = conv(dilations = dl, groups = gr, "
            + "pad = pd, pad_type = pt, strides = st, weight = W3, x = x)[name = string(\"c3\")];\n"
    )
    lines.append("        tensor<fp16, [1, \(hidden), 1, \(spatial)]> sig = sigmoid(x = h1)[name = string(\"sg\")];\n")
    lines.append("        tensor<fp16, [1, \(hidden), 1, \(spatial)]> silu = mul(x = h1, y = sig)[name = string(\"si\")];\n")
    lines.append("        tensor<fp16, [1, \(hidden), 1, \(spatial)]> gate = mul(x = silu, y = h3)[name = string(\"gt\")];\n")
    lines.append(
        "        tensor<fp16, [1, \(dim), 1, \(spatial)]> out = conv(dilations = dl, groups = gr, "
            + "pad = pd, pad_type = pt, strides = st, weight = W2, x = gate)[name = string(\"c2\")];\n"
    )
    lines.append("    } -> (out);\n")
    lines.append("}\n")
    return lines.joined()
}

func milSwiGLUTiled(dim: Int, hiddenTiles: [Int], spatial: Int) -> String {
    var lines: [String] = [milHeader(), "{\n",
                           "    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(spatial)]> x) {\n"]
    lines.append(contentsOf: commonConvConsts())
    var acc = ""
    for (i, tile) in hiddenTiles.enumerated() {
        lines.append(
            "        tensor<fp16, [\(tile), \(dim), 1, 1]> W1_\(i) = const()[name = string(\"W1_\(i)\"), "
                + "val = tensor<fp16, [\(tile), \(dim), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w1_\(i).bin\"), offset = uint64(64)))];\n"
        )
        lines.append(
            "        tensor<fp16, [\(tile), \(dim), 1, 1]> W3_\(i) = const()[name = string(\"W3_\(i)\"), "
                + "val = tensor<fp16, [\(tile), \(dim), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w3_\(i).bin\"), offset = uint64(64)))];\n"
        )
        lines.append(
            "        tensor<fp16, [\(dim), \(tile), 1, 1]> W2_\(i) = const()[name = string(\"W2_\(i)\"), "
                + "val = tensor<fp16, [\(dim), \(tile), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w2_\(i).bin\"), offset = uint64(64)))];\n"
        )
        lines.append(
            "        tensor<fp16, [1, \(tile), 1, \(spatial)]> h1_\(i) = conv(dilations = dl, groups = gr, "
                + "pad = pd, pad_type = pt, strides = st, weight = W1_\(i), x = x)[name = string(\"c1_\(i)\")];\n"
        )
        lines.append(
            "        tensor<fp16, [1, \(tile), 1, \(spatial)]> h3_\(i) = conv(dilations = dl, groups = gr, "
                + "pad = pd, pad_type = pt, strides = st, weight = W3_\(i), x = x)[name = string(\"c3_\(i)\")];\n"
        )
        lines.append("        tensor<fp16, [1, \(tile), 1, \(spatial)]> sig_\(i) = sigmoid(x = h1_\(i))[name = string(\"sg_\(i)\")];\n")
        lines.append("        tensor<fp16, [1, \(tile), 1, \(spatial)]> silu_\(i) = mul(x = h1_\(i), y = sig_\(i))[name = string(\"si_\(i)\")];\n")
        lines.append("        tensor<fp16, [1, \(tile), 1, \(spatial)]> gate_\(i) = mul(x = silu_\(i), y = h3_\(i))[name = string(\"gt_\(i)\")];\n")
        lines.append(
            "        tensor<fp16, [1, \(dim), 1, \(spatial)]> out_\(i) = conv(dilations = dl, groups = gr, "
                + "pad = pd, pad_type = pt, strides = st, weight = W2_\(i), x = gate_\(i))[name = string(\"c2_\(i)\")];\n"
        )
        if i == 0 {
            acc = "out_0"
        } else {
            let next = "sum_\(i)"
            lines.append(
                "        tensor<fp16, [1, \(dim), 1, \(spatial)]> \(next) = add(x = \(acc), y = out_\(i))[name = string(\"add_\(i)\")];\n"
            )
            acc = next
        }
    }
    lines.append("    } -> (\(acc));\n")
    lines.append("}\n")
    return lines.joined()
}

final class ANEKernel {
    private let ptr: OpaquePointer
    private let inputChannels: Int
    private let outputChannels: Int
    private let spatial: Int

    init(
        ptr: OpaquePointer,
        inputChannels: Int,
        outputChannels: Int,
        spatial: Int
    ) {
        self.ptr = ptr
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.spatial = spatial
    }

    deinit {
        ane_bridge_free(ptr)
    }

    func runFP16(_ x: [Float16]) throws -> [Float16] {
        let expected = inputChannels * spatial
        guard x.count == expected else {
            throw AppError.runtime("expected \(expected) elems, got \(x.count)")
        }
        var inArr = x
        var outArr = Array<Float16>(repeating: 0, count: outputChannels * spatial)
        inArr.withUnsafeBytes { raw in
            ane_bridge_write_input(ptr, 0, raw.baseAddress, raw.count)
        }
        let ok = ane_bridge_eval(ptr)
        if !ok {
            throw AppError.runtime("ane_bridge_eval failed")
        }
        outArr.withUnsafeMutableBytes { raw in
            ane_bridge_read_output(ptr, 0, raw.baseAddress, raw.count)
        }
        return outArr
    }

    func runVecFP16(_ x: [Float16]) throws -> [Float16] {
        guard x.count == inputChannels else {
            throw AppError.runtime("expected vector of \(inputChannels), got \(x.count)")
        }
        if spatial == 1 {
            return try runFP16(x)
        }
        var buf = Array<Float16>(repeating: 0, count: inputChannels * spatial)
        for i in 0..<inputChannels {
            buf[i * spatial] = x[i]
        }
        let outFull = try runFP16(buf)
        var out = Array<Float16>(repeating: 0, count: outputChannels)
        for i in 0..<outputChannels {
            out[i] = outFull[i * spatial]
        }
        return out
    }
}

final class ANEBridge {
    init() throws {
        if ane_bridge_init() != 0 {
            throw AppError.runtime("ane_bridge_init failed")
        }
    }

    func compileCount() -> Int {
        Int(ane_bridge_get_compile_count())
    }

    func compileLinear(weightOutIn: [[Float]], spatial: Int) throws -> ANEKernel {
        let out = weightOutIn.count
        guard let first = weightOutIn.first else { throw AppError.runtime("empty weight") }
        let inp = first.count
        let flat = weightOutIn.flatMap { $0 }
        let blob = try buildWeightBlob(weights: flat, rows: out, cols: inp)
        let mil = milLinearConv(inFeatures: inp, outFeatures: out, spatial: spatial)
        return try compileSingle(
            mil: mil,
            weights: ["@model_path/weights/weight.bin": blob],
            inputChannels: inp,
            outputChannels: out,
            spatial: spatial
        )
    }

    func compileFusedSwiGLU(
        w1HiddenDim: [[Float]],
        w3HiddenDim: [[Float]],
        w2DimHidden: [[Float]],
        spatial: Int
    ) throws -> ANEKernel {
        let (dim, hidden) = try validateShapes(w1: w1HiddenDim, w3: w3HiddenDim, w2: w2DimHidden)
        let mil = milSwiGLUFused(dim: dim, hidden: hidden, spatial: spatial)
        return try compileSingle(
            mil: mil,
            weights: [
                "@model_path/weights/w1.bin": try buildWeightBlob(weights: w1HiddenDim.flatMap { $0 }, rows: hidden, cols: dim),
                "@model_path/weights/w3.bin": try buildWeightBlob(weights: w3HiddenDim.flatMap { $0 }, rows: hidden, cols: dim),
                "@model_path/weights/w2.bin": try buildWeightBlob(weights: w2DimHidden.flatMap { $0 }, rows: dim, cols: hidden),
            ],
            inputChannels: dim,
            outputChannels: dim,
            spatial: spatial
        )
    }

    func compileTiledSwiGLU(
        w1HiddenDim: [[Float]],
        w3HiddenDim: [[Float]],
        w2DimHidden: [[Float]],
        hiddenTile: Int,
        spatial: Int
    ) throws -> ANEKernel {
        let (dim, hidden) = try validateShapes(w1: w1HiddenDim, w3: w3HiddenDim, w2: w2DimHidden)
        guard hiddenTile > 0 else { throw AppError.runtime("hiddenTile must be >= 1") }
        var tiles: [Int] = []
        var at = 0
        while at < hidden {
            let t = min(hiddenTile, hidden - at)
            tiles.append(t)
            at += t
        }

        var weights: [String: Data] = [:]
        var offset = 0
        for (i, tile) in tiles.enumerated() {
            let r = offset..<(offset + tile)
            let w1Chunk = Array(w1HiddenDim[r]).flatMap { $0 }
            let w3Chunk = Array(w3HiddenDim[r]).flatMap { $0 }
            let w2ChunkRows = w2DimHidden.map { Array($0[r]) }
            let w2Chunk = w2ChunkRows.flatMap { $0 }
            weights["@model_path/weights/w1_\(i).bin"] = try buildWeightBlob(weights: w1Chunk, rows: tile, cols: dim)
            weights["@model_path/weights/w3_\(i).bin"] = try buildWeightBlob(weights: w3Chunk, rows: tile, cols: dim)
            weights["@model_path/weights/w2_\(i).bin"] = try buildWeightBlob(weights: w2Chunk, rows: dim, cols: tile)
            offset += tile
        }

        let mil = milSwiGLUTiled(dim: dim, hiddenTiles: tiles, spatial: spatial)
        return try compileSingle(
            mil: mil,
            weights: weights,
            inputChannels: dim,
            outputChannels: dim,
            spatial: spatial
        )
    }

    private func validateShapes(w1: [[Float]], w3: [[Float]], w2: [[Float]]) throws -> (Int, Int) {
        guard !w1.isEmpty, !w3.isEmpty, !w2.isEmpty else {
            throw AppError.runtime("empty weight")
        }
        let hidden = w1.count
        let dim = w1[0].count
        guard w3.count == hidden, w3[0].count == dim else {
            throw AppError.runtime("w1/w3 shape mismatch")
        }
        guard w2.count == dim, w2[0].count == hidden else {
            throw AppError.runtime("w2 shape mismatch")
        }
        return (dim, hidden)
    }

    private func buildWeightBlob(weights: [Float], rows: Int, cols: Int) throws -> Data {
        var len: Int = 0
        let ptr = weights.withUnsafeBufferPointer { bp -> UnsafeMutablePointer<UInt8>? in
            guard let base = bp.baseAddress else { return nil }
            return ane_bridge_build_weight_blob(base, Int32(rows), Int32(cols), &len)
        }
        guard let ptr else { throw AppError.runtime("ane_bridge_build_weight_blob failed") }
        defer { free(ptr) }
        return Data(bytes: ptr, count: len)
    }

    private func compileSingle(
        mil: String,
        weights: [String: Data],
        inputChannels: Int,
        outputChannels: Int,
        spatial: Int
    ) throws -> ANEKernel {
        let inSize = inputChannels * spatial * 2
        let outSize = outputChannels * spatial * 2
        var inSizes = [inSize]
        var outSizes = [outSize]
        var milBytes = Array(mil.utf8)

        let handle: OpaquePointer? = milBytes.withUnsafeMutableBufferPointer { mbp in
            let cMil = UnsafeRawPointer(mbp.baseAddress!).assumingMemoryBound(to: CChar.self)
            if weights.count <= 1 {
                let blob = weights.values.first ?? Data()
                var blobArr = Array(blob)
                return blobArr.withUnsafeMutableBufferPointer { bbuf in
                    inSizes.withUnsafeBufferPointer { ibuf in
                        outSizes.withUnsafeBufferPointer { obuf in
                            ane_bridge_compile(
                                cMil,
                                mbp.count,
                                bbuf.baseAddress,
                                bbuf.count,
                                1,
                                ibuf.baseAddress,
                                1,
                                obuf.baseAddress
                            )
                        }
                    }
                }
            }

            let sorted = weights.keys.sorted()
            var names = sorted.map { strdup($0) }
            defer {
                for n in names { if let n { free(n) } }
            }
            var blobs = sorted.map { Array(weights[$0] ?? Data()) }
            var namePtrs = names.map { UnsafePointer<CChar>($0) }
            var dataPtrs = blobs.map { blob -> UnsafePointer<UInt8>? in
                blob.withUnsafeBufferPointer { $0.baseAddress }
            }
            var lens = blobs.map { $0.count }
            return namePtrs.withUnsafeMutableBufferPointer { nbp in
                dataPtrs.withUnsafeMutableBufferPointer { dbp in
                    lens.withUnsafeMutableBufferPointer { lbp in
                        inSizes.withUnsafeBufferPointer { ibuf in
                            outSizes.withUnsafeBufferPointer { obuf in
                                ane_bridge_compile_multi_weights(
                                    cMil,
                                    mbp.count,
                                    nbp.baseAddress,
                                    dbp.baseAddress,
                                    lbp.baseAddress,
                                    Int32(sorted.count),
                                    1,
                                    ibuf.baseAddress,
                                    1,
                                    obuf.baseAddress
                                )
                            }
                        }
                    }
                }
            }
        }

        guard let handle else {
            throw AppError.runtime(
                "ANE compile failed [in=\(inputChannels), out=\(outputChannels), spatial=\(spatial), n_weights=\(weights.count)]"
            )
        }
        return ANEKernel(ptr: handle, inputChannels: inputChannels, outputChannels: outputChannels, spatial: spatial)
    }
}

struct BenchOptions {
    var shape: String = "2560:9216"
    var warmup: Int = 5
    var iters: Int = 20
    var spatial: Int = 32
    var mode: String = "fused_tiled"
    var tileHidden: Int = 2048
    var peakTFLOPS: Double = 15.8
    var seed: UInt64 = 7
    var powermetrics: Bool = false
    var powermetricsSampleRateMs: Int = 500
    var powermetricsSamplers: String = "cpu_power,gpu_power,ane_power"
}

func parseShape(_ s: String) throws -> (Int, Int) {
    let parts = s.split(separator: ":", maxSplits: 1).map(String.init)
    guard parts.count == 2, let a = Int(parts[0]), let b = Int(parts[1]) else {
        throw AppError.usage("shape must be dim:hidden, got \(s)")
    }
    return (a, b)
}

struct LCRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 1 : seed }
    mutating func next() -> UInt64 {
        state = 2862933555777941757 &* state &+ 3037000493
        return state
    }
}

func randFloat(_ rng: inout LCRNG, scale: Float) -> Float {
    let x = Float(rng.next() & 0xffff_ffff) / Float(UInt32.max)
    return (x * 2 - 1) * scale
}

func makeMatrix(rows: Int, cols: Int, scale: Float, rng: inout LCRNG) -> [[Float]] {
    (0..<rows).map { _ in (0..<cols).map { _ in randFloat(&rng, scale: scale) } }
}

func makeVector(n: Int, scale: Float, rng: inout LCRNG) -> [Float16] {
    (0..<n).map { _ in Float16(randFloat(&rng, scale: scale)) }
}

@inline(__always)
func sigmoid(_ x: Float) -> Float {
    1 / (1 + exp(-x))
}

func benchLinearStack(
    bridge: ANEBridge,
    w1: [[Float]],
    w3: [[Float]],
    w2: [[Float]],
    x: [Float16],
    options: BenchOptions
) throws -> (Double, Double) {
    let t0 = CFAbsoluteTimeGetCurrent()
    let k1 = try bridge.compileLinear(weightOutIn: w1, spatial: options.spatial)
    let k3 = try bridge.compileLinear(weightOutIn: w3, spatial: options.spatial)
    let k2 = try bridge.compileLinear(weightOutIn: w2, spatial: options.spatial)
    let compileMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

    for _ in 0..<options.warmup {
        let h1 = try k1.runVecFP16(x).map(Float.init)
        let h3 = try k3.runVecFP16(x).map(Float.init)
        var gate = Array<Float16>(repeating: 0, count: h1.count)
        for i in 0..<h1.count {
            gate[i] = Float16(h1[i] * sigmoid(h1[i]) * h3[i])
        }
        _ = try k2.runVecFP16(gate)
    }

    let t1 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<options.iters {
        let h1 = try k1.runVecFP16(x).map(Float.init)
        let h3 = try k3.runVecFP16(x).map(Float.init)
        var gate = Array<Float16>(repeating: 0, count: h1.count)
        for i in 0..<h1.count {
            gate[i] = Float16(h1[i] * sigmoid(h1[i]) * h3[i])
        }
        _ = try k2.runVecFP16(gate)
    }
    let evalMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000 / Double(options.iters)
    return (compileMs, evalMs)
}

func benchKernel(
    compile: () throws -> ANEKernel,
    x: [Float16],
    warmup: Int,
    iters: Int
) throws -> (Double, Double) {
    let t0 = CFAbsoluteTimeGetCurrent()
    let k = try compile()
    let compileMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
    for _ in 0..<warmup {
        _ = try k.runVecFP16(x)
    }
    let t1 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        _ = try k.runVecFP16(x)
    }
    let evalMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000 / Double(iters)
    return (compileMs, evalMs)
}

func printBenchRow(
    dim: Int,
    hidden: Int,
    mode: String,
    compileMs: Double,
    evalMs: Double,
    peakTFLOPS: Double,
    perf: TurnPerfMetrics,
    power: PowerMetrics?
) {
    let flops = 6.0 * Double(dim) * Double(hidden)
    let gflops = flops / (evalMs * 1e6)
    let tflops = gflops / 1000.0
    let util = 100.0 * tflops / peakTFLOPS
    print(
        "mode=\(mode) "
            + "compile=\(String(format: "%.1f", compileMs))ms "
            + "eval=\(String(format: "%.3f", evalMs))ms "
            + "gflops=\(String(format: "%.2f", gflops)) "
            + "tflops=\(String(format: "%.3f", tflops)) "
            + "util=\(String(format: "%.2f", util))%"
    )
    printTurnMetrics(perf, power: power)
}

func runBench(_ options: BenchOptions) throws {
    guard options.warmup >= 0, options.iters > 0 else {
        throw AppError.usage("--warmup must be >= 0 and --iters must be > 0")
    }
    guard options.spatial >= 1 else { throw AppError.usage("--spatial must be >= 1") }
    guard options.tileHidden >= 1 else { throw AppError.usage("--tile-hidden must be >= 1") }

    let (dim, hidden) = try parseShape(options.shape)
    var rng = LCRNG(seed: options.seed)
    let w1 = makeMatrix(rows: hidden, cols: dim, scale: 0.02, rng: &rng)
    let w3 = makeMatrix(rows: hidden, cols: dim, scale: 0.02, rng: &rng)
    let w2 = makeMatrix(rows: dim, cols: hidden, scale: 0.02, rng: &rng)
    let x = makeVector(n: dim, scale: 0.2, rng: &rng)

    let bridge = try ANEBridge()
    print("shape dim=\(dim) hidden=\(hidden) spatial=\(options.spatial) mode=\(options.mode)")

    let powerSession = PowerMetricsSession(
        enabled: options.powermetrics,
        sampleRateMs: options.powermetricsSampleRateMs,
        samplers: options.powermetricsSamplers
    )
    powerSession.start()

    let (compileMs, evalMs): (Double, Double)
    switch options.mode {
    case "linear_stack":
        (compileMs, evalMs) = try benchLinearStack(bridge: bridge, w1: w1, w3: w3, w2: w2, x: x, options: options)
    case "fused_mlp":
        (compileMs, evalMs) = try benchKernel(
            compile: { try bridge.compileFusedSwiGLU(w1HiddenDim: w1, w3HiddenDim: w3, w2DimHidden: w2, spatial: options.spatial) },
            x: x,
            warmup: options.warmup,
            iters: options.iters
        )
    case "fused_tiled":
        (compileMs, evalMs) = try benchKernel(
            compile: {
                try bridge.compileTiledSwiGLU(
                    w1HiddenDim: w1,
                    w3HiddenDim: w3,
                    w2DimHidden: w2,
                    hiddenTile: options.tileHidden,
                    spatial: options.spatial
                )
            },
            x: x,
            warmup: options.warmup,
            iters: options.iters
        )
    default:
        throw AppError.usage("unsupported --mode \(options.mode)")
    }

    let power = powerSession.stop()
    let perf = TurnPerfMetrics(
        generatedTokens: options.iters,
        prefillTokens: max(options.warmup, 1),
        decodeTokens: options.iters,
        prefillSeconds: (Double(options.warmup) * evalMs) / 1000.0,
        decodeSeconds: (Double(options.iters) * evalMs) / 1000.0,
        totalSeconds: (Double(options.warmup + options.iters) * evalMs) / 1000.0
    )
    printBenchRow(
        dim: dim,
        hidden: hidden,
        mode: options.mode,
        compileMs: compileMs,
        evalMs: evalMs,
        peakTFLOPS: options.peakTFLOPS,
        perf: perf,
        power: power
    )
    print("[run] bridge_compiles=\(bridge.compileCount())")
}

func printUsage() {
    print(
        """
        qwen-ane-swift (native Swift ANE runtime port)

        Usage:
          qwen-ane-swift bench [options]

        bench options:
          --shape <dim:hidden>              default: 2560:9216
          --mode <linear_stack|fused_mlp|fused_tiled>  default: fused_tiled
          --spatial <N>                     default: 32
          --tile-hidden <N>                 default: 2048
          --warmup <N>                      default: 5
          --iters <N>                       default: 20
          --peak-tflops <N>                 default: 15.8
          --seed <N>                        default: 7
          --powermetrics                    requires sudo/root
          --powermetrics-sample-rate-ms <N> default: 500
          --powermetrics-samplers <list>    default: cpu_power,gpu_power,ane_power
        """
    )
}

func parseBenchOptions(_ args: [String]) throws -> BenchOptions {
    var opts = BenchOptions()
    var i = 0
    while i < args.count {
        let a = args[i]
        switch a {
        case "--shape":
            i += 1
            opts.shape = args[safe: i] ?? ""
        case "--warmup":
            i += 1
            opts.warmup = Int(args[safe: i] ?? "") ?? opts.warmup
        case "--iters":
            i += 1
            opts.iters = Int(args[safe: i] ?? "") ?? opts.iters
        case "--spatial":
            i += 1
            opts.spatial = Int(args[safe: i] ?? "") ?? opts.spatial
        case "--mode":
            i += 1
            opts.mode = args[safe: i] ?? opts.mode
        case "--tile-hidden":
            i += 1
            opts.tileHidden = Int(args[safe: i] ?? "") ?? opts.tileHidden
        case "--peak-tflops":
            i += 1
            opts.peakTFLOPS = Double(args[safe: i] ?? "") ?? opts.peakTFLOPS
        case "--seed":
            i += 1
            opts.seed = UInt64(args[safe: i] ?? "") ?? opts.seed
        case "--powermetrics":
            opts.powermetrics = true
        case "--powermetrics-sample-rate-ms":
            i += 1
            opts.powermetricsSampleRateMs = Int(args[safe: i] ?? "") ?? opts.powermetricsSampleRateMs
        case "--powermetrics-samplers":
            i += 1
            opts.powermetricsSamplers = args[safe: i] ?? opts.powermetricsSamplers
        default:
            throw AppError.usage("unknown argument: \(a)")
        }
        i += 1
    }
    return opts
}

extension Array {
    subscript(safe index: Int) -> Element? {
        guard index >= 0 && index < count else { return nil }
        return self[index]
    }
}

@main
struct Main {
    static func main() {
        do {
            var args = CommandLine.arguments
            _ = args.removeFirst()
            guard let cmd = args.first else {
                printUsage()
                return
            }
            let tail = Array(args.dropFirst())
            switch cmd {
            case "bench":
                let opts = try parseBenchOptions(tail)
                try runBench(opts)
            case "help", "--help", "-h":
                printUsage()
            default:
                throw AppError.usage("unknown command: \(cmd)")
            }
        } catch let err as AppError {
            fputs("error: \(err.description)\n", stderr)
            printUsage()
            exit(2)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }
}
