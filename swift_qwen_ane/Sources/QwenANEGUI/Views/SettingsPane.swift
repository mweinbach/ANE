import SwiftUI
import Charts

struct SettingsPane: View {
    @ObservedObject var model: ChatViewModel

    var body: some View {
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
                        pickerRow(title: "KV Cache DType", selection: $model.kvCacheDtype, values: ["auto", "fp16", "bf16"])
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
                            statsRow("KV Cache", model.latestKvCacheDtype)
                            statsRow("Quant Mode", model.latestModelQuantMode)
                            statsRow("Runtime Model", model.runtimeModelID.isEmpty ? "-" : model.runtimeModelID)
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
            .padding(16)
        }
        .background(Color(NSColor.windowBackgroundColor))
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
}
