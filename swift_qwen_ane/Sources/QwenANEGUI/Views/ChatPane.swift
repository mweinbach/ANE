import SwiftUI

struct ChatPane: View {
    @ObservedObject var model: ChatViewModel

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()
            transcript
            Divider()
            ComposerView(model: model)
        }
        .background(Color(NSColor.textBackgroundColor))
    }

    private var headerBar: some View {
        HStack(spacing: 16) {
            HStack(spacing: 6) {
                Circle()
                    .fill(model.backendReady ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(model.backendStatus)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.gray.opacity(0.1))
            .clipShape(Capsule())

            Spacer()

            Button {
                Task { await model.startBackend() }
            } label: {
                Label("Restart", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.gray.opacity(0.1))
            .clipShape(Capsule())
            .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel)

            Button {
                model.stopBackend()
            } label: {
                Label("Stop", systemImage: "stop.fill")
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.gray.opacity(0.1))
            .clipShape(Capsule())
            .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel)

            Button {
                model.clearChat()
            } label: {
                Label("Clear", systemImage: "trash")
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.gray.opacity(0.1))
            .clipShape(Capsule())
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.regularMaterial)
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
}
