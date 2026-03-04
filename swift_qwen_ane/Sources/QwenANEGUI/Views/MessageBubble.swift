import SwiftUI

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
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: message.role == .assistant ? "sparkles" : "person.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(message.role == .assistant ? .purple : .blue)
                Text(headerText)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            if showReasoning, message.role == .assistant, let reasoning = trimmedReasoning {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Image(systemName: "brain")
                            .font(.system(size: 10))
                        Text("Reasoning")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .foregroundStyle(.secondary)
                    Text(reasoning)
                        .textSelection(.enabled)
                        .font(.system(size: 13, weight: .regular, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(12)
                .background(Color.black.opacity(0.04))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
                )
            }

            if !message.imagePaths.isEmpty {
                AttachmentStrip(paths: message.imagePaths)
            }

            if !trimmedContent.isEmpty {
                Text(trimmedContent)
                    .textSelection(.enabled)
                    .font(.system(size: 15, weight: .regular, design: .default))
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
            } else if message.isStreaming {
                HStack(spacing: 6) {
                    Text("Thinking")
                        .font(.system(size: 15, weight: .regular, design: .default))
                        .foregroundStyle(.secondary)
                    ProgressView()
                        .controlSize(.small)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: 680, alignment: .leading)
        .background(message.role == .assistant ? Color.gray.opacity(0.08) : Color.blue.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: Color.black.opacity(0.03), radius: 3, x: 0, y: 1)
    }

    private var headerText: String {
        if message.role == .assistant {
            return message.isStreaming ? "Qwen (Thinking)" : "Qwen"
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
