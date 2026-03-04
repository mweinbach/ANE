import SwiftUI
import UniformTypeIdentifiers

struct ComposerView: View {
    @ObservedObject var model: ChatViewModel
    @FocusState private var composerFocused: Bool
    @State private var isImageDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.lastError.isEmpty {
                Text(model.lastError)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
            }

            if !model.pendingImagePaths.isEmpty {
                AttachmentStrip(
                    paths: model.pendingImagePaths,
                    removable: true,
                    onRemove: { path in
                        model.removePendingImage(path: path)
                    }
                )
                .padding(.horizontal, 16)
                .padding(.top, 8)
            }

            HStack(alignment: .bottom, spacing: 12) {
                Button {
                    model.selectImagesForCurrentInput()
                    focusComposer()
                } label: {
                    Image(systemName: "plus.circle.fill")
                        .font(.system(size: 24))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .disabled(model.isGenerating || model.isAutoTuning || model.isDownloadingModel)
                .padding(.bottom, 8)

                ZStack(alignment: .topLeading) {
                    if model.inputText.isEmpty {
                        Text("Message Qwen...")
                            .font(.system(size: 15))
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 10)
                            .allowsHitTesting(false)
                    }
                    
                    TextEditor(text: $model.inputText)
                        .font(.system(size: 15))
                        .frame(minHeight: 40, maxHeight: 120)
                        .scrollContentBackground(.hidden)
                        .background(Color.clear)
                        .focused($composerFocused)
                        .padding(4)
                }
                .background(Color.gray.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
                )

                Button {
                    model.sendCurrentInput()
                    focusComposer()
                } label: {
                    Image(systemName: model.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(
                            (model.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && model.pendingImagePaths.isEmpty)
                                ? AnyShapeStyle(.secondary.opacity(0.5)) : AnyShapeStyle(.blue)
                        )
                }
                .buttonStyle(.plain)
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
                .padding(.bottom, 6)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(NSColor.windowBackgroundColor))
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
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .overlay(
                        Text("Drop images to attach")
                            .font(.caption.weight(.medium))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(.regularMaterial)
                            .clipShape(Capsule())
                            .shadow(color: .black.opacity(0.1), radius: 4, y: 2)
                    )
            }
        }
    }
    
    private func focusComposer() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) {
            composerFocused = true
        }
    }
}
