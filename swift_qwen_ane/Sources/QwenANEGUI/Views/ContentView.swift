import SwiftUI

struct ContentView: View {
    @ObservedObject var model: ChatViewModel
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            ChatPane(model: model)
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            showSettings.toggle()
                        } label: {
                            Label("Settings", systemImage: "slider.horizontal.3")
                        }
                        .help("Toggle Settings")
                    }
                }
                .inspector(isPresented: $showSettings) {
                    SettingsPane(model: model)
                        .inspectorColumnWidth(min: 300, ideal: 360, max: 420)
                }
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}
