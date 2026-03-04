// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "QwenANESwift",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "qwen-ane-swift", targets: ["QwenANESwift"]),
        .executable(name: "qwen-ane-gui", targets: ["QwenANEGUI"]),
    ],
    targets: [
        .target(
            name: "CAneBridge",
            path: "Sources/CAneBridge",
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(["-fobjc-arc"]),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("CoreML"),
                .linkedFramework("IOSurface"),
                .linkedLibrary("objc"),
                .linkedLibrary("dl"),
            ]
        ),
        .executableTarget(
            name: "QwenANESwift",
            dependencies: ["CAneBridge"],
            path: "Sources/QwenANESwift",
            linkerSettings: [
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "QwenANEGUI",
            path: "Sources/QwenANEGUI",
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("AppKit"),
            ]
        ),
    ]
)
