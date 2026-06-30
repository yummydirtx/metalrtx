// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MetalRTX",
    platforms: [
        .macOS(.v15)
    ],
    targets: [
        .executableTarget(
            name: "MetalRTX",
            path: "Sources/MetalRTX",
            resources: [
                .copy("Shaders")
            ],
            swiftSettings: [
                .unsafeFlags(["-suppress-warnings"], .when(configuration: .release))
            ]
        )
    ]
)
