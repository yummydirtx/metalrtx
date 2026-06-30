import Metal
import Foundation

/// Loads `.metal` source files bundled as resources and compiles them at runtime into a
/// single `MTLLibrary`. Runtime compilation is used (instead of a precompiled `.metallib`)
/// because Swift Package Manager does not reliably compile `.metal` files from the command
/// line. Sources are concatenated in a defined order so shared declarations in `Common.metal`
/// are visible to every kernel without relying on `#include` resolution.
enum ShaderLibrary {
    /// Order matters: shared declarations must come before the kernels that use them.
    private static let compilationOrder = [
        "Common",
        "Sky",
        "PathTrace",
        "Denoise",
        "Tonemap",
        "Blit"
    ]

    static func make(device: MTLDevice) -> MTLLibrary {
        guard let shadersURL = Bundle.module.url(forResource: "Shaders", withExtension: nil) else {
            fatalError("Could not locate the bundled Shaders directory.")
        }

        var combinedSource = ""
        for name in compilationOrder {
            let fileURL = shadersURL.appendingPathComponent("\(name).metal")
            guard FileManager.default.fileExists(atPath: fileURL.path) else { continue }
            do {
                let source = try String(contentsOf: fileURL, encoding: .utf8)
                combinedSource += "\n// ===== \(name).metal =====\n"
                combinedSource += source
            } catch {
                fatalError("Failed to read shader \(name).metal: \(error)")
            }
        }

        let options = MTLCompileOptions()
        options.mathMode = .fast
        options.languageVersion = .version3_2

        do {
            return try device.makeLibrary(source: combinedSource, options: options)
        } catch {
            fatalError("Shader compilation failed:\n\(error)")
        }
    }
}
