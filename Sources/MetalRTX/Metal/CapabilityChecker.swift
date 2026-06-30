import Metal
import MetalKit

/// Verifies the GPU supports the features this demo requires and logs a summary.
enum CapabilityChecker {
    static func check(device: MTLDevice) {
        let name = device.name
        let rt = device.supportsRaytracing
        let apple9 = device.supportsFamily(.apple9) // M3 generation and later
        let apple8 = device.supportsFamily(.apple8) // M2 generation
        let apple7 = device.supportsFamily(.apple7) // M1 generation

        let generation: String
        if apple9 {
            generation = "Apple9+ (M3 or later — hardware ray tracing)"
        } else if apple8 {
            generation = "Apple8 (M2 — software ray tracing)"
        } else if apple7 {
            generation = "Apple7 (M1 — software ray tracing)"
        } else {
            generation = "Unknown / pre-Apple Silicon"
        }

        print("""
        ──────────────────────────────────────────────
        Metal RTX — GPU capability report
          Device:            \(name)
          GPU family:        \(generation)
          Ray tracing:       \(rt ? "supported" : "NOT supported")
          Function pointers:  \(device.supportsFunctionPointers ? "yes" : "no")
        ──────────────────────────────────────────────
        """)

        guard rt else {
            fatalError("""
            This GPU does not support Metal ray tracing. \
            A Mac with Apple Silicon (M-series) is required.
            """)
        }

        if !apple9 {
            print("⚠️  Hardware ray tracing requires an M3 or later chip. " +
                  "The demo will still run, but tracing falls back to shader cores.")
        }
    }
}
