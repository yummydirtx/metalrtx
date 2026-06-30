import Metal
import AppKit

/// Reads back a shared Metal texture and writes it to disk as a PNG. Used for headless
/// verification of the renderer (enabled via the `METALRTX_SCREENSHOT_PATH` env var).
enum ScreenshotSaver {
    static func save(texture: MTLTexture, to path: String) {
        let width = texture.width
        let height = texture.height
        let bytesPerRow = width * 4
        var bgra = [UInt8](repeating: 0, count: bytesPerRow * height)

        let region = MTLRegionMake2D(0, 0, width, height)
        bgra.withUnsafeMutableBytes { ptr in
            texture.getBytes(ptr.baseAddress!,
                             bytesPerRow: bytesPerRow,
                             from: region,
                             mipmapLevel: 0)
        }

        // Swizzle BGRA -> RGBA for NSBitmapImageRep.
        var rgba = bgra
        for i in stride(from: 0, to: rgba.count, by: 4) {
            rgba.swapAt(i, i + 2)
        }

        guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil,
                                         pixelsWide: width,
                                         pixelsHigh: height,
                                         bitsPerSample: 8,
                                         samplesPerPixel: 4,
                                         hasAlpha: true,
                                         isPlanar: false,
                                         colorSpaceName: .deviceRGB,
                                         bytesPerRow: bytesPerRow,
                                         bitsPerPixel: 32) else {
            print("Screenshot: failed to create bitmap representation.")
            return
        }

        rgba.withUnsafeBytes { src in
            _ = memcpy(rep.bitmapData!, src.baseAddress!, rgba.count)
        }

        guard let png = rep.representation(using: .png, properties: [:]) else {
            print("Screenshot: failed to encode PNG.")
            return
        }
        do {
            try png.write(to: URL(fileURLWithPath: path))
            print("Screenshot saved to \(path)")
        } catch {
            print("Screenshot: write failed: \(error)")
        }
    }
}
