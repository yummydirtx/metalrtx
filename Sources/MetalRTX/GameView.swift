import MetalKit
import AppKit

/// An `MTKView` subclass that owns keyboard and mouse input for the free-fly camera and
/// forwards per-frame input state to the renderer.
final class GameView: MTKView {
    weak var renderer: Renderer?

    /// Keys currently held down, identified by their lowercase character.
    private(set) var pressedKeys: Set<String> = []
    /// Accumulated mouse movement (in points) since the last time the renderer consumed it.
    private(set) var mouseDelta: SIMD2<Float> = .zero
    /// Whether the cursor is captured for mouse-look.
    private var mouseCaptured = false

    /// Set when the player double-taps space (Minecraft-style flight toggle); consumed by the renderer.
    private var flightTogglePending = false
    /// Set when the player presses F to toggle the flashlight; consumed by the renderer.
    private var flashlightTogglePending = false
    /// Set when the player presses R to freeze/unfreeze the flashlight; consumed by the renderer.
    private var freezeFlashlightTogglePending = false
    /// Timestamp of the last discrete space press, for double-tap detection.
    private var lastSpaceTap: TimeInterval = 0
    /// Maximum gap between two space taps to count as a double-tap.
    private let doubleTapWindow: TimeInterval = 0.3

    init(frame: CGRect, device: MTLDevice) {
        super.init(frame: frame, device: device)
        self.colorPixelFormat = .bgra8Unorm
        self.depthStencilPixelFormat = .invalid
        self.framebufferOnly = false // path tracer writes to the drawable via a blit/compute
        self.preferredFramesPerSecond = 120
        self.isPaused = false
        self.enableSetNeedsDisplay = false
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("init(coder:) is not used")
    }

    override var acceptsFirstResponder: Bool { true }

    // MARK: - Input consumption

    /// Returns and clears the accumulated mouse delta. Called once per frame by the renderer.
    func consumeMouseDelta() -> SIMD2<Float> {
        let delta = mouseDelta
        mouseDelta = .zero
        return delta
    }

    /// Returns whether a flight-toggle (double-tap space) occurred since the last call, clearing it.
    func consumeFlightToggle() -> Bool {
        defer { flightTogglePending = false }
        return flightTogglePending
    }

    /// Returns whether the flashlight was toggled (F key) since the last call, clearing it.
    func consumeFlashlightToggle() -> Bool {
        defer { flashlightTogglePending = false }
        return flashlightTogglePending
    }

    /// Returns whether the flashlight freeze was toggled (R key) since the last call, clearing it.
    func consumeFreezeFlashlightToggle() -> Bool {
        defer { freezeFlashlightTogglePending = false }
        return freezeFlashlightTogglePending
    }

    // MARK: - Keyboard

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { // Escape releases the mouse
            releaseMouse()
            return
        }
        // Detect a double-tap of the space bar to toggle flight (ignoring key-repeat events).
        if event.keyCode == 49 && !event.isARepeat {
            let now = ProcessInfo.processInfo.systemUptime
            if now - lastSpaceTap <= doubleTapWindow {
                flightTogglePending = true
                lastSpaceTap = 0
            } else {
                lastSpaceTap = now
            }
        }
        // F toggles the flashlight.
        if event.keyCode == 3 && !event.isARepeat {
            flashlightTogglePending = true
        }
        // R freezes / unfreezes the flashlight in place.
        if event.keyCode == 15 && !event.isARepeat {
            freezeFlashlightTogglePending = true
        }
        if let chars = event.charactersIgnoringModifiers?.lowercased() {
            pressedKeys.insert(chars)
        }
    }

    override func keyUp(with event: NSEvent) {
        if let chars = event.charactersIgnoringModifiers?.lowercased() {
            pressedKeys.remove(chars)
        }
    }

    override func flagsChanged(with event: NSEvent) {
        // Track Shift (descend) and Space is handled as a normal key.
        if event.modifierFlags.contains(.shift) {
            pressedKeys.insert("shift")
        } else {
            pressedKeys.remove("shift")
        }
    }

    // MARK: - Mouse

    override func mouseDown(with event: NSEvent) {
        captureMouse()
    }

    override func mouseDragged(with event: NSEvent) {
        guard mouseCaptured else { return }
        mouseDelta += SIMD2<Float>(Float(event.deltaX), Float(event.deltaY))
    }

    override func mouseMoved(with event: NSEvent) {
        guard mouseCaptured else { return }
        mouseDelta += SIMD2<Float>(Float(event.deltaX), Float(event.deltaY))
    }

    private func captureMouse() {
        guard !mouseCaptured else { return }
        mouseCaptured = true
        NSCursor.hide()
        CGAssociateMouseAndMouseCursorPosition(0)
    }

    private func releaseMouse() {
        guard mouseCaptured else { return }
        mouseCaptured = false
        NSCursor.unhide()
        CGAssociateMouseAndMouseCursorPosition(1)
        pressedKeys.removeAll()
    }
}
