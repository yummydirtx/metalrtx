import AppKit

/// A floating HUD panel with live sliders and toggles bound to the renderer.
/// Adjusting any control updates the renderer immediately and lets the denoiser
/// reconverge, so the scene responds in real time.
@MainActor
final class ControlsPanel {
    private weak var renderer: Renderer?
    private var panel: NSPanel!

    // Value labels kept so we can refresh the numeric readout as sliders move.
    private var timeValue: NSTextField!
    private var exposureValue: NSTextField!
    private var sunValue: NSTextField!
    private var bouncesValue: NSTextField!
    private var waterValue: NSTextField!

    init(renderer: Renderer) {
        self.renderer = renderer
        build()
    }

    // MARK: - Layout

    private func build() {
        let width: CGFloat = 300
        let rect = NSRect(x: 0, y: 0, width: width, height: 384)
        panel = NSPanel(
            contentRect: rect,
            styleMask: [.titled, .closable, .utilityWindow, .hudWindow, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.title = "Controls"
        panel.isFloatingPanel = true
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true

        guard let content = panel.contentView else { return }
        var y = rect.height - 36

        func addRow(_ title: String,
                    min: Double, max: Double, value: Double,
                    action: Selector) -> (NSSlider, NSTextField) {
            let label = NSTextField(labelWithString: title)
            label.frame = NSRect(x: 16, y: y, width: 150, height: 18)
            label.font = .systemFont(ofSize: 11, weight: .semibold)
            content.addSubview(label)

            let readout = NSTextField(labelWithString: "")
            readout.frame = NSRect(x: 170, y: y, width: 114, height: 18)
            readout.alignment = .right
            readout.font = .monospacedDigitSystemFont(ofSize: 11, weight: .regular)
            content.addSubview(readout)

            y -= 22
            let slider = NSSlider(value: value, minValue: min, maxValue: max,
                                  target: self, action: action)
            slider.frame = NSRect(x: 16, y: y, width: 268, height: 20)
            content.addSubview(slider)
            y -= 34
            return (slider, readout)
        }

        let (_, tVal) = addRow("Time of Day", min: 0, max: 1,
                               value: Double(renderer?.timeOfDay ?? 0.32),
                               action: #selector(timeChanged(_:)))
        timeValue = tVal

        let (_, eVal) = addRow("Exposure", min: 0.1, max: 4,
                               value: Double(renderer?.exposure ?? 1.3),
                               action: #selector(exposureChanged(_:)))
        exposureValue = eVal

        let (_, sVal) = addRow("Sun Strength", min: 0, max: 30,
                               value: Double(renderer?.sunStrength ?? 11),
                               action: #selector(sunChanged(_:)))
        sunValue = sVal

        let (bSlider, bVal) = addRow("Max Bounces", min: 1, max: 12,
                                     value: Double(renderer?.maxBounces ?? 5),
                                     action: #selector(bouncesChanged(_:)))
        bSlider.numberOfTickMarks = 12
        bSlider.allowsTickMarkValuesOnly = true
        bouncesValue = bVal

        let (_, wVal) = addRow("Water Roughness", min: 0, max: 0.3,
                               value: Double(renderer?.waterRoughness ?? 0),
                               action: #selector(waterChanged(_:)))
        waterValue = wVal

        let toggle = NSButton(checkboxWithTitle: "Denoiser",
                              target: self, action: #selector(denoiseToggled(_:)))
        toggle.frame = NSRect(x: 16, y: 48, width: 160, height: 20)
        toggle.state = (renderer?.denoiseEnabled ?? true) ? .on : .off
        content.addSubview(toggle)

        let hint = NSTextField(labelWithString: "WASD move · QE/Space/Shift · mouse look · Esc release")
        hint.frame = NSRect(x: 16, y: 12, width: 268, height: 30)
        hint.font = .systemFont(ofSize: 10)
        hint.textColor = .secondaryLabelColor
        hint.lineBreakMode = .byWordWrapping
        hint.maximumNumberOfLines = 2
        content.addSubview(hint)

        refreshReadouts()
    }

    func show(relativeTo window: NSWindow) {
        let origin = NSPoint(
            x: window.frame.maxX + 16,
            y: window.frame.maxY - panel.frame.height
        )
        panel.setFrameOrigin(origin)
        panel.orderFront(nil)
    }

    // MARK: - Actions

    @objc private func timeChanged(_ sender: NSSlider) {
        renderer?.timeOfDay = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func exposureChanged(_ sender: NSSlider) {
        renderer?.exposure = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func sunChanged(_ sender: NSSlider) {
        renderer?.sunStrength = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func bouncesChanged(_ sender: NSSlider) {
        renderer?.maxBounces = UInt32(sender.doubleValue.rounded())
        refreshReadouts()
    }

    @objc private func waterChanged(_ sender: NSSlider) {
        renderer?.waterRoughness = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func denoiseToggled(_ sender: NSButton) {
        renderer?.denoiseEnabled = (sender.state == .on)
    }

    private func refreshReadouts() {
        guard let r = renderer else { return }
        timeValue?.stringValue = String(format: "%@  %.2f", clockString(r.timeOfDay), r.timeOfDay)
        exposureValue?.stringValue = String(format: "%.2f", r.exposure)
        sunValue?.stringValue = String(format: "%.1f", r.sunStrength)
        bouncesValue?.stringValue = "\(r.maxBounces)"
        waterValue?.stringValue = String(format: "%.3f", r.waterRoughness)
    }

    /// Maps a 0…1 time-of-day to a 24-hour clock label (0.25 = sunrise/06:00).
    private func clockString(_ t: Float) -> String {
        let hours = (Double(t) * 24.0).truncatingRemainder(dividingBy: 24.0)
        let h = Int(hours)
        let m = Int((hours - Double(h)) * 60)
        return String(format: "%02d:%02d", h, m)
    }
}
