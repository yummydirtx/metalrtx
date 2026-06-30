import AppKit

/// A floating HUD panel with live sliders and toggles bound to the renderer.
/// Adjusting any control updates the renderer immediately and lets the denoiser
/// reconverge, so the scene responds in real time. The water-wave parameters live
/// in a collapsible section so the panel stays compact until they are needed.
@MainActor
final class ControlsPanel {
    private weak var renderer: Renderer?
    private var panel: NSPanel!

    /// A label + numeric readout + slider laid out as one row.
    private struct Row {
        let label: NSTextField
        let readout: NSTextField
        let slider: NSSlider
    }

    // General controls.
    private var timeRow: Row!
    private var exposureRow: Row!
    private var sunRow: Row!
    private var bouncesRow: Row!
    private var denoiseToggle: NSButton!
    private var flashlightToggle: NSButton!
    private var freezeFlashlightToggle: NSButton!

    // Collapsible water section.
    private var waterHeader: NSButton!
    private var waterRows: [Row] = []
    private var waterExpanded = false

    private var hint: NSTextField!

    // Layout metrics (points).
    private let panelWidth: CGFloat = 300
    private let rowHeight: CGFloat = 56
    private let topPad: CGFloat = 36
    private let toggleHeight: CGFloat = 28
    private let headerHeight: CGFloat = 28
    private let tailGap: CGFloat = 50

    init(renderer: Renderer) {
        self.renderer = renderer
        build()
    }

    // MARK: - Construction

    private func build() {
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: panelWidth, height: collapsedHeight),
            styleMask: [.titled, .closable, .utilityWindow, .hudWindow, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.title = "Controls"
        panel.isFloatingPanel = true
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true

        guard let content = panel.contentView else { return }

        timeRow = makeRow(in: content, "Time of Day", min: 0, max: 1,
                          value: Double(renderer?.timeOfDay ?? 0.32),
                          action: #selector(timeChanged(_:)))
        exposureRow = makeRow(in: content, "Exposure", min: 0.1, max: 4,
                              value: Double(renderer?.exposure ?? 1.3),
                              action: #selector(exposureChanged(_:)))
        sunRow = makeRow(in: content, "Sun Strength", min: 0, max: 30,
                         value: Double(renderer?.sunStrength ?? 11),
                         action: #selector(sunChanged(_:)))
        bouncesRow = makeRow(in: content, "Max Bounces", min: 1, max: 12,
                             value: Double(renderer?.maxBounces ?? 5),
                             action: #selector(bouncesChanged(_:)))
        bouncesRow.slider.numberOfTickMarks = 12
        bouncesRow.slider.allowsTickMarkValuesOnly = true

        denoiseToggle = NSButton(checkboxWithTitle: "Denoiser",
                                 target: self, action: #selector(denoiseToggled(_:)))
        denoiseToggle.state = (renderer?.denoiseEnabled ?? true) ? .on : .off
        content.addSubview(denoiseToggle)

        flashlightToggle = NSButton(checkboxWithTitle: "Flashlight (F)",
                                    target: self, action: #selector(flashlightToggled(_:)))
        flashlightToggle.state = (renderer?.flashlightOn ?? false) ? .on : .off
        content.addSubview(flashlightToggle)

        // Keep the checkbox in sync when the flashlight is toggled with the F key.
        renderer?.onFlashlightChanged = { [weak self] on in
            self?.flashlightToggle.state = on ? .on : .off
        }

        freezeFlashlightToggle = NSButton(checkboxWithTitle: "Freeze Flashlight (R)",
                                          target: self, action: #selector(freezeFlashlightToggled(_:)))
        freezeFlashlightToggle.state = (renderer?.flashlightFrozen ?? false) ? .on : .off
        content.addSubview(freezeFlashlightToggle)

        // Keep the freeze checkbox in sync when toggled with the R key.
        renderer?.onFlashlightFrozenChanged = { [weak self] frozen in
            self?.freezeFlashlightToggle.state = frozen ? .on : .off
        }

        // Collapsible water section header (disclosure-style button).
        waterHeader = NSButton(title: waterHeaderTitle,
                               target: self, action: #selector(toggleWater(_:)))
        waterHeader.isBordered = false
        waterHeader.alignment = .left
        waterHeader.font = .systemFont(ofSize: 11, weight: .bold)
        waterHeader.contentTintColor = .labelColor
        content.addSubview(waterHeader)

        waterRows = [
            makeRow(in: content, "Wave Height", min: 0, max: 0.6,
                    value: Double(renderer?.waveAmplitude ?? 0.16),
                    action: #selector(waveAmplitudeChanged(_:))),
            makeRow(in: content, "Choppiness", min: 0, max: 2.5,
                    value: Double(renderer?.waveChoppiness ?? 0.85),
                    action: #selector(waveChoppinessChanged(_:))),
            makeRow(in: content, "Wave Speed", min: 0, max: 1.5,
                    value: Double(renderer?.waveSpeed ?? 0.55),
                    action: #selector(waveSpeedChanged(_:))),
            makeRow(in: content, "Surface Detail", min: 0, max: 0.3,
                    value: Double(renderer?.waterRoughness ?? 0),
                    action: #selector(waterChanged(_:))),
        ]

        hint = NSTextField(labelWithString: "WASD move · double-tap Space fly/walk · F flashlight · R freeze · mouse look · Esc release")
        hint.font = .systemFont(ofSize: 10)
        hint.textColor = .secondaryLabelColor
        hint.lineBreakMode = .byWordWrapping
        hint.maximumNumberOfLines = 2
        content.addSubview(hint)

        relayout()
        refreshReadouts()
    }

    /// Creates a titled slider row and adds its views to `content`. Frames are set later
    /// by `relayout()`.
    private func makeRow(in content: NSView, _ title: String,
                         min: Double, max: Double, value: Double,
                         action: Selector) -> Row {
        let label = NSTextField(labelWithString: title)
        label.font = .systemFont(ofSize: 11, weight: .semibold)
        content.addSubview(label)

        let readout = NSTextField(labelWithString: "")
        readout.alignment = .right
        readout.font = .monospacedDigitSystemFont(ofSize: 11, weight: .regular)
        content.addSubview(readout)

        let slider = NSSlider(value: value, minValue: min, maxValue: max,
                              target: self, action: action)
        content.addSubview(slider)

        return Row(label: label, readout: readout, slider: slider)
    }

    // MARK: - Layout

    private var collapsedHeight: CGFloat {
        topPad + 4 * rowHeight + 2 * toggleHeight + headerHeight + tailGap
    }

    private var expandedHeight: CGFloat {
        collapsedHeight + CGFloat(waterRows.count) * rowHeight
    }

    private var waterHeaderTitle: String {
        waterExpanded ? "▾  Water" : "▸  Water"
    }

    /// Positions every control top-to-bottom for the current expanded/collapsed state and
    /// resizes the panel, keeping its top edge anchored so it grows downward.
    private func relayout() {
        guard let content = panel.contentView else { return }

        let newHeight = waterExpanded ? expandedHeight : collapsedHeight
        let maxY = panel.frame.maxY
        var frame = panel.frame
        frame.size.height = newHeight
        frame.origin.y = maxY - newHeight
        panel.setFrame(frame, display: true, animate: false)

        let width = content.bounds.width
        var y = newHeight - topPad

        func place(_ row: Row) {
            row.label.frame = NSRect(x: 16, y: y, width: 150, height: 18)
            row.readout.frame = NSRect(x: width - 130, y: y, width: 114, height: 18)
            row.slider.frame = NSRect(x: 16, y: y - 22, width: width - 32, height: 20)
            y -= rowHeight
        }

        place(timeRow)
        place(exposureRow)
        place(sunRow)
        place(bouncesRow)

        denoiseToggle.frame = NSRect(x: 16, y: y - 2, width: 120, height: 20)
        flashlightToggle.frame = NSRect(x: 144, y: y - 2, width: 150, height: 20)
        y -= toggleHeight

        freezeFlashlightToggle.frame = NSRect(x: 16, y: y - 2, width: 200, height: 20)
        y -= toggleHeight

        waterHeader.frame = NSRect(x: 14, y: y - 2, width: 160, height: 20)
        y -= headerHeight

        for row in waterRows {
            let hidden = !waterExpanded
            row.label.isHidden = hidden
            row.readout.isHidden = hidden
            row.slider.isHidden = hidden
            if !hidden { place(row) }
        }

        hint.frame = NSRect(x: 16, y: 12, width: width - 32, height: 30)
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

    @objc private func toggleWater(_ sender: NSButton) {
        waterExpanded.toggle()
        waterHeader.title = waterHeaderTitle
        relayout()
    }

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

    @objc private func waveAmplitudeChanged(_ sender: NSSlider) {
        renderer?.waveAmplitude = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func waveChoppinessChanged(_ sender: NSSlider) {
        renderer?.waveChoppiness = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func waveSpeedChanged(_ sender: NSSlider) {
        renderer?.waveSpeed = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func waterChanged(_ sender: NSSlider) {
        renderer?.waterRoughness = Float(sender.doubleValue)
        refreshReadouts()
    }

    @objc private func denoiseToggled(_ sender: NSButton) {
        renderer?.denoiseEnabled = (sender.state == .on)
    }

    @objc private func flashlightToggled(_ sender: NSButton) {
        renderer?.flashlightOn = (sender.state == .on)
    }

    @objc private func freezeFlashlightToggled(_ sender: NSButton) {
        renderer?.flashlightFrozen = (sender.state == .on)
    }

    private func refreshReadouts() {
        guard let r = renderer else { return }
        timeRow.readout.stringValue = String(format: "%@  %.2f", clockString(r.timeOfDay), r.timeOfDay)
        exposureRow.readout.stringValue = String(format: "%.2f", r.exposure)
        sunRow.readout.stringValue = String(format: "%.1f", r.sunStrength)
        bouncesRow.readout.stringValue = "\(r.maxBounces)"
        waterRows[0].readout.stringValue = String(format: "%.2f", r.waveAmplitude)
        waterRows[1].readout.stringValue = String(format: "%.2f", r.waveChoppiness)
        waterRows[2].readout.stringValue = String(format: "%.2f", r.waveSpeed)
        waterRows[3].readout.stringValue = String(format: "%.3f", r.waterRoughness)
    }

    /// Maps a 0…1 time-of-day to a 24-hour clock label (0.25 = sunrise/06:00).
    private func clockString(_ t: Float) -> String {
        let hours = (Double(t) * 24.0).truncatingRemainder(dividingBy: 24.0)
        let h = Int(hours)
        let m = Int((hours - Double(h)) * 60)
        return String(format: "%02d:%02d", h, m)
    }
}
