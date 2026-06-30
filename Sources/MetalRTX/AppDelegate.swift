import AppKit
import MetalKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var view: GameView!
    private var renderer: Renderer!
    private var controlsPanel: ControlsPanel!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this Mac.")
        }

        // Verify hardware ray tracing support before proceeding.
        CapabilityChecker.check(device: device)

        let contentRect = NSRect(x: 0, y: 0, width: 1280, height: 800)
        window = NSWindow(
            contentRect: contentRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Metal RTX — Voxel Path Tracer"
        window.center()

        view = GameView(frame: contentRect, device: device)
        renderer = Renderer(view: view, device: device)
        view.renderer = renderer
        view.delegate = renderer

        window.contentView = view
        window.makeFirstResponder(view)
        window.makeKeyAndOrderFront(nil)

        controlsPanel = ControlsPanel(renderer: renderer)
        controlsPanel.show(relativeTo: window)

        buildMenu()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func buildMenu() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenuItem.submenu = appMenu
        appMenu.addItem(
            withTitle: "Quit Metal RTX",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )

        NSApplication.shared.mainMenu = mainMenu
    }
}
