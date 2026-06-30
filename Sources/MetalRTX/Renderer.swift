import MetalKit
import simd

/// Owns the Metal device, command queue, pipelines, scene, and per-frame render loop.
/// Drives a progressive hardware-ray-traced path tracer: each frame traces one sample per
/// pixel into an HDR accumulation buffer and tone maps it to the drawable. Accumulation
/// resets whenever the camera or scene settings change.
@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    private weak var view: GameView?

    // Scene.
    let world: VoxelWorld
    let scene: Scene
    let camera: Camera

    // Pipelines.
    private let pathTracePipeline: MTLComputePipelineState
    private let temporalPipeline: MTLComputePipelineState
    private let atrousPipeline: MTLComputePipelineState
    private let tonemapPipeline: MTLComputePipelineState

    // Render targets.
    private var colorTex: MTLTexture?          // 1-spp path traced radiance
    private var normalDepthTex: MTLTexture?    // xyz normal, w view depth
    private var posTex: MTLTexture?            // xyz world position, w material id
    private var prevPosTex: MTLTexture?
    private var prevNormalTex: MTLTexture?
    private var histColor: [MTLTexture?] = [nil, nil]   // ping-pong across frames
    private var atrousTex: [MTLTexture?] = [nil, nil]   // ping-pong within a frame
    private var outputTexture: MTLTexture?     // tone-mapped LDR (shared, readable)
    private var frameParity = 0
    private var targetWidth = 0
    private var targetHeight = 0

    // Headless verification.
    private let screenshotPath = ProcessInfo.processInfo.environment["METALRTX_SCREENSHOT_PATH"]
    private var screenshotSaved = false

    // Frame state.
    private var frameIndex: UInt32 = 0
    private var accumulatedFrames: UInt32 = 0
    private var lastFrameTime = CACurrentMediaTime()
    private let startTime = CACurrentMediaTime()

    // Live, user-tunable settings (bound by the controls panel).
    var timeOfDay: Float = 0.32 { didSet { settingsDirty = true } }
    var exposure: Float = 1.3 { didSet { settingsDirty = true } }
    var maxBounces: UInt32 = 5 { didSet { settingsDirty = true } }
    var waterRoughness: Float = 0.0 { didSet { settingsDirty = true } }
    var denoiseEnabled = true { didSet { settingsDirty = true } }
    var sunStrength: Float = 11 { didSet { settingsDirty = true } }
    private var settingsDirty = true

    init(view: GameView, device: MTLDevice) {
        self.view = view
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create a Metal command queue.")
        }
        self.commandQueue = queue
        self.library = ShaderLibrary.make(device: device)

        self.pathTracePipeline = Renderer.makePipeline(library: library, device: device, name: "pathTrace")
        self.temporalPipeline = Renderer.makePipeline(library: library, device: device, name: "temporalReproject")
        self.atrousPipeline = Renderer.makePipeline(library: library, device: device, name: "atrous")
        self.tonemapPipeline = Renderer.makePipeline(library: library, device: device, name: "tonemap")

        // Generate the voxel world and build its acceleration structures.
        let world = VoxelWorld()
        TerrainGenerator().generate(into: world)
        self.world = world
        self.scene = AccelerationStructureBuilder.build(world: world,
                                                        device: device,
                                                        commandQueue: queue)

        // Place the camera above the terrain looking toward the world center.
        let center = SIMD3<Float>(Float(world.sizeX) * 0.5,
                                  Float(world.seaLevel) + 22,
                                  Float(world.sizeZ) * 0.5)
        let cam = Camera(position: center + SIMD3(-40, 8, 40), yaw: -0.7, pitch: -0.18)
        self.camera = cam

        super.init()
        print("Renderer initialized. Runtime shader library compiled successfully.")
    }

    private static func makePipeline(library: MTLLibrary, device: MTLDevice, name: String) -> MTLComputePipelineState {
        guard let fn = library.makeFunction(name: name),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            fatalError("Could not create the \(name) compute pipeline.")
        }
        return pipeline
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        resizeTargets(width: Int(size.width), height: Int(size.height))
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        let width = drawable.texture.width
        let height = drawable.texture.height
        if width != targetWidth || height != targetHeight {
            resizeTargets(width: width, height: height)
        }
        guard let colorTex, let normalDepthTex, let posTex,
              let prevPosTex, let prevNormalTex, let output = outputTexture,
              let histPrev = histColor[frameParity],
              let histCurr = histColor[1 - frameParity],
              let atrous0 = atrousTex[0], let atrous1 = atrousTex[1] else { return }

        // --- Update camera from input ------------------------------------------
        let now = CACurrentMediaTime()
        let dt = Float(min(now - lastFrameTime, 0.1))
        lastFrameTime = now

        let gameView = view as? GameView
        let keys = gameView?.pressedKeys ?? []
        let mouseDelta = gameView?.consumeMouseDelta() ?? .zero
        let moved = camera.update(deltaTime: dt, pressedKeys: keys, mouseDelta: mouseDelta)

        if moved || settingsDirty {
            accumulatedFrames = 0
            settingsDirty = false
        }

        let aspect = Float(width) / Float(height)
        var uniforms = camera.makeUniforms(aspect: aspect,
                                           frameIndex: frameIndex,
                                           accumulatedFrames: accumulatedFrames)
        var settings = makeRenderSettings(width: UInt32(width), height: UInt32(height))

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // --- 1. Path tracing pass (writes color + G-buffer) ---------------------
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(pathTracePipeline)
            encoder.setAccelerationStructure(scene.tlas, bufferIndex: 0)
            encoder.setBytes(&uniforms, length: MemoryLayout<CameraUniforms>.stride, index: 1)
            encoder.setBytes(&settings, length: MemoryLayout<RenderSettings>.stride, index: 2)
            encoder.setBuffer(scene.primitiveBuffer, offset: 0, index: 3)
            encoder.setBuffer(scene.instanceOffsetBuffer, offset: 0, index: 4)
            encoder.setBuffer(scene.materialBuffer, offset: 0, index: 5)
            encoder.setTexture(colorTex, index: 0)
            encoder.setTexture(normalDepthTex, index: 1)
            encoder.setTexture(posTex, index: 2)
            for blas in scene.blases { encoder.useResource(blas, usage: .read) }
            dispatch(encoder, pipeline: pathTracePipeline, width: width, height: height)
            encoder.endEncoding()
        }

        var tonemapInput: MTLTexture = colorTex

        if denoiseEnabled {
            // --- 2. Temporal reprojection ---------------------------------------
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(temporalPipeline)
                encoder.setTexture(colorTex, index: 0)
                encoder.setTexture(normalDepthTex, index: 1)
                encoder.setTexture(posTex, index: 2)
                encoder.setTexture(prevPosTex, index: 3)
                encoder.setTexture(prevNormalTex, index: 4)
                encoder.setTexture(histPrev, index: 5)
                encoder.setTexture(histCurr, index: 6)
                encoder.setBytes(&uniforms, length: MemoryLayout<CameraUniforms>.stride, index: 0)
                dispatch(encoder, pipeline: temporalPipeline, width: width, height: height)
                encoder.endEncoding()
            }

            // --- 3. Edge-avoiding à-trous (3 iterations) ------------------------
            let steps: [UInt32] = [1, 2, 4]
            var src = histCurr
            var dst = atrous0
            for (i, var step) in steps.enumerated() {
                if let encoder = commandBuffer.makeComputeCommandEncoder() {
                    encoder.setComputePipelineState(atrousPipeline)
                    encoder.setTexture(src, index: 0)
                    encoder.setTexture(normalDepthTex, index: 1)
                    encoder.setTexture(posTex, index: 2)
                    encoder.setTexture(dst, index: 3)
                    encoder.setBytes(&step, length: MemoryLayout<UInt32>.stride, index: 0)
                    dispatch(encoder, pipeline: atrousPipeline, width: width, height: height)
                    encoder.endEncoding()
                }
                src = dst
                dst = (i % 2 == 0) ? atrous1 : atrous0
            }
            tonemapInput = src
        }

        // --- 4. Tone mapping into the shared output texture --------------------
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(tonemapPipeline)
            encoder.setTexture(tonemapInput, index: 0)
            encoder.setTexture(output, index: 1)
            var exp = exposure
            encoder.setBytes(&exp, length: MemoryLayout<Float>.stride, index: 0)
            dispatch(encoder, pipeline: tonemapPipeline, width: width, height: height)
            encoder.endEncoding()
        }

        // --- 5. Present + stash this frame's G-buffer for next-frame reprojection
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: output, to: drawable.texture)
            blit.copy(from: posTex, to: prevPosTex)
            blit.copy(from: normalDepthTex, to: prevNormalTex)
            blit.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()

        // Headless screenshot once the image has converged a little.
        if let path = screenshotPath, !screenshotSaved, accumulatedFrames >= 64 {
            commandBuffer.waitUntilCompleted()
            ScreenshotSaver.save(texture: output, to: path)
            screenshotSaved = true
        }

        frameParity = 1 - frameParity
        frameIndex &+= 1
        accumulatedFrames &+= 1
    }

    // MARK: - Helpers

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          width: Int, height: Int) {
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let tg = MTLSize(width: w, height: h, depth: 1)
        let groups = MTLSize(width: (width + w - 1) / w,
                             height: (height + h - 1) / h,
                             depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }

    private func resizeTargets(width: Int, height: Int) {
        guard width > 0, height > 0 else { return }
        targetWidth = width
        targetHeight = height

        func make(_ format: MTLPixelFormat, shared: Bool = false, _ label: String) -> MTLTexture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = shared ? .shared : .private
            guard let tex = device.makeTexture(descriptor: desc) else {
                fatalError("Failed to allocate texture \(label).")
            }
            tex.label = label
            return tex
        }

        colorTex = make(.rgba16Float, "Color1spp")
        normalDepthTex = make(.rgba16Float, "NormalDepth")
        posTex = make(.rgba32Float, "WorldPosMatId")
        prevPosTex = make(.rgba32Float, "PrevWorldPos")
        prevNormalTex = make(.rgba16Float, "PrevNormal")
        histColor = [make(.rgba16Float, "HistColorA"), make(.rgba16Float, "HistColorB")]
        atrousTex = [make(.rgba16Float, "AtrousA"), make(.rgba16Float, "AtrousB")]
        outputTexture = make(.bgra8Unorm, shared: true, "OutputLDR")

        accumulatedFrames = 0
    }

    /// Marks accumulation dirty so the next frame restarts convergence.
    func invalidateAccumulation() { settingsDirty = true }

    private func makeRenderSettings(width: UInt32, height: UInt32) -> RenderSettings {
        let sun = computeSun()
        return RenderSettings(
            sunDirection: PackedFloat3(sun.direction),
            sunIntensity: sun.intensity,
            sunColor: PackedFloat3(sun.color),
            turbidity: 2.5,
            timeOfDay: timeOfDay,
            exposure: exposure,
            maxBounces: maxBounces,
            denoiseEnabled: denoiseEnabled ? 1 : 0,
            waterRoughness: waterRoughness,
            elapsedTime: Float(CACurrentMediaTime() - startTime),
            width: width,
            height: height
        )
    }

    /// Computes the sun's direction, color, and intensity from the time of day.
    private func computeSun() -> (direction: SIMD3<Float>, color: SIMD3<Float>, intensity: Float) {
        let dayAngle = (timeOfDay - 0.25) * 2 * .pi
        let dir = normalize(SIMD3<Float>(cos(dayAngle), sin(dayAngle), 0.35))
        let elev = dir.y

        let t = smoothstepF(0.0, 0.35, elev)
        let warm = SIMD3<Float>(1.0, 0.46, 0.22)
        let noon = SIMD3<Float>(1.0, 0.97, 0.92)
        let color = mixVec(warm, noon, t: t)

        let intensity: Float = elev > 0 ? (0.15 + 0.85 * t) * sunStrength : 0
        return (dir, color, intensity)
    }

    private func smoothstepF(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    }
}

private func mixVec(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
    a + (b - a) * t
}
