import simd

/// A camera that supports two movement modes: free-fly (noclip) and walking with gravity and
/// voxel collision. Yaw/pitch mouse look and WASD movement work in both. Tracks the previous
/// frame's view-projection so the denoiser can reproject history.
final class Camera {
    var position: SIMD3<Float>
    var yaw: Float      // radians, around world up (Y)
    var pitch: Float    // radians, clamped to avoid gimbal flip

    var fovYRadians: Float = 70 * .pi / 180
    var moveSpeed: Float = 18    // fly speed, voxels/second
    var walkSpeed: Float = 7     // walk speed, voxels/second
    var lookSensitivity: Float = 0.0022

    /// When true the camera flies freely with no collision (the original behavior). When false
    /// the camera walks: gravity pulls it down, it collides with solid voxels, and space jumps.
    var flying: Bool = true

    // --- Walking physics ------------------------------------------------------
    /// Player capsule, approximated as an axis-aligned box. `position` is the eye location.
    private let eyeHeight: Float = 1.62
    private let playerHeight: Float = 1.8
    private let playerHalfWidth: Float = 0.3
    private let gravity: Float = -28        // voxels/second^2
    private let jumpSpeed: Float = 9.0      // initial upward velocity on jump
    private let terminalFallSpeed: Float = -56

    private var verticalVelocity: Float = 0
    private var grounded: Bool = false

    /// The world used for collision queries while walking. Set by the renderer.
    weak var world: VoxelWorld?

    private(set) var prevViewProj: simd_float4x4 = matrix_identity_float4x4
    private(set) var didMove: Bool = false

    init(position: SIMD3<Float>, yaw: Float = 0, pitch: Float = 0) {
        self.position = position
        self.yaw = yaw
        self.pitch = pitch
    }

    /// Toggles between flying and walking (driven by a double-tap of the space bar). When
    /// dropping into walk mode the vertical velocity is reset so the player doesn't lurch.
    func toggleFlight() {
        flying.toggle()
        verticalVelocity = 0
        grounded = false
    }

    var forward: SIMD3<Float> {
        SIMD3(cos(pitch) * sin(yaw),
              sin(pitch),
              -cos(pitch) * cos(yaw))
    }

    var right: SIMD3<Float> {
        normalize(cross(forward, SIMD3(0, 1, 0)))
    }

    var up: SIMD3<Float> {
        normalize(cross(right, forward))
    }

    /// Advances the camera from input. Returns whether the view changed this frame.
    @discardableResult
    func update(deltaTime: Float, pressedKeys: Set<String>, mouseDelta: SIMD2<Float>) -> Bool {
        var moved = false
        let startPos = position

        // Mouse look.
        if mouseDelta != .zero {
            yaw += mouseDelta.x * lookSensitivity
            pitch -= mouseDelta.y * lookSensitivity
            let limit = Float.pi / 2 - 0.01
            pitch = max(-limit, min(limit, pitch))
            moved = true
        }

        if flying {
            updateFlying(deltaTime: deltaTime, pressedKeys: pressedKeys)
        } else {
            updateWalking(deltaTime: deltaTime, pressedKeys: pressedKeys)
        }

        if length(position - startPos) > 1e-5 { moved = true }
        didMove = moved
        return moved
    }

    /// Free-fly movement: WASD plus space/E up and shift/Q down, no collision.
    private func updateFlying(deltaTime: Float, pressedKeys: Set<String>) {
        var velocity = SIMD3<Float>(0, 0, 0)
        let f = forward
        let r = right
        if pressedKeys.contains("w") { velocity += f }
        if pressedKeys.contains("s") { velocity -= f }
        if pressedKeys.contains("d") { velocity += r }
        if pressedKeys.contains("a") { velocity -= r }
        if pressedKeys.contains("e") || pressedKeys.contains(" ") { velocity += SIMD3(0, 1, 0) }
        if pressedKeys.contains("q") || pressedKeys.contains("shift") { velocity -= SIMD3(0, 1, 0) }

        let boost: Float = pressedKeys.contains("shift") && pressedKeys.contains("w") ? 2.5 : 1.0
        if velocity != .zero {
            position += normalize(velocity) * moveSpeed * boost * deltaTime
        }
    }

    /// Walking movement: WASD glides along the ground plane, gravity pulls down, solid voxels
    /// block movement, and space jumps when grounded.
    private func updateWalking(deltaTime: Float, pressedKeys: Set<String>) {
        // Horizontal intent, flattened onto the XZ plane so looking up/down doesn't slow you.
        let flatForward = normalizeSafe(SIMD3(forward.x, 0, forward.z))
        let flatRight = normalizeSafe(SIMD3(right.x, 0, right.z))
        var horiz = SIMD3<Float>(0, 0, 0)
        if pressedKeys.contains("w") { horiz += flatForward }
        if pressedKeys.contains("s") { horiz -= flatForward }
        if pressedKeys.contains("d") { horiz += flatRight }
        if pressedKeys.contains("a") { horiz -= flatRight }
        if horiz != .zero { horiz = normalize(horiz) }

        let sprint: Float = pressedKeys.contains("shift") ? 1.6 : 1.0
        let horizontalStep = horiz * walkSpeed * sprint * deltaTime

        // Jump on space when standing on the ground.
        if grounded && pressedKeys.contains(" ") {
            verticalVelocity = jumpSpeed
            grounded = false
        }

        // Integrate gravity.
        verticalVelocity = max(verticalVelocity + gravity * deltaTime, terminalFallSpeed)
        let verticalStep = verticalVelocity * deltaTime

        // Resolve movement per axis against solid voxels.
        moveAxis(SIMD3(horizontalStep.x, 0, 0))
        moveAxis(SIMD3(0, 0, horizontalStep.z))
        grounded = false
        moveAxisVertical(verticalStep)
    }

    /// Attempts a horizontal move along a single axis, cancelling it if the player would end up
    /// inside a solid voxel.
    private func moveAxis(_ delta: SIMD3<Float>) {
        let candidate = position + delta
        if !collides(at: candidate) {
            position = candidate
        }
    }

    /// Attempts a vertical move, stopping against floors/ceilings and updating the grounded flag.
    private func moveAxisVertical(_ dy: Float) {
        let candidate = position + SIMD3(0, dy, 0)
        if collides(at: candidate) {
            if dy < 0 { grounded = true }   // landed on a floor
            verticalVelocity = 0
        } else {
            position = candidate
        }
    }

    /// True if the player's axis-aligned box would overlap any solid voxel at `eye`.
    private func collides(at eye: SIMD3<Float>) -> Bool {
        guard let world else { return false }
        let hw = playerHalfWidth
        let feet = eye.y - eyeHeight
        let head = feet + playerHeight

        let x0 = Int(floor(eye.x - hw)), x1 = Int(floor(eye.x + hw))
        let z0 = Int(floor(eye.z - hw)), z1 = Int(floor(eye.z + hw))
        let y0 = Int(floor(feet)), y1 = Int(floor(head - 1e-4))

        for x in x0...x1 {
            for y in y0...y1 {
                for z in z0...z1 {
                    if Camera.isSolid(world.block(x, y, z)) { return true }
                }
            }
        }
        return false
    }

    /// Solid for collision purposes: everything except air and water (water is wadeable).
    private static func isSolid(_ block: BlockType) -> Bool {
        block != .air && block != .water
    }

    private func normalizeSafe(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let len = length(v)
        return len > 1e-6 ? v / len : .zero
    }

    func viewMatrix() -> simd_float4x4 {
        let f = normalize(forward)
        let r = normalize(cross(f, SIMD3(0, 1, 0)))
        let u = cross(r, f)
        let t = SIMD3(-dot(r, position), -dot(u, position), dot(f, position))
        return simd_float4x4(columns: (
            SIMD4(r.x, u.x, -f.x, 0),
            SIMD4(r.y, u.y, -f.y, 0),
            SIMD4(r.z, u.z, -f.z, 0),
            SIMD4(t.x, t.y, t.z, 1)
        ))
    }

    func projectionMatrix(aspect: Float) -> simd_float4x4 {
        let f = 1 / tan(fovYRadians / 2)
        let near: Float = 0.05
        let far: Float = 2000
        let zRange = far - near
        return simd_float4x4(columns: (
            SIMD4(f / aspect, 0, 0, 0),
            SIMD4(0, f, 0, 0),
            SIMD4(0, 0, -(far + near) / zRange, -1),
            SIMD4(0, 0, -2 * far * near / zRange, 0)
        ))
    }

    /// Builds the GPU uniform block and records this frame's matrices for next-frame reprojection.
    func makeUniforms(aspect: Float, frameIndex: UInt32, accumulatedFrames: UInt32) -> CameraUniforms {
        let view = viewMatrix()
        let proj = projectionMatrix(aspect: aspect)
        let viewProj = proj * view
        let invViewProj = viewProj.inverse

        let uniforms = CameraUniforms(
            viewProj: viewProj,
            invViewProj: invViewProj,
            prevViewProj: prevViewProj,
            position: PackedFloat3(position),
            forward: PackedFloat3(normalize(forward)),
            right: PackedFloat3(normalize(right)),
            up: PackedFloat3(normalize(up)),
            tanHalfFovY: tan(fovYRadians / 2),
            aspect: aspect,
            frameIndex: frameIndex,
            accumulatedFrames: accumulatedFrames
        )

        prevViewProj = viewProj
        return uniforms
    }
}
