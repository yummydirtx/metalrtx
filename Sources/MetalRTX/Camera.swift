import simd

/// A free-fly camera with yaw/pitch mouse look and WASD+QE movement. Tracks the previous
/// frame's view-projection so the denoiser can reproject history.
final class Camera {
    var position: SIMD3<Float>
    var yaw: Float      // radians, around world up (Y)
    var pitch: Float    // radians, clamped to avoid gimbal flip

    var fovYRadians: Float = 70 * .pi / 180
    var moveSpeed: Float = 18    // voxels/second
    var lookSensitivity: Float = 0.0022

    private(set) var prevViewProj: simd_float4x4 = matrix_identity_float4x4
    private(set) var didMove: Bool = false

    init(position: SIMD3<Float>, yaw: Float = 0, pitch: Float = 0) {
        self.position = position
        self.yaw = yaw
        self.pitch = pitch
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

        // Mouse look.
        if mouseDelta != .zero {
            yaw += mouseDelta.x * lookSensitivity
            pitch -= mouseDelta.y * lookSensitivity
            let limit = Float.pi / 2 - 0.01
            pitch = max(-limit, min(limit, pitch))
            moved = true
        }

        // Movement.
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
            moved = true
        }

        didMove = moved
        return moved
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
