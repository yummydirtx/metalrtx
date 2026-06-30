import simd

// =============================================================================
// Swift mirrors of the GPU struct layouts declared in Common.metal. Field order,
// sizes, and alignment MUST match exactly. All members are 4-byte aligned and
// `packed_float3` is represented by `PackedFloat3` (12 bytes, no padding).
// =============================================================================

/// 12-byte tightly packed float3, matching Metal's `packed_float3`.
struct PackedFloat3 {
    var x: Float
    var y: Float
    var z: Float

    init(_ x: Float, _ y: Float, _ z: Float) {
        self.x = x; self.y = y; self.z = z
    }
    init(_ v: SIMD3<Float>) {
        self.x = v.x; self.y = v.y; self.z = v.z
    }
    init() { self.x = 0; self.y = 0; self.z = 0 }
}

/// Mirrors `Material` (68 bytes).
struct GPUMaterial {
    var albedo: PackedFloat3
    var roughness: Float
    var emission: PackedFloat3
    var metallic: Float
    var transparency: Float
    var ior: Float
    var flags: UInt32
    var detailScale: Float
    var detailStrength: Float
    var bumpStrength: Float
    var absorption: PackedFloat3

    init(_ def: MaterialDef) {
        albedo = PackedFloat3(def.albedo)
        roughness = def.roughness
        emission = PackedFloat3(def.emission)
        metallic = def.metallic
        transparency = def.transparency
        ior = def.ior
        flags = def.isWater ? 1 : 0
        detailScale = def.detailScale
        detailStrength = def.detailStrength
        bumpStrength = def.bumpStrength
        absorption = PackedFloat3(def.absorption)
    }
}

/// Mirrors `PrimitiveData` (16 bytes).
struct GPUPrimitiveData {
    var normal: PackedFloat3
    var materialIndex: UInt32
}

/// Mirrors `CameraUniforms` (272 bytes, 16-byte aligned).
struct CameraUniforms {
    var viewProj: simd_float4x4
    var invViewProj: simd_float4x4
    var prevViewProj: simd_float4x4
    var position: PackedFloat3
    var _pad0: Float = 0
    var forward: PackedFloat3
    var _pad1: Float = 0
    var right: PackedFloat3
    var _pad2: Float = 0
    var up: PackedFloat3
    var _pad3: Float = 0
    var tanHalfFovY: Float
    var aspect: Float
    var frameIndex: UInt32
    var accumulatedFrames: UInt32
}

/// Mirrors `RenderSettings` (108 bytes).
struct RenderSettings {
    var sunDirection: PackedFloat3
    var sunIntensity: Float
    var sunColor: PackedFloat3
    var turbidity: Float
    var timeOfDay: Float
    var exposure: Float
    var maxBounces: UInt32
    var denoiseEnabled: UInt32
    var waterRoughness: Float
    var waveAmplitude: Float
    var waveChoppiness: Float
    var waveSpeed: Float
    var elapsedTime: Float
    var width: UInt32
    var height: UInt32
    var flashlightEnabled: UInt32
    var flashlightPos: PackedFloat3
    var flashlightDir: PackedFloat3
    var fogEnabled: UInt32
    var sunAngularRadius: Float
    var emitterCount: UInt32
}

/// Mirrors `Emitter` (32 bytes).
struct GPUEmitter {
    var position: PackedFloat3
    var radius: Float
    var emission: PackedFloat3
    var pad: Float = 0

    init(position: SIMD3<Float>, radius: Float, emission: SIMD3<Float>) {
        self.position = PackedFloat3(position)
        self.radius = radius
        self.emission = PackedFloat3(emission)
    }
}
