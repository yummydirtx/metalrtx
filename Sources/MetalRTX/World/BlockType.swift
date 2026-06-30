import simd

/// CPU-side description of a block's surface appearance. Converted to the packed GPU
/// `Material` layout when the material table is uploaded.
struct MaterialDef {
    var albedo: SIMD3<Float>
    var roughness: Float
    var emission: SIMD3<Float>
    var metallic: Float
    var transparency: Float
    var ior: Float
    var isWater: Bool
    var detailScale: Float
    var detailStrength: Float
    var bumpStrength: Float
    var absorption: SIMD3<Float>

    init(albedo: SIMD3<Float>,
         roughness: Float = 0.9,
         emission: SIMD3<Float> = .zero,
         metallic: Float = 0.0,
         transparency: Float = 0.0,
         ior: Float = 1.5,
         isWater: Bool = false,
         detailScale: Float = 0.0,
         detailStrength: Float = 0.0,
         bumpStrength: Float = 0.0,
         absorption: SIMD3<Float> = .zero) {
        self.albedo = albedo
        self.roughness = roughness
        self.emission = emission
        self.metallic = metallic
        self.transparency = transparency
        self.ior = ior
        self.isWater = isWater
        self.detailScale = detailScale
        self.detailStrength = detailStrength
        self.bumpStrength = bumpStrength
        self.absorption = absorption
    }
}

/// All block types in the world. The raw value doubles as the material-table index.
enum BlockType: UInt8, CaseIterable {
    case air = 0
    case grass
    case dirt
    case stone
    case sand
    case water
    case glass
    case glowstone
    case wood
    case leaves
    case snow
    case lantern

    /// Render-time opacity rank used by the greedy mesher. A face is emitted between two
    /// voxels only when their ranks differ, and it belongs to the higher-ranked voxel. This
    /// yields exactly one face per visible surface with no double-sided overdraw.
    /// air (empty) < water/glass (transparent) < opaque.
    var occlusionRank: Int {
        switch self {
        case .air: return 0
        case .water, .glass: return 1
        default: return 2
        }
    }

    var isAir: Bool { self == .air }

    /// Appearance for each block. Colors are linear (not sRGB).
    var material: MaterialDef {
        switch self {
        case .air:
            return MaterialDef(albedo: .zero)
        case .grass:
            return MaterialDef(albedo: SIMD3(0.20, 0.55, 0.16), roughness: 0.95,
                               detailScale: 0.7, detailStrength: 0.35, bumpStrength: 0.12)
        case .dirt:
            return MaterialDef(albedo: SIMD3(0.36, 0.24, 0.14), roughness: 0.98,
                               detailScale: 1.1, detailStrength: 0.45, bumpStrength: 0.25)
        case .stone:
            return MaterialDef(albedo: SIMD3(0.42, 0.42, 0.45), roughness: 0.85,
                               detailScale: 0.9, detailStrength: 0.40, bumpStrength: 0.30)
        case .sand:
            return MaterialDef(albedo: SIMD3(0.80, 0.72, 0.48), roughness: 0.92,
                               detailScale: 2.5, detailStrength: 0.20, bumpStrength: 0.10)
        case .water:
            return MaterialDef(albedo: SIMD3(0.02, 0.10, 0.16),
                               roughness: 0.02,
                               transparency: 0.92,
                               ior: 1.33,
                               isWater: true,
                               absorption: SIMD3(0.45, 0.10, 0.06))
        case .glass:
            return MaterialDef(albedo: SIMD3(0.85, 0.92, 0.95),
                               roughness: 0.02,
                               transparency: 0.95,
                               ior: 1.5,
                               absorption: SIMD3(0.03, 0.01, 0.02))
        case .glowstone:
            return MaterialDef(albedo: SIMD3(0.95, 0.78, 0.45),
                               roughness: 0.7,
                               emission: SIMD3(7.0, 4.6, 1.8))
        case .wood:
            return MaterialDef(albedo: SIMD3(0.30, 0.20, 0.10), roughness: 0.9,
                               detailScale: 1.0, detailStrength: 0.30, bumpStrength: 0.15)
        case .leaves:
            return MaterialDef(albedo: SIMD3(0.12, 0.36, 0.10), roughness: 0.95,
                               detailScale: 1.8, detailStrength: 0.40, bumpStrength: 0.10)
        case .snow:
            return MaterialDef(albedo: SIMD3(0.90, 0.92, 0.96), roughness: 0.6,
                               detailScale: 1.5, detailStrength: 0.10, bumpStrength: 0.06)
        case .lantern:
            return MaterialDef(albedo: SIMD3(0.95, 0.85, 0.55),
                               roughness: 0.5,
                               emission: SIMD3(9.0, 6.0, 2.5))
        }
    }

    /// The full material table in raw-value order, ready for GPU upload.
    static var materialTable: [MaterialDef] {
        allCases.sorted { $0.rawValue < $1.rawValue }.map(\.material)
    }
}
