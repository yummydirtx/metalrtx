import simd
import Foundation

/// Procedurally fills a `VoxelWorld` with rolling terrain, water bodies, beaches, trees, and
/// scattered emissive blocks. Uses deterministic value-noise fBm so a given seed always
/// produces the same world.
struct TerrainGenerator {
    var seed: UInt32 = 1337

    func generate(into world: VoxelWorld) {
        let w = world.sizeX
        let d = world.sizeZ
        let sea = world.seaLevel

        // --- Heightmap via fractal value noise --------------------------------------
        var height = [Int](repeating: 0, count: w * d)
        let baseFreq: Float = 1.0 / 48.0
        for z in 0..<d {
            for x in 0..<w {
                let fx = Float(x)
                let fz = Float(z)
                var h = fbm(fx * baseFreq, fz * baseFreq, octaves: 5)   // 0..1
                // Reshape: flatten lowlands, sharpen peaks for visual interest.
                h = pow(h, 1.6)
                let amplitude: Float = 38.0
                let elevation = Float(sea) - 8.0 + h * amplitude
                height[x + z * w] = Int(elevation.rounded())
            }
        }

        // --- Fill columns ------------------------------------------------------------
        for z in 0..<d {
            for x in 0..<w {
                let columnHeight = height[x + z * w]
                for y in 0...columnHeight where y < world.sizeY {
                    let block = surfaceBlock(x: x, y: y, z: z,
                                             columnHeight: columnHeight, sea: sea)
                    world.setBlock(x, y, z, block)
                }
                // Flood water up to sea level over low terrain.
                if columnHeight < sea {
                    for y in (columnHeight + 1)...sea where y < world.sizeY {
                        world.setBlock(x, y, z, .water)
                    }
                }
            }
        }

        // --- Decorate: trees + emissive accents -------------------------------------
        placeTrees(in: world, height: height)
        scatterLights(in: world, height: height)
    }

    // MARK: - Column composition

    private func surfaceBlock(x: Int, y: Int, z: Int, columnHeight: Int, sea: Int) -> BlockType {
        let depthFromTop = columnHeight - y
        let isTop = (y == columnHeight)

        if isTop {
            if columnHeight <= sea + 1 {
                return .sand                    // beach line
            } else if columnHeight > sea + 26 {
                return .snow                    // snowy peaks
            } else {
                return .grass
            }
        }
        if depthFromTop <= 3 {
            return columnHeight <= sea + 1 ? .sand : .dirt
        }
        return .stone
    }

    // MARK: - Trees

    private func placeTrees(in world: VoxelWorld, height: [Int]) {
        let w = world.sizeX
        let d = world.sizeZ
        let sea = world.seaLevel
        for z in 2..<(d - 2) {
            for x in 2..<(w - 2) {
                let top = height[x + z * w]
                guard top > sea + 2, top < world.sizeY - 8 else { continue }
                guard world.block(x, top, z) == .grass else { continue }
                // Sparse, deterministic placement.
                if hash3(x, z, 7919) > 0.987 {
                    growTree(in: world, x: x, baseY: top + 1, z: z)
                }
            }
        }
    }

    private func growTree(in world: VoxelWorld, x: Int, baseY: Int, z: Int) {
        let trunkHeight = 4 + Int(hash3(x, z, 17) * 3.0)
        for i in 0..<trunkHeight {
            world.setBlock(x, baseY + i, z, .wood)
        }
        let crownY = baseY + trunkHeight
        // Spherical-ish leaf crown.
        for dy in -2...2 {
            for dz in -2...2 {
                for dx in -2...2 {
                    let r = Float(dx * dx + dy * dy + dz * dz)
                    if r <= 5.2 {
                        let lx = x + dx, ly = crownY + dy, lz = z + dz
                        if world.block(lx, ly, lz) == .air {
                            world.setBlock(lx, ly, lz, .leaves)
                        }
                    }
                }
            }
        }
        world.setBlock(x, crownY + 2, z, .leaves)
    }

    // MARK: - Emissive accents

    private func scatterLights(in world: VoxelWorld, height: [Int]) {
        let w = world.sizeX
        let d = world.sizeZ
        let sea = world.seaLevel
        for z in 0..<d {
            for x in 0..<w {
                let top = height[x + z * w]
                guard top > sea + 1, top < world.sizeY - 2 else { continue }
                if hash3(x, z, 4211) > 0.9955 {
                    // A lantern post for night-time ambience.
                    world.setBlock(x, top + 1, z, .wood)
                    world.setBlock(x, top + 2, z, .lantern)
                } else if hash3(x, z, 2749) > 0.997 {
                    world.setBlock(x, top + 1, z, .glowstone)
                }
            }
        }
    }

    // MARK: - Noise

    /// Hash a 2D integer coordinate (+salt) to a float in [0, 1).
    private func hash3(_ x: Int, _ z: Int, _ salt: Int) -> Float {
        var h = UInt32(bitPattern: Int32(truncatingIfNeeded: x &* 374761393
            &+ z &* 668265263 &+ salt &* 362437 &+ Int(seed)))
        h = (h ^ (h >> 13)) &* 1274126177
        h = h ^ (h >> 16)
        return Float(h) / Float(UInt32.max)
    }

    /// Smooth value noise in [0, 1).
    private func valueNoise(_ x: Float, _ z: Float) -> Float {
        let x0 = Int(floor(x)), z0 = Int(floor(z))
        let fx = x - Float(x0), fz = z - Float(z0)
        let ux = fx * fx * (3 - 2 * fx)
        let uz = fz * fz * (3 - 2 * fz)
        let n00 = hash3(x0,     z0,     11)
        let n10 = hash3(x0 + 1, z0,     11)
        let n01 = hash3(x0,     z0 + 1, 11)
        let n11 = hash3(x0 + 1, z0 + 1, 11)
        let nx0 = n00 + (n10 - n00) * ux
        let nx1 = n01 + (n11 - n01) * ux
        return nx0 + (nx1 - nx0) * uz
    }

    private func fbm(_ x: Float, _ z: Float, octaves: Int) -> Float {
        var sum: Float = 0
        var amp: Float = 0.5
        var freq: Float = 1
        var norm: Float = 0
        for _ in 0..<octaves {
            sum += amp * valueNoise(x * freq, z * freq)
            norm += amp
            amp *= 0.5
            freq *= 2.0
        }
        return sum / norm
    }
}
