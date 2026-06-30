import simd

/// Dense voxel storage for the whole world. Small enough for a tech demo to keep entirely
/// resident, which keeps neighbor sampling (needed by the greedy mesher) trivial and exact
/// across chunk boundaries. The world is logically divided into cubic chunks; each chunk
/// becomes one bottom-level acceleration structure instanced into the scene.
final class VoxelWorld {
    let chunkSize: Int
    let chunksX: Int
    let chunksY: Int
    let chunksZ: Int

    let sizeX: Int
    let sizeY: Int
    let sizeZ: Int

    /// Sea level in voxels; everything below is flooded with water during generation.
    let seaLevel: Int

    private var blocks: [UInt8]

    init(chunkSize: Int = 32,
         chunksX: Int = 4,
         chunksY: Int = 3,
         chunksZ: Int = 4,
         seaLevel: Int = 34) {
        self.chunkSize = chunkSize
        self.chunksX = chunksX
        self.chunksY = chunksY
        self.chunksZ = chunksZ
        self.sizeX = chunkSize * chunksX
        self.sizeY = chunkSize * chunksY
        self.sizeZ = chunkSize * chunksZ
        self.seaLevel = seaLevel
        self.blocks = [UInt8](repeating: 0, count: sizeX * sizeY * sizeZ)
    }

    @inline(__always)
    func inBounds(_ x: Int, _ y: Int, _ z: Int) -> Bool {
        x >= 0 && y >= 0 && z >= 0 && x < sizeX && y < sizeY && z < sizeZ
    }

    @inline(__always)
    private func index(_ x: Int, _ y: Int, _ z: Int) -> Int {
        x + y * sizeX + z * sizeX * sizeY
    }

    /// Returns the block at world coordinates, treating anything outside the world as air.
    @inline(__always)
    func block(_ x: Int, _ y: Int, _ z: Int) -> BlockType {
        guard inBounds(x, y, z) else { return .air }
        return BlockType(rawValue: blocks[index(x, y, z)]) ?? .air
    }

    @inline(__always)
    func setBlock(_ x: Int, _ y: Int, _ z: Int, _ type: BlockType) {
        guard inBounds(x, y, z) else { return }
        blocks[index(x, y, z)] = type.rawValue
    }

    /// The list of chunks in generation/iteration order, as integer origins (in voxels).
    var chunkOrigins: [SIMD3<Int>] {
        var origins: [SIMD3<Int>] = []
        origins.reserveCapacity(chunksX * chunksY * chunksZ)
        for cz in 0..<chunksZ {
            for cy in 0..<chunksY {
                for cx in 0..<chunksX {
                    origins.append(SIMD3(cx, cy, cz) &* chunkSize)
                }
            }
        }
        return origins
    }
}
