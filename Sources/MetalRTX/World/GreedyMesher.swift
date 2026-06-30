import simd

/// Triangle geometry for a single chunk, in chunk-local coordinates (0...chunkSize). The
/// instance transform translates it to world space. Positions feed the acceleration structure;
/// per-triangle normals and material indices are looked up by the path tracer at hit time.
struct ChunkMesh {
    var positions: [SIMD3<Float>] = []
    var indices: [UInt32] = []
    var triNormals: [SIMD3<Float>] = []
    var triMaterials: [UInt32] = []

    var triangleCount: Int { indices.count / 3 }
    var isEmpty: Bool { indices.isEmpty }
}

/// Greedy mesher: merges coplanar, same-material voxel faces into the largest possible quads,
/// dramatically reducing triangle count versus naive per-face meshing. Each visible surface is
/// emitted exactly once, owned by the higher-"occlusion rank" voxel, so opaque/water/glass
/// boundaries are handled without double-sided overdraw or light-leaking seams.
enum GreedyMesher {
    /// Maps a sweep direction d to its two in-plane axes (u, v).
    private static let axisU = [1, 2, 0]
    private static let axisV = [2, 0, 1]

    static func mesh(world: VoxelWorld, chunkOrigin: SIMD3<Int>) -> ChunkMesh {
        let size = world.chunkSize
        var mesh = ChunkMesh()

        for d in 0..<3 {
            let u = axisU[d]
            let v = axisV[d]

            for s in [1, -1] {
                for layer in 0..<size {
                    // Build the (u, v) mask of faces owned by cells in this layer/side.
                    var mask = [Int32](repeating: -1, count: size * size)
                    for vi in 0..<size {
                        for ui in 0..<size {
                            var c = SIMD3<Int>(0, 0, 0)
                            c[d] = layer; c[u] = ui; c[v] = vi
                            let cell = world.block(chunkOrigin.x + c.x,
                                                   chunkOrigin.y + c.y,
                                                   chunkOrigin.z + c.z)
                            if cell.isAir { continue }

                            var nc = c; nc[d] = layer + s
                            let neighbor = world.block(chunkOrigin.x + nc.x,
                                                       chunkOrigin.y + nc.y,
                                                       chunkOrigin.z + nc.z)

                            if cell.occlusionRank > neighbor.occlusionRank {
                                mask[ui + vi * size] = Int32(cell.rawValue)
                            }
                        }
                    }

                    greedyMerge(mask: &mask, size: size) { i, j, wdt, hgt, material in
                        appendQuad(into: &mesh,
                                   d: d, u: u, v: v, s: s, layer: layer,
                                   i: i, j: j, wdt: wdt, hgt: hgt,
                                   material: UInt32(material))
                    }
                }
            }
        }

        return mesh
    }

    /// Standard greedy rectangle extraction over a 2D material mask. Calls `emit` for each
    /// merged rectangle and clears it from the mask.
    private static func greedyMerge(mask: inout [Int32],
                                    size: Int,
                                    emit: (_ i: Int, _ j: Int, _ wdt: Int, _ hgt: Int, _ material: Int32) -> Void) {
        for j in 0..<size {
            var i = 0
            while i < size {
                let m = mask[i + j * size]
                if m < 0 { i += 1; continue }

                var wdt = 1
                while i + wdt < size && mask[(i + wdt) + j * size] == m { wdt += 1 }

                var hgt = 1
                heightLoop: while j + hgt < size {
                    for k in 0..<wdt where mask[(i + k) + (j + hgt) * size] != m {
                        break heightLoop
                    }
                    hgt += 1
                }

                emit(i, j, wdt, hgt, m)

                for dj in 0..<hgt {
                    for di in 0..<wdt {
                        mask[(i + di) + (j + dj) * size] = -1
                    }
                }
                i += wdt
            }
        }
    }

    private static func appendQuad(into mesh: inout ChunkMesh,
                                   d: Int, u: Int, v: Int, s: Int, layer: Int,
                                   i: Int, j: Int, wdt: Int, hgt: Int,
                                   material: UInt32) {
        var base = SIMD3<Float>(repeating: 0)
        base[d] = Float(s > 0 ? layer + 1 : layer)
        base[u] = Float(i)
        base[v] = Float(j)

        var eu = SIMD3<Float>(repeating: 0); eu[u] = Float(wdt)
        var ev = SIMD3<Float>(repeating: 0); ev[v] = Float(hgt)

        let p0 = base
        let p1 = base + eu
        let p2 = base + eu + ev
        let p3 = base + ev

        var normal = SIMD3<Float>(repeating: 0)
        normal[d] = Float(s)

        let baseIndex = UInt32(mesh.positions.count)
        mesh.positions.append(contentsOf: [p0, p1, p2, p3])

        // Wind triangles so the front face matches the outward normal.
        if s > 0 {
            mesh.indices.append(contentsOf: [
                baseIndex, baseIndex + 1, baseIndex + 2,
                baseIndex, baseIndex + 2, baseIndex + 3
            ])
        } else {
            mesh.indices.append(contentsOf: [
                baseIndex, baseIndex + 2, baseIndex + 1,
                baseIndex, baseIndex + 3, baseIndex + 2
            ])
        }

        mesh.triNormals.append(normal)
        mesh.triNormals.append(normal)
        mesh.triMaterials.append(material)
        mesh.triMaterials.append(material)
    }
}
