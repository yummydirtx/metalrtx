import Metal
import simd

/// The GPU-resident scene the path tracer traverses: a top-level acceleration structure over
/// per-chunk bottom-level structures, plus the buffers needed to shade hits.
struct Scene {
    let tlas: MTLAccelerationStructure
    /// Bottom-level structures, retained because the TLAS references them.
    let blases: [MTLAccelerationStructure]
    /// Per-triangle normal + material index for the whole world, indexed by
    /// `instanceOffset[instance_id] + primitive_id`.
    let primitiveBuffer: MTLBuffer
    /// Per-instance base offset (in triangles) into `primitiveBuffer`.
    let instanceOffsetBuffer: MTLBuffer
    /// Packed material table indexed by `materialIndex`.
    let materialBuffer: MTLBuffer

    let instanceCount: Int
    let triangleCount: Int
}

/// Builds hardware ray-tracing acceleration structures from a voxel world.
enum AccelerationStructureBuilder {
    static func build(world: VoxelWorld,
                      device: MTLDevice,
                      commandQueue: MTLCommandQueue) -> Scene {
        // --- 1. Greedy-mesh every chunk; keep only non-empty ones -------------------
        var positionBuffers: [MTLBuffer] = []
        var indexBuffers: [MTLBuffer] = []
        var triangleCounts: [Int] = []
        var instanceTranslations: [SIMD3<Float>] = []
        var primitiveData: [GPUPrimitiveData] = []
        var instanceOffsets: [UInt32] = []

        var totalTriangles = 0

        for origin in world.chunkOrigins {
            let mesh = GreedyMesher.mesh(world: world, chunkOrigin: origin)
            if mesh.isEmpty { continue }

            // Pack positions tightly (12-byte stride) for the AS vertex buffer.
            var packedPositions = [PackedFloat3]()
            packedPositions.reserveCapacity(mesh.positions.count)
            for p in mesh.positions { packedPositions.append(PackedFloat3(p)) }

            guard let posBuf = device.makeBuffer(bytes: packedPositions,
                                                 length: packedPositions.count * MemoryLayout<PackedFloat3>.stride,
                                                 options: .storageModeShared),
                  let idxBuf = device.makeBuffer(bytes: mesh.indices,
                                                 length: mesh.indices.count * MemoryLayout<UInt32>.stride,
                                                 options: .storageModeShared) else {
                fatalError("Failed to allocate chunk geometry buffers.")
            }
            posBuf.label = "ChunkPositions"
            idxBuf.label = "ChunkIndices"

            instanceOffsets.append(UInt32(totalTriangles))
            for t in 0..<mesh.triangleCount {
                primitiveData.append(GPUPrimitiveData(normal: PackedFloat3(mesh.triNormals[t]),
                                                      materialIndex: mesh.triMaterials[t]))
            }

            positionBuffers.append(posBuf)
            indexBuffers.append(idxBuf)
            triangleCounts.append(mesh.triangleCount)
            instanceTranslations.append(SIMD3<Float>(Float(origin.x), Float(origin.y), Float(origin.z)))
            totalTriangles += mesh.triangleCount
        }

        precondition(!positionBuffers.isEmpty, "World produced no geometry.")

        // --- 2. Build a bottom-level acceleration structure per chunk ---------------
        var blasDescriptors: [MTLPrimitiveAccelerationStructureDescriptor] = []
        for i in positionBuffers.indices {
            let geo = MTLAccelerationStructureTriangleGeometryDescriptor()
            geo.vertexBuffer = positionBuffers[i]
            geo.vertexBufferOffset = 0
            geo.vertexStride = MemoryLayout<PackedFloat3>.stride
            geo.vertexFormat = .float3
            geo.indexBuffer = indexBuffers[i]
            geo.indexType = .uint32
            geo.triangleCount = triangleCounts[i]
            geo.opaque = true

            let desc = MTLPrimitiveAccelerationStructureDescriptor()
            desc.geometryDescriptors = [geo]
            blasDescriptors.append(desc)
        }

        var blases: [MTLAccelerationStructure] = []
        var maxScratch = 0
        var blasSizes: [MTLAccelerationStructureSizes] = []
        for desc in blasDescriptors {
            let sizes = device.accelerationStructureSizes(descriptor: desc)
            blasSizes.append(sizes)
            maxScratch = max(maxScratch, sizes.buildScratchBufferSize)
            guard let blas = device.makeAccelerationStructure(size: sizes.accelerationStructureSize) else {
                fatalError("Failed to allocate a bottom-level acceleration structure.")
            }
            blas.label = "ChunkBLAS"
            blases.append(blas)
        }

        guard let scratch = device.makeBuffer(length: max(maxScratch, 1),
                                              options: .storageModePrivate) else {
            fatalError("Failed to allocate BLAS scratch buffer.")
        }

        // Build each BLAS. Reusing one scratch buffer forces serialization, which is fine
        // for a one-time build and keeps memory low.
        for (i, desc) in blasDescriptors.enumerated() {
            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeAccelerationStructureCommandEncoder() else {
                fatalError("Failed to create AS build command encoder.")
            }
            enc.build(accelerationStructure: blases[i],
                      descriptor: desc,
                      scratchBuffer: scratch,
                      scratchBufferOffset: 0)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        // --- 3. Build the instanced top-level acceleration structure ----------------
        let instanceCount = blases.count
        let instanceStride = MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride
        guard let instanceBuffer = device.makeBuffer(length: instanceStride * instanceCount,
                                                     options: .storageModeShared) else {
            fatalError("Failed to allocate instance descriptor buffer.")
        }
        let instancePtr = instanceBuffer.contents()
            .bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: instanceCount)

        for i in 0..<instanceCount {
            var inst = MTLAccelerationStructureInstanceDescriptor()
            inst.transformationMatrix = packedTranslation(instanceTranslations[i])
            inst.options = .opaque
            inst.mask = 0xFF
            inst.intersectionFunctionTableOffset = 0
            inst.accelerationStructureIndex = UInt32(i)
            instancePtr[i] = inst
        }

        let tlasDesc = MTLInstanceAccelerationStructureDescriptor()
        tlasDesc.instancedAccelerationStructures = blases
        tlasDesc.instanceCount = instanceCount
        tlasDesc.instanceDescriptorBuffer = instanceBuffer

        let tlasSizes = device.accelerationStructureSizes(descriptor: tlasDesc)
        guard let tlas = device.makeAccelerationStructure(size: tlasSizes.accelerationStructureSize) else {
            fatalError("Failed to allocate the top-level acceleration structure.")
        }
        tlas.label = "WorldTLAS"
        guard let tlasScratch = device.makeBuffer(length: max(tlasSizes.buildScratchBufferSize, 1),
                                                  options: .storageModePrivate) else {
            fatalError("Failed to allocate TLAS scratch buffer.")
        }

        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeAccelerationStructureCommandEncoder() else {
            fatalError("Failed to create TLAS build command encoder.")
        }
        enc.build(accelerationStructure: tlas,
                  descriptor: tlasDesc,
                  scratchBuffer: tlasScratch,
                  scratchBufferOffset: 0)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // --- 4. Upload shading buffers ----------------------------------------------
        let materials = BlockType.materialTable.map { GPUMaterial($0) }
        guard let materialBuffer = device.makeBuffer(bytes: materials,
                                                     length: materials.count * MemoryLayout<GPUMaterial>.stride,
                                                     options: .storageModeShared),
              let primitiveBuffer = device.makeBuffer(bytes: primitiveData,
                                                      length: max(primitiveData.count, 1) * MemoryLayout<GPUPrimitiveData>.stride,
                                                      options: .storageModeShared),
              let instanceOffsetBuffer = device.makeBuffer(bytes: instanceOffsets,
                                                          length: instanceOffsets.count * MemoryLayout<UInt32>.stride,
                                                          options: .storageModeShared) else {
            fatalError("Failed to allocate shading buffers.")
        }

        print("Scene built: \(instanceCount) chunks, \(totalTriangles) triangles.")

        return Scene(tlas: tlas,
                     blases: blases,
                     primitiveBuffer: primitiveBuffer,
                     instanceOffsetBuffer: instanceOffsetBuffer,
                     materialBuffer: materialBuffer,
                     instanceCount: instanceCount,
                     triangleCount: totalTriangles)
    }

    /// Builds a 3×4 affine transform (column-major) that is pure translation.
    private static func packedTranslation(_ t: SIMD3<Float>) -> MTLPackedFloat4x3 {
        var m = MTLPackedFloat4x3()
        m.columns.0 = MTLPackedFloat3Make(1, 0, 0)
        m.columns.1 = MTLPackedFloat3Make(0, 1, 0)
        m.columns.2 = MTLPackedFloat3Make(0, 0, 1)
        m.columns.3 = MTLPackedFloat3Make(t.x, t.y, t.z)
        return m
    }
}
