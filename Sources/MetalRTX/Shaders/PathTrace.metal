// =============================================================================
// PathTrace.metal — progressive Monte-Carlo path tracer driven by Metal's
// hardware ray-tracing intersector. One sample per pixel per frame is traced and
// accumulated into an HDR buffer; the image converges while the camera holds still.
// Supports Lambertian diffuse (with sun next-event estimation), rough metal, and
// dielectric (water/glass) surfaces.
// =============================================================================

// Returns true if a shadow ray from `o` along `d` hits any geometry within `maxDist`.
inline bool traceShadow(instance_acceleration_structure tlas,
                        float3 o, float3 d, float maxDist) {
    intersector<triangle_data, instancing> sh;
    sh.assume_geometry_type(geometry_type::triangle);
    sh.force_opacity(forced_opacity::opaque);
    sh.accept_any_intersection(true);

    ray r;
    r.origin = o;
    r.direction = d;
    r.min_distance = 1e-3f;
    r.max_distance = maxDist;

    auto hit = sh.intersect(r, tlas);
    return hit.type != intersection_type::none;
}

// Traces a single Monte-Carlo sub-path starting at (ro, rd) and returns the
// radiance it carries back. This is the shared body used both for the camera
// path and for the reflection / refraction sub-paths spawned by the primary
// water surface. Deeper dielectric hits stay stochastic (low contribution).
inline float3 traceRadiance(instance_acceleration_structure tlas,
                            device const PrimitiveData *primitives,
                            device const uint *instanceOffsets,
                            device const Material *materials,
                            constant RenderSettings &settings,
                            float3 sunDir,
                            float3 ro, float3 rd, float3 throughput,
                            uint maxBounces,
                            thread float &firstHitT,
                            thread uint &seed) {
    float3 radiance = float3(0.0f);

    // Distance from the sub-path origin to its first hit. Used to build a
    // parallax-correct "virtual" position for specular reprojection. Defaults to
    // a far distance so sky-only reflections reproject like a distant skybox.
    firstHitT = 1.0e4f;

    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    for (uint bounce = 0; bounce <= maxBounces; ++bounce) {
        ray r;
        r.origin = ro;
        r.direction = rd;
        r.min_distance = 1e-3f;
        r.max_distance = 1e4f;

        auto hit = isect.intersect(r, tlas);
        if (hit.type == intersection_type::none) {
            if (bounce == 0u) firstHitT = 1.0e4f;
            radiance += throughput * skyColor(rd, settings);
            break;
        }

        if (bounce == 0u) firstHitT = hit.distance;

        uint primIndex = instanceOffsets[hit.instance_id] + hit.primitive_id;
        PrimitiveData prim = primitives[primIndex];
        Material mat = materials[prim.materialIndex];

        float3 N = normalize(float3(prim.normal));
        float3 hitP = ro + rd * hit.distance;
        bool backface = dot(N, rd) > 0.0f;
        float3 Ns = backface ? -N : N;

        radiance += throughput * float3(mat.emission);

        if (mat.transparency > 0.5f) {
            if (mat.flags & MATERIAL_FLAG_WATER) {
                float up = clamp(Ns.y, 0.0f, 1.0f);
                float3 wn = waterNormal(hitP.xz, settings.elapsedTime, settings.waterRoughness);
                Ns = normalize(mix(Ns, wn, up));
            }

            float etaI = 1.0f, etaT = mat.ior;
            if (backface) { float tmp = etaI; etaI = etaT; etaT = tmp; }
            float eta = etaI / etaT;

            float cosi = clamp(dot(-rd, Ns), 0.0f, 1.0f);
            float r0 = (etaI - etaT) / (etaI + etaT);
            r0 = r0 * r0;
            float fres = r0 + (1.0f - r0) * pow(1.0f - cosi, 5.0f);

            float3 refr = refract(rd, Ns, eta);
            bool tir = dot(refr, refr) < 1e-6f;

            float3 newDir;
            if (tir || randFloat(seed) < fres) {
                newDir = reflect(rd, Ns);
            } else {
                newDir = normalize(refr);
                throughput *= mix(float3(1.0f), float3(mat.albedo) * 4.0f + 0.6f, 0.25f);
            }
            ro = hitP + newDir * 1e-3f;
            rd = newDir;
        } else if (mat.metallic > 0.5f) {
            float roughness = max(mat.roughness, 0.02f);
            float3 H = sampleGGX(Ns, roughness, rand2(seed));
            float3 newDir = reflect(rd, H);
            if (dot(newDir, Ns) <= 0.0f) break;
            ro = hitP + Ns * 1e-3f;
            rd = newDir;
            throughput *= float3(mat.albedo);
        } else {
            float3 albedo = float3(mat.albedo);
            float NdotL = max(dot(Ns, sunDir), 0.0f);
            if (NdotL > 0.0f) {
                float3 shadowOrigin = hitP + Ns * 1e-3f;
                if (!traceShadow(tlas, shadowOrigin, sunDir, 1e4f)) {
                    radiance += throughput * albedo * INV_PI * sunRadiance(settings) * NdotL;
                }
            }
            rd = cosineSampleHemisphere(Ns, rand2(seed));
            ro = hitP + Ns * 1e-3f;
            throughput *= albedo;
        }

        if (bounce > 3u) {
            float p = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05f, 1.0f);
            if (randFloat(seed) > p) break;
            throughput /= p;
        }
    }

    return radiance;
}

kernel void pathTrace(instance_acceleration_structure tlas        [[buffer(0)]],
                      constant CameraUniforms &cam                [[buffer(1)]],
                      constant RenderSettings &settings           [[buffer(2)]],
                      device const PrimitiveData *primitives      [[buffer(3)]],
                      device const uint *instanceOffsets          [[buffer(4)]],
                      device const Material *materials            [[buffer(5)]],
                      texture2d<float, access::write> outColor       [[texture(0)]],
                      texture2d<float, access::write> outNormalDepth [[texture(1)]],
                      texture2d<float, access::write> outPos         [[texture(2)]],
                      texture2d<float, access::write> outReflPos     [[texture(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= settings.width || gid.y >= settings.height) return;

    uint seed = initSeed(gid, settings.width, cam.frameIndex);

    // --- Primary ray (with sub-pixel jitter for anti-aliasing) ------------------
    float2 jitter = rand2(seed) - 0.5f;
    float2 px = float2(gid) + 0.5f + jitter;
    float ndcX = (2.0f * px.x / float(settings.width) - 1.0f) * cam.aspect * cam.tanHalfFovY;
    float ndcY = (1.0f - 2.0f * px.y / float(settings.height)) * cam.tanHalfFovY;

    float3 ro = float3(cam.position);
    float3 rd = normalize(float3(cam.forward) + ndcX * float3(cam.right) + ndcY * float3(cam.up));

    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    float3 sunDir = normalize(float3(settings.sunDirection));

    // Primary-surface attributes for the denoiser G-buffer.
    float3 gNormal = float3(0.0f);
    float3 gPos = ro + rd * 1.0e4f;
    float gMaterialId = -1.0f;          // < 0 marks sky
    float gViewZ = 1.0e4f;

    // Virtual position of the reflected feature (xyz) + valid flag (w). Lets the
    // denoiser reproject specular reflections with parallax-correct motion so
    // they accumulate a long, stable history instead of smearing while moving.
    float4 gReflPos = float4(0.0f);

    float3 radiance = float3(0.0f);

    // --- Primary intersection (fills the G-buffer) ------------------------------
    ray pr;
    pr.origin = ro;
    pr.direction = rd;
    pr.min_distance = 1e-3f;
    pr.max_distance = 1e4f;
    auto phit = isect.intersect(pr, tlas);

    if (phit.type == intersection_type::none) {
        radiance = skyColor(rd, settings);
    } else {
        uint primIndex = instanceOffsets[phit.instance_id] + phit.primitive_id;
        PrimitiveData prim = primitives[primIndex];
        Material mat = materials[prim.materialIndex];

        float3 N = normalize(float3(prim.normal));
        float3 hitP = ro + rd * phit.distance;
        bool backface = dot(N, rd) > 0.0f;
        float3 Ns = backface ? -N : N;

        gNormal = Ns;
        gPos = hitP;
        gMaterialId = float(prim.materialIndex);
        gViewZ = phit.distance;

        radiance += float3(mat.emission);

        if (mat.transparency > 0.5f) {
            // ---- Deterministic Fresnel split on the primary water/glass hit ----
            // Trace BOTH the reflection and refraction sub-paths and weight them
            // by Fresnel, instead of randomly picking one per frame. This removes
            // the reflect/refract coin-flip that was the dominant source of the
            // grainy water noise, so it converges in a fraction of the samples.
            if (mat.flags & MATERIAL_FLAG_WATER) {
                float up = clamp(Ns.y, 0.0f, 1.0f);
                float3 wn = waterNormal(hitP.xz, settings.elapsedTime, settings.waterRoughness);
                Ns = normalize(mix(Ns, wn, up));
            }

            float etaI = 1.0f, etaT = mat.ior;
            if (backface) { float tmp = etaI; etaI = etaT; etaT = tmp; }
            float eta = etaI / etaT;

            float cosi = clamp(dot(-rd, Ns), 0.0f, 1.0f);
            float r0 = (etaI - etaT) / (etaI + etaT);
            r0 = r0 * r0;
            float fres = r0 + (1.0f - r0) * pow(1.0f - cosi, 5.0f);

            float3 refl = reflect(rd, Ns);
            float3 refr = refract(rd, Ns, eta);
            bool tir = dot(refr, refr) < 1e-6f;
            uint subBounces = settings.maxBounces > 0u ? settings.maxBounces - 1u : 0u;

            // Under total internal reflection all energy reflects.
            float reflW = tir ? 1.0f : fres;
            float3 reflO = hitP + refl * 1e-3f;
            float reflHitT = 1.0e4f;
            radiance += reflW * traceRadiance(tlas, primitives, instanceOffsets,
                                              materials, settings, sunDir,
                                              reflO, refl, float3(1.0f), subBounces,
                                              reflHitT, seed);

            // Build the virtual reflected position by elongating the primary view
            // ray by the reflection's hit distance (P = O + V·(t_surface + t_refl)).
            // Because the Fresnel split is deterministic, reflHitT is stable frame
            // to frame, so this reprojects cleanly (the technique behind NRD/ReLAX).
            float3 virtualPos = ro + rd * (phit.distance + reflHitT);
            gReflPos = float4(virtualPos, 1.0f);

            if (!tir) {
                float3 tint = mix(float3(1.0f), float3(mat.albedo) * 4.0f + 0.6f, 0.25f);
                float3 rdir = normalize(refr);
                float3 refrO = hitP + rdir * 1e-3f;
                float refrHitT = 1.0e4f;
                radiance += (1.0f - fres) * traceRadiance(tlas, primitives, instanceOffsets,
                                                          materials, settings, sunDir,
                                                          refrO, rdir, tint, subBounces,
                                                          refrHitT, seed);
            }
        } else {
            // Metal / diffuse: the unified tracer handles shading and bounces.
            float primHitT = 1.0e4f;
            radiance += traceRadiance(tlas, primitives, instanceOffsets,
                                      materials, settings, sunDir,
                                      ro, rd, float3(1.0f), settings.maxBounces,
                                      primHitT, seed);
        }
    }

    // Clamp the worst fireflies before they enter the denoiser.
    radiance = min(radiance, float3(48.0f));

    outColor.write(float4(radiance, 1.0f), gid);
    outNormalDepth.write(float4(gNormal, gViewZ), gid);
    outPos.write(float4(gPos, gMaterialId), gid);
    outReflPos.write(gReflPos, gid);
}
