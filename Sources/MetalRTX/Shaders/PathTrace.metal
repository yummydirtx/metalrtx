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

kernel void pathTrace(instance_acceleration_structure tlas        [[buffer(0)]],
                      constant CameraUniforms &cam                [[buffer(1)]],
                      constant RenderSettings &settings           [[buffer(2)]],
                      device const PrimitiveData *primitives      [[buffer(3)]],
                      device const uint *instanceOffsets          [[buffer(4)]],
                      device const Material *materials            [[buffer(5)]],
                      texture2d<float, access::write> outColor       [[texture(0)]],
                      texture2d<float, access::write> outNormalDepth [[texture(1)]],
                      texture2d<float, access::write> outPos         [[texture(2)]],
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

    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);

    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    float3 sunDir = normalize(float3(settings.sunDirection));

    // Primary-surface attributes for the denoiser G-buffer.
    float3 gNormal = float3(0.0f);
    float3 gPos = ro + rd * 1.0e4f;
    float gMaterialId = -1.0f;          // < 0 marks sky
    float gViewZ = 1.0e4f;

    for (uint bounce = 0; bounce <= settings.maxBounces; ++bounce) {
        ray r;
        r.origin = ro;
        r.direction = rd;
        r.min_distance = 1e-3f;
        r.max_distance = 1e4f;

        auto hit = isect.intersect(r, tlas);
        if (hit.type == intersection_type::none) {
            radiance += throughput * skyColor(rd, settings);
            break;
        }

        uint primIndex = instanceOffsets[hit.instance_id] + hit.primitive_id;
        PrimitiveData prim = primitives[primIndex];
        Material mat = materials[prim.materialIndex];

        float3 N = normalize(float3(prim.normal));
        float3 hitP = ro + rd * hit.distance;
        bool backface = dot(N, rd) > 0.0f;
        float3 Ns = backface ? -N : N;

        if (bounce == 0u) {
            gNormal = Ns;
            gPos = hitP;
            gMaterialId = float(prim.materialIndex);
            gViewZ = hit.distance;
        }

        // Emission contributes regardless of surface type.
        radiance += throughput * float3(mat.emission);

        if (mat.transparency > 0.5f) {
            // ---- Dielectric (water / glass): Fresnel reflect or refract --------
            // Animate the water surface with rolling ripples.
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
                // Subtle tint on transmission for a watery/glassy feel.
                throughput *= mix(float3(1.0f), float3(mat.albedo) * 4.0f + 0.6f, 0.25f);
            }
            ro = hitP + newDir * 1e-3f;
            rd = newDir;
        } else if (mat.metallic > 0.5f) {
            // ---- Rough metal: GGX specular reflection --------------------------
            float roughness = max(mat.roughness, 0.02f);
            float3 H = sampleGGX(Ns, roughness, rand2(seed));
            float3 newDir = reflect(rd, H);
            if (dot(newDir, Ns) <= 0.0f) break;
            ro = hitP + Ns * 1e-3f;
            rd = newDir;
            throughput *= float3(mat.albedo);
        } else {
            // ---- Lambertian diffuse with sun next-event estimation -------------
            float3 albedo = float3(mat.albedo);
            float NdotL = max(dot(Ns, sunDir), 0.0f);
            if (NdotL > 0.0f) {
                float3 shadowOrigin = hitP + Ns * 1e-3f;
                if (!traceShadow(tlas, shadowOrigin, sunDir, 1e4f)) {
                    radiance += throughput * albedo * INV_PI * sunRadiance(settings) * NdotL;
                }
            }
            // Indirect bounce: cosine-weighted hemisphere (pdf cancels the cosine).
            rd = cosineSampleHemisphere(Ns, rand2(seed));
            ro = hitP + Ns * 1e-3f;
            throughput *= albedo;
        }

        // Russian roulette after a few bounces keeps paths unbiased but cheap.
        if (bounce > 3u) {
            float p = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05f, 1.0f);
            if (randFloat(seed) > p) break;
            throughput /= p;
        }
    }

    // Clamp the worst fireflies before they enter the denoiser.
    radiance = min(radiance, float3(48.0f));

    outColor.write(float4(radiance, 1.0f), gid);
    outNormalDepth.write(float4(gNormal, gViewZ), gid);
    outPos.write(float4(gPos, gMaterialId), gid);
}
