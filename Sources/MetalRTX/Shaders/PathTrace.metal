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

// Direct lighting from the camera-mounted flashlight: a soft-edged spotlight at the
// camera position aimed along its forward axis. Returns the (unshadowed-if-visible)
// diffuse contribution for a hit point, or zero when disabled / outside the cone.
inline float3 flashlightDirect(instance_acceleration_structure tlas,
                               constant RenderSettings &s,
                               float3 P, float3 Ns, float3 albedo) {
    if (s.flashlightEnabled == 0u) return float3(0.0f);

    float3 lpos = float3(s.flashlightPos);
    float3 ldir = normalize(float3(s.flashlightDir));
    float3 toL = lpos - P;
    float dist = length(toL);
    if (dist < 1e-4f) return float3(0.0f);
    float3 L = toL / dist;

    float NdotL = max(dot(Ns, L), 0.0f);
    if (NdotL <= 0.0f) return float3(0.0f);

    // Spot cone: full brightness inside the inner angle, feathered to the outer angle.
    float spotCos = dot(-L, ldir);
    const float cosInner = 0.965f;   // ~15 degrees
    const float cosOuter = 0.90f;    // ~26 degrees
    float spot = smoothstep(cosOuter, cosInner, spotCos);
    if (spot <= 0.0f) return float3(0.0f);

    // Inverse-square falloff with a soft range cutoff so it fades out naturally.
    const float range = 110.0f;
    float atten = 1.0f / (1.0f + 0.012f * dist * dist);
    atten *= clamp(1.0f - dist / range, 0.0f, 1.0f);
    if (atten <= 0.0f) return float3(0.0f);

    // Shadow test toward the light.
    float3 origin = P + Ns * 1e-3f;
    if (traceShadow(tlas, origin, L, dist - 2e-3f)) return float3(0.0f);

    const float3 color = float3(1.0f, 0.95f, 0.86f);
    const float intensity = 16.0f;
    return albedo * INV_PI * color * intensity * NdotL * spot * atten;
}

// Visible response of a water surface to the camera flashlight. Because the light is
// Visible response of a water surface to the camera flashlight. A specular reflection of
// the bulb is NOT the right model here: the light is at the eye, so its mirror reflection
// only lands straight down at your feet (never in the middle of the lake) — unlike the moon,
// which is at infinity and streaks to the horizon. What you actually see when you shine a
// flashlight into water is the beam ENTERING and scattering within it: a soft, water-tinted
// glow that is strongest exactly where you look down into the surface (high transmission),
// plus a faint Fresnel sparkle off grazing wavelets. `rd` is the incoming view ray; `Ns` is
// the wave-perturbed normal.
inline float3 flashlightWater(instance_acceleration_structure tlas,
                              constant RenderSettings &s,
                              float3 P, float3 Ns, float3 rd) {
    if (s.flashlightEnabled == 0u) return float3(0.0f);

    float3 lpos = float3(s.flashlightPos);
    float3 ldir = normalize(float3(s.flashlightDir));
    float3 toL = lpos - P;
    float dist = length(toL);
    if (dist < 1e-4f) return float3(0.0f);
    float3 L = toL / dist;

    // Spot cone gate (the lit point must be inside the beam).
    float spotCos = dot(-L, ldir);
    const float cosInner = 0.965f;   // ~15 degrees
    const float cosOuter = 0.86f;    // ~31 degrees, soft edge
    float spot = smoothstep(cosOuter, cosInner, spotCos);
    if (spot <= 0.0f) return float3(0.0f);

    // The beam must hit the top of the water surface.
    float NdotL = max(dot(Ns, L), 0.0f);
    if (NdotL <= 0.0f) return float3(0.0f);

    // Inverse-square-ish falloff with a soft range cutoff.
    const float range = 160.0f;
    float atten = 1.0f / (1.0f + 0.006f * dist * dist);
    atten *= clamp(1.0f - dist / range, 0.0f, 1.0f);
    if (atten <= 0.0f) return float3(0.0f);

    // The surface point must actually see the light.
    float3 origin = P + Ns * 1e-3f;
    if (traceShadow(tlas, origin, L, dist - 2e-3f)) return float3(0.0f);

    float3 V = -rd;                                  // surface -> eye

    // Transmission: how much we see INTO the water (Schlick, ~98% looking straight down,
    // dropping at grazing where the surface turns mirror-like). The beam that gets through
    // scatters off particulate / the lakebed and glows back up to the eye.
    float cosV = clamp(dot(Ns, V), 0.0f, 1.0f);
    float fresV = 0.02f + 0.98f * pow(1.0f - cosV, 5.0f);
    float transmit = 1.0f - fresV;
    const float3 scatterTint = float3(0.05f, 0.22f, 0.30f);   // murky blue-green water
    float3 glow = scatterTint * (transmit * NdotL);

    // Faint surface sparkle: a Fresnel-weighted highlight off grazing wavelets so the beam
    // also throws a few lively glints on the surface without washing it out.
    float3 H = normalize(L + V);
    float NdotH = max(dot(Ns, H), 0.0f);
    float fresH = 0.02f + 0.98f * pow(1.0f - max(dot(V, H), 0.0f), 5.0f);
    float3 sparkle = float3(1.0f, 0.97f, 0.9f) * (pow(NdotH, 80.0f) * fresH);

    const float glowStrength = 10.0f;
    const float sparkleStrength = 6.0f;
    return (glow * glowStrength + sparkle * sparkleStrength) * spot * atten;
}

// -----------------------------------------------------------------------------
// Visible flashlight body. The light is otherwise an invisible point source, so
// we draw a small analytic prop (a capped cylinder with an emissive lens) at the
// flashlight origin, aimed along its beam. It is intersected directly by the
// primary camera ray — when held it reads as a viewmodel in the lower view, and
// when frozen it stays put in the world. `flashlightPos`/`flashlightDir` already
// hold the live hand pose or the frozen pose, so this follows both automatically.
struct FlashlightHit {
    bool   hit;
    float  t;
    float3 normal;
    bool   isLens;   // front face that glows when the light is on
};

inline FlashlightHit intersectFlashlight(constant RenderSettings &s, float3 ro, float3 rd) {
    FlashlightHit res;
    res.hit = false;
    res.t = 1.0e30f;
    res.normal = float3(0.0f, 1.0f, 0.0f);
    res.isLens = false;

    float3 axis = normalize(float3(s.flashlightDir));   // points out the lens
    float3 front = float3(s.flashlightPos);             // lens center
    const float bodyLen = 0.17f;
    const float radius = 0.035f;
    float3 back = front - axis * bodyLen;               // tail end

    // ---- Side wall: ray vs finite cylinder (infinite solve, clamped to length).
    float3 ba = front - back;
    float baLen2 = dot(ba, ba);
    float3 oc = ro - back;
    float baoc = dot(ba, oc);
    float bard = dot(ba, rd);
    float a = baLen2 - bard * bard;
    float b = baLen2 * dot(oc, rd) - baoc * bard;
    float c = baLen2 * dot(oc, oc) - baoc * baoc - radius * radius * baLen2;
    float disc = b * b - a * c;
    if (disc >= 0.0f && fabs(a) > 1e-9f) {
        float sq = sqrt(disc);
        float t = (-b - sq) / a;
        float y = baoc + t * bard;
        if (t > 1e-3f && t < res.t && y >= 0.0f && y <= baLen2) {
            res.hit = true;
            res.t = t;
            float3 p = ro + rd * t;
            res.normal = normalize((p - back) - ba * (y / baLen2));
            res.isLens = false;
        }
    }

    // ---- End caps: front lens disc and back disc.
    float3 capC[2] = { front, back };
    float3 capN[2] = { axis, -axis };
    for (int cap = 0; cap < 2; ++cap) {
        float denom = dot(rd, capN[cap]);
        if (fabs(denom) > 1e-6f) {
            float t = dot(capC[cap] - ro, capN[cap]) / denom;
            if (t > 1e-3f && t < res.t) {
                float3 p = ro + rd * t;
                if (length(p - capC[cap]) <= radius) {
                    res.hit = true;
                    res.t = t;
                    res.normal = capN[cap];
                    res.isLens = (cap == 0);
                }
            }
        }
    }
    return res;
}

// Shades the flashlight prop: a glowing lens when lit, otherwise a dark gunmetal
// body with simple sun + ambient lighting (and a sun shadow test against the world).
inline float3 shadeFlashlight(instance_acceleration_structure tlas,
                              constant RenderSettings &s,
                              float3 sunDir, float3 P, float3 Ns, bool isLens) {
    if (isLens) {
        if (s.flashlightEnabled != 0u) {
            return float3(1.0f, 0.95f, 0.86f) * 7.0f;   // bright emissive lens
        }
        return float3(0.03f, 0.035f, 0.04f);            // dark glass when off
    }

    const float3 albedo = float3(0.06f, 0.065f, 0.075f); // dark gunmetal
    float3 col = albedo * 0.20f;                         // sky ambient term
    float NdotL = max(dot(Ns, sunDir), 0.0f);
    if (NdotL > 0.0f) {
        float3 o = P + Ns * 1e-3f;
        if (!traceShadow(tlas, o, sunDir, 1e4f)) {
            col += albedo * INV_PI * sunRadiance(s) * NdotL;
        }
    }
    return col;
}

// Volumetric in-scatter of the flashlight beam through a thin fog. Ray-marches the
// view ray and accumulates light scattered toward the camera from inside the cone,
// shadow-testing each in-cone sample so the shaft is occluded by world geometry.
// Makes the beam itself visible as a soft cone of light. `tMax` is the depth of the
// nearest opaque surface (so the fog does not bleed past walls).
inline float3 flashlightVolumetric(instance_acceleration_structure tlas,
                                   constant RenderSettings &s,
                                   float3 ro, float3 rd, float tMax,
                                   thread uint &seed) {
    if (s.flashlightEnabled == 0u || s.fogEnabled == 0u) return float3(0.0f);

    float3 lpos = float3(s.flashlightPos);
    float3 ldir = normalize(float3(s.flashlightDir));
    const float cosInner = 0.965f;
    const float cosOuter = 0.90f;
    const float range = 110.0f;

    float marchEnd = min(tMax, range);
    if (marchEnd <= 1e-3f) return float3(0.0f);

    const int STEPS = 32;
    float dt = marchEnd / float(STEPS);
    float jitter = randFloat(seed);          // break up slice banding

    // Henyey-Greenstein forward scatter makes the cone read more like a beam.
    float g = 0.35f;
    float g2 = g * g;

    const float3 color = float3(1.0f, 0.95f, 0.86f);
    const float intensity = 16.0f;
    const float scatter = 0.012f;            // very slight fog density

    float3 accum = float3(0.0f);
    for (int i = 0; i < STEPS; ++i) {
        float t = (float(i) + jitter) * dt;
        float3 X = ro + rd * t;
        float3 toL = lpos - X;
        float dist = length(toL);
        if (dist < 1e-4f) continue;
        float3 L = toL / dist;

        float spotCos = dot(-L, ldir);
        float spot = smoothstep(cosOuter, cosInner, spotCos);
        if (spot <= 0.0f) continue;          // outside the cone: skip the shadow ray

        float atten = 1.0f / (1.0f + 0.012f * dist * dist);
        atten *= clamp(1.0f - dist / range, 0.0f, 1.0f);
        if (atten <= 0.0f) continue;

        // True volumetric shadow: only lit if the bulb can see this point.
        if (traceShadow(tlas, X, L, dist - 2e-3f)) continue;

        float cosTheta = dot(rd, -L);
        float phase = (1.0f - g2) /
                      (4.0f * M_PI_F * pow(1.0f + g2 - 2.0f * g * cosTheta, 1.5f));

        accum += spot * atten * phase;
    }
    return accum * color * intensity * scatter * dt;
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
                float3 wn = waterNormal(hitP.xz, settings.elapsedTime,
                                        settings.waveAmplitude, settings.waveChoppiness,
                                        settings.waveSpeed, settings.waterRoughness);
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

            // Flashlight response on this water surface (visible regardless of whether this
            // sample reflects or refracts), evaluated as a Fresnel specular highlight.
            if (mat.flags & MATERIAL_FLAG_WATER) {
                radiance += throughput * flashlightWater(tlas, settings, hitP, Ns, rd);
            }

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
            radiance += throughput * flashlightDirect(tlas, settings, hitP, Ns, albedo);
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

    // The flashlight prop is analytic geometry (not in the TLAS), so test it here
    // against the primary ray and let it occlude the scene when it is in front.
    FlashlightHit fl = intersectFlashlight(settings, ro, rd);
    float sceneT = (phit.type == intersection_type::none) ? 1.0e30f : phit.distance;

    if (fl.hit && fl.t < sceneT) {
        float3 P = ro + rd * fl.t;
        float3 Ns = fl.normal;
        if (dot(Ns, rd) > 0.0f) Ns = -Ns;   // face the camera
        radiance = shadeFlashlight(tlas, settings, sunDir, P, Ns, fl.isLens);
        gNormal = Ns;
        gPos = P;
        gMaterialId = -2.0f;                 // distinct id keeps the denoiser from blending it
        gViewZ = fl.t;
    } else if (phit.type == intersection_type::none) {
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
                float3 wn = waterNormal(hitP.xz, settings.elapsedTime,
                                        settings.waveAmplitude, settings.waveChoppiness,
                                        settings.waveSpeed, settings.waterRoughness);
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

            // Explicit flashlight response on the water surface (point lights are never hit
            // by the reflection ray, so this is what makes the beam visible on water).
            radiance += flashlightWater(tlas, settings, hitP, Ns, rd);

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

    // Volumetric in-scatter of the flashlight beam through the thin fog (if enabled).
    radiance += flashlightVolumetric(tlas, settings, ro, rd, gViewZ, seed);

    // Clamp the worst fireflies before they enter the denoiser.
    radiance = min(radiance, float3(48.0f));

    outColor.write(float4(radiance, 1.0f), gid);
    outNormalDepth.write(float4(gNormal, gViewZ), gid);
    outPos.write(float4(gPos, gMaterialId), gid);
    outReflPos.write(gReflPos, gid);
}
