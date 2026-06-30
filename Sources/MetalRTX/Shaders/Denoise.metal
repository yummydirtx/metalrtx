// =============================================================================
// Denoise.metal — real-time denoising for the 1-spp path traced image:
//   1. temporalReproject — reprojects the previous frame's accumulated color into
//      the current frame using world-space hit positions, validating reuse by
//      position / normal / material so moving the camera doesn't smear.
//   2. atrous — edge-avoiding à-trous wavelet filter (Dammertz et al. 2010) that
//      removes residual noise while preserving voxel edges via normal, world
//      position, and material-id edge-stopping functions.
// =============================================================================

// ---- Temporal reprojection -------------------------------------------------

kernel void temporalReproject(texture2d<float, access::read>  curColor      [[texture(0)]],
                              texture2d<float, access::read>  curNormalDepth [[texture(1)]],
                              texture2d<float, access::read>  curPos        [[texture(2)]],
                              texture2d<float, access::read>  prevPos       [[texture(3)]],
                              texture2d<float, access::read>  prevNormal    [[texture(4)]],
                              texture2d<float, access::read>  histColorIn   [[texture(5)]],
                              texture2d<float, access::write> histColorOut  [[texture(6)]],
                              texture2d<float, access::read>  curReflPos    [[texture(7)]],
                              texture2d<float, access::read>  prevReflPos   [[texture(8)]],
                              constant CameraUniforms &cam                  [[buffer(0)]],
                              device const Material *materials              [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint w = curColor.get_width();
    uint h = curColor.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float4 cur = curColor.read(gid);
    float4 pos = curPos.read(gid);
    float3 wp = pos.xyz;
    float matId = pos.w;
    float3 curN = curNormalDepth.read(gid).xyz;

    float3 result = cur.rgb;
    float historyLen = 1.0f;

    bool specular = false;
    if (matId >= 0.0f) {
        Material m = materials[uint(matId)];
        specular = (m.metallic > 0.5f) || (m.transparency > 0.5f);
    }

    // For specular surfaces, reproject the *reflected feature* by its virtual
    // world position rather than the surface position. This parallax-correct
    // motion vector (the technique behind SVGF / NVIDIA NRD / ReLAX specular
    // reprojection) keeps reflection history valid while the camera moves, so
    // reflections accumulate many frames again instead of smearing. It is robust
    // here because the deterministic Fresnel split makes the reflection hitT (and
    // thus the virtual position) stable from frame to frame.
    float4 rp = curReflPos.read(gid);
    bool useVirtual = specular && rp.w > 0.5f;
    float3 reproPos = useVirtual ? rp.xyz : wp;

    // Sky (matId < 0) is noise-free; pass it straight through.
    if (matId >= 0.0f) {
        float4 clip = cam.prevViewProj * float4(reproPos, 1.0f);
        if (clip.w > 0.0f) {
            float2 ndc = clip.xy / clip.w;
            float2 uv = ndc * 0.5f + 0.5f;
            uv.y = 1.0f - uv.y;
            float2 fp = uv * float2(w, h);
            int2 prevPix = int2(floor(fp));

            if (prevPix.x >= 0 && prevPix.y >= 0 &&
                prevPix.x < int(w) && prevPix.y < int(h)) {
                uint2 pp = uint2(prevPix);
                float4 ppos = prevPos.read(pp);

                bool reuse;
                float maxHistory;
                if (useVirtual) {
                    // Validate by matching the previous frame's virtual hit
                    // position. The tolerance scales with reflection distance so
                    // far reflections (sky) are accepted generously while near
                    // reflections stay tight.
                    float4 prp = prevReflPos.read(pp);
                    float vdist = distance(rp.xyz, float3(cam.position));
                    float thresh = max(0.75f, 0.05f * vdist);
                    bool virtOk = (prp.w > 0.5f) &&
                                  (distance(prp.xyz, rp.xyz) < thresh);
                    bool matOk  = abs(ppos.w - matId) < 0.5f;
                    reuse = virtOk && matOk;
                    maxHistory = 48.0f;
                } else {
                    float3 pn = prevNormal.read(pp).xyz;
                    bool distOk = distance(ppos.xyz, wp) < 0.6f;   // ~½ a voxel
                    bool normOk = dot(pn, curN) > 0.85f;
                    bool matOk  = abs(ppos.w - matId) < 0.5f;
                    reuse = distOk && normOk && matOk;
                    // Specular without a virtual hit keeps a short history to stay
                    // responsive; diffuse accumulates long.
                    maxHistory = specular ? 8.0f : 64.0f;
                }

                if (reuse) {
                    float4 hist = histColorIn.read(pp);
                    historyLen = min(hist.a + 1.0f, maxHistory);
                    float alpha = 1.0f / historyLen;
                    result = mix(hist.rgb, cur.rgb, alpha);
                }
            }
        }
    }

    histColorOut.write(float4(result, historyLen), gid);
}

// ---- Edge-avoiding à-trous wavelet -----------------------------------------

kernel void atrous(texture2d<float, access::read>  inColor       [[texture(0)]],
                   texture2d<float, access::read>  gNormalDepth  [[texture(1)]],
                   texture2d<float, access::read>  gPos          [[texture(2)]],
                   texture2d<float, access::write> outColor      [[texture(3)]],
                   constant uint &stepSize                       [[buffer(0)]],
                   device const Material *materials              [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint w = inColor.get_width();
    uint h = inColor.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float4 centerC = inColor.read(gid);
    float4 centerP = gPos.read(gid);
    float centerMat = centerP.w;
    float3 centerN = gNormalDepth.read(gid).xyz;

    // Sky / background: leave untouched.
    if (centerMat < 0.0f) {
        outColor.write(centerC, gid);
        return;
    }

    const float kernelW[3] = { 3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f };

    // Specular (metal/water) surfaces can no longer rely on long temporal
    // accumulation, so they carry much more per-frame noise. Relax the spatial
    // edge-stopping for them — wider position support and a far looser
    // luminance test — so the à-trous filter actually blurs the noise away
    // instead of preserving it. Diffuse surfaces keep the tight values that
    // protect their detail.
    bool specular = (materials[uint(centerMat)].metallic > 0.5f) ||
                    (materials[uint(centerMat)].transparency > 0.5f);
    float sigmaPos    = specular ? 4.0f : 0.8f;
    float sigmaNormal = specular ? 8.0f : 32.0f;
    float sigmaColor  = specular ? 8.0f : 1.2f;
    float centerLum = luminance(centerC.rgb);

    float3 sum = float3(0.0f);
    float wsum = 0.0f;

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 q = int2(gid) + int2(dx, dy) * int(stepSize);
            if (q.x < 0 || q.y < 0 || q.x >= int(w) || q.y >= int(h)) continue;
            uint2 qu = uint2(q);

            float weight;
            if (dx == 0 && dy == 0) {
                weight = kernelW[0] * kernelW[0];
            } else {
                float4 qc = inColor.read(qu);
                float4 qp = gPos.read(qu);
                float3 qn = gNormalDepth.read(qu).xyz;

                // Hard material edge stop preserves voxel boundaries.
                if (abs(qp.w - centerMat) > 0.5f) continue;

                float wn = pow(max(dot(centerN, qn), 0.0f), sigmaNormal);
                float pd = distance(qp.xyz, centerP.xyz);
                float wp = exp(-(pd * pd) / sigmaPos);
                float ld = fabs(luminance(qc.rgb) - centerLum);
                float wl = exp(-ld / sigmaColor);

                float hw = kernelW[abs(dx)] * kernelW[abs(dy)];
                weight = hw * wn * wp * wl;
            }

            float4 qc = inColor.read(qu);
            sum += qc.rgb * weight;
            wsum += weight;
        }
    }

    float3 outc = (wsum > 0.0f) ? sum / wsum : centerC.rgb;
    outColor.write(float4(outc, centerC.a), gid);
}
