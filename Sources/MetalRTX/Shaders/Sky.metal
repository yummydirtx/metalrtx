// =============================================================================
// Sky.metal — analytic day/night sky. Provides hemispherical radiance used both
// for ray misses (image-based lighting) and for direct viewing, including a sun
// disk and sunset warming driven by the sun's elevation.
// =============================================================================

inline float3 skyColor(float3 dir, constant RenderSettings &s) {
    float3 sunDir = normalize(float3(s.sunDirection));
    float sunElev = sunDir.y;                       // -1 (midnight) .. 1 (noon)
    float day      = smoothstep(-0.18f, 0.22f, sunElev); // 0 night, 1 day
    // How close the sun is to the horizon: peaks at sunrise/sunset, 0 at noon/midnight.
    float twilight = smoothstep(0.46f, 0.0f, fabs(sunElev));
    // Golden-hour weight: only counts while the sun is actually up (or just below).
    float golden   = twilight * smoothstep(-0.12f, 0.06f, sunElev);

    float up = max(dir.y, 0.0f);

    // ---- Base vertical gradient (zenith -> horizon) -------------------------
    float3 zenithDay    = float3(0.05f, 0.17f, 0.46f);
    float3 horizonDay   = float3(0.62f, 0.74f, 0.92f);
    float3 zenithNight  = float3(0.004f, 0.010f, 0.028f);
    float3 horizonNight = float3(0.015f, 0.022f, 0.05f);

    float3 zenith  = mix(zenithNight, zenithDay, day);
    float3 horizon = mix(horizonNight, horizonDay, day);

    // ---- Golden-hour tinting of the gradient --------------------------------
    // Deepen the zenith toward dusk blue/indigo and warm the horizon band.
    float3 duskZenith   = float3(0.07f, 0.10f, 0.32f);   // indigo overhead
    float3 duskHorizon  = float3(1.05f, 0.52f, 0.22f);   // warm amber band
    zenith  = mix(zenith,  duskZenith,  golden * 0.85f);
    horizon = mix(horizon, duskHorizon, golden);

    // Smooth vertical falloff for the horizon glow.
    float horizonBlend = pow(1.0f - up, 4.0f);
    float3 sky = mix(zenith, horizon, horizonBlend);

    // ---- Atmospheric perspective near the horizon (Rayleigh-ish haze) -------
    // A thin bright band hugging the horizon line, strongest during twilight.
    float horizonHaze = pow(1.0f - up, 12.0f);
    float3 hazeTint = mix(float3(0.78f, 0.84f, 0.95f),
                          float3(1.15f, 0.62f, 0.32f), golden);
    sky = mix(sky, hazeTint, horizonHaze * (0.35f + 0.45f * golden) * day);

    // ---- Azimuthal sun warmth ----------------------------------------------
    // Warmth that wraps horizontally around the sun's compass bearing, not just a
    // radial halo. This paints the whole sunward quarter of the sky orange at dusk.
    float mu = dot(dir, sunDir);              // -1 .. 1, angular proximity to sun
    float muP = max(mu, 0.0f);
    float sunwardAz = pow(muP, 1.5f);         // broad, soft sunward bias
    float3 sunsetGlow = float3(1.25f, 0.55f, 0.20f);
    sky += sunsetGlow * sunwardAz * horizonBlend * golden * 1.1f;

    // Layered sun glow: a wide soft halo plus a tighter hot core.
    float3 warmHalo = mix(float3(1.0f, 0.62f, 0.32f), s.sunColor, day);
    sky += warmHalo * pow(muP, 6.0f) * (0.45f + golden * 0.9f) * day;
    sky += s.sunColor * pow(muP, 120.0f) * 0.35f * day;

    // ---- Belt of Venus -------------------------------------------------------
    // The pinkish band opposite the sun, sitting just above the bluish Earth-shadow
    // near the horizon during twilight.
    float antiSun = clamp(-mu, 0.0f, 1.0f);
    float beltBand = smoothstep(0.0f, 0.10f, dir.y) * smoothstep(0.34f, 0.08f, dir.y);
    float3 beltColor = float3(0.85f, 0.45f, 0.55f);
    sky += beltColor * beltBand * antiSun * golden * 0.35f;

    // ---- Sun disk (sharp) — visible directly and in reflections -------------
    float disk = smoothstep(0.9993f, 0.9997f, muP);
    // The disk reddens as it sinks, just like a real low sun.
    float3 diskColor = mix(s.sunColor, float3(1.0f, 0.42f, 0.18f), golden * 0.8f);
    sky += disk * diskColor * s.sunIntensity * 0.18f;

    // ---- Stars at night ------------------------------------------------------
    if (day < 0.25f && dir.y > 0.0f) {
        float3 sd = dir * 200.0f;
        float star = fract(sin(dot(floor(sd), float3(12.9898f, 78.233f, 37.719f))) * 43758.5453f);
        float twinkle = step(0.9985f, star);
        sky += twinkle * (1.0f - day) * 0.6f;
    }

    return max(sky, 0.0f);
}

// ---- Procedural cloud layer -------------------------------------------------
// A single animated 2D cloud sheet at a fixed altitude, composited over the
// analytic sky. fBm coverage gives soft, drifting cumulus shapes; a cheap
// directional self-shadow and a sunward "silver lining" rim give them volume.
// Evaluated only for primary rays and first-bounce reflections (not GI) to keep
// the cost down.
inline float3 renderClouds(float3 ro, float3 rd, constant RenderSettings &s, float3 skyCol) {
    if (rd.y <= 0.02f) return skyCol;

    const float cloudH = 140.0f;                       // cloud sheet altitude
    float t = (cloudH - ro.y) / rd.y;
    if (t <= 0.0f) return skyCol;
    float3 hit = ro + rd * t;

    // Animated coverage field.
    float2 uv = hit.xz * 0.008f + float2(0.05f, 0.018f) * s.elapsedTime;
    float n = fbm3(float3(uv.x, 0.0f, uv.y), 5);

    float3 sunDir = normalize(float3(s.sunDirection));
    float day = smoothstep(-0.05f, 0.25f, sunDir.y);

    const float coverage = 0.46f;
    float density = smoothstep(coverage, coverage + 0.20f, n);
    density *= smoothstep(0.02f, 0.22f, rd.y);          // thin out toward the horizon
    if (density <= 0.001f) return skyCol;

    // Directional self-shadow: compare density a step toward the sun.
    float n2 = fbm3(float3(uv.x + sunDir.x * 0.04f, 0.0f, uv.y + sunDir.z * 0.04f), 4);
    float shade = clamp(0.55f + (n - n2) * 2.5f, 0.25f, 1.15f);

    float3 baseLit = mix(float3(0.55f, 0.58f, 0.65f), float3(1.0f, 0.98f, 0.94f), day);
    float golden = smoothstep(0.35f, 0.0f, fabs(sunDir.y)) * day;
    baseLit = mix(baseLit, float3(1.1f, 0.7f, 0.45f), golden * 0.6f);
    float3 cloudCol = baseLit * shade;

    // Silver lining toward the sun, strongest through thin cloud.
    float vdotl = max(dot(rd, sunDir), 0.0f);
    cloudCol += float3(1.0f, 0.92f, 0.78f) * pow(vdotl, 6.0f) * (1.0f - density) * day;

    return mix(skyCol, cloudCol, density * (0.5f + 0.5f * day));
}

/// Full sky including clouds, for primary rays and first-bounce reflections.
inline float3 skyWithClouds(float3 ro, float3 rd, constant RenderSettings &s) {
    return renderClouds(ro, rd, s, skyColor(rd, s));
}

/// Radiance of the sun itself, for next-event estimation toward the directional light.
inline float3 sunRadiance(constant RenderSettings &s) {
    return s.sunColor * s.sunIntensity;
}

// ---- Ocean wave field -------------------------------------------------------
// Ported from theJunkyard's ocean simulator: instead of three fixed sine waves we
// sum a small spectrum of Gerstner waves spread around a prevailing wind, sized by
// a wavelength/amplitude distribution and animated with the finite-depth dispersion
// relation omega = sqrt(g k tanh(k d)). The surface normal is taken analytically
// from the displacement tangents, and an advected gust field plus capillary detail
// add the lively, non-repeating chop the old three-wave version lacked.

inline float oceanHash2(float2 p) {
    return fract(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453123f);
}

inline float oceanValueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0f - 2.0f * f);
    float a = oceanHash2(i);
    float b = oceanHash2(i + float2(1.0f, 0.0f));
    float c = oceanHash2(i + float2(0.0f, 1.0f));
    float d = oceanHash2(i + float2(1.0f, 1.0f));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

inline float oceanFbm(float2 p) {
    float v = 0.0f, a = 0.5f;
    for (int i = 0; i < 4; ++i) { v += a * oceanValueNoise(p); p *= 2.0f; a *= 0.5f; }
    return v;
}

/// Deterministic per-wave pseudo-random, matching theJunkyard's spectrum builder.
inline float oceanRand(float s) {
    float x = sin(s * 12.9898f + s * s * 78.233f) * 43758.5453123f;
    return x - floor(x);
}

inline float2 oceanRot2(float2 v, float ang) {
    float c = cos(ang), s = sin(ang);
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

/// Animated water surface normal from a Gerstner wave spectrum. `p` is the world XZ
/// position; `amplitude`, `choppiness`, and `speed` shape the swell and `rough` adds
/// capillary micro-detail. Returns a unit normal around +Y.
inline float3 waterNormal(float2 p, float t,
                          float amplitude, float choppiness, float speed, float rough) {
    constexpr int   WAVE_COUNT = 8;
    constexpr float GRAV       = 9.81f;
    constexpr float DEPTH      = 11.0f;          // ocean depth in voxels
    const float2 wind      = normalize(float2(1.0f, 0.35f));
    const float2 crossWind = float2(-wind.y, wind.x);
    const float2 crossSwell = oceanRot2(wind, -PI / 3.15f);

    // Wind-gust field: advected fbm that swells and relaxes wave height in patches.
    float gustField = oceanFbm(p * 0.018f + wind * (t * 0.05f));
    gustField += oceanFbm(p * 0.045f + crossWind * 3.2f - wind * (t * 0.08f)) * 0.5f;
    float gust = mix(0.72f, 1.28f, clamp(gustField, 0.0f, 1.0f));

    float master    = amplitude;                  // overall wave steepness
    float chop      = choppiness;                 // Gerstner horizontal sharpening (q)
    float timeScale = speed;                       // wave animation time scale

    // Analytic surface tangents, starting from the flat plane basis.
    float3 dPdx = float3(1.0f, 0.0f, 0.0f);
    float3 dPdz = float3(0.0f, 0.0f, 1.0f);

    for (int i = 0; i < WAVE_COUNT; ++i) {
        float f = float(i) / float(WAVE_COUNT - 1);

        float wavelength = mix(26.0f, 2.2f, pow(f, 0.7f));
        float peak = exp(-pow((f - 0.28f) / 0.24f, 2.0f));
        float ampW = mix(1.0f, 0.06f, pow(f, 0.58f)) * (0.5f + 0.5f * peak);

        // Spread the direction around the wind, biasing the mid band toward a cross swell.
        float spread = mix(0.08f, 0.94f, pow(f, 1.04f));
        float dirBlend = clamp(0.43f - fabs(f - 0.3f) * 2.2f, 0.0f, 0.43f);
        float2 baseDir = normalize(mix(wind, crossSwell, dirBlend));
        float jitter = (oceanRand(10.7f + float(i) * 17.17f) - 0.5f) * spread
                     + (oceanRand(55.2f + float(i) * 7.19f) - 0.5f) * spread * 0.55f;
        float2 dir = normalize(oceanRot2(baseDir, jitter));

        // Damp waves travelling against the wind so the swell stays directional.
        float dirDamp = pow(clamp(dot(dir, wind) * 0.5f + 0.5f, 0.0f, 1.0f),
                            mix(4.0f, 1.5f, f));
        float amp = master * ampW * mix(0.38f, 1.0f, dirDamp) * gust;

        float speedScale = mix(1.24f, 0.78f, f);
        float phase0 = oceanRand(92.0f + float(i) * 13.17f) * TWO_PI;

        float k = TWO_PI / wavelength;
        float omega = sqrt(GRAV * k * max(tanh(k * DEPTH), 0.16f)) * speedScale * timeScale;
        float phase = k * dot(dir, p) - omega * t + phase0;
        float s = sin(phase), c = cos(phase);

        float ak = amp * k;
        float waveTerm = chop * ak * s;

        // d(displacement)/dx and d(displacement)/dz for a Gerstner wave.
        dPdx.x += -dir.x * dir.x * waveTerm;
        dPdx.y +=  dir.x * ak * c;
        dPdx.z += -dir.x * dir.y * waveTerm;

        dPdz.x += -dir.x * dir.y * waveTerm;
        dPdz.y +=  dir.y * ak * c;
        dPdz.z += -dir.y * dir.y * waveTerm;
    }

    float3 n = normalize(cross(dPdz, dPdx));

    // Capillary detail: two high-frequency fbm gradient layers add micro-chop that
    // keeps sun glints lively without disturbing the large-scale wave shape.
    float detail = 0.05f + rough * 0.18f;
    float2 g1 = float2(oceanFbm(p * 0.9f + float2( t * 0.18f, -t * 0.12f)) - 0.5f,
                       oceanFbm(p * 0.9f + float2(40.0f, -22.0f) - float2(t * 0.11f, -t * 0.09f)) - 0.5f);
    float2 g2 = float2(oceanFbm(p * 2.3f - float2( t * 0.28f, -t * 0.21f)) - 0.5f,
                       oceanFbm(p * 2.3f + float2(-83.0f, 62.0f) + float2(t * 0.24f, t * 0.23f)) - 0.5f);
    n = normalize(n + float3(g1.x, 0.0f, g1.y) * detail + float3(g2.x, 0.0f, g2.y) * detail * 0.58f);

    return n;
}

