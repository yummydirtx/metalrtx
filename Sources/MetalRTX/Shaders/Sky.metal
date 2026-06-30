// =============================================================================
// Sky.metal — analytic day/night sky. Provides hemispherical radiance used both
// for ray misses (image-based lighting) and for direct viewing, including a sun
// disk and sunset warming driven by the sun's elevation.
// =============================================================================

inline float3 skyColor(float3 dir, constant RenderSettings &s) {
    float3 sunDir = normalize(float3(s.sunDirection));
    float sunElev = sunDir.y;                       // -1 (midnight) .. 1 (noon)
    float day = smoothstep(-0.18f, 0.22f, sunElev); // 0 night, 1 day

    // Vertical gradient between horizon and zenith.
    float3 zenithDay   = float3(0.09f, 0.21f, 0.52f);
    float3 horizonDay  = float3(0.64f, 0.72f, 0.88f);
    float3 zenithNight = float3(0.004f, 0.010f, 0.028f);
    float3 horizonNight = float3(0.015f, 0.022f, 0.05f);

    float3 zenith  = mix(zenithNight, horizonNight, 0.0f);
    zenith  = mix(zenithNight, zenithDay, day);
    float3 horizon = mix(horizonNight, horizonDay, day);

    float horizonBlend = pow(1.0f - max(dir.y, 0.0f), 4.0f);
    float3 sky = mix(zenith, horizon, horizonBlend);

    // Sun glow + warm sunset tint when the sun is low.
    float mu = max(dot(dir, sunDir), 0.0f);
    float lowSun = smoothstep(0.0f, 0.35f, 1.0f - abs(sunElev)) * day;
    float3 warm = float3(0.95f, 0.45f, 0.18f);
    sky += warm * pow(mu, 5.0f) * lowSun * 0.7f;
    sky += s.sunColor * pow(mu, 80.0f) * 0.25f * day;

    // Sun disk (sharp) — visible directly and in reflections/refractions.
    float disk = smoothstep(0.9993f, 0.9997f, mu);
    sky += disk * s.sunColor * s.sunIntensity * 0.18f;

    // A few stars at night.
    if (day < 0.25f && dir.y > 0.0f) {
        float3 sd = dir * 200.0f;
        float star = fract(sin(dot(floor(sd), float3(12.9898f, 78.233f, 37.719f))) * 43758.5453f);
        float twinkle = step(0.9985f, star);
        sky += twinkle * (1.0f - day) * 0.6f;
    }

    return max(sky, 0.0f);
}

/// Radiance of the sun itself, for next-event estimation toward the directional light.
inline float3 sunRadiance(constant RenderSettings &s) {
    return s.sunColor * s.sunIntensity;
}

/// Animated water surface normal built from a few directional sine waves. `p` is the world
/// XZ position; `rough` increases choppiness. Returns a unit normal around +Y.
inline float3 waterNormal(float2 p, float t, float rough) {
    float2 d1 = normalize(float2(1.0f, 0.35f));
    float2 d2 = normalize(float2(-0.6f, 1.0f));
    float2 d3 = normalize(float2(0.2f, -1.0f));

    float k1 = 0.55f, k2 = 1.1f, k3 = 2.1f;
    float c1 = cos(dot(p, d1) * k1 + t * 1.4f);
    float c2 = cos(dot(p, d2) * k2 + t * 1.05f);
    float c3 = cos(dot(p, d3) * k3 + t * 1.9f);

    float amp = 0.05f + rough * 0.30f;
    float2 grad = amp * (d1 * k1 * c1 + d2 * k2 * c2 + d3 * k3 * c3 * 0.5f);
    return normalize(float3(-grad.x, 1.0f, -grad.y));
}

