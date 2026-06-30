// =============================================================================
// Tonemap.metal — converts the accumulated HDR radiance buffer to a display-ready
// image using exposure control and the ACES filmic curve, then encodes to gamma.
// =============================================================================

inline float3 acesFilmic(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

kernel void tonemap(texture2d<float, access::read>  accum    [[texture(0)]],
                    texture2d<float, access::write> outColor [[texture(1)]],
                    constant float &exposure                 [[buffer(0)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outColor.get_width() || gid.y >= outColor.get_height()) return;

    float3 hdr = accum.read(gid).rgb * exposure;
    float3 mapped = acesFilmic(hdr);
    mapped = pow(mapped, float3(1.0f / 2.2f));   // gamma encode for the 8-bit drawable
    outColor.write(float4(mapped, 1.0f), gid);
}
