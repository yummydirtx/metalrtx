// =============================================================================
// Blit.metal — final presentation. For now (Phase 0) it draws an animated sky
// gradient so we can verify runtime shader compilation and the compute→drawable
// path end-to-end. Later phases replace the body with a tonemap of the traced
// + denoised radiance buffer.
// =============================================================================

kernel void presentGradient(texture2d<float, access::write> outColor [[texture(0)]],
                            constant float &time                     [[buffer(0)]],
                            uint2 gid                                [[thread_position_in_grid]]) {
    uint w = outColor.get_width();
    uint h = outColor.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float2 uv = float2(gid) / float2(w, h);

    // A simple vertical sky gradient with a slow horizontal shimmer.
    float3 top = float3(0.18f, 0.42f, 0.78f);
    float3 bottom = float3(0.92f, 0.76f, 0.52f);
    float t = uv.y + 0.03f * sin(uv.x * 12.0f + time);
    float3 col = mix(bottom, top, clamp(t, 0.0f, 1.0f));

    // Mild vignette so the window obviously contains rendered content.
    float2 d = uv - 0.5f;
    col *= 1.0f - 0.35f * dot(d, d);

    outColor.write(float4(col, 1.0f), gid);
}
