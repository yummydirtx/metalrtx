#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// =============================================================================
// Common.metal — shared declarations used across every kernel in the demo.
// This file is concatenated first by ShaderLibrary, so everything declared here
// is visible to the path tracer, denoiser, sky, and post-processing kernels.
// =============================================================================

// ---- Constants --------------------------------------------------------------

constant float PI       = 3.14159265358979323846f;
constant float TWO_PI   = 6.28318530717958647692f;
constant float INV_PI   = 0.31830988618379067154f;

// ---- Per-frame uniforms (must match ShaderTypes.swift layout) ---------------

struct CameraUniforms {
    float4x4 viewProj;          // current view-projection
    float4x4 invViewProj;       // inverse, for reconstructing world rays
    float4x4 prevViewProj;      // previous frame, for temporal reprojection
    packed_float3 position;     // camera world position
    float _pad0;
    packed_float3 forward;
    float _pad1;
    packed_float3 right;
    float _pad2;
    packed_float3 up;
    float _pad3;
    float tanHalfFovY;
    float aspect;
    uint  frameIndex;           // increments every frame, used to seed RNG
    uint  accumulatedFrames;    // resets to 0 when the camera/scene changes
};

struct RenderSettings {
    packed_float3 sunDirection; // normalized, points toward the sun
    float sunIntensity;
    packed_float3 sunColor;
    float turbidity;
    float timeOfDay;            // 0..1 across a full day
    float exposure;
    uint  maxBounces;
    uint  denoiseEnabled;
    float waterRoughness;
    float waveAmplitude;        // master Gerstner wave steepness
    float waveChoppiness;       // horizontal sharpening (Gerstner q)
    float waveSpeed;            // wave animation time scale
    float elapsedTime;          // seconds, drives water animation
    uint  width;
    uint  height;
    uint  flashlightEnabled;    // 1 = camera flashlight on
    packed_float3 flashlightPos; // world-space light position (camera)
    packed_float3 flashlightDir; // normalized aim direction (camera forward)
    uint  fogEnabled;           // 1 = thin volumetric fog so the beam cone is visible
};

// ---- Material (must match ShaderTypes.swift) --------------------------------

struct Material {
    packed_float3 albedo;
    float roughness;
    packed_float3 emission;
    float metallic;
    float transparency;     // 0 = opaque, 1 = fully transparent (glass/water)
    float ior;              // index of refraction
    uint  flags;            // bit0 = isWater
};

constant uint MATERIAL_FLAG_WATER = 1u;

// ---- Geometry buffers -------------------------------------------------------

// Per-triangle shading data, indexed by (instanceBaseOffset + primitive_id). Voxel faces
// are flat, so a single normal + material per triangle is exact.
struct PrimitiveData {
    packed_float3 normal;
    uint materialIndex;
};

// ---- RNG: PCG hash + uniform floats ----------------------------------------

inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline uint initSeed(uint2 pixel, uint width, uint frame) {
    return pcgHash(pixel.x + pixel.y * width + frame * 0x9E3779B9u);
}

inline float randFloat(thread uint &seed) {
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word) * (1.0f / 4294967296.0f);
}

inline float2 rand2(thread uint &seed) {
    return float2(randFloat(seed), randFloat(seed));
}

// ---- Sampling helpers -------------------------------------------------------

// Build an orthonormal basis around a unit normal (Duff et al. 2017).
inline void onb(float3 n, thread float3 &t, thread float3 &b) {
    float s = (n.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere sample around normal n.
inline float3 cosineSampleHemisphere(float3 n, float2 u) {
    float r = sqrt(u.x);
    float phi = TWO_PI * u.y;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0f, 1.0f - u.x));
    float3 t, b;
    onb(n, t, b);
    return normalize(t * x + b * y + n * z);
}

// GGX importance sampling — returns a microfacet half-vector around normal n.
inline float3 sampleGGX(float3 n, float roughness, float2 u) {
    float a = roughness * roughness;
    float phi = TWO_PI * u.x;
    float cosTheta = sqrt((1.0f - u.y) / (1.0f + (a * a - 1.0f) * u.y));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float3 h = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    float3 t, b;
    onb(n, t, b);
    return normalize(t * h.x + b * h.y + n * h.z);
}

// Schlick Fresnel.
inline float3 fresnelSchlick(float cosTheta, float3 f0) {
    float m = clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float m2 = m * m;
    return f0 + (1.0f - f0) * (m2 * m2 * m);
}

inline float luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}
