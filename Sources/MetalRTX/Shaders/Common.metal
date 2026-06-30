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
    float sunAngularRadius;     // angular radius of the sun disk (radians) for soft shadows
    uint  emitterCount;         // number of emissive blocks sampled by next-event estimation
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
    float detailScale;      // procedural noise frequency (0 disables surface detail)
    float detailStrength;   // albedo variation amount
    float bumpStrength;     // normal-perturbation amount
    packed_float3 absorption; // Beer-Lambert extinction per unit distance (dielectrics)
};

constant uint MATERIAL_FLAG_WATER = 1u;

// ---- Emissive blocks for next-event estimation (must match ShaderTypes.swift) ----

// A spherical light approximating one emissive voxel, sampled directly so emissive
// blocks cast clean local lighting and soft shadows instead of relying on random GI.
struct Emitter {
    packed_float3 position;   // block center, world space
    float radius;             // sphere radius
    packed_float3 emission;   // emitted radiance
    float _pad;
};

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

// Uniformly sample a direction inside a cone of half-angle acos(cosThetaMax) around
// `axis`. Used to sample the sun disk (soft shadows) and spherical emitters.
inline float3 sampleCone(float3 axis, float cosThetaMax, float2 u) {
    float cosTheta = 1.0f - u.x * (1.0f - cosThetaMax);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = TWO_PI * u.y;
    float3 t, b;
    onb(axis, t, b);
    return normalize(t * (cos(phi) * sinTheta) + b * (sin(phi) * sinTheta) + axis * cosTheta);
}

// Schlick Fresnel.
inline float3 fresnelSchlick(float cosTheta, float3 f0) {
    float m = clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float m2 = m * m;
    return f0 + (1.0f - f0) * (m2 * m2 * m);
}

// ---- Procedural surface detail (value-noise fBm) ---------------------------

inline float hash13(float3 p) {
    p = fract(p * 0.1031f);
    p += dot(p, p.yzx + 33.33f);
    return fract((p.x + p.y) * p.z);
}

// Smooth trilinear value noise in [0, 1).
inline float valueNoise3(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    float n000 = hash13(i + float3(0.0f, 0.0f, 0.0f));
    float n100 = hash13(i + float3(1.0f, 0.0f, 0.0f));
    float n010 = hash13(i + float3(0.0f, 1.0f, 0.0f));
    float n110 = hash13(i + float3(1.0f, 1.0f, 0.0f));
    float n001 = hash13(i + float3(0.0f, 0.0f, 1.0f));
    float n101 = hash13(i + float3(1.0f, 0.0f, 1.0f));
    float n011 = hash13(i + float3(0.0f, 1.0f, 1.0f));
    float n111 = hash13(i + float3(1.0f, 1.0f, 1.0f));
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
}

inline float fbm3(float3 p, int octaves) {
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * valueNoise3(p);
        norm += amp;
        p *= 2.02f;
        amp *= 0.5f;
    }
    return sum / norm;
}

// Modulates albedo + roughness and perturbs the shading normal using world-space
// value-noise fBm, turning flat voxel faces into varied, slightly bumpy surfaces.
// Returns the perturbed normal; `albedo` and `roughness` are modified in place.
inline float3 surfaceDetail(float3 worldPos, float3 N,
                            thread float3 &albedo, thread float &roughness,
                            float detailScale, float detailStrength, float bumpStrength) {
    if (detailScale <= 0.0f) return N;
    float3 p = worldPos * detailScale;
    float h = fbm3(p, 4);
    float centered = h - 0.5f;
    albedo *= clamp(1.0f + centered * detailStrength, 0.0f, 2.0f);
    roughness = clamp(roughness - centered * 0.20f, 0.03f, 1.0f);
    if (bumpStrength <= 0.0f) return N;
    float3 t, b;
    onb(N, t, b);
    const float eps = 0.5f;
    float hT = fbm3((worldPos + t * eps) * detailScale, 4);
    float hB = fbm3((worldPos + b * eps) * detailScale, 4);
    float2 grad = float2(hT - h, hB - h) / eps;
    return normalize(N - (t * grad.x + b * grad.y) * bumpStrength);
}

// ---- Microfacet BRDF terms (GGX NDF / Smith masking-shadowing) -------------

inline float ggxD(float NdotH, float a) {
    float a2 = a * a;
    float d = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(PI * d * d, 1e-7f);
}

inline float smithG1(float NdotX, float a) {
    float a2 = a * a;
    return 2.0f * NdotX / max(NdotX + sqrt(a2 + (1.0f - a2) * NdotX * NdotX), 1e-7f);
}

inline float smithG(float NdotV, float NdotL, float a) {
    return smithG1(NdotV, a) * smithG1(NdotL, a);
}

inline float luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}
