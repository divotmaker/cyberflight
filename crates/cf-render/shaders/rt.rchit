#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "floor_shading.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

// The terrain's vertex normals: 3 floats per vertex, 3 vertices per triangle,
// in the order of its triangles.
layout(set = 0, binding = 4, std430) readonly buffer TerrainNormals {
    float terrain_normals[];
};

layout(push_constant) uniform PushConstants {
    vec4 camera_pos;  // xyz = position, w = viewport X offset (pixels)
    mat4 inv_view_proj;
    vec4 ball_pos;    // xyz = ball center, w = viewport Y offset (pixels)
    layout(offset = 100) float ball_radius;
} pc;

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT vec4 reflPayload;

// Barycentrics of the hit: the weights of the triangle's second and third
// vertices.
hitAttributeEXT vec2 hit_bary;

// Geometry types (gl_InstanceCustomIndexEXT):
//   0 = floor (reflective glass surface, normal +Y)
//   1 = terrain (the same glass, normal interpolated from its vertices)
//   2 = tee box fill (dark surface)
//   4 = tee box border (cyan)
//
// The tracer tube (3) is translucent triangles with its own hit group
// (rt_trail.rahit).
//
// The tee box instances (2, 4) are in the TLAS, but no ray's cull mask
// includes theirs, so their branches below are never reached.
//
// Ball reflections are computed analytically in the floor shader
// (no ball geometry in the TLAS).

const vec3 TEE_COLOR = vec3(0.0, 0.0, 0.0);

// The terrain's normal at the hit: its triangle's vertex normals weighted by
// the hit's barycentrics.
vec3 terrain_normal() {
    int base = gl_PrimitiveID * 9;
    vec3 n0 = vec3(terrain_normals[base], terrain_normals[base + 1], terrain_normals[base + 2]);
    vec3 n1 = vec3(terrain_normals[base + 3], terrain_normals[base + 4], terrain_normals[base + 5]);
    vec3 n2 = vec3(terrain_normals[base + 6], terrain_normals[base + 7], terrain_normals[base + 8]);
    float w0 = 1.0 - hit_bary.x - hit_bary.y;
    return normalize(n0 * w0 + n1 * hit_bary.x + n2 * hit_bary.y);
}

// Dark glass with Fresnel reflections at `hit_pos`, surface normal `N`.
vec4 glass(vec3 hit_pos, vec3 N) {
    vec3 view_dir = normalize(gl_WorldRayDirectionEXT);

    // Trace reflection ray for the tracer tube only (mask 0x04), without
    // the opaque flag so its any-hit runs. The tube adds its light to
    // rgb and the light-weighted sum of the heights it crossed to alpha.
    vec3 refl_dir = reflect(view_dir, N);
    reflPayload = vec4(0.0);
    traceRayEXT(
        tlas,
        gl_RayFlagsNoneEXT,
        0x04, 0, 0, 0,
        hit_pos + N * 0.001, 0.01, refl_dir, 1000.0, 1
    );

    return shade_ground(pc.camera_pos.xyz, hit_pos, view_dir, reflPayload, pc.ball_pos.xyz,
                        pc.ball_radius, N);
}

void main() {
    int geom_type = gl_InstanceCustomIndexEXT;
    vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    if (geom_type == 0) {
        // Floor: flat glass.
        payload = glass(hit_pos, vec3(0.0, 1.0, 0.0));

    } else if (geom_type == 1) {
        // Terrain: the same glass, following the ground's shape.
        payload = glass(hit_pos, terrain_normal());

    } else if (geom_type == 2) {
        // Tee box fill: dark surface.
        payload = vec4(TEE_COLOR, 1.0);

    } else if (geom_type == 4) {
        // Tee box border: bright cyan.
        payload = vec4(CYAN, 1.0);

    } else {
        payload = vec4(1.0, 0.0, 1.0, 1.0); // debug: hot pink
    }
}
