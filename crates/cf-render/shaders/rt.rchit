#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "floor_shading.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    vec4 camera_pos;  // xyz = position, w = viewport X offset (pixels)
    mat4 inv_view_proj;
    vec4 ball_pos;    // xyz = ball center, w = viewport Y offset (pixels)
} pc;

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT vec4 reflPayload;

hitAttributeEXT vec2 attribs;

// Geometry types (gl_InstanceCustomIndexEXT):
//   0 = floor (reflective glass surface)
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

void main() {
    int geom_type = gl_InstanceCustomIndexEXT;
    vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    if (geom_type == 0) {
        // ── Floor: dark glass with Fresnel reflections ──
        vec3 N = vec3(0.0, 1.0, 0.0);
        vec3 view_dir = normalize(gl_WorldRayDirectionEXT);

        // Trace reflection ray for the tracer tube only (mask 0x04), without
        // the opaque flag so its any-hit runs. The tube adds its light to
        // rgb and the light-weighted sum of the heights it crossed to alpha.
        vec3 refl_dir = reflect(view_dir, N);
        reflPayload = vec4(0.0);
        if (gl_HitTEXT > 0.0) {
            traceRayEXT(
                tlas,
                gl_RayFlagsNoneEXT,
                0x04, 0, 0, 0,
                hit_pos + N * 0.001, 0.01, refl_dir, 1000.0, 1
            );
        }

        payload = shade_floor(pc.camera_pos.xyz, hit_pos, view_dir, reflPayload, pc.ball_pos.xyz);

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
