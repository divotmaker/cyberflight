#version 450
#extension GL_GOOGLE_include_directive : require

#include "floor_shading.glsl"

// Fragment stage of the mirror pass's floor: shades each floor pixel as
// rt.rchit does, with the mirrored tracer's light that the tracer pass
// accumulated in this pixel in place of the light along a reflected ray.

layout(set = 0, binding = 0) uniform sampler2D tracer;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 camera_pos; // xyz = camera position
    vec4 ball_pos;   // xyz = ball centre, w = ball radius
} pc;

layout(location = 0) in vec3 frag_position;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 view_dir = normalize(frag_position - pc.camera_pos.xyz);
    vec4 trail = texelFetch(tracer, ivec2(gl_FragCoord.xy), 0);
    out_color = reflection_texel(
        shade_floor(pc.camera_pos.xyz, frag_position, view_dir, trail, pc.ball_pos.xyz,
                    pc.ball_pos.w));
}
