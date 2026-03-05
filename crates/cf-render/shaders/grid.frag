#version 450

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 color;
    vec4 clip_bounds; // [min_x, min_z, max_x, max_z]
} pc;

layout(location = 0) in float frag_fade;
layout(location = 1) in vec3 frag_world_pos;

layout(location = 0) out vec4 out_color;

void main() {
    // Clip to grid bounds (world-space XZ)
    if (frag_world_pos.x < pc.clip_bounds.x || frag_world_pos.x > pc.clip_bounds.z ||
        frag_world_pos.z < pc.clip_bounds.y || frag_world_pos.z > pc.clip_bounds.w) {
        discard;
    }

    // Fade grid lines toward transparent at distance
    float alpha = 1.0 - frag_fade;
    // Smooth falloff for neon glow feel
    alpha = alpha * alpha;
    out_color = vec4(pc.color.rgb, pc.color.a * alpha);
}
