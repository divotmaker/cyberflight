#version 450

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 color;
} pc;

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_fade;

layout(location = 0) out float frag_fade;

void main() {
    gl_Position = pc.view_proj * vec4(in_position, 1.0);
    frag_fade = in_fade;
}
