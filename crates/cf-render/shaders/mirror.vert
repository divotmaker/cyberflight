#version 450

// Vertex stage of the mirrored tracer: the tube and centre line the raster
// pass draws, reflected across the floor plane y = 0.

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_fade;

layout(location = 0) out float frag_fade;
// Height of the vertex above the floor, before the reflection.
layout(location = 1) out float frag_height;

void main() {
    gl_Position = pc.view_proj * vec4(in_position.x, -in_position.y, in_position.z, 1.0);
    frag_fade = in_fade;
    frag_height = in_position.y;
}
