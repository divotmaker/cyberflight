#version 450

// Vertex stage of the mirror pass's floor: the floor's triangles, which
// cover the pixels that show the floor.

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 camera_pos; // xyz = camera position
    vec4 ball_pos;   // xyz = ball centre, w = ball radius
} pc;

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 frag_position;

void main() {
    gl_Position = pc.view_proj * vec4(in_position, 1.0);
    frag_position = in_position;
}
