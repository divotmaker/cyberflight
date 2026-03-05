#version 460

// Fullscreen triangle — no vertex buffer needed.
// Three vertices cover the entire screen when drawn as a single triangle.

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
