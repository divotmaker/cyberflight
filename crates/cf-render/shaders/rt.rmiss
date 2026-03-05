#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
    // Black background (cyberflight aesthetic)
    payload = vec4(0.0, 0.0, 0.0, 0.0);
}
