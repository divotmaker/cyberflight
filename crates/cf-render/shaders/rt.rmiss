#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
    // Nothing to add: every trace starts from a zeroed payload (black
    // background, cyberflight aesthetic), and the tracer's any-hit has
    // accumulated into it by the time a ray misses.
}
