#version 460
#extension GL_EXT_ray_tracing : require

// Closest hit of the trail tube (rt.rint). Only floor reflection rays reach
// the trail (mask 0x04), so this runs at the pipeline's maximum recursion
// depth and traces nothing.

layout(location = 0) rayPayloadInEXT vec4 payload;

// Written by rt.rint.
struct TubeHit {
    float height;   // world height of the closest point on the trail axis
    float fade;     // tracer fade value t there (1 at the ball)
    float rel_dist; // ray-to-axis distance / tube radius, in [0, 1)
    float rel_edge; // width of the soft edge / tube radius, in (0, 1]
    float weight;   // brightness of the tip taper where the width cannot taper
};
hitAttributeEXT TubeHit hit;

const vec3 MAGENTA = vec3(1.0, 0.0, 1.0);

void main() {
    // Soft edge: opaque at the axis, fading to transparent at the radius, so
    // the tube's reflection is never wider than the raster core.
    float coverage = 1.0 - smoothstep(1.0 - hit.rel_edge, 1.0, hit.rel_dist);

    // Raster core alpha (1 - 0.7 (1 - t))^2.
    float core = 1.0 - 0.7 * (1.0 - hit.fade);
    float alpha = core * core * coverage * hit.weight;

    // Alpha carries the hit's world height for the floor's height fade.
    payload = vec4(MAGENTA * alpha, hit.height);
}
