#version 450
#extension GL_GOOGLE_include_directive : require

#include "floor_shading.glsl"

// Fragment stage of the mirrored tracer. Adds the light of a tracer surface
// to the pixel, in the form rt_trail.rahit gives the floor shader along a
// reflected ray (rgb = light, a = light × height of the surface above the
// floor); the mirror pass blends it additively. The part of the tracer below
// the floor has no reflection, as no ray leaving the floor reaches it.

layout(location = 0) in float frag_fade;
layout(location = 1) in float frag_height;

layout(location = 0) out vec4 out_light;

void main() {
    if (frag_height < 0.0) {
        discard;
    }
    // The raster fragment shader's alpha, (1 - fade)².
    float alpha = (1.0 - frag_fade) * (1.0 - frag_fade);
    out_light = vec4(MAGENTA * alpha, alpha * frag_height);
}
