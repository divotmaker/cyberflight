#version 450

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 color;
} pc;

layout(location = 0) in float frag_fade;

layout(location = 0) out vec4 out_color;

void main() {
    // Negative fade encodes alert color (bright red, full opacity).
    if (frag_fade < 0.0) {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    // Fade grid lines toward transparent at distance
    float alpha = 1.0 - frag_fade;
    // Smooth falloff for neon glow feel
    alpha = alpha * alpha;
    out_color = vec4(pc.color.rgb, pc.color.a * alpha);
}
