#version 460

layout(set = 0, binding = 0) uniform sampler2D reflectionImage;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(reflectionImage, 0));
    vec2 uv = gl_FragCoord.xy * texel;

    // Slightly diffuse glass: 5x5 Gaussian blur on reflections.
    // Simulates subtle surface roughness — reflections are soft, not mirror-sharp.
    // Blur radius scaled by 1.5 texels for a gentle ~4px diameter softening.
    vec2 step = texel * 1.5;

    // 5x5 Gaussian kernel (sigma ~1.2), weights sum to 273
    vec4 color = vec4(0.0);
    // Row -2
    color += texture(reflectionImage, uv + vec2(-2, -2) * step) *  1.0;
    color += texture(reflectionImage, uv + vec2(-1, -2) * step) *  4.0;
    color += texture(reflectionImage, uv + vec2( 0, -2) * step) *  7.0;
    color += texture(reflectionImage, uv + vec2( 1, -2) * step) *  4.0;
    color += texture(reflectionImage, uv + vec2( 2, -2) * step) *  1.0;
    // Row -1
    color += texture(reflectionImage, uv + vec2(-2, -1) * step) *  4.0;
    color += texture(reflectionImage, uv + vec2(-1, -1) * step) * 16.0;
    color += texture(reflectionImage, uv + vec2( 0, -1) * step) * 26.0;
    color += texture(reflectionImage, uv + vec2( 1, -1) * step) * 16.0;
    color += texture(reflectionImage, uv + vec2( 2, -1) * step) *  4.0;
    // Row 0 (center)
    color += texture(reflectionImage, uv + vec2(-2,  0) * step) *  7.0;
    color += texture(reflectionImage, uv + vec2(-1,  0) * step) * 26.0;
    color += texture(reflectionImage, uv + vec2( 0,  0) * step) * 41.0;
    color += texture(reflectionImage, uv + vec2( 1,  0) * step) * 26.0;
    color += texture(reflectionImage, uv + vec2( 2,  0) * step) *  7.0;
    // Row 1
    color += texture(reflectionImage, uv + vec2(-2,  1) * step) *  4.0;
    color += texture(reflectionImage, uv + vec2(-1,  1) * step) * 16.0;
    color += texture(reflectionImage, uv + vec2( 0,  1) * step) * 26.0;
    color += texture(reflectionImage, uv + vec2( 1,  1) * step) * 16.0;
    color += texture(reflectionImage, uv + vec2( 2,  1) * step) *  4.0;
    // Row 2
    color += texture(reflectionImage, uv + vec2(-2,  2) * step) *  1.0;
    color += texture(reflectionImage, uv + vec2(-1,  2) * step) *  4.0;
    color += texture(reflectionImage, uv + vec2( 0,  2) * step) *  7.0;
    color += texture(reflectionImage, uv + vec2( 1,  2) * step) *  4.0;
    color += texture(reflectionImage, uv + vec2( 2,  2) * step) *  1.0;

    outColor = color / 273.0;
}
