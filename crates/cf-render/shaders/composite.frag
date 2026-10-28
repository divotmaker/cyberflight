#version 460

layout(set = 0, binding = 0) uniform sampler2D reflectionImage;

layout(location = 0) out vec4 outColor;

void main() {
    // One storage-image texel per pixel: the storage image and the target
    // share one extent, so the reflections land exactly where they were
    // traced. Their colour is already zero where no floor was hit, so it is
    // added at full weight (alpha 1).
    outColor = vec4(texelFetch(reflectionImage, ivec2(gl_FragCoord.xy), 0).rgb, 1.0);
}
