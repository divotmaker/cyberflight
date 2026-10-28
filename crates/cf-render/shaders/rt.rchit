#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    vec4 camera_pos;  // xyz = position, w = viewport X offset (pixels)
    mat4 inv_view_proj;
    vec4 ball_pos;    // xyz = ball center, w = viewport Y offset (pixels)
} pc;

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT vec4 reflPayload;

hitAttributeEXT vec2 attribs;

// Geometry types (gl_InstanceCustomIndexEXT):
//   0 = floor (reflective glass surface)
//   2 = tee box fill (dark surface)
//   4 = tee box border (cyan)
//
// The tracer tube (3) is translucent triangles with its own hit group
// (rt_trail.rahit).
//
// The tee box instances (2, 4) are in the TLAS, but no ray's cull mask
// includes theirs, so their branches below are never reached.
//
// Ball reflections are computed analytically in the floor shader
// (no ball geometry in the TLAS).

const vec3 CYAN    = vec3(0.0, 1.0, 1.0);
const vec3 MAGENTA = vec3(1.0, 0.0, 1.0);
const vec3 TEE_COLOR = vec3(0.0, 0.0, 0.0);
const float BALL_RADIUS = 0.021335; // golf ball radius in meters

// ── Glass material parameters ──

// Schlick F0 for polished dark glass (higher than real glass ~0.04
// for a more dramatic Tron floor look).
const float F0_GLASS = 0.10;

// Reflectance of the floor for the tracer and ball: one constant at every
// distance and view angle, so their reflection is equally strong near and
// far. They are light sources brighter than display white, hence > 1.
// Glass tint and specular use the Fresnel term.
const float EMISSIVE_REFLECTANCE = 1.1;

// Colour filter of the tinted glass on emissive reflections: they lose
// REFL_DESATURATE of their saturation and are multiplied by the cool
// REFL_TINT, so the tracer's magenta reads as violet in the navy glass.
const vec3 REFL_TINT = vec3(0.55, 0.75, 1.0);
const float REFL_DESATURATE = 0.45;

// Tracer and ball reflections fade with the reflected point's height above
// the floor: full strength at the floor, falling linearly to REFLECTION_FADE_MIN
// of it at REFLECTION_FADE_HEIGHT metres and staying there above.
const float REFLECTION_FADE_HEIGHT = 60.0; // metres above the floor
const float REFLECTION_FADE_MIN = 0.1;     // strength at and above that height

// Dark navy glass tint, stronger at grazing angles.
const vec3 GLASS_TINT = vec3(0.006, 0.014, 0.030);

// Overhead specular intensity (cyan gleam on floor surface).
const float SPEC_INTENSITY = 0.18;
const float SPEC_POWER = 200.0;

// Analytical ball reflection: glow halo multiplier relative to ball radius.
const float BALL_GLOW_MULT = 6.0;

// Strength of the reflection of a point h metres above the floor.
float height_fade(float h) {
    return mix(1.0, REFLECTION_FADE_MIN, clamp(h / REFLECTION_FADE_HEIGHT, 0.0, 1.0));
}

void main() {
    int geom_type = gl_InstanceCustomIndexEXT;
    vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    if (geom_type == 0) {
        // ── Floor: dark glass with Fresnel reflections ──
        vec3 V = -normalize(gl_WorldRayDirectionEXT); // toward camera
        vec3 N = vec3(0.0, 1.0, 0.0);

        // Schlick Fresnel approximation
        float NdotV = max(dot(N, V), 0.0);
        float fresnel = F0_GLASS + (1.0 - F0_GLASS) * pow(1.0 - NdotV, 5.0);

        // Trace reflection ray for the tracer tube only (mask 0x04), without
        // the opaque flag so its any-hit runs. The tube adds its light to
        // rgb and the light-weighted sum of the heights it crossed to alpha.
        vec3 refl_dir = reflect(gl_WorldRayDirectionEXT, N);
        reflPayload = vec4(0.0);
        if (gl_HitTEXT > 0.0) {
            traceRayEXT(
                tlas,
                gl_RayFlagsNoneEXT,
                0x04, 0, 0, 0,
                hit_pos + N * 0.001, 0.01, refl_dir, 1000.0, 1
            );
        }

        vec3 trail_reflection = reflPayload.rgb * EMISSIVE_REFLECTANCE;
        // Height of the reflected light above the floor, averaged by light.
        float trail_height = reflPayload.r > 0.0 ? reflPayload.a / reflPayload.r : 0.0;
        float trail_fade = height_fade(trail_height - hit_pos.y);

        // Analytical ball reflection: glow intensity from the closest approach
        // of the reflected ray, which starts at the floor, to the ball center
        // (the ray's start if the ball is behind it).
        vec3 to_center = pc.ball_pos.xyz - hit_pos;
        float t_closest = max(dot(to_center, refl_dir), 0.0);
        float d = length(to_center - refl_dir * t_closest);
        float glow_radius = BALL_RADIUS * BALL_GLOW_MULT;
        float glow_t = clamp(d / glow_radius, 0.0, 1.0);
        float ball_glow = pow(1.0 - glow_t, 3.0);
        // Core brightens near the center
        float core_d = clamp(d / BALL_RADIUS, 0.0, 1.0);
        float core_glow = pow(1.0 - core_d, 4.0) * 0.6;
        // Suppress glow directly under the ball where the raster ball occludes
        // the floor — the additive composite would otherwise bleed through.
        // Use the viewing ray to check if this floor point is within the ball's
        // apparent disc from the camera's perspective.
        vec3 to_ball = pc.ball_pos.xyz - hit_pos;
        float proj = dot(to_ball, V);
        float perp = length(to_ball - V * proj);
        float occlude = smoothstep(BALL_RADIUS * 1.5, BALL_RADIUS * 3.0, perp);
        vec3 ball_reflection = (MAGENTA * ball_glow * 3.0 + vec3(1.0) * core_glow * 3.0) * EMISSIVE_REFLECTANCE * occlude;
        float ball_fade = height_fade(pc.ball_pos.y - hit_pos.y);

        // Glass base tint: dark navy, stronger at grazing angles. Far floor
        // fades into dark blue instead of pure black.
        float cam_dist = length(hit_pos.xz - pc.camera_pos.xz);
        float haze = smoothstep(150.0, 400.0, cam_dist);
        vec3 glass_base = GLASS_TINT * fresnel + GLASS_TINT * 1.5 * (haze * 0.6);

        // Specular highlight: faint cyan gleam from an overhead light source.
        vec3 L = normalize(vec3(0.15, 1.0, 0.4));
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), SPEC_POWER);
        vec3 specular = CYAN * SPEC_INTENSITY * spec * fresnel;

        vec3 reflection = trail_reflection * trail_fade + ball_reflection * ball_fade;

        // Tinted glass: the reflection loses saturation and takes the glass hue.
        float lum = dot(reflection, vec3(0.2126, 0.7152, 0.0722));
        reflection = mix(reflection, vec3(lum), REFL_DESATURATE);
        reflection *= REFL_TINT;

        payload = vec4(glass_base + reflection + specular, 1.0);

    } else if (geom_type == 2) {
        // Tee box fill: dark surface.
        payload = vec4(TEE_COLOR, 1.0);

    } else if (geom_type == 4) {
        // Tee box border: bright cyan.
        payload = vec4(CYAN, 1.0);

    } else {
        payload = vec4(1.0, 0.0, 1.0, 1.0); // debug: hot pink
    }
}
