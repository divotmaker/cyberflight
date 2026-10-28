#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    vec4 camera_pos;  // xyz = position, w = viewport X offset (pixels)
    mat4 inv_view_proj;
    vec4 params;      // x = trail tail fade value, yz = unused, w = viewport Y offset (pixels)
    vec4 ball_pos;    // xyz = ball center, w = trail fade distance (meters)
} pc;

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT vec4 reflPayload;

hitAttributeEXT vec2 attribs;

// Geometry types (gl_InstanceCustomIndexEXT):
//   0 = floor (reflective glass surface)
//   2 = tee box fill (dark surface)
//   3 = trail ribbon (emissive magenta)
//   4 = tee box border (cyan)
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

// Emissive gain of the tracer and ball as seen in the floor: they are light
// sources several times brighter than display white, so their Fresnel
// reflection (~0.10 at a 45° view) stays visible. Glass tint and specular
// use the plain Fresnel term.
const float EMISSIVE_GAIN = 4.0;

// Colour filter of the tinted glass on emissive reflections: they lose
// REFL_DESATURATE of their saturation and are multiplied by the cool
// REFL_TINT, so the tracer's magenta reads as violet in the navy glass.
const vec3 REFL_TINT = vec3(0.55, 0.75, 1.0);
const float REFL_DESATURATE = 0.45;

// Dark navy glass tint, stronger at grazing angles.
const vec3 GLASS_TINT = vec3(0.006, 0.014, 0.030);

// Overhead specular intensity (cyan gleam on floor surface).
const float SPEC_INTENSITY = 0.18;
const float SPEC_POWER = 200.0;

// Analytical ball reflection: glow halo multiplier relative to ball radius.
const float BALL_GLOW_MULT = 6.0;

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

        // Trace reflection ray for trail geometry only (mask 0x04)
        vec3 refl_dir = reflect(gl_WorldRayDirectionEXT, N);
        reflPayload = vec4(0.0);
        if (gl_HitTEXT > 0.0) {
            traceRayEXT(
                tlas,
                gl_RayFlagsOpaqueEXT,
                0x04, 0, 0, 0,
                hit_pos + N * 0.001, 0.01, refl_dir, 1000.0, 1
            );
        }

        // Trail reflection with distance fade
        float cam_dist = length(hit_pos.xz - pc.camera_pos.xz);
        float dist_fade = 1.0 - smoothstep(100.0, 350.0, cam_dist);
        float emissive_refl = fresnel * EMISSIVE_GAIN;
        vec3 trail_reflection = reflPayload.rgb * emissive_refl * dist_fade;

        // Analytical ball reflection: glow intensity from the closest approach
        // of the reflected ray to the ball center (none if the ball is behind it).
        vec3 to_center = pc.ball_pos.xyz - hit_pos;
        float t_closest = dot(to_center, refl_dir);
        float d = t_closest > 0.0 ? length(to_center - refl_dir * t_closest) : 1e9;
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
        vec3 ball_reflection = (MAGENTA * ball_glow * 3.0 + vec3(1.0) * core_glow * 3.0) * emissive_refl * dist_fade * occlude;

        // Glass base tint: dark navy, stronger at grazing angles.
        vec3 glass_base = GLASS_TINT * fresnel;

        // Specular highlight: faint cyan gleam from an overhead light source.
        vec3 L = normalize(vec3(0.15, 1.0, 0.4));
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), SPEC_POWER);
        vec3 specular = CYAN * SPEC_INTENSITY * spec * fresnel;

        // Subtle ground haze at distance: reflections fade into dark blue
        // instead of pure black, adding atmospheric depth.
        float haze = smoothstep(150.0, 400.0, cam_dist);
        vec3 haze_color = GLASS_TINT * 1.5;
        vec3 reflection = trail_reflection + ball_reflection;

        // Tinted glass: the reflection loses saturation and takes the glass hue.
        float lum = dot(reflection, vec3(0.2126, 0.7152, 0.0722));
        reflection = mix(reflection, vec3(lum), REFL_DESATURATE);
        reflection *= REFL_TINT;

        reflection = mix(reflection, haze_color, haze * 0.6);

        payload = vec4(glass_base + reflection + specular, 1.0);

    } else if (geom_type == 2) {
        // Tee box fill: dark surface.
        payload = vec4(TEE_COLOR, 1.0);

    } else if (geom_type == 3) {
        // Trail: emissive magenta with the raster tracer's fade. The tracer
        // fade value t runs from 1 at the ball to params.x at the trail's
        // oldest point, interpolated by distance from the ball over the
        // trail's arc length (ball_pos.w).
        float fade_dist = pc.ball_pos.w > 0.0 ? pc.ball_pos.w : 50.0;
        float u = clamp(distance(hit_pos, pc.ball_pos.xyz) / fade_dist, 0.0, 1.0);
        float t = 1.0 - u * (1.0 - pc.params.x);
        // Raster core alpha (1 - 0.7 (1 - t))^2, times its tip taper over t < 0.15.
        float core = 1.0 - 0.7 * (1.0 - t);
        float alpha = core * core * min(t / 0.15, 1.0);

        // Only floor reflection rays reach the trail (mask 0x04), so this
        // runs at the pipeline's maximum recursion depth and traces nothing.
        payload = vec4(MAGENTA * alpha, 1.0);

    } else if (geom_type == 4) {
        // Tee box border: bright cyan.
        payload = vec4(CYAN, 1.0);

    } else {
        payload = vec4(1.0, 0.0, 1.0, 1.0); // debug: hot pink
    }
}
