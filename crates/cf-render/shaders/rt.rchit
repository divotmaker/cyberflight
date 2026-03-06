#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    vec4 camera_pos;
    mat4 inv_view_proj;
    vec4 grid_params; // x = spacing_m, y = line_half_width, z = max_fade_dist, w = viewport_y_offset
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

// Subtle dark blue-cyan tint visible at grazing angles.
const vec3 GLASS_TINT = vec3(0.006, 0.014, 0.022);

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

        // Fresnel-modulated trail reflection with distance fade
        float cam_dist = length(hit_pos.xz - pc.camera_pos.xz);
        float dist_fade = 1.0 - smoothstep(100.0, 350.0, cam_dist);
        vec3 trail_reflection = reflPayload.rgb * fresnel * dist_fade;

        // Analytical ball reflection: mirror ball center across floor (Y=0)
        // and compute glow intensity based on distance from the floor hit point
        // to the mirrored ball position.
        vec3 ball_mirror = vec3(pc.ball_pos.x, -pc.ball_pos.y, pc.ball_pos.z);
        float d = distance(hit_pos, ball_mirror);
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
        vec3 ball_reflection = (MAGENTA * ball_glow * 3.0 + vec3(1.0) * core_glow * 3.0) * fresnel * dist_fade * occlude;

        // Glass base tint: subtle dark blue-cyan, stronger at grazing angles.
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
        reflection = mix(reflection, haze_color, haze * 0.6);

        payload = vec4(glass_base + reflection + specular, 1.0);

    } else if (geom_type == 2) {
        // Tee box fill: dark surface.
        payload = vec4(TEE_COLOR, 1.0);

    } else if (geom_type == 3) {
        // Trail: emissive magenta, fading to transparent at the tail.
        float fade_dist = pc.ball_pos.w > 0.0 ? pc.ball_pos.w : 50.0;
        float d = distance(hit_pos, pc.ball_pos.xyz);
        float alpha = 1.0 - clamp(d / fade_dist, 0.0, 1.0);

        // Continuation ray through trail to see what's behind (floor only).
        reflPayload = vec4(0.0);
        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            0x01, 0, 0, 0,
            hit_pos + gl_WorldRayDirectionEXT * 0.001,
            0.01, gl_WorldRayDirectionEXT, 1000.0, 1
        );
        vec3 background = reflPayload.rgb;

        vec3 trail_color = mix(background, MAGENTA, alpha);
        payload = vec4(trail_color, 1.0);

    } else if (geom_type == 4) {
        // Tee box border: bright cyan.
        payload = vec4(CYAN, 1.0);

    } else {
        payload = vec4(1.0, 0.0, 1.0, 1.0); // debug: hot pink
    }
}
