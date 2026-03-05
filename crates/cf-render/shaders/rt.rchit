#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    vec4 camera_pos;
    mat4 inv_view_proj;
    vec4 grid_params; // x = spacing_m, y = line_half_width, z = max_fade_dist, w = unused
    vec4 ball_pos;    // xyz = ball center, w = trail fade distance (meters)
} pc;

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT vec4 reflPayload;

hitAttributeEXT vec2 attribs;

// Geometry types (gl_InstanceCustomIndexEXT):
//   0 = floor (reflective glass surface)
//   1 = ball (emissive magenta glow sphere)
//   2 = tee box fill (dark surface)
//   3 = trail ribbon (emissive magenta)
//   4 = tee box border (cyan)
//
// Instance mask scheme:
//   Floor:  0x01 (bit 0 — visible to primary rays)
//   Ball:   0x02 (bit 1 — visible to reflection rays only)
//   Tee box:  0x02
//   Border: 0x02
//   Trail:  0x04 (bit 2 — visible to reflections, skipped by trail continuation)

const vec3 CYAN    = vec3(0.0, 1.0, 1.0);
const vec3 MAGENTA = vec3(1.0, 0.0, 1.0);
const vec3 TEE_COLOR = vec3(0.0, 0.0, 0.0);

// ── Glass material parameters ──

// Schlick F0 for polished dark glass (higher than real glass ~0.04
// for a more dramatic Tron floor look).
const float F0_GLASS = 0.10;

// Subtle dark blue-cyan tint visible at grazing angles.
const vec3 GLASS_TINT = vec3(0.006, 0.014, 0.022);

// Overhead specular intensity (cyan gleam on floor surface).
const float SPEC_INTENSITY = 0.18;
const float SPEC_POWER = 200.0;

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

        // Trace single reflection ray
        vec3 refl_dir = reflect(gl_WorldRayDirectionEXT, N);
        reflPayload = vec4(0.0);
        if (gl_HitTEXT > 0.0) {
            traceRayEXT(
                tlas,
                gl_RayFlagsOpaqueEXT,
                0x06, 0, 0, 0,
                hit_pos + N * 0.001, 0.01, refl_dir, 1000.0, 1
            );
        }

        // Fresnel-modulated reflection with distance fade
        float cam_dist = length(hit_pos.xz - pc.camera_pos.xz);
        float dist_fade = 1.0 - smoothstep(100.0, 350.0, cam_dist);
        vec3 reflection = reflPayload.rgb * fresnel * dist_fade;

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
        reflection = mix(reflection, haze_color, haze * 0.6);

        payload = vec4(glass_base + reflection + specular, 1.0);

    } else if (geom_type == 1) {
        // Ball: emissive magenta halo sphere (6x physical ball radius in TLAS).
        // Two-layer glow: hot white core fading to magenta outer halo.
        float ball_r = pc.ball_pos.w > 0.0 ? pc.ball_pos.w : 0.021335;
        float rt_sphere_r = ball_r * 6.0; // must match rt_offscreen.rs multiplier
        float dist_to_center = distance(hit_pos, pc.ball_pos.xyz);
        float halo_t = clamp(dist_to_center / rt_sphere_r, 0.0, 1.0);

        // Outer magenta halo: cubic falloff across full sphere
        float outer = pow(1.0 - halo_t, 3.0);
        // Inner white-hot core: very steep falloff, only visible within ~1x radius
        float core_t = clamp(dist_to_center / ball_r, 0.0, 1.0);
        float core = pow(1.0 - core_t, 4.0);

        vec3 color = MAGENTA * outer + vec3(1.0) * core * 0.6;
        payload = vec4(color, 1.0);

    } else if (geom_type == 2) {
        // Tee box fill: dark surface.
        payload = vec4(TEE_COLOR, 1.0);

    } else if (geom_type == 3) {
        // Trail: emissive magenta, fading to transparent at the tail.
        float fade_dist = pc.ball_pos.w > 0.0 ? pc.ball_pos.w : 50.0;
        float d = distance(hit_pos, pc.ball_pos.xyz);
        float alpha = 1.0 - clamp(d / fade_dist, 0.0, 1.0);

        // Continuation ray through trail to see what's behind.
        reflPayload = vec4(0.0);
        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            0x03, 0, 0, 0,
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
