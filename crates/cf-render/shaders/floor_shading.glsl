// The floor's material and the light it reflects, shared by the ray-traced
// floor (rt.rchit) and the raster mirror (mirror_floor.frag), so the two
// reflection paths shade a floor point with the same code.
//
// A path supplies what its way of finding the tracer gives: `trail`, the
// summed light of every tracer surface the reflected ray crosses
//   rgb = Σ light,  a = Σ light × world height of the crossing
// (rt_trail.rahit and rt_line.rahit accumulate it along the ray; the mirror
// accumulates the mirrored tracer per pixel), and the ball, which is
// reflected analytically.

const vec3 CYAN    = vec3(0.0, 1.0, 1.0);
const vec3 MAGENTA = vec3(1.0, 0.0, 1.0);

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

// Upper bound of the stored colour, well above the brightest reflection.
const float MAX_RADIANCE = 64.0;

// Strength of the reflection of a point h metres above the floor.
float height_fade(float h) {
    return mix(1.0, REFLECTION_FADE_MIN, clamp(h / REFLECTION_FADE_HEIGHT, 0.0, 1.0));
}

// What the floor point `hit_pos` (on the plane y = 0) shows to a camera at
// `camera_pos` looking along the unit vector `view_dir`: glass tint, the
// reflected tracer (`trail`) and ball (centre `ball_pos`, radius
// `ball_radius`, both in metres), and the specular
// gleam. Alpha 1.
vec4 shade_floor(vec3 camera_pos, vec3 hit_pos, vec3 view_dir, vec4 trail, vec3 ball_pos,
                 float ball_radius) {
    vec3 V = -view_dir; // toward camera
    vec3 N = vec3(0.0, 1.0, 0.0);

    // Schlick Fresnel approximation
    float NdotV = max(dot(N, V), 0.0);
    float fresnel = F0_GLASS + (1.0 - F0_GLASS) * pow(1.0 - NdotV, 5.0);

    vec3 refl_dir = reflect(view_dir, N);

    vec3 trail_reflection = trail.rgb * EMISSIVE_REFLECTANCE;
    // Height of the reflected light above the floor, averaged by light.
    float trail_height = trail.r > 0.0 ? trail.a / trail.r : 0.0;
    float trail_fade = height_fade(trail_height - hit_pos.y);

    // Analytical ball reflection: glow intensity from the closest approach
    // of the reflected ray, which starts at the floor, to the ball center
    // (the ray's start if the ball is behind it).
    vec3 to_center = ball_pos - hit_pos;
    float t_closest = max(dot(to_center, refl_dir), 0.0);
    float d = length(to_center - refl_dir * t_closest);
    float glow_radius = ball_radius * BALL_GLOW_MULT;
    float glow_t = clamp(d / glow_radius, 0.0, 1.0);
    float ball_glow = pow(1.0 - glow_t, 3.0);
    // Core brightens near the center
    float core_d = clamp(d / ball_radius, 0.0, 1.0);
    float core_glow = pow(1.0 - core_d, 4.0) * 0.6;
    // Suppress glow directly under the ball where the raster ball occludes
    // the floor — the additive composite would otherwise bleed through.
    // Use the viewing ray to check if this floor point is within the ball's
    // apparent disc from the camera's perspective.
    vec3 to_ball = ball_pos - hit_pos;
    float proj = dot(to_ball, V);
    float perp = length(to_ball - V * proj);
    float occlude = smoothstep(ball_radius * 1.5, ball_radius * 3.0, perp);
    vec3 ball_reflection = (MAGENTA * ball_glow * 3.0 + vec3(1.0) * core_glow * 3.0) * EMISSIVE_REFLECTANCE * occlude;
    float ball_fade = height_fade(ball_pos.y - hit_pos.y);

    // Glass base tint: dark navy, stronger at grazing angles. Far floor
    // fades into dark blue instead of pure black.
    float cam_dist = length(hit_pos.xz - camera_pos.xz);
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

    return vec4(glass_base + reflection + specular, 1.0);
}

// A reflection-image texel: colour in [0, MAX_RADIANCE] (emissive
// reflections are brighter than display white; the composite's target clips
// them), alpha in [0, 1]. NaN is stored as 0, so the composite never adds
// unbounded or NaN values.
vec4 reflection_texel(vec4 color) {
    return clamp(mix(color, vec4(0.0), isnan(color)), vec4(0.0),
                 vec4(vec3(MAX_RADIANCE), 1.0));
}
