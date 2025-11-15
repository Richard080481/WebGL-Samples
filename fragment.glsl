precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform vec2 iMouse;
uniform float DRAG_MULT;
uniform float WATER_DEPTH;
uniform float CAMERA_HEIGHT;
uniform int ITERATIONS_NORMAL;
uniform int ITERATIONS_RAYMARCH;
uniform vec3 shipPos;
uniform float shipRadius;

#define NormalizedMouse (iMouse / iResolution)
#define DEBUG_MODE (0) // 0 = normal render, 1 = wave height map, 2 = normal vectors

struct BoatHit {
    bool hit;
    vec3 color;      // Final boat color
    float t;         // Distance along ray
};

// SDF basic shapes
// Shpere SDF return distance from point to shpere
float sdfShpere(vec3 p, float r) {
    return length(p) - r;
}

// Box SDF return distance from point to box
// b = vec3(width/2, height/2, depth/2)
float sdfBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Cylinder SDF return distance from point to box
float sdfCylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Create a smooth transition at the contact point between the two SDFs.
float sm_union(float a, float b, float t) {
    float h = clamp(0.5 + 0.5 * (b - a) / t, 0.0, 1.0);
    return mix(b, a, h) - t * h * (1.0 - h);
}

// Boat Hull SDF
float sdfBoatHull(vec3 p) {
    float L = 1.2;                                         // Boat half-length
    float H = 0.3;                                         // Boat half-height
    float keel = p.y + 0.22;                               // Boat bottom plane
    float width = 0.25 + 0.3 * smoothstep(-1.2, 0.5, p.z); // Boat width
    float side = abs(p.x) - width;
    float body = max(keel, side);
    float frontBack = abs(p.z) - L;
    float heightLimit = abs(p.y) - H;
    return max(max(body, frontBack), heightLimit);
}

// Boat Body SDF
float sdfBoatBody(vec3 p) {
    return sdfBox(p - vec3(0.0, 0.3, 0.0), vec3(0.6, 0.3, 1.2));
}

// Boat Nose SDF
float sdfBoatNose(vec3 p) {
    vec3 q = p;
    q.z *= 1.5;
    float taper = p.z * 0.8;
    return sdfBox(q, vec3(0.35 - taper, 0.25, 0.9));
}

// Boat Cockpit SDF
float sdfBoatCockpit(vec3 p) {
    return sdfBox(p - vec3(0.0, 0.45, -0.3), vec3(0.3, 0.2, 0.4));
}

// Boat Windscreen SDF
float sdfBoatWindscreen(vec3 p) {
    vec3 q = p - vec3(0.0, 0.5, -0.2);
    q = q * mat3(1.0, 0.0, 0.0, 0.0, 0.8, 0.3, 0.0, -0.3, 0.8);
    return sdfBox(q, vec3(0.25, 0.1, 0.05));
}

// Combine all part into SpeedBoat
float sdfSpeedBoat(vec3 p) {
    float d = sdfBoatHull(p);
    d = sm_union(d, sdfBoatBody(p), 0.5);
    d = sm_union(d, sdfBoatNose(p), 0.3);
    d = sm_union(d, sdfBoatCockpit(p), 0.2);
    d = sm_union(d, sdfBoatWindscreen(p), 0.15);
    return d;
}

// Compute normal of the boat SDF using central difference
vec3 boatNormal(vec3 p) {
    float e = 0.001;
    return normalize(vec3(sdfSpeedBoat(p + vec3(e, 0, 0)) - sdfSpeedBoat(p - vec3(e, 0, 0)), sdfSpeedBoat(p + vec3(0, e, 0)) - sdfSpeedBoat(p - vec3(0, e, 0)), sdfSpeedBoat(p + vec3(0, 0, e)) - sdfSpeedBoat(p - vec3(0, 0, e))));
}

// star generation function
float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 generateStars(vec3 rayDir)
{
    if(rayDir.y < 0.05) return vec3(0.0);

    vec2 starCoord = rayDir.xz / max(rayDir.y, 0.01) * 15.0;
    vec2 gridId = floor(starCoord);
    vec2 gridUv = fract(starCoord);

    float starRandom = hash(gridId);
    float star = 0.0;

    if(starRandom > 0.88)
    {
        vec2 starPos = vec2(hash(gridId + vec2(1.0, 0.0)), hash(gridId + vec2(0.0, 1.0)));
        float dist = length(gridUv - starPos);
        star = 1.0 / (1.0 + dist * 40.0);
        star = pow(star, 1.5) * 12.0;
        star *= (0.5 + 0.5 * hash(gridId + vec2(2.0, 3.0)));
    }

    if(starRandom > 0.96)
    {
        star *= 4.0;
    }

    star *= 0.9 + 0.1 * sin(iTime * 2.0 + hash(gridId) * 100.0);
    float horizonFade = smoothstep(0.0, 0.2, rayDir.y);

    vec3 starColor = vec3(0.9, 0.95, 1.0);
    return starColor * star * horizonFade;
}

// Calculates wave value and its derivative,
// for the wave direction, position in space, wave frequency and time
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift)
{
    float x = dot(direction, position) * frequency + timeshift;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}

// Calculates waves by summing octaves of various waves with various parameters
float getwaves(vec2 position, int iterations)
{
    float wavePhaseShift = length(position) * 0.1; // this is to avoid every octave having exactly the same phase everywhere
    float iter           = 0.0;                    // this will help generating well distributed wave directions
    float frequency      = 1.0;                    // frequency of the wave, this will change every iteration
    float timeMultiplier = 2.0;                    // time multiplier for the wave, this will change every iteration
    float weight         = 1.0;                    // weight in final sum for the wave, this will change every iteration
    float sumOfValues    = 0.0;                    // will store final sum of values
    float sumOfWeights   = 0.0;                    // will store final sum of weights

    for(int i=0; i < 64; i++)
    {
        // generate some wave direction that looks kind of random
        vec2 p = vec2(sin(iter), cos(iter));

        // calculate wave data
        vec2 res = wavedx(position, p, frequency, iTime * timeMultiplier + wavePhaseShift);

        // shift position around according to wave drag and derivative of the wave
        position += p * res.y * weight * DRAG_MULT;

        // add the results to sums
        sumOfValues += res.x * weight;
        sumOfWeights += weight;

        // modify next octave
        weight = mix(weight, 0.0, 0.2);
        frequency *= 1.18;
        timeMultiplier *= 1.07;

        // add some kind of random value to make next wave look random too
        iter += 1232.399963;
    }
    // calculate and return
    return sumOfValues / sumOfWeights;
}
// Raymarches the ray from top water layer boundary to low water layer boundary
float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth)
{
    vec3 pos = start;
    vec3 dir = normalize(end - start);
    for(int i=0; i < 64; i++)
    {
        // the height is from 0 to -depth
        float height = getwaves(pos.xz, ITERATIONS_RAYMARCH) * depth - depth;
        // if the waves height almost nearly matches the ray height, assume its a hit and return the hit distance
        if(height + 0.01 > pos.y)
        {
            return distance(pos, camera);
        }
        // iterate forwards according to the height mismatch
        pos += dir * (pos.y - height);
    }
    // if hit was not registered, just assume hit the top layer,
    // this makes the raymarching faster and looks better at higher distances
    return distance(start, camera);
}

// Calculate normal at point by calculating the height at the pos and 2 additional points very close to pos
vec3 normal(vec2 pos, float e, float depth)
{
    vec2 ex = vec2(e, 0.0);
    float H = getwaves(pos, ITERATIONS_NORMAL) * depth;
    vec3 a = vec3(pos.x, H, pos.y);
    return normalize(cross(
        a - vec3(pos.x - e, getwaves(pos - ex, ITERATIONS_NORMAL) * depth, pos.y),
        a - vec3(pos.x, getwaves(pos + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)
    ));
}

// Helper function generating a rotation matrix around the axis by the angle
mat3 createRotationMatrixAxisAngle(vec3 axis, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    return mat3(
        oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
        oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
        oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c
    );
}

// Helper function that generates camera ray based on UV and mouse
vec3 getRay(vec2 fragCoord)
{
    vec2 uv = ((fragCoord / iResolution) * 2.0 - 1.0) * vec2(iResolution.x / iResolution.y, 1.0);
    // for fisheye, uncomment following line and comment the next one
    // vec3 proj = normalize(vec3(uv.x, uv.y, 1.0) + vec3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);
    vec3 proj = normalize(vec3(uv.x, uv.y, 1.5));
    return createRotationMatrixAxisAngle(vec3(0.0, -1.0, 0.0), 3.0 * ((NormalizedMouse.x + 0.5) * 2.0 - 1.0))
        * createRotationMatrixAxisAngle(vec3(1.0, 0.0, 0.0), 0.5 + 1.5 * (((NormalizedMouse.y == 0.0 ? 0.27 : NormalizedMouse.y) * 1.0) * 2.0 - 1.0))
        * proj;
}

// Ray-Plane intersection checker
float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal)
{
    return clamp(dot(point - origin, normal) / dot(direction, normal), -1.0, 9991999.0);
}

// Some very barebones but fast atmosphere approximation
vec3 extra_cheap_atmosphere(vec3 raydir, vec3 sundir)
{
    float special_trick = 1.0 / (raydir.y * 1.0 + 0.1);
    float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
    float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
    float sundt = pow(max(0.0, dot(sundir, raydir)), 8.0);
    float mymie = sundt * special_trick * 0.2;
    vec3 suncolor = mix(vec3(1.0), max(vec3(0.0), vec3(1.0) - vec3(5.5, 13.0, 22.4) / 22.4), special_trick2);
    vec3 bluesky = vec3(5.5, 13.0, 22.4) / 22.4 * suncolor;
    vec3 bluesky2 = max(vec3(0.0), bluesky - vec3(5.5, 13.0, 22.4) * 0.002 * (special_trick + -6.0 * sundir.y * sundir.y));
    bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
    return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0));
}

// sun motion, just fake it, going up and down vertically
vec3 getSunDirection()
{
    float cycleSpeed = 0.15;  // day/night switching speed
    float phase = fract(iTime * cycleSpeed);

    float height;

    if(phase < 0.35)  //daylight
    {
        height = (phase / 0.35) * 0.8;  // 0 → 0.8
    }
    else if(phase < 0.75)  // night (sun under the surface)
    {
        height = -0.5;
    }
    else  //sun get back to surface from under surface area
    {
        float transition = (phase - 0.75) / 0.25;
        height = mix(-0.5, 0.0, transition);
    }

    // fixed sun position
    return normalize(vec3(0.7, height, 0.6));
}

// Get atmosphere color for given direction
vec3 getAtmosphere(vec3 dir)
{
    return extra_cheap_atmosphere(dir, getSunDirection()) * 0.5;
}

// Get sun color for given direction
float getSun(vec3 dir)
{
    return pow(max(0.0, dot(dir, getSunDirection())), 720.0) * 210.0;
}

// Great tonemapping function from other shader: https://www.shadertoy.com/view/XsGfWV
vec3 aces_tonemap(vec3 color)
{
    mat3 m1 = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    mat3 m2 = mat3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    vec3 v = m1 * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

BoatHit raymarchBoat(vec3 origin, vec3 ray) {
    BoatHit result;
    result.hit = false;
    result.color = vec3(0.0);
    result.t = 0.0;
    float tBoat = 0.0;
    float maxDist = 200.0;
    float dBoat;
    float boatScale = 1.0;

    // Raymarch boat
    for(int i = 0; i < 100; i++) {
        vec3 p = origin + ray * tBoat;
        vec3 localP = (p - shipPos) / boatScale;
        dBoat = sdfSpeedBoat(localP) * boatScale;
        if(dBoat < 0.001) {
            vec3 Nboat = boatNormal(localP);
            vec3 light = getSunDirection();
            float diff = max(dot(Nboat, light), 0.2);

            vec3 boatColor = vec3(1.0);

            result.hit = true;
            result.color = boatColor * diff;
            result.t = tBoat;
            return result;
        }
        tBoat += dBoat;
        if(tBoat > maxDist)
            break;
    }
    return result;
}

void main()
{
    vec3 ray = getRay(gl_FragCoord.xy);

    if(ray.y >= 0.0)
    {
        // ===== Simplified Day/Night Cycle =====
        vec3 sunDir = getSunDirection();
        float sunHeight = sunDir.y;

        // Determine if it's day or night
        bool isNight = (sunHeight < 0.0);  // Sun below horizon = night

        // Smooth day/night transition factor
        float dayNightFactor = smoothstep(-0.1, 0.1, sunHeight);

        // Atmosphere rendering (bright during day, dark at night)
        vec3 atmosphere = getAtmosphere(ray) * dayNightFactor;

        // Sun rendering (only visible during daytime)
        float sun = getSun(ray) * dayNightFactor;

        // Starfield rendering (only visible at night)
        vec3 stars = vec3(0.0);
        if(isNight)
        {
            stars = generateStars(ray) * (1.0 - dayNightFactor);
        }

        vec3 C = atmosphere + sun + stars;
        gl_FragColor = vec4(aces_tonemap(C * 2.0), 1.0);
        return;
    }

    // === NORMAL MODE: Full Water Rendering ===
    // calculate normal at the hit position
    vec3 waterPlaneHigh = vec3(0.0, 0.0, 0.0);
    vec3 waterPlaneLow = vec3(0.0, -WATER_DEPTH, 0.0);
    vec3 origin = vec3(iTime * 0.2, CAMERA_HEIGHT, 1.0);

    float highPlaneHit = intersectPlane(origin, ray, waterPlaneHigh, vec3(0.0, 1.0, 0.0));
    float lowPlaneHit = intersectPlane(origin, ray, waterPlaneLow, vec3(0.0, 1.0, 0.0));
    vec3 highHitPos = origin + ray * highPlaneHit;
    vec3 lowHitPos = origin + ray * lowPlaneHit;

    float dist = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH);
    vec3 waterHitPos = origin + ray * dist;

    // === Boat Rendering ===
    // Call boat raymarch
    BoatHit boat = raymarchBoat(origin, ray);
    // Boat is closer than water
    if(boat.hit && boat.t < dist) {
        gl_FragColor = vec4(boat.color, 1.0);
        return;
    }

    if(DEBUG_MODE == 1)
    {
        // Sample wave height at the hit position (no raymarching, just direct query)
        float waveHeight = getwaves(waterHitPos.xz, ITERATIONS_NORMAL) * WATER_DEPTH;

        // Normalize to [0, 1]: assume wave height range is [-WATER_DEPTH, WATER_DEPTH]
        float heightNorm = (waveHeight + WATER_DEPTH) / (2.0 * WATER_DEPTH);
        heightNorm = clamp(heightNorm, 0.0, 1.0);

        // Simple grayscale
        gl_FragColor = vec4(vec3(heightNorm), 1.0);
        return;
    }

    // === DEBUG MODE: Normal Vectors ===
    if(DEBUG_MODE == 2)
    {
        // Calculate normal at the hit position
        vec3 N = normal(waterHitPos.xz, 0.01, WATER_DEPTH);

        // Visualize normals: map from [-1,1] to [0,1]
        vec3 normalVis = N * 0.5 + 0.5;
        gl_FragColor = vec4(normalVis, 1.0);
        return;
    }

    // === NORMAL MODE: Full Water Rendering ===
    // calculate normal at the hit position
    vec3 N = normal(waterHitPos.xz, 0.01, WATER_DEPTH);
    // smooth the normal with distance to avoid disturbing high frequency noise
    N = mix(N, vec3(0.0, 1.0, 0.0), 0.8 * min(1.0, sqrt(dist * 0.01) * 1.1));

    // calculate fresnel coefficient
    float fresnel = (0.04 + (1.0 - 0.04) * (pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));

    // reflect the ray and make sure it bounces up
    vec3 R = normalize(reflect(ray, N));
    R.y = abs(R.y);

    // calculate the reflection and approximate subsurface scattering
    vec3 reflection = getAtmosphere(R) + getSun(R);
    vec3 scattering = vec3(0.0293, 0.0698, 0.1717) * 0.1 * (0.2 + (waterHitPos.y + WATER_DEPTH) / WATER_DEPTH);

    vec3 C = fresnel * reflection + scattering;
    gl_FragColor = vec4(aces_tonemap(C * 2.0), 1.0);
}