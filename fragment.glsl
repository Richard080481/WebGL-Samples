precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform vec2 iMouse;
uniform float DRAG_MULT;
uniform float WATER_DEPTH;
uniform float CAMERA_HEIGHT;
uniform int ITERATIONS_NORMAL;
uniform int ITERATIONS_RAYMARCH;
uniform float SUN_ROTATION_SPEED;
uniform vec3 shipPos;
uniform float shipRadius;
uniform float shipRotation;
uniform float STAR_DENSITY;
uniform vec3 camPos;
uniform vec3 camTarget;

#define NormalizedMouse (iMouse / iResolution)
#define DEBUG_MODE (0) // 0 = normal render, 1 = wave height map, 2 = normal vectors

struct BoatHit {
    bool hit;
    vec3 color;      // Final boat color
    float t;         // Distance along ray
};

// SDF basic shapes
// Box SDF return distance from point to box
// b = vec3(width/2, height/2, depth/2)
float sdfBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Create a smooth transition at the contact point between the two SDFs.
vec2 sm_union(vec2 a, vec2 b, float t) {
    float h = clamp(0.5 + 0.5 * (b.x - a.x) / t, 0.0, 1.0);
    float d = mix(b.x, a.x, h) - t * h * (1.0 - h);
    float material = (a.x < b.x) ? a.y : b.y;
    return vec2(d, material);
}

// Boat Hull SDF
vec2 sdfBoatHull(vec3 p) {
    float L = 1.2;                                         // Boat half-length
    float H = 0.3;                                         // Boat half-height
    float keel = p.y + 0.22;                               // Boat bottom plane
    float width = 0.25 + 0.3 * smoothstep(0.5, -1.2, p.z); // Boat width
    float side = abs(p.x) - width;
    float body = max(keel, side);
    float frontBack = abs(p.z) - L;
    float heightLimit = abs(p.y) - H;
    float d = max(max(body, frontBack), heightLimit);
    return vec2(d, 1.0);
}

float sdfRoundedBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) - r;
}

// Boat Body SDF
vec2 sdfBoatBody(vec3 p) {
    float d = sdfRoundedBox(p - vec3(0.0, 0.0, 0.0), vec3(0.55, 0.26, 1.2), 0.06);
    return vec2(d, 1.0);
}

// Boat Nose SDF
vec2 sdfBoatNose(vec3 p) {
    vec3 q = p - vec3(0.0, 0.0, 1.2);
    float L = 0.7;
    float t = clamp(q.z / L, 0.0, 1.0);
    float taper = t * t;
    float halfWidth  = mix(0.55, 0.0, taper);
    float halfHeight = mix(0.26, 0.0, taper);
    vec2 d = abs(q.xy) - vec2(halfWidth, halfHeight);
    float crossSection = length(max(d, 0.0)) - 0.06;
    float zBound = max(-q.z, q.z - L);
    return vec2(max(crossSection, zBound), 1.0);
}

// Boat Cockpit SDF
vec2 sdfBoatCockpit(vec3 p) {
    return vec2(sdfBox(p - vec3(0.0, 0.4, -0.3), vec3(0.4, 0.2, 0.4)), 3.0);
}

// Boat Windscreen SDF
vec2 sdfBoatWindscreen(vec3 p) {
    vec3 q = p - vec3(0.0, 0.4, 0.6);
    q = q * mat3(1.0, 0.0, 0.0, 0.0, -0.8, 0.7, 0.0, 0.7, 0.8);
    return vec2(sdfBox(q, vec3(0.4, 0.2, 0.01)), 2.0);
}

// Combine all part into SpeedBoat
vec2 sdfSpeedBoat(vec3 p) {
    vec2 d = sdfBoatHull(p);
    d = sm_union(d, sdfBoatBody(p), 0.3);
    d = vec2(min(d.x, sdfBoatNose(p).x), 1.0);
    d = sm_union(d, sdfBoatCockpit(p), 0.2);
    d = sm_union(d, sdfBoatWindscreen(p), 0.15);
    return d;
}

// Compute normal of the boat SDF using central difference
vec3 boatNormal(vec3 p) {
    float e = 0.001;
    return normalize(vec3(sdfSpeedBoat(p + vec3(e, 0, 0)).x - sdfSpeedBoat(p - vec3(e, 0, 0)).x, sdfSpeedBoat(p + vec3(0, e, 0)).x - sdfSpeedBoat(p - vec3(0, e, 0)).x, sdfSpeedBoat(p + vec3(0, 0, e)).x - sdfSpeedBoat(p - vec3(0, 0, e)).x));
}

//simple noise functions

// Simple 2D noise (Perlin-like)
float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // Smoothstep
    
    float a = fract(sin(dot(i, vec2(127.1, 311.7))) * 43758.5453);
    float b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7))) * 43758.5453);
    float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
    float d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

//star sky generation functions
float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

//fbm noise for nebula
float fbm(vec2 p)
{
    float sum  = 0.0;
    float amp  = 0.5;  // Initial amplitude of the first octave
    float freq = 1.0;  // Initial frequency of the first octave
    
    // 4–5 octaves are enough to get detail without being too expensive
    for(int i = 0; i < 5; i++)
    {
        sum += amp * noise(p * freq);
        freq *= 2.03;  // Increase frequency every octave
        amp  *= 0.55;  // Decrease amplitude every octave
    }
    return sum;
}

// Procedural nebula in the sky (uses ray direction + sun height)
vec3 generateNebula(vec3 rayDir, float sunHeight)
{
    // Only visible during night / near-night conditions
    float nightFactor = smoothstep(-0.1, -0.35, sunHeight); // Stronger when sun goes below horizon
    if(nightFactor <= 0.0) return vec3(0.0);
    if(rayDir.y < 0.02) return vec3(0.0);                   // Avoid foggy band right at the horizon

    // Project ray direction into a 2D "sky coordinate" (similar to stars)
    vec2 skyCoord = vec2(
        atan(rayDir.z, rayDir.x),               // Azimuth: [-PI, PI]
        asin(clamp(rayDir.y, -1.0, 1.0))        // Elevation: [-PI/2, PI/2]
    );

    // Controls the approximate position of the nebula on the sky dome
    vec2 nebulaCenter = vec2(0.6, 0.15);        // Change this to move the main nebula lobe
    vec2 uv = skyCoord * 0.7 + nebulaCenter;    // Smaller scale = larger, smoother clouds

    // fbm noise for soft cloudy structure
    float n = fbm(uv * 3.0);

    // Turn noise into a soft cloud mask using threshold + smoothing
    float density = smoothstep(0.55, 0.80, n);

    // Radial falloff from the nebula center to give a blob-like shape
    float r = length(uv - nebulaCenter);
    float radial = exp(-r * 1.8);              // Larger r → darker / more faded

    float finalMask = density * radial * nightFactor;

    // Very small contributions can be skipped for performance
    if(finalMask <= 0.001) return vec3(0.0);

    // Two-color gradient nebula (feel free to tweak colors)
    vec3 colorA = vec3(0.40, 0.10, 0.65);      // Purple tint
    vec3 colorB = vec3(0.10, 0.60, 0.85);      // Cyan-blue tint
    float t = clamp(n * 1.3, 0.0, 1.0);
    vec3 nebulaColor = mix(colorA, colorB, t);

    // Add an extra brightness variation to make the core brighter and edges darker
    float brightness = 0.6 + 0.6 * fbm(uv * 6.0 + 13.0);

    return nebulaColor * finalMask * brightness * 0.7;
}

vec3 generateStars(vec3 rayDir)
{
    if(rayDir.y < 0.02) return vec3(0.0);
    
    // Convert ray to 2D sky coordinates
    vec2 skyCoord = vec2(
        atan(rayDir.z, rayDir.x) * 2.0,
        asin(clamp(rayDir.y, -1.0, 1.0)) * 2.0
    );
    
    // Create grid
    vec2 gridCoord = skyCoord * 50.0;
    vec2 gridId = floor(gridCoord);
    vec2 gridUv = fract(gridCoord);  // Position within cell [0, 1]
    
    // Random value for this grid cell
    float random = hash(gridId);
    // Map STAR_DENSITY (0.0– ~0.1) → probability of a star per cell
    float density = clamp(STAR_DENSITY, 0.0, 0.2);   // safety clamp

    if(random > 1.0 - density)
    {
        // Star position within the cell (also random)
        vec2 starPos = vec2(
            hash(gridId + vec2(1.0, 0.0)),
            hash(gridId + vec2(0.0, 1.0))
        );
        
        // Distance from current position to star center
        float dist = length(gridUv - starPos);
        
        // Create small point star
        float star = smoothstep(0.08, 0.0, dist);  // Sharp falloff
        
        return vec3(star);
    }
    
    return vec3(0.0);
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
    float t = iTime * 0.5 * SUN_ROTATION_SPEED;
    float r = 2.0;
    float cx = 0.0;
    float cy = 0.0;

    float x = cx + r * cos(t);
    float y = cy + r * sin(t);
    float z = 3.0;

    // Camera is at (0,0,0) looking towards +Z
    // return normalize(vec3(0, 0, 1));
    return normalize(vec3(x, y, z));
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

vec3 shadeMaterial(float id, vec3 N, vec3 V, vec3 L, vec3 R) {
    float NV = max(dot(N, -V), 0.0);
    // Blinn-Phong
    float specPower = 80.0;
    float spec = pow(max(dot(N, normalize(L + V)), 0.0), specPower);
    // env reflection
    vec3 env = getAtmosphere(R) + getSun(R);
    // material colors
    if (id == 1.0) {
        return vec3(0.7, 0.05, 0.05) * max(dot(N, L), 0.0) + spec*0.5 + env*0.15;
    }
    else if (id == 2.0) {
        float F0 = 0.04;
        float fresnel = F0 + (1.0 - F0) * pow(1.0 - NV, 5.0);
        vec3 refractDir = normalize(refract(V, N, 1.0 / 1.5));
        vec3 refracted = getAtmosphere(refractDir);
        vec3 tint = vec3(0.35, 0.45, 0.6);
        refracted *= tint * 0.7;
        vec3 reflected = env * 1.3 + spec * 2.0;
        return mix(refracted, reflected, fresnel);
    }
    else if (id == 3.0) {
        return vec3(0.75, 0.75, 0.75) * max(dot(N, L), 0.0) + spec*0.1 + env*0.1;
    }   
    return vec3(1.0);
}

BoatHit raymarchBoat(vec3 origin, vec3 ray) {
    BoatHit result;
    result.hit = false;
    result.color = vec3(0.0);
    result.t = 0.0;
    float tBoat = 0.0;
    float maxDist = 200.0;
    float dBoat;
    float matID = 1.0;
    float boatScale = 1.0;
    float angle = shipRotation;
    mat3 rot = mat3(
        cos(angle), 0.0, -sin(angle),
        0.0, 1.0, 0.0,
        sin(angle), 0.0, cos(angle)
    );

    // Raymarch boat
    for(int i = 0; i < 100; i++) {
        vec3 p = origin + ray * tBoat;
        vec3 localP = rot * (p - shipPos) / boatScale;
        vec2 speedBoat = sdfSpeedBoat(localP) * boatScale;
        dBoat = speedBoat.x;
        matID = speedBoat.y;
        if(dBoat < 0.001) {
            vec3 Nboat = boatNormal(localP);
            vec3 L = normalize(getSunDirection());
            vec3 V = normalize(-ray);        // view direction
            vec3 H = normalize(L + V);       // half vector
            vec3 R = reflect(ray, Nboat);
            vec3 color = shadeMaterial(matID, Nboat, V, L, R);
            result.hit = true;
            result.color = color;
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
        vec3 sunDir = getSunDirection();
        float sunHeight = sunDir.y;

        //ranges to match actual sun motion
        vec3 daySky = getAtmosphere(ray);
        vec3 deepBlueSky = vec3(0.01, 0.03, 0.15);
        vec3 nightSky = vec3(0.0);
        
        // fixed ranges (sun can only reach ~-0.55 minimum)
        float dayFactor = smoothstep(-0.4, 0.3, sunHeight);
        float deepBlueFactor = smoothstep(-0.55, -0.2, sunHeight);
        float pureBlackFactor = smoothstep(-0.25, -0.55, sunHeight);
        
        vec3 baseColor = mix(deepBlueSky, daySky, dayFactor);
        baseColor = mix(nightSky, baseColor, 1.0 - pureBlackFactor);

        //twilight Colors
        vec3 twilightColor = vec3(0.0);
        if(sunHeight < 0.4 && sunHeight > -0.5)
        {
            float horizonGlow = exp(-pow(ray.y * 2.5, 2.0));
            
            // Golden phase
            vec3 golden = vec3(1.0, 0.8, 0.4);
            float goldenIntensity = smoothstep(-0.15, 0.4, sunHeight) * 
                                    smoothstep(0.4, -0.05, sunHeight);
            
            // Orange/Red phase
            vec3 orangeRed = vec3(1.0, 0.4, 0.2);
            float orangeIntensity = smoothstep(-0.3, 0.15, sunHeight) * 
                                    smoothstep(0.3, -0.2, sunHeight);
            
            // Purple phase
            vec3 purple = vec3(0.4, 0.15, 0.6);
            float purpleIntensity = smoothstep(-0.5, -0.1, sunHeight) * 
                                    smoothstep(0.1, -0.4, sunHeight);
            
            twilightColor = (golden * goldenIntensity * 0.5 + 
                            orangeRed * orangeIntensity * 0.4 + 
                            purple * purpleIntensity * 0.3) * horizonGlow;
        }

        //Sun 
        float sun = getSun(ray) * max(0.0, smoothstep(-0.25, 0.3, sunHeight));

        //tars 
        vec3 stars = vec3(0.0);
        if(sunHeight < -0.2)
        {
            float starFactor = smoothstep(-0.2, -0.5, sunHeight);
            stars = generateStars(ray) * starFactor;
        }
        //Nebula
        vec3 nebula = generateNebula(ray, sunHeight);

        vec3 C = baseColor + twilightColor + sun + stars + nebula;
        gl_FragColor = vec4(aces_tonemap(C * 2.0), 1.0);
        return;
    }
    // === NORMAL MODE: Full Water Rendering ===
    // calculate normal at the hit position
    vec3 waterPlaneHigh = vec3(0.0, 0.0, 0.0);
    vec3 waterPlaneLow = vec3(0.0, -WATER_DEPTH, 0.0);
    vec3 origin = camPos;

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