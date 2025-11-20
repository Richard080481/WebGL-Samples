precision highp float;

varying vec3 vNormal;
varying vec3 vWorldPos;

uniform vec3 uColor;   // base color of the boat
uniform vec3 uLightDir; // light direction in world space

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(-vWorldPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 base = uColor;
    vec3 color = base * (0.15 + 0.85 * diff) + vec3(1.0) * spec * 0.7;

    gl_FragColor = vec4(color, 1.0);
}