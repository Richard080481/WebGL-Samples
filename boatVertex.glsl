precision highp float;

attribute vec3 position;
attribute vec3 normal;

// obejct space -> world space
uniform mat4 uModel;
// world space -> view space
uniform mat4 uView;
// view space -> image space
uniform mat4 uProj;

varying vec3 vNormal;
varying vec3 vWorldPos;

void main() {
    vec4 worldPos = uModel * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;

    vNormal = mat3(uModel) * normal;

    gl_Position = uProj * uView * worldPos;
}