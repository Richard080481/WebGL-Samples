const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl");

if (!gl) {
  alert("WebGL not supported!");
}

// Vertex shader
const vsSource = `
attribute vec2 aPosition;
attribute vec3 aColor;
varying vec3 vColor;
uniform float uAngle;

void main(void) {
  float c = cos(uAngle);
  float s = sin(uAngle);
  mat2 rotation = mat2(c, -s, s, c);
  gl_Position = vec4(rotation * aPosition, 0.0, 1.0);
  vColor = aColor;
}
`;

// Fragment shader
const fsSource = `
precision mediump float;
varying vec3 vColor;
void main(void) {
  gl_FragColor = vec4(vColor, 1.0);
}
`;

// Helper to compile shader
function compileShader(source, type) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
  }
  return shader;
}

const vertexShader = compileShader(vsSource, gl.VERTEX_SHADER);
const fragmentShader = compileShader(fsSource, gl.FRAGMENT_SHADER);

const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);
gl.useProgram(program);

// Triangle vertices and colors
const vertices = new Float32Array([
  0.0,  0.8,  1.0, 0.0, 0.0, // top (red)
 -0.8, -0.8,  0.0, 1.0, 0.0, // left (green)
  0.8, -0.8,  0.0, 0.0, 1.0  // right (blue)
]);

const buffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

const FSIZE = vertices.BYTES_PER_ELEMENT;

const aPosition = gl.getAttribLocation(program, "aPosition");
gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, FSIZE * 5, 0);
gl.enableVertexAttribArray(aPosition);

const aColor = gl.getAttribLocation(program, "aColor");
gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, FSIZE * 5, FSIZE * 2);
gl.enableVertexAttribArray(aColor);

const uAngle = gl.getUniformLocation(program, "uAngle");

let angle = 0;
function render() {
  angle += 0.01;
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.uniform1f(uAngle, angle);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  requestAnimationFrame(render);
}

render();
