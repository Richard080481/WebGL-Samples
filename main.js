const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl2');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
gl.viewport(0, 0, canvas.width, canvas.height);

// Setup uniforms
const uniforms = {
    dragMult: 0.38,
    waterDepth: 1.0,
    camHeight: 1.5,
    rayIter: 12,
    normIter: 36,
    sunRotationSpeed: 1.0,
    starDensity: 0.03
};

// Update display values
document.getElementById('dragMult').addEventListener('input', e => {
    uniforms.dragMult = parseFloat(e.target.value);
    document.getElementById('dragVal').textContent = uniforms.dragMult.toFixed(2);
});
document.getElementById('waterDepth').addEventListener('input', e => {
    uniforms.waterDepth = parseFloat(e.target.value);
    document.getElementById('depthVal').textContent = uniforms.waterDepth.toFixed(1);
});
document.getElementById('camHeight').addEventListener('input', e => {
    uniforms.camHeight = parseFloat(e.target.value);
    document.getElementById('heightVal').textContent = uniforms.camHeight.toFixed(1);
});
document.getElementById('rayIter').addEventListener('input', e => {
    uniforms.rayIter = parseInt(e.target.value);
    document.getElementById('rayVal').textContent = uniforms.rayIter;
});
document.getElementById('normIter').addEventListener('input', e => {
    uniforms.normIter = parseInt(e.target.value);
    document.getElementById('normVal').textContent = uniforms.normIter;
});
document.getElementById('sunRotationSpeed').addEventListener('input', e => {
    uniforms.sunRotationSpeed = parseFloat(e.target.value);
    document.getElementById('sunSpeedVal').textContent = uniforms.sunRotationSpeed.toFixed(1);
});
document.getElementById('boatRotationSpeed').addEventListener('input', e => {
    uniforms.boatRotationSpeed = parseFloat(e.target.value);
    document.getElementById('boatSpeedVal').textContent = uniforms.boatRotationSpeed.toFixed(1);
});
document.getElementById('starDensity').addEventListener('input', e => {
    uniforms.starDensity = parseFloat(e.target.value);
    document.getElementById('starDensityVal').textContent = uniforms.starDensity.toFixed(3);
});

function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(shader));
    }
    return shader;
}

// Load shader files
// If the page is opened via file://, fetch will fail in most browsers.
if (location.protocol === 'file:') {
    console.warn('Warning: page loaded using file://. Fetching shader files usually fails when opened directly from the filesystem. Run a local server (e.g. `python -m http.server`) and open via http://localhost:8000');
}

Promise.all([
    fetch('vertex.glsl').then(r => { if (!r.ok) throw new Error('Failed to fetch vertex.glsl: ' + r.status + ' ' + r.statusText); return r.text(); }),
    fetch('fragment.glsl').then(r => { if (!r.ok) throw new Error('Failed to fetch fragment.glsl: ' + r.status + ' ' + r.statusText); return r.text(); })
]).then(([vertexSrc, fragmentSrc]) => {
    const vertShader = compileShader(gl.VERTEX_SHADER, vertexSrc);
    const fragShader = compileShader(gl.FRAGMENT_SHADER, fragmentSrc);

    const program = gl.createProgram();
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vertices = [-1, -1, 1, -1, -1, 1, 1, 1];
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 8, 0);

    let mouseX = 0, mouseY = 0;
    let lastMouseX = 0, lastMouseY = 0;
    let isMouseDown = false;

    window.addEventListener('mousedown', () => {
        isMouseDown = true;
        lastMouseX = window.innerWidth / 2;
        lastMouseY = window.innerHeight / 2;
    });

    window.addEventListener('mouseup', () => {
        isMouseDown = false;
    });

    window.addEventListener('mousemove', e => {
        if (isMouseDown) {
            mouseX = e.clientX;
            mouseY = e.clientY;
        }
    });

    const startTime = Date.now();
    function animate() {
        const time = (Date.now() - startTime) * 0.001;

        gl.uniform2f(gl.getUniformLocation(program, 'iResolution'), canvas.width, canvas.height);
        gl.uniform1f(gl.getUniformLocation(program, 'iTime'), time);
        gl.uniform2f(gl.getUniformLocation(program, 'iMouse'), mouseX, mouseY);
        gl.uniform1f(gl.getUniformLocation(program, 'DRAG_MULT'), uniforms.dragMult);
        gl.uniform1f(gl.getUniformLocation(program, 'WATER_DEPTH'), uniforms.waterDepth);
        gl.uniform1f(gl.getUniformLocation(program, 'CAMERA_HEIGHT'), uniforms.camHeight);
        gl.uniform1i(gl.getUniformLocation(program, 'ITERATIONS_RAYMARCH'), uniforms.rayIter);
        gl.uniform1i(gl.getUniformLocation(program, 'ITERATIONS_NORMAL'), uniforms.normIter);
        gl.uniform1f(gl.getUniformLocation(program, 'SUN_ROTATION_SPEED'), uniforms.sunRotationSpeed);
        gl.uniform1f(gl.getUniformLocation(program, 'STAR_DENSITY'), uniforms.starDensity);


        // Calculate ship position moving in circle
        const shipSpeed = uniforms.boatRotationSpeed / 10.0 || 0.5;
        const shipRadius = 8;
        const shipX = Math.cos(time * shipSpeed) * shipRadius;
        const shipZ = Math.sin(time * shipSpeed) * shipRadius;
        const shipRot = time * shipSpeed;
        gl.uniform3f(gl.getUniformLocation(program, 'shipPos'), shipX, -0.5, shipZ);
        gl.uniform1f(gl.getUniformLocation(program, 'shipRadius'), 2.0);
        gl.uniform1f(gl.getUniformLocation(program, 'shipRotation'), shipRot);

        gl.bindVertexArray(vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        requestAnimationFrame(animate);
    }

    animate();
}).catch(err => {
    console.error('Failed to load shaders:', err);
    // Add an extra hint for the common file:// case
    if (location.protocol === 'file:') {
        console.error('Hint: You are opening the page directly from the filesystem (file://). Start a local HTTP server and open via http:// to allow fetch() to load external shader files.');
    }
});

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
});