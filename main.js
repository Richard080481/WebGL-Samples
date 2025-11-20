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

const camera = {
    eye:    [0, 1, -3],
    center: [0, 0, 0],
    up:     [0, 1, 0],
};

let boatProgram = null;
let boatVAO = null;
let boatIndexCount = 0;
let boatModel = mat4Identity();
let boatView = mat4Identity();
let boatProj = mat4Identity();
let boat_uModel, boat_uView, boat_uProj, boat_uColor, boat_uLightDir;

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

function mat4Identity() {
    return new Float32Array([
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    ]);
}

function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
            out[r*4 + c] =
                a[r*4 + 0]*b[0*4 + c] +
                a[r*4 + 1]*b[1*4 + c] +
                a[r*4 + 2]*b[2*4 + c] +
                a[r*4 + 3]*b[3*4 + c];
        }
    }
    return out;
}

function mat4Perspective(fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;

    out[8]  = 0;
    out[9]  = 0;
    out[10] = (far + near) * nf;
    out[11] = -1;

    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) * nf;
    out[15] = 0;
    return out;
}

function mat4LookAt(eye, center, up) {
    const [ex, ey, ez] = eye;
    const [cx, cy, cz] = center;
    const [ux, uy, uz] = up;

    let zx = ex - cx;
    let zy = ey - cy;
    let zz = ez - cz;
    let zl = Math.hypot(zx, zy, zz);
    zx /= zl; zy /= zl; zz /= zl;

    let xx = uy*zz - uz*zy;
    let xy = uz*zx - ux*zz;
    let xz = ux*zy - uy*zx;
    let xl = Math.hypot(xx, xy, xz);
    xx /= xl; xy /= xl; xz /= xl;

    let yx = zy*xz - zz*xy;
    let yy = zz*xx - zx*xz;
    let yz = zx*xy - zy*xx;

    const out = new Float32Array(16);
    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx*ex + xy*ey + xz*ez);
    out[13] = -(yx*ex + yy*ey + yz*ez);
    out[14] = -(zx*ex + zy*ey + zz*ez);
    out[15] = 1;
    return out;
}

function mat4RotateX(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([
        1, 0, 0, 0,
        0, c,-s, 0,
        0, s, c, 0,
        0, 0, 0, 1
    ]);
}

function mat4RotateY(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([
         c, 0, s, 0,
         0, 1, 0, 0,
        -s, 0, c, 0,
         0, 0, 0, 1
    ]);
}

function parseOBJ(text) {
    const positions = [];
    const normals = [];
    const finalPositions = [];
    const finalNormals = [];
    const indices = [];

    const lines = text.split('\n');
    const vertexMap = new Map();
    let indexCounter = 0;

    function getVertexIndex(vStr) {
        // 支持 v//n 或 v/vt/n 或 v
        let [vIdxStr, vtIdxStr, nIdxStr] = vStr.split('/');
        const vIdx = parseInt(vIdxStr, 10);
        const nIdx = nIdxStr ? parseInt(nIdxStr, 10) : null;

        const key = vStr;
        if (vertexMap.has(key)) {
            return vertexMap.get(key);
        }
        const px = positions[(vIdx - 1)*3 + 0];
        const py = positions[(vIdx - 1)*3 + 1];
        const pz = positions[(vIdx - 1)*3 + 2];

        let nx = 0, ny = 0, nz = 1;
        if (nIdx != null && normals.length >= nIdx*3) {
            nx = normals[(nIdx - 1)*3 + 0];
            ny = normals[(nIdx - 1)*3 + 1];
            nz = normals[(nIdx - 1)*3 + 2];
        }

        finalPositions.push(px, py, pz);
        finalNormals.push(nx, ny, nz);

        vertexMap.set(key, indexCounter);
        return indexCounter++;
    }

    for (let line of lines) {
        line = line.trim();
        if (line.startsWith('#') || line === '') continue;
        const parts = line.split(/\s+/);
        if (parts[0] === 'v') {
            positions.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
        } else if (parts[0] === 'vn') {
            normals.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
        } else if (parts[0] === 'f') {
            if (parts.length < 4) continue;
            const i0 = getVertexIndex(parts[1]);
            const i1 = getVertexIndex(parts[2]);
            const i2 = getVertexIndex(parts[3]);
            indices.push(i0, i1, i2);

            if (parts.length === 5) {
                const i3 = getVertexIndex(parts[4]);
                indices.push(i0, i2, i3);
            }
        }
    }

    return {
        positions: new Float32Array(finalPositions),
        normals: new Float32Array(finalNormals),
        indices: new Uint16Array(indices)
    };
}

// Load shader files
// If the page is opened via file://, fetch will fail in most browsers.
if (location.protocol === 'file:') {
    console.warn('Warning: page loaded using file://. Fetching shader files usually fails when opened directly from the filesystem. Run a local server (e.g. `python -m http.server`) and open via http://localhost:8000');
}

Promise.all([
    fetch('vertex.glsl').then(r => { if (!r.ok) throw new Error('Failed to fetch vertex.glsl: ' + r.status + ' ' + r.statusText); return r.text(); }),
    fetch('fragment.glsl').then(r => { if (!r.ok) throw new Error('Failed to fetch fragment.glsl: ' + r.status + ' ' + r.statusText); return r.text(); }),
    fetch('boatVertex.glsl').then(r => { if (!r.ok) throw new Error('Failed to fetch boatVertex.glsl: ' + r.status + ' ' + r.statusText); return r.text(); }),
    fetch('boatFragment.glsl').then(r => { if (!r.ok) throw new Error('Failed to fetch boatFragment.glsl: ' + r.status + ' ' + r.statusText); return r.text(); }),
    fetch('boat.obj').then(r => { if (!r.ok) throw new Error('Failed to fetch boat.obj: ' + r.status + ' ' + r.statusText); return r.text(); })
]).then(([vertexSrc, fragmentSrc, boatVertSrc, boatFragSrc, boatObjText]) => {
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

    // boat shader
    const boatVertShader = compileShader(gl.VERTEX_SHADER, boatVertSrc);
    const boatFragShader = compileShader(gl.FRAGMENT_SHADER, boatFragSrc);

    boatProgram = gl.createProgram();
    gl.attachShader(boatProgram, boatVertShader);
    gl.attachShader(boatProgram, boatFragShader);
    gl.linkProgram(boatProgram);

    // parse boat model
    const boatMesh = parseOBJ(boatObjText);

    // create boat VAO/VBO/IBO
    boatVAO = gl.createVertexArray();
    gl.bindVertexArray(boatVAO);

    const boatVBO = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, boatVBO);
    const boatVertexData = new Float32Array(boatMesh.positions.length + boatMesh.normals.length);
    for (let i = 0, j = 0; i < boatMesh.positions.length/3; i++) {
        boatVertexData[j++] = boatMesh.positions[i*3 + 0];
        boatVertexData[j++] = boatMesh.positions[i*3 + 1];
        boatVertexData[j++] = boatMesh.positions[i*3 + 2];
        boatVertexData[j++] = boatMesh.normals[i*3 + 0];
        boatVertexData[j++] = boatMesh.normals[i*3 + 1];
        boatVertexData[j++] = boatMesh.normals[i*3 + 2];
    }
    gl.bufferData(gl.ARRAY_BUFFER, boatVertexData, gl.STATIC_DRAW);

    const boatEBO = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boatEBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, boatMesh.indices, gl.STATIC_DRAW);
    boatIndexCount = boatMesh.indices.length;

    // setup attributes
    const boatPosLoc = gl.getAttribLocation(boatProgram, 'position');
    const boatNormalLoc = gl.getAttribLocation(boatProgram, 'normal');
    const stride = 6 * 4; // 6 floats * 4 bytes
    gl.enableVertexAttribArray(boatPosLoc);
    gl.vertexAttribPointer(boatPosLoc, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(boatNormalLoc);
    gl.vertexAttribPointer(boatNormalLoc, 3, gl.FLOAT, false, stride, 3*4);

    gl.bindVertexArray(null);

    // get uniform locations
    gl.useProgram(boatProgram);
    boat_uModel = gl.getUniformLocation(boatProgram, 'uModel');
    boat_uView  = gl.getUniformLocation(boatProgram, 'uView');
    boat_uProj  = gl.getUniformLocation(boatProgram, 'uProj');
    boat_uColor = gl.getUniformLocation(boatProgram, 'uColor');
    boat_uLightDir = gl.getUniformLocation(boatProgram, 'uLightDir');

    // simple camera setup
    boatView = mat4LookAt(camera.eye, camera.center, camera.up);
    boatProj = mat4Perspective(Math.PI/4, canvas.width/canvas.height, 0.1, 500.0);
    boatModel = mat4Identity();

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
        camera.eye[1] = uniforms.camHeight;
        boatView = mat4LookAt(camera.eye, camera.center, camera.up);

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(program); // SDF shader

        gl.uniform3f(gl.getUniformLocation(program, "camPos"),camera.eye[0], camera.eye[1], camera.eye[2]);
        gl.uniform3f(gl.getUniformLocation(program, "camTarget"),camera.center[0], camera.center[1], camera.center[2]);
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
        const shipRadius = 20.0;
        const speedshipRadius = 10.0;
        const shipX = Math.cos(time * shipSpeed) * shipRadius;
        const shipZ = Math.sin(time * shipSpeed) * shipRadius;
        const shipRot = time * shipSpeed;
        gl.uniform3f(gl.getUniformLocation(program, 'shipPos'), shipX, -0.5, shipZ);
        gl.uniform1f(gl.getUniformLocation(program, 'shipRadius'), 2.0);
        gl.uniform1f(gl.getUniformLocation(program, 'shipRotation'), shipRot);

        gl.bindVertexArray(vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindVertexArray(null);

        if (boatProgram && boatVAO) {
            gl.enable(gl.DEPTH_TEST);
            gl.useProgram(boatProgram);
            gl.bindVertexArray(boatVAO);

            const s = 1.0 / 1000.0; 
            const scale = new Float32Array([
                s,0,0,0,
                0,s,0,0,
                0,0,s,0,
                0,0,0,1
            ]);
            const rotX = mat4RotateX(Math.PI / 2);     // 90°
            const rotY = mat4RotateY(Math.PI / 2);     // 90°
            let model = mat4Multiply(rotX, rotY);
            model = mat4Multiply(scale, model);

            model[12] = 0.0;  // x
            model[13] = 0.0;  // y
            model[14] = 180.0 * s;  // z
            boatModel = model;

            gl.uniformMatrix4fv(boat_uModel, false, boatModel);
            gl.uniformMatrix4fv(boat_uView,  false, boatView);
            gl.uniformMatrix4fv(boat_uProj,  false, boatProj);

            gl.uniform3f(boat_uColor, 1.0, 1.0, 1.0);

            gl.uniform3f(boat_uLightDir, 0.0, 1.0, 0.0);

            gl.drawElements(gl.TRIANGLES, boatIndexCount, gl.UNSIGNED_SHORT, 0);
            gl.bindVertexArray(null);
        }

        requestAnimationFrame(animate);
    }

    animate();
}).catch(err => {
    console.error('Failed to load shaders or models:', err);
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