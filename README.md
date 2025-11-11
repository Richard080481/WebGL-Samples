# Wave-Simulator

## Requirements

- Modern browser with WebGL2 support (Chrome, Edge, Firefox recent versions).
- A local HTTP server to avoid `fetch()` failing when the page is opened from the filesystem.

## Run (quick)

Open a terminal (cmd.exe) in the project directory `c:\Users\Richard\Documents\Wave-Simulator` and run one of the following:

Using Python 3 (built-in):

```cmd
python -m http.server 8000
```

Then open in your browser:

```http
http://localhost:8000/
```
## Controls

The demo includes UI sliders (in the controls panel) for the following variables. These map to uniforms passed into the shaders from `main.js`:

Drag the mouse while holding the mouse button to rotate the view.