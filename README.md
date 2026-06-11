# ◉ BLACK HOLE — Relativistic Ray-Tracer

**Alexandre Almeida** · Universidade de Aveiro, 2026  
*Bachelor's Dissertation in Computational Engineering*  
*Supervisor: Prof. Pedro Cunha*

> GPU-accelerated relativistic ray-tracing engine for generating physically accurate images and animations of Schwarzschild black holes — with relativistic Doppler effect, volumetric radiative transfer, procedural gas (FBM), and Tidal Disruption Event (TDE) simulation.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square) ![Taichi](https://img.shields.io/badge/Taichi-GPU-orange?style=flat-square) ![Universidade de Aveiro](https://img.shields.io/badge/UA-2026-red?style=flat-square)

---

<p align="center">
  <img src="results/rotation_disk.gif" alt="Black hole rotation animation" width="600px">
</p>


---

## Results

| File | Description |
|----------|-----------|
| `results/bh.png` | Black hole with accretion disk and Doppler effect (1024px) |
| `results/bh_fbm.png` | Volumetric gas disk with FBM turbulence (2048px) |
| `results/bh_fbm_20d.png` | Render at 20° inclination (for EHT comparison) |
| `results/bh_fbm_20d_processed.png` | Simulation filtered under EHT resolution |
| `results/rotation_disk.gif` | Keplerian disk rotation animation (60 frames, 30fps) |
| `results/tde_animation_final.mp4` | Complete tidal disruption animation (spaghettification) |

---

## The Physics

### Spacetime and the Schwarzschild Metric

The geometry around a static, spherically symmetric black hole is described by the **Schwarzschild metric** (in natural units G = c = 1):

$$ds^2 = -\left(1 - \frac{2M}{r}\right)dt^2 + \left(1 - \frac{2M}{r}\right)^{-1}dr^2 + r^2\left(d\theta^2 + \sin^2\theta d\varphi^2\right)$$

From this metric three fundamental radial distances emerge:

- **Singularity** (r = 0): infinite curvature, breakdown of known physics
- **Event Horizon** (r = 2M): point of no return
- **Photon Sphere** (r = 3M): unstable circular light orbits — defines the apparent size of the shadow

### Geodesics — The Path of Light and Matter

In General Relativity, light and matter follow **geodesics** — the "straight lines" of curved spacetime. The system of equations of motion is:

$$\ddot{t} = -\frac{2M}{r(r-2M)}\dot{r}\dot{t}$$

$$\ddot{r} = -\frac{M}{r^2}\left(1-\frac{2M}{r}\right)\dot{t}^2 + \frac{M}{r^2\left(1-\frac{2M}{r}\right)}\dot{r}^2 + (r-2M)\left(\dot{\theta}^2 + \sin^2\theta\dot{\varphi}^2\right)$$

$$\ddot{\theta} = -\frac{2}{r}\dot{r}\dot{\theta} + \sin\theta\cos\theta\dot{\varphi}^2$$

$$\ddot{\varphi} = -\frac{2}{r}\dot{r}\dot{\varphi} - 2\cot\theta\dot{\theta}\dot{\varphi}$$

The distinction between **photons** (null geodesics, $ds^2 = 0$) and **matter** (timelike geodesics, $ds^2 = -1$) lies solely in the normalisation of initial conditions — the same integrator serves both cases.

### Relativistic Doppler Effect

The accretion disk orbits with Keplerian angular velocity $\omega = \sqrt{M/r^3}$. The local frequency of radiation measured by the moving plasma is:

$$\nu_\text{local} = -p_\mu u^\mu = A v^t u^t - r^2 \sin^2\theta v^\varphi u^\varphi$$

where $A = 1 - 2M/r$. The side of the disk approaching the observer appears brighter (*blueshift* + relativistic beaming); the opposite side appears darker (*redshift*).

### Radiative Transfer

The disk is not a solid surface but a **semi-translucent participating medium**. The evolution of the invariant intensity $y = I_\nu / \nu^3$ along the geodesic obeys:

$$\frac{dy}{d\lambda} = -p_\mu u^\mu \left( -\chi_\nu y + \frac{j_\nu}{\nu^3} \right)$$

The engine solves this equation simultaneously with the accumulated optical depth $L$, applying at the end an **algebraic correction** $\delta y = -y_\text{final} e^{L_\text{final}}$ that ensures physical consistency without additional iterations.

### Volumetric Gas with FBM

Disk turbulence is simulated with **Fractional Brownian Motion** (4 octaves of 3D sinusoidal noise), deformed by Keplerian differential rotation ($\omega \propto r^{-3/2}$). Inner layers rotate faster than outer ones, creating the characteristic spiral pattern.

### Tidal Disruption Event (TDE)

A star represented by N = 1000 particles approaches the black hole. The differential gravity tears the star apart — **spaghettification** — in two phases:

1. **Phase 1 — Dynamics:** Integration of the timelike geodesics of each particle; trajectories saved in `data/trajetorias_tde.npy`
2. **Phase 2 — Render:** Density reconstruction via **SPH** (Gaussian kernel $W = e^{-d^2/h^2}$) and volumetric ray-tracing rendering

---

## Installation

### Requirements

- **Python 3.13** (Fully tested. Other versions have not been verified).
- **GPU drivers** (optional but strongly recommended):
  - NVIDIA: CUDA 11.0+ (Taichi uses CUDA backend)
  - AMD / Intel: Vulkan drivers installed on the system
  - Without a compatible GPU, set `ti.init(arch=ti.cpu)` in each script — renders will be significantly slower
- **FFmpeg** (system-level, not a Python package) — required only if you use the `ffmpeg` CLI command directly to compile frames into video. The Python-based `frames_to_video.py` uses `imageio-ffmpeg` which bundles its own binary, so FFmpeg does not need to be installed separately for that script.

### Python packages

```bash
git clone <repo-url>
cd _BLACK_HOLE_
pip install -r requirements.txt
```

### FFmpeg (optional, for CLI encoding)

If you prefer the direct `ffmpeg` command over `frames_to_video.py`:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

> **GPU recommended.** For CPU, replace `ti.init(arch=ti.gpu)` with `ti.init(arch=ti.cpu)` at the top of each script.

---

## How to Use

### Basic render — Doppler
```bash
python code/rendering/render_doppler.py
# → results/bh.png  (1024px, seconds on GPU)
```

### Advanced render — FBM Gas
```bash
python code/rendering/render_fbm.py
# → results/bh_fbm.png  (2048px)
```

### Disk animation
```bash
python code/rendering/render_gif.py
# → results/rotation_disk.mp4  (60 frames)
```

> To export as GIF instead of MP4, set `out_format = 'gif'` at the top of `render_gif.py`.

### Solid opaque disk (high resolution)
```bash
python code/rendering/render_solid_disk.py
# → 5000px, geodesic structure very sharp
```

### TDE — Full pipeline

> **Note:** Step 1 must be completed before Step 2. `render_tde.py` loads `data/trajetorias_tde.npy` and will raise a `FileNotFoundError` if the simulation has not been run first.

```bash
# Step 1: simulate the trajectories (generates ~15 MB)
python code/simulation/tde_fase1_simulacao.py

# Step 2: render frame by frame (has crash recovery)
python code/rendering/render_tde.py

# Step 3: compile into video
python code/util/frames_to_video.py

# Fast alternative with FFmpeg:
ffmpeg -framerate 30 -i results/frames_tde/frame_%04d.png \
       -c:v libx264 -pix_fmt yuv420p -crf 18 \
       results/tde_animation_final.mp4
```

### Particle visualisers
```bash
python code/util/particles2D.py   # equatorial plane
python code/util/particles3D.py   # 3D view
```

### EHT comparison
```bash
python code/util/invert_to_real.py
# Applies Gaussian blur (σ=40) and maps afmhot colour
# to simulate Event Horizon Telescope resolution
```

> Requires `results/bh_fbm_20d.png` to exist. Run `render_fbm.py` with `theta_obs` set to 20° (≈ 0.349 rad) first.

---

## Project Structure

```
_BLACK_HOLE_/
├── code/
│   ├── rendering/
│   │   ├── render_doppler.py       # Render with Doppler effect
│   │   ├── render_fbm.py           # Volumetric FBM gas disk
│   │   ├── render_gif.py           # Disk rotation animation
│   │   ├── render_solid_disk.py    # Solid opaque disk
│   │   └── render_tde.py           # TDE animation with SPH
│   ├── simulation/
│   │   └── tde_fase1_simulacao.py  # Relativistic N-body simulation
│   └── util/
│       ├── frames_to_video.py      # Frame → MP4 compiler
│       ├── invert_to_real.py       # EHT resolution emulation
│       ├── particles2D.py          # 2D particle visualiser
│       └── particles3D.py          # 3D particle visualiser
├── data/
│   └── trajetorias_tde.npy         # Simulated trajectories (generated)
├── results/                        # Generated images and videos
├── textures/                       # Star backgrounds and real EHT image
└── requirements.txt
```

---

## Main Parameters

| Parameter | Description | Default |
|-----------|-----------|---------|
| `M` | Black hole mass (geometric units) | `1.0` |
| `r_obs` | Observer radial distance | `50 M` |
| `theta_obs` | Observer polar angle | `1.4 rad` |
| `res` | Image resolution in pixels | `1024` |
| `steps` | Max RK4 integration steps per ray | `7000–17000` |
| `r_disk_in/out` | Inner/outer disk radius | `4–20 M` |
| `eps` | Disk thickness | `0.01–0.2` |
| `n_frames` | Frames for animations | `60` |

---

## Performance

| Resolution | Max steps | Time (GPU RX 560X) | Throughput |
|-----------|-------------|----------------------|------------|
| 512 × 512 | 10 000 | 1.1 s | ~232 000 photons/s |
| 1024 × 1024 | 15 000 | 2.6 s | ~399 000 photons/s |
| 2048 × 2048 | 17 000 | 10.5 s | ~397 000 photons/s |

The TDE render at 1024px takes ~7 min/frame (GPU NVIDIA T1000) due to SPH evaluation over 1000 particles at each integration step.

---

## References

- Schwarzschild, K. (1916) — Exact solution of Einstein's equations
- EHT Collaboration (2019) — First image of M87\*
- James et al. (2015) — Relativistic ray-tracing in *Interstellar*
- Fuerst & Wu (2004) — Radiative transfer in curved spacetime
- Monaghan (1992) — Smoothed Particle Hydrodynamics
- Hu et al. (2019) — Taichi framework

---

*"Space is not simply the setting for a star. The star bends space."*