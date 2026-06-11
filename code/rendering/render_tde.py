"""
Relativistic Raytracer - Schwarzschild Black Hole
Calculates Null Geodesics, Relativistic Doppler Effect, and Radiative Transfer.
Renders a Tidal Disruption Event (TDE) from pre-calculated N-body trajectories
using SPH (Smoothed-Particle Hydrodynamics) volumetric mapping.

@author: Alexandre Almeida

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import taichi as ti
import time
import imageio
import os

# Initialize Taichi. Disabled advanced_optimization to prevent compilation timeouts on complex loops.
ti.init(arch=ti.gpu)  # ou ti.cpu


# -- PARAMETERS --

# Black Hole
M = 1.0

# Camera / Observer
r_obs = 110.0 
theta_obs = 1.2
phi_obs = np.pi
res = 100           # Image resolution
screen_size = 17.0  # Field of view scaling
D = 35.0            # Distance from camera to projection screen

# Integrator (Runge-Kutta 4)
h_min = -(10**(-3))*M
h_max = -(10**(-1))*M
sig = 3*M           # Controls adaptive step size near the black hole
steps = 10000

# Aesthetics
target_color_orange = ti.Vector([1.7, 0.4, 0.05])

# Animation
n_frames = 60
fps_out = 30



# -- FILE PATHS & DATA LOADING --

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load pre-calculated Star Trajectories safely
data_path = os.path.join(ROOT_DIR, 'data', 'trajetorias_tde.npy')

if not os.path.exists(data_path):

    raise FileNotFoundError(f"Error: Data file not found at '{data_path}'. Please run the Phase 1 Simulation script first.")


print("Loading and interpolating TDE data...")
cache_data = np.load(data_path).astype(np.float32)
N_PART, N_CUTS, _ = cache_data.shape

t_min_global = np.min(cache_data[:, :, 0])
t_max_global = np.max(cache_data[:, :, 0])

# Pre-interpolate trajectories into an uniform grid 
N_GRID = 2000
t_grid_np = np.linspace(t_min_global, t_max_global, N_GRID)
dt_grid = (t_max_global - t_min_global) / (N_GRID - 1)

uniform_data_np = np.zeros((N_PART, N_GRID, 3), dtype=np.float32)

for k in range(N_PART):
    t_k  = cache_data[k, :, 0]
    r_k  = cache_data[k, :, 1]
    th_k = cache_data[k, :, 2]
    ph_k = cache_data[k, :, 3]
    
    # Prevent duplicate times from the adaptive integrator
    t_k, unique_indices = np.unique(t_k, return_index=True)
    
    # SciPy Interpolation
    f_r  = interpolate.interp1d(t_k, r_k[unique_indices],  bounds_error=False, fill_value=(r_k[0], r_k[-1]))
    f_th = interpolate.interp1d(t_k, th_k[unique_indices], bounds_error=False, fill_value=(th_k[0], th_k[-1]))
    f_ph = interpolate.interp1d(t_k, ph_k[unique_indices], bounds_error=False, fill_value=(ph_k[0], ph_k[-1]))
    
    uniform_data_np[k, :, 0] = f_r(t_grid_np)
    uniform_data_np[k, :, 1] = f_th(t_grid_np)
    uniform_data_np[k, :, 2] = f_ph(t_grid_np)


# Load Background Image safely
bg_path = os.path.join(ROOT_DIR, 'textures', 'galaxy1.jpg')
try:
    bg_img_np = plt.imread(bg_path)
except FileNotFoundError:
    print(f"Error: Background image not found at {bg_path}")
    bg_img_np = np.zeros((512, 512, 3), dtype=np.float32) # Fallback to black

if bg_img_np.ndim == 2:
    bg_img_np = np.stack((bg_img_np,)*3, axis=-1)
elif bg_img_np.shape[2] == 4:
    bg_img_np = bg_img_np[:, :, :3]
if bg_img_np.max() > 1.0: 
    bg_img_np = bg_img_np / 255.0



# -- TAICHI FIELDS --

# Interpolated Particle Data
tde_grid = ti.Vector.field(3, dtype=ti.f32, shape=(N_PART, N_GRID))
tde_grid.from_numpy(uniform_data_np)

t_min_ti = ti.field(dtype=ti.f32, shape=())
t_min_ti[None] = t_min_global

dt_grid_ti = ti.field(dtype=ti.f32, shape=())
dt_grid_ti[None] = dt_grid

# RGB Image & Coordinates
image = ti.Vector.field(3, ti.f32, shape=(res, res))
x_range = ti.field(ti.f32, shape=res)
y_range = ti.field(ti.f32, shape=res)

h_img, w_img, _ = bg_img_np.shape
bg_img_ti = ti.Vector.field(3, dtype=ti.f32, shape=(h_img, w_img))
bg_img_ti.from_numpy(bg_img_np.astype(np.float32))


@ti.kernel
def init_ranges():
    for i in range(res):
        t = i / (res - 1)
        value = -screen_size + t * (2 * screen_size)
        x_range[i] = value
        y_range[i] = value



# -- PHYSICS HELPERS --

@ti.func
def interpolate_particle(t, k):
    """
    Linearly interpolates the position of particle 'k' at photon time 't'
    from the pre-calculated uniform grid.
    """
    idx_float = (t - t_min_ti[None]) / dt_grid_ti[None]
    idx = ti.cast(ti.floor(idx_float), ti.i32)
    idx = ti.math.clamp(idx, 0, N_GRID - 2)
        
    # Linear Interpolation
    weight1 = idx_float - ti.cast(idx, ti.f32)
    weight1 = ti.math.clamp(weight1, 0.0, 1.0) 
    weight0 = 1.0 - weight1
    
    pos0 = tde_grid[k, idx]
    pos1 = tde_grid[k, idx + 1]
    
    return pos0 * weight0 + pos1 * weight1


h_star_base = 6.0 * M

@ti.func
def kernel_sph(dist2, h2):
    """ Gaussian SPH (Smoothed-Particle Hydrodynamics) Kernel: W(r,h) = exp(-r^2/h^2) """
    return ti.exp(-dist2 / h2)

@ti.func
def get_star_density(r, theta, phi, photon_time):
    """
    Reconstructs the volumetric gas density from the discrete N-body particles
    using the SPH method. The photon evaluates the local density as it travels.
    """
    rho = 0.0
    photon_pos = ti.Vector([
        r * ti.sin(theta) * ti.cos(phi),
        r * ti.sin(theta) * ti.sin(phi),
        r * ti.cos(theta)
    ])

    h2 = h_star_base * h_star_base

    # Accumulate density from all nearby particles
    for k in range(N_PART):
        p = interpolate_particle(photon_time, k)
        r_p, th_p, ph_p = p[0], p[1], p[2]

        if r_p > 2.1 * M and ti.abs(r - r_p) < 25.0:
            part_pos = ti.Vector([
                r_p * ti.sin(th_p) * ti.cos(ph_p),
                r_p * ti.sin(th_p) * ti.sin(ph_p),
                r_p * ti.cos(th_p)
            ])

            dist2 = (photon_pos - part_pos).norm_sqr()
            rho += kernel_sph(dist2, h2)

    return rho


@ti.func
def f(t, r, theta, phi, vt, vr, vtheta, vphi, y, L):
    """
    Calculates the derivatives for the Null Geodesics and Radiative Transfer.
    Solves 2nd order ODEs reduced to 1st order using Christoffel Symbols.
    """
    
    A = 1 - 2*M/r

    r_2 = r * r
    sin_theta = ti.sin(theta)
    sin_theta_2 = sin_theta * sin_theta
    cos_theta = ti.cos(theta)
    
    # Geodesic Equations
    dvt = - (2*M / (r_2 * A)) * vt * vr
    dvr = - (M*A/r_2)*vt*vt + (M/(r_2*A))*vr*vr + (r-2*M)*vtheta* vtheta + (r-2*M)*sin_theta_2 * vphi * vphi
    dvth = - (2/r)*vr*vtheta + sin_theta*cos_theta*(vphi*vphi)
    dvph = - (2/r)*vr*vphi - 2*(cos_theta/sin_theta)*vtheta*vphi

    # Gas density
    density = get_star_density(r, theta, phi, t)
    density = ti.math.min(density, 30.0)  # Clamp to prevent overflow

    # Doppler simplification
    p_dot_u = 1.0

    b_x = density                 # Emission coefficient
    a_x = 0.05 * density          # Absorption coefficient 

    dy = -a_x * p_dot_u * y + b_x * p_dot_u
    dL = a_x * p_dot_u

    return ti.Vector([vt, vr, vtheta, vphi, dvt, dvr, dvth, dvph, dy, dL])


@ti.func
def rk4_step_3d(t, r, theta, phi, vt, vr, vtheta, vphi, y, L, h):
    """
    4th Order Runge-Kutta integrator.
    """
    k1 = f(t, r, theta, phi, vt, vr, vtheta, vphi, y, L)
    k2 = f(t + 0.5*h*k1[0], r+0.5*h*k1[1], theta+0.5*h*k1[2], phi+0.5*h*k1[3], vt+0.5*h*k1[4], vr+0.5*h*k1[5], vtheta+0.5*h*k1[6], vphi+0.5*h*k1[7], y + 0.5*h*k1[8], L + 0.5*h*k1[9])
    k3 = f(t + 0.5*h*k2[0], r+0.5*h*k2[1], theta+0.5*h*k2[2], phi+0.5*h*k2[3], vt+0.5*h*k2[4], vr+0.5*h*k2[5], vtheta+0.5*h*k2[6], vphi+0.5*h*k2[7], y + 0.5*h*k2[8], L + 0.5*h*k2[9])
    k4 = f(t + h*k3[0], r+h*k3[1], theta+h*k3[2], phi+h*k3[3], vt+h*k3[4], vr+h*k3[5], vtheta+h*k3[6], vphi+h*k3[7], y + h*k3[8], L + h*k3[9])

    t_n = t + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    r_n = r + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    th_n = theta + h/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    ph_n = phi + h/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    vt_n = vt + h/6*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
    vr_n = vr + h/6*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
    vth_n = vtheta + h/6*(k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
    vph_n = vphi + h/6*(k1[7] + 2*k2[7] + 2*k3[7] + k4[7])

    y_n = y + h/6*(k1[8] + 2*k2[8] + 2*k3[8] + k4[8])
    L_n = L + h/6*(k1[9] + 2*k2[9] + 2*k3[9] + k4[9])
    
    return ti.Vector([t_n, r_n, th_n, ph_n, vt_n , vr_n, vth_n, vph_n, y_n, L_n])


@ti.func
def initial_conditions(sin_alpha, cos_alpha_sin_beta, cos_alpha_cos_beta, t_current_frame : ti.f32):
    """
    Sets the initial 4-momentum (p^mu) of the photon at the observer's camera.
    Rays are traced backwards in time (from the observer to the scene).
    """

    E = 1.0 # Photon Energy at infinity

    pt = E / ti.sqrt(1 - 2*M/r_obs)
    P = pt

    t = t_current_frame
    r, theta, phi = r_obs, theta_obs, phi_obs

    # Initial velocities 
    vt = P / ti.sqrt(1 - 2*M/r)
    vr = (P * ti.sqrt(1 - 2*M/r)) * cos_alpha_cos_beta
    vtheta = (P / r) * sin_alpha
    vphi = (P / (r*ti.sin(theta)) ) * cos_alpha_sin_beta

    # Radiative Transfer Equation
    y = 0.0
    L = 0.0

    return ti.Vector([t, r, theta, phi, vt, vr, vtheta, vphi, y, L])



# -- MAIN RENDERING -- 

@ti.kernel
def main_calculation(t_current_frame : ti.f32):

    for i, j in image:
        
        image[i, j] = ti.Vector([0.0, 0.0, 0.0])

        bx = x_range[j]
        by = y_range[i]
    
        dist = ti.sqrt(bx*bx + by*by + D*D)

        # Ray angles mapping
        sin_alpha = by / dist
        cos_alpha_sin_beta = bx / dist
        cos_alpha_cos_beta = D / dist

        initial_conditions_vec = initial_conditions(sin_alpha , cos_alpha_sin_beta, cos_alpha_cos_beta, t_current_frame)

        t = initial_conditions_vec[0]
        r = initial_conditions_vec[1]
        theta = initial_conditions_vec[2]
        phi = initial_conditions_vec[3]
        vt = initial_conditions_vec[4]
        vr = initial_conditions_vec[5]
        vtheta = initial_conditions_vec[6]
        vphi = initial_conditions_vec[7]
        y = initial_conditions_vec[8]
        L = initial_conditions_vec[9]

        adaptive_h_val = h_max

        # Raytracing step loop
        for s in range(steps):

            state = rk4_step_3d(t, r, theta, phi, vt, vr, vtheta, vphi, y, L, adaptive_h_val)

            t = state[0]
            r = state[1]
            theta = state[2]
            phi = state[3]
            vt = state[4]
            vr = state[5]
            vtheta = state[6]
            vphi = state[7]
            y = state[8]
            L = state[9]

            # Adaptive step size: Smaller steps closer to the black hole
            rsin = r * ti.sin(theta)
            adaptive_h_val = h_min + (h_max - h_min) * (1.0 - ti.exp(-(rsin*rsin) / (2.0*sig*sig)))

            # CONDITION 1: Ray fell into the Event Horizon
            if r <= 2.05 * M:
                
                delta_y = -y * ti.exp(L) 

                image[i, j] = delta_y * target_color_orange
                break
                
            # CONDITION 2: Ray escaped to infinity (Celestial Sphere)
            if r > r_obs:
                
                # Map spherical coordinates to 2D background image coordinates
                u_norm = (phi % (2.0 * ti.math.pi)) / (2.0 * ti.math.pi)
                v_norm = (theta % ti.math.pi) / ti.math.pi
                
                u_px = int(u_norm * (w_img - 1))
                v_px = int(v_norm * (h_img - 1))

                delta_y = -y * ti.exp(L) 
                
                image[i, j] = (ti.exp(L) * bg_img_ti[v_px, u_px]) + (delta_y * target_color_orange)
                break

        image[i, j] = ti.math.clamp(image[i, j], 0.0, 1.0)



# -- EXECUTION --

if __name__ == "__main__":

    print("=" * 60)
    print("Starting TDE Animation Rendering to Frames...")
    print("=" * 60)

    # Calculate physical times to observe
    t_obs_start = t_min_global + r_obs + 10.0 
    delta_t_physical_per_frame = 12.0 

    frames_dir = os.path.join(ROOT_DIR, 'results', 'frames_tde')
    os.makedirs(frames_dir, exist_ok=True)

    init_ranges()

    start_time = time.time()

    for frame in range(n_frames):
        frame_path = os.path.join(frames_dir, f'frame_{frame:04d}.png')
        
        # Crash recovery / Resume capability
        if os.path.exists(frame_path):
            print(f"  Frame {frame+1:3d}/{n_frames}  — already exists, skipping...")
            continue

        t_current_frame = t_obs_start + frame * delta_t_physical_per_frame

        t0 = time.time()

        # Clear image memory between frames
        image.fill(0)

        main_calculation(t_current_frame)
        
        ti.sync() 

        # Flip and save image
        img = np.flipud(image.to_numpy())
        imageio.imwrite(frame_path, (img * 255).astype(np.uint8))

        t1 = time.time()
        eta_seconds = (t1 - t0) * (n_frames - frame - 1)
        print(f"  Frame {frame+1:3d}/{n_frames}  ({t1-t0:.1f}s)  — ETA: {eta_seconds:.1f} s")

    end_time = time.time()
    print(f"\nRender completed successfully in: {end_time - start_time:.2f} seconds")
    print(f"Frames saved to: {frames_dir}")