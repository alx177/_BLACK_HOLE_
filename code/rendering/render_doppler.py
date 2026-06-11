"""
Relativistic Raytracer - Schwarzschild Black Hole
Calculates Null Geodesics, Relativistic Doppler Effect, and Radiative Transfer.

@author: Alexandre Almeida

"""


import imageio
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import time
import os

# Initialize Taichi. Use ti.cpu if you don't have a dedicated GPU.
ti.init(arch=ti.gpu)  # ou ti.cpu


# -- PARAMETERS --

# Black Hole
M = 1.0

# Camera / Observer
r_obs = 50.0 
theta_obs = 1.4
phi_obs = 0.0
res = 1024          # Image resolution
screen_size = 15.0  # Field of view scaling
D = 35.0            # Distance from camera to projection screen

# Accretion Disk
r_disco_in = 6.0 * M
r_disco_out = 15.0 * M
eps = 0.2           # Disk thickness

# Integrator (Runge-Kutta 4)
h_min = -(10**-3)*M
h_max = -(10**-1)*M
sig = 3*M         # Controls adaptive step size near the black hole
steps = 7000

# Aesthetics
target_color_orange  = ti.Vector([1.5, 0.4, 0.05])



# -- FILE PATHS --

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(ROOT_DIR, 'results', 'bh.png')
bg_path = os.path.join(ROOT_DIR, 'textures', 'galaxy1.jpg')


# Load Background Image safely
try:
    bg_img_np = plt.imread(bg_path)
except FileNotFoundError:
    print(f"Error: Background image not found at {bg_path}")
    bg_img_np = np.zeros((512, 512, 3), dtype=np.float32) # Fallback to black


if bg_img_np.ndim == 2:
    bg_img_np = np.stack((bg_img_np,)*3, axis=-1)

elif bg_img_np.shape[2] == 4:
    bg_img_np = bg_img_np[:, :, :3]

if bg_img_np.max() > 1.0: # Normalize 0-255 to 0.0-1.0
    bg_img_np = bg_img_np / 255.0



# -- TAICHI FIELDS --


# Imagem RGB
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
def f(t, r, theta, phi, vt, vr, vtheta, vphi, y, L):
    """
    Calculates the derivatives for the Null Geodesics and Radiative Transfer.
    Solves 2nd order ODEs reduced to 1st order using Christoffel Symbols.
    """

    A = 1 - 2*M/r

    r_2 = r * r
    sin_theta = ti.sin(theta)
    sin_theta_2 = sin_theta * sin_theta

    # Geodesic Equations
    dvt = - (2*M / (r**2 * A)) * vt * vr
    dvr = - (M*A/r**2)*vt**2 + (M/(r**2*A))*vr**2 + (r-2*M)*vtheta**2 + (r-2*M)*ti.sin(theta)**2 * vphi**2
    dvth = - (2/r)*vr*vtheta + ti.sin(theta)*ti.cos(theta)*vphi**2
    dvph = - (2/r)*vr*vphi - 2*(ti.cos(theta)/ti.sin(theta))*vtheta*vphi

    # Accretion Disk Physics

    fator_radial = ti.math.smoothstep(r_disco_in - 1.0, r_disco_in, r) * \
    (1.0 - ti.math.smoothstep(r_disco_out, r_disco_out + 2.0, r))

    density = ti.exp(-((r*ti.cos(theta))**2) / (2*eps**2)) * fator_radial

    # Relativistic Doppler Effect

    p_dot_u  = 1.0

    if r > 3.0 * M:

        # Keplerian angular velocity (w = sqrt(M/r^3))
        w = ti.sqrt(M / (r**(3)))

        ut = 1 / ti.sqrt(A - w**2 * r**2 * ti.sin(theta)**2)
        uphi = w*ut

        # Inner product
        p_dot_u  = A * vt * ut - r_2 * sin_theta_2 * vphi * uphi

    # Radiative Transfer Equation
    v = p_dot_u  
    b_x = 80.0 / (r * r + 0.1) * density  # Emission coefficient
    a_x = 0.5 * density                   # Absorption coefficient

    dy = -a_x * p_dot_u  * y + (b_x/(v)**3) * p_dot_u 
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
def initial_conditions(sin_alpha , cos_alpha_sin_beta, cos_alpha_cos_beta):
    """
    Sets the initial 4-momentum (p^mu) of the photon at the observer's camera.
    Rays are traced backwards in time (from the observer to the scene).
    """

    E = 1.0 # Photon Energy at infinity

    pt = E / ti.sqrt(1 - 2*M/r_obs)
    P = pt

    t = 0.0
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
def main_calculation():

    for i, j in image:
        
        image[i, j] = ti.Vector([0.0, 0.0, 0.0])

        bx = x_range[j]
        by = y_range[i]
    
        dist = ti.sqrt(bx**2 + by**2 + D**2)

        # Ray angles mapping
        sin_alpha = by / dist
        cos_alpha_sin_beta = bx / dist
        cos_alpha_cos_beta = D / dist

        initial_conditions_vec = initial_conditions(sin_alpha , cos_alpha_sin_beta, cos_alpha_cos_beta)

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
            adaptive_h_val = (h_min + (h_max - h_min) * (1 - ti.exp(-((r*ti.sin(theta))**2) / (2*sig**2))))

            # CONDITION 1: Ray fell into the Event Horizon
            if r <= 2.05 * M:
                
                delta_y = -y * ti.exp(L) 

                image[i, j] = delta_y * target_color_orange 

                break
        
            
            # CONDITION 2: Ray escaped to infinity (Celestial Sphere)
            if r > r_obs * 2:
                # Map spherical coordinates to 2D background image coordinates

                u_norm = (phi % (2.0 * ti.math.pi)) / (2.0 * ti.math.pi)
                v_norm = (theta % ti.math.pi) / ti.math.pi
                
                u_px = int(u_norm * (w_img - 1))
                v_px = int(v_norm * (h_img - 1))


                delta_y = -y * ti.exp(L) 
                image[i, j] = (ti.exp(L) * bg_img_ti[v_px, u_px]) + (delta_y * target_color_orange )


                break



# -- EXECUTION --

if __name__ == "__main__":

    plt.figure(figsize=(10, 10))

    print("Starting raytracing calculation...")
    start = time.time()

    init_ranges()
    main_calculation()
    img = image.to_numpy()

    end = time.time()
    print(f"Render completed in: {end - start:.2f} seconds")

    # Display image
    plt.imshow(img, origin='lower')
    plt.axis('off')
    plt.show()

    # Save to file
    img = np.flipud(image.to_numpy())
    img = np.clip(img, 0.0, 1.0)
    imageio.imwrite(results_path, (img * 255).astype(np.uint8))
    print(f"Image saved to: {results_path}")
