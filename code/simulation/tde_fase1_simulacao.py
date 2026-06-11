"""
Relativistic Raytracer & Simulator - Schwarzschild Black Hole
Phase 1: Tidal Disruption Event (TDE) Simulation.
Calculates Timelike Geodesics for a cluster of massive particles (a star)
being disrupted by the supermassive black hole.

@author: Alexandre Almeida

"""


import numpy as np
import taichi as ti
import time
import os

# Initialize Taichi. Use ti.cpu if you don't have a dedicated GPU.
ti.init(arch=ti.gpu)  # ou ti.cpu


# -- PARAMETERS --

# Black Hole
M = 1.0

# Star Initial Conditions
r_star_init = 60.0 * M
theta_star_init = 0.0
phi_star_init = 0.0

R_star = 10.0 * M
N_particles = 1000

# Integrator (Runge-Kutta 4)
h_min = (10**(-3)) * M
h_max = 0.5 * (10**(-2)) * M
sig = 3 * M         # Controls adaptive step size near the black hole
steps = 500

# Animation / Simulation Outputs
n_cuts_tde = 1000



# -- FILE PATHS --

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.join(ROOT_DIR, 'data'), exist_ok=True)
output_data_path = os.path.join(ROOT_DIR, 'data', 'trajetorias_tde.npy')



# -- TAICHI FIELDS --

# Particle State: [t, r, theta, phi, vt, vr, vtheta, vphi]
particle_states = ti.Vector.field(8, dtype=ti.f32, shape=N_particles)

# Trajectories cache to save to disk: [N_particles, n_cuts_tde, 4 (t, r, theta, phi)]
trajectories_np = np.zeros((N_particles, n_cuts_tde, 4), dtype=np.float32)

# Particle Status Flags: 0 = Active, 1 = Captured by Horizon, 2 = Escaped to Infinity
particle_flags = ti.field(dtype=ti.i32, shape=N_particles)



# -- PHYSICS HELPERS --

@ti.func
def f_timelike(t, r, theta, phi, vt, vr, vtheta, vphi):
    """
    Calculates the derivatives for Timelike Geodesics (massive particles).
    The geodesic equation form is identical to null geodesics, the difference
    lies purely in the initial 4-velocity normalization.
    """

    A = 1.0 - 2.0 * M / r

    r_2 = r * r
    sin_theta = ti.sin(theta)
    sin_theta_2 = sin_theta * sin_theta
    cos_theta = ti.cos(theta)
    
    # Geodesic Equations
    dvt = - (2.0*M / (r_2 * A)) * vt * vr
    dvr = - (M*A/r_2)*vt*vt + (M/(r_2*A))*vr*vr + (r-2.0*M)*vtheta*vtheta + (r-2.0*M)*sin_theta_2 * vphi*vphi
    dvth = - (2.0/r)*vr*vtheta + sin_theta*cos_theta*(vphi*vphi)
    dvph = - (2.0/r)*vr*vphi - 2.0*(cos_theta/sin_theta)*vtheta*vphi

    return ti.Vector([vt, vr, vtheta, vphi, dvt, dvr, dvth, dvph])

@ti.func
def rk4_step_3d(t, r, theta, phi, vt, vr, vtheta, vphi, h):
    """
    4th Order Runge-Kutta integrator.
    """
    
    k1 = f_timelike(t, r, theta, phi, vt, vr, vtheta, vphi)
    k2 = f_timelike(t + 0.5*h*k1[0], r+0.5*h*k1[1], theta+0.5*h*k1[2], phi+0.5*h*k1[3], vt+0.5*h*k1[4], vr+0.5*h*k1[5], vtheta+0.5*h*k1[6], vphi+0.5*h*k1[7])
    k3 = f_timelike(t + 0.5*h*k2[0], r+0.5*h*k2[1], theta+0.5*h*k2[2], phi+0.5*h*k2[3], vt+0.5*h*k2[4], vr+0.5*h*k2[5], vtheta+0.5*h*k2[6], vphi+0.5*h*k2[7])
    k4 = f_timelike(t + h*k3[0], r+h*k3[1], theta+h*k3[2], phi+h*k3[3], vt+h*k3[4], vr+h*k3[5], vtheta+h*k3[6], vphi+h*k3[7])

    t_n = t + h/6.0*(k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
    r_n = r + h/6.0*(k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
    th_n = theta + h/6.0*(k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])
    ph_n = phi + h/6.0*(k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3])

    vt_n = vt + h/6.0*(k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4])
    vr_n = vr + h/6.0*(k1[5] + 2.0*k2[5] + 2.0*k3[5] + k4[5])
    vth_n = vtheta + h/6.0*(k1[6] + 2.0*k2[6] + 2.0*k3[6] + k4[6])
    vph_n = vphi + h/6.0*(k1[7] + 2.0*k2[7] + 2.0*k3[7] + k4[7])
    
    return ti.Vector([t_n, r_n, th_n, ph_n, vt_n , vr_n, vth_n, vph_n])


@ti.kernel
def load_initial_state(states_np: ti.types.ndarray()):
    """ Copies the Numpy initial states into the Taichi Fields. """
    for k in range(N_particles):
        for j in ti.static(range(8)):
            particle_states[k][j] = states_np[k, j]
        particle_flags[k] = 0


def initialize_star():
    """
    Initializes a cluster of particles representing the star.
    Enforces the Timelike constraint: g_mu_nu * u^mu * u^nu = -1
    """
    # Star Center Position & Velocity (Cartesian)
    x_ini, y_ini, z_ini = 60.0, 0.0, 0.0
    vx_ini, vy_ini, vz_ini = -0.05, 0.12, 0.0

    rng = np.random.default_rng()
    states_np = np.zeros((N_particles, 8), dtype=np.float32)

    for k in range(N_particles):
        
        phi_rand = rng.uniform(0, 2*np.pi)
        costheta_rand = rng.uniform(-1, 1)
        u_rand = rng.uniform(0, 1)
        
        theta_rand = np.arccos(costheta_rand)
        r_rand = R_star * (u_rand**(1/3)) 

        # Particle Coordinates relative to the black hole
        dx = r_rand * np.sin(theta_rand) * np.cos(phi_rand)
        dy = r_rand * np.sin(theta_rand) * np.sin(phi_rand)
        dz = r_rand * np.cos(theta_rand)

        xk, yk, zk = x_ini + dx, y_ini + dy, z_ini + dz

        # Convert to Spherical Coordinates
        rk = np.sqrt(xk**2 + yk**2 + zk**2)
        thetak = np.arccos(zk / rk)
        phik = np.arctan2(yk, xk)

        # Convert Cartesian Velocities to Spherical Velocities
        vr_k = (xk*vx_ini + yk*vy_ini + zk*vz_ini) / rk
        rho2 = xk**2 + yk**2
        vtheta_k = ( (xk*vx_ini + yk*vy_ini)*zk - rho2*vz_ini ) / (rk**2 * np.sqrt(rho2))
        vphi_k = (xk*vy_ini - yk*vx_ini) / (rho2)

        # Timelike Constraint Normalization (Massive particles)
        A = 1.0 - 2.0 * M / rk
        spatial_terms = (vr_k**2 / A) + (rk**2 * vtheta_k**2) + (rk**2 * np.sin(thetak)**2 * vphi_k**2)
        
        # vt is calculated so that g_ab * u^a * u^b = -1
        vt_k = np.sqrt((1.0 + spatial_terms) / A)

        states_np[k] = np.array([0.0, rk, thetak, phik, vt_k, vr_k, vtheta_k, vphi_k], dtype=np.float32)

    return states_np



# -- MAIN SIMULATION -- 

@ti.kernel
def main_calculation(n_steps: ti.i32):

    for k in range(N_particles):
        
        # Skip if particle is captured or escaped
        if particle_flags[k] != 0:
            continue

        p = particle_states[k]

        t      = p[0]
        r      = p[1]
        theta  = p[2]
        phi    = p[3]
        vt     = p[4]
        vr     = p[5]
        vtheta = p[6]
        vphi   = p[7]

        adaptive_h_val = h_max

        # Integration step loop
        for _ in range(n_steps):
            if particle_flags[k] != 0:
                break

            state = rk4_step_3d(t, r, theta, phi, vt, vr, vtheta, vphi, adaptive_h_val)

            t = state[0]
            r = state[1]
            theta = state[2]
            phi = state[3]
            vt = state[4]
            vr = state[5]
            vtheta = state[6]
            vphi = state[7]
            
            # Adaptive step size: Smaller steps closer to the black hole
            rsin = r * ti.sin(theta)
            adaptive_h_val = h_min + (h_max - h_min) * (1.0 - ti.exp(-(rsin*rsin) / (2.0*sig*sig)))

            # CONDITION 1: Particle fell into the Event Horizon
            if r <= 2.05 * M:
                particle_flags[k] = 1
                break

            # CONDITION 2: Particle escaped to infinity (Ejected)
            if r > 3.0 * r_star_init:
                particle_flags[k] = 2
                break

        particle_states[k] = ti.Vector([t, r, theta, phi, vt, vr, vtheta, vphi])



# -- EXECUTION --

if __name__ == "__main__":

    print("=" * 60)
    print("Tidal Disruption Event (TDE) Simulation - Phase 1")
    print("=" * 60)
    print(f"  N_particles    = {N_particles}")
    print(f"  r_star_init    = {r_star_init} M")
    print(f"  R_star         = {R_star} M")
    print()
    
    print("Initializing Star Cluster...")
    states_np = initialize_star()
    load_initial_state(states_np)
    
    start_time = time.time()

    for frame in range(n_cuts_tde):
        t0 = time.time()
    
        main_calculation(steps)
    
        current_state = particle_states.to_numpy()   # Shape: (N, 8)
        current_flags = particle_flags.to_numpy()    # Shape: (N,)
    
        trajectories_np[:, frame, 0] = current_state[:, 0]   # t
        trajectories_np[:, frame, 1] = current_state[:, 1]   # r
        trajectories_np[:, frame, 2] = current_state[:, 2]   # theta
        trajectories_np[:, frame, 3] = current_state[:, 3]   # phi
    
        dt  = time.time() - t0
        eta = dt * (n_cuts_tde - frame - 1)
        
        active   = int(np.sum(current_flags == 0))
        captured = int(np.sum(current_flags == 1))
        escaped  = int(np.sum(current_flags == 2))

        print(f"  Frame {frame+1:4d}/{n_cuts_tde}  ({dt:.2f}s)  "
              f"Active: {active:3d}  "
              f"Captured: {captured:3d}  "
              f"Escaped: {escaped:3d}  "
              f"ETA: {eta:.0f}s")
    
    print(f"\nSimulation completed in {time.time() - start_time:.2f}s")
    
    # Save the huge numpy array to disk
    np.save(output_data_path, trajectories_np)
    
    print("\nFiles saved successfully:")
    print(f"  {output_data_path}   — shape (N_particles, n_cuts_tde, 4)")