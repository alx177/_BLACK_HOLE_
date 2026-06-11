"""
Relativistic Raytracer & Simulator - Schwarzschild Black Hole
Utility: 2D Particle Visualizer.
Generates an equatorial plane 2D animation (GIF) of the Tidal Disruption Event,
displaying the Event Horizon (2M) and the ISCO (6M) boundaries.

@author: Alexandre Almeida

"""


import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import os


# -- PARAMETERS --

# Black Hole
M = 1.0
r_star_init = 60.0 * M

# Camera / Observer
r_obs = 50.0 
theta_obs = 1.4
phi_obs = np.pi

# Project Observer coordinates onto the 2D equatorial plane (X, Y)
x_obs = r_obs * np.sin(theta_obs) * np.cos(phi_obs)
y_obs = r_obs * np.sin(theta_obs) * np.sin(phi_obs)

# Animation Settings
fps_tde = 30



# -- FILE PATHS & DATA LOADING --

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(ROOT_DIR, 'data', 'trajetorias_tde.npy')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Error: Simulation data not found at '{data_path}'. Please run the simulation script first.")

print("Loading simulation trajectories...")
trajectories_np = np.load(data_path)
N_particles, n_frames_tde, _ = trajectories_np.shape



# -- VISUALIZATION --

def render_frame_2d(frame_idx):
    """ Renders a single frame of the 2D equatorial particle distribution. """

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#050a1f')
    ax.set_facecolor('#050a1f')
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_aspect('equal')
    ax.axis('off')
 
    # Draw Schwarzschild Event Horizon (R = 2M)
    ax.add_patch(plt.Circle((0, 0), 2.0 * M, color='black', zorder=5))
    
    # Draw ISCO - Innermost Stable Circular Orbit (R = 6M)
    ax.add_patch(plt.Circle((0, 0), 6.0 * M, fill=False, color='white',
                             linewidth=0.4, alpha=0.4, linestyle='--', zorder=3)) 
 
    # Plot Observer Location
    ax.scatter(x_obs, y_obs, color='lime', s=100, marker='*', label='Observer', zorder=10)

    frame_state = trajectories_np[:, frame_idx, :]
    r_arr     = frame_state[:, 1]
    theta_arr = frame_state[:, 2]
    phi_arr   = frame_state[:, 3]
 
    x = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
    y = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
 
    for k in range(N_particles):
        r_k = r_arr[k]

        if r_k <= 2.1 * M:
            continue
        else:

            d = np.clip((r_arr[k] - 2.0*M) / (r_star_init - 2.0*M), 0.0, 1.0)
            color = (1.0, 0.3 + 0.5*d, 0.2*d)
            size = 4.0
 
        ax.scatter(x[k], y[k], s=size, color=color, zorder=6, linewidths=0)
 
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    
    plt.close(fig)
    return img



# -- EXECUTION --

if __name__ == "__main__":

    print("Generating 2D particle animation frames...")
    start_time = time.time()
    frames_gif = []

    for frame in range(n_frames_tde):
        img = render_frame_2d(frame)
        frames_gif.append(img)
        print(f"  Frame {frame+1:3d}/{n_frames_tde} processed.")

    # Save compile sequence as a GIF
    print("Compiling and saving GIF animation...")
    output_path = os.path.join(ROOT_DIR, 'results', 'tde_simulacao_2D.gif')
    imageio.mimsave(output_path, frames_gif, fps=fps_tde, loop=0)

    print(f"\nRender completed successfully in {time.time() - start_time:.2f}s!")
    print(f"Animation saved to: {output_path}")