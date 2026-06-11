"""
Relativistic Raytracer & Simulator - Schwarzschild Black Hole
Utility: 3D Particle Visualizer.
Generates a 3D spatial animation (GIF) of the Tidal Disruption Event,
mapping the Event Horizon as a 3D sphere and aligning the camera angle
with the relativistic observer coordinates.

@author: Alexandre Almeida

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# Animation Settings
fps_tde = 30

u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
bh_x = 2.0 * M * np.cos(u) * np.sin(v)
bh_y = 2.0 * M * np.sin(u) * np.sin(v)
bh_z = 2.0 * M * np.cos(v)



# -- FILE PATHS & DATA LOADING --

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(ROOT_DIR, 'data', 'trajetorias_tde.npy')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Error: Simulation data not found at '{data_path}'. Please run the simulation script first.")

print("Loading simulation trajectories...")
trajectories_np = np.load(data_path)
N_particles, n_frames_tde, _ = trajectories_np.shape



# -- VISUALIZATION KERNEL --

def render_frame_3d(frame_idx):
    """ Renders a single frame of the 3D particle distribution with perspective camera. """
    
    fig = plt.figure(figsize=(8, 8), facecolor='#050a1f')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050a1f')
    ax.axis('off')
    
    ax.set_box_aspect([1, 1, 1])
    
    limit = 70
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    # Draw Schwarzschild Event Horizon 3D Surface Mesh
    ax.plot_surface(bh_x, bh_y, bh_z, color='black', alpha=1.0, zorder=10)

    # Position the 3D Camera to match the Observer's coordinate parameters
    camera_elevation = 90.0 - np.degrees(theta_obs)
    camera_azimuth = np.degrees(phi_obs)
    ax.view_init(elev=camera_elevation, azim=camera_azimuth)

    frame_state = trajectories_np[:, frame_idx, :]
    r_arr     = frame_state[:, 1]
    theta_arr = frame_state[:, 2]
    phi_arr   = frame_state[:, 3]
 
    x = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
    y = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
    z = r_arr * np.cos(theta_arr)
 
    for k in range(N_particles):
        r_k = r_arr[k]

        if r_k <= 2.1 * M:
            continue
        else:
            d = np.clip((r_arr[k] - 2.0*M) / (r_star_init - 2.0*M), 0.0, 1.0)
            color = (1.0, 0.3 + 0.5*d, 0.2*d)
            size = 10.0
 
        ax.scatter(x[k], y[k], z[k], s=size, color=color, zorder=6, alpha=0.9)
 
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    
    plt.close(fig)
    return img



# -- EXECUTION --

if __name__ == "__main__":

    print("Generating 3D particle animation frames...")
    start_time = time.time()
    frames_gif = []

    for frame in range(n_frames_tde):
        img = render_frame_3d(frame)
        frames_gif.append(img)
        print(f"  Frame {frame+1:3d}/{n_frames_tde} processed.")

    # Save compile sequence as a GIF
    print("Compiling and saving GIF animation...")
    output_path = os.path.join(ROOT_DIR, 'results', 'tde_simulacao_3D.gif')
    imageio.mimsave(output_path, frames_gif, fps=fps_tde, loop=0)

    print(f"\nRender completed successfully in {time.time() - start_time:.2f}s!")
    print(f"Animation saved to: {output_path}")