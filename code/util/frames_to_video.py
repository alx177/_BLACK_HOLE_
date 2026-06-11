"""
Relativistic Raytracer & Simulator - Schwarzschild Black Hole
Utility: Frames to Video Compiler.
Stitches rendered PNG frames into an MP4 video using imageio (Python) or FFmpeg (CLI).

@author: Alexandre Almeida

"""

# If you have FFmpeg installed on your system terminal, you can run this command 
# directly from the root directory of the project for faster and more optimized encoding:
#
# ffmpeg -framerate 30 -i results/frames_tde/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 results/tde_animation_final.mp4

import imageio.v2 as imageio
import glob
import os


# -- PARAMETERS --

fps_out = 30

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frames_dir = os.path.join(ROOT_DIR, 'results', 'frames_tde')
output_video_path = os.path.join(ROOT_DIR, 'results', 'tde_animation_final.mp4')


# -- EXECUTION --

if __name__ == "__main__":

    print("=" * 60)
    print("Compiling TDE Rendered Frames to Video...")
    print("=" * 60)

    # Fetch and sort all rendered frames
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
    
    if len(frame_paths) == 0:
        print(f"Error: No rendered frames found at '{frames_dir}'.")
        print("Please render the frames using 'code/rendering/render_tde.py' first.")
        exit()

    print(f"Found {len(frame_paths)} frames.")
    print("Compiling video file (this may take a few seconds)...")

    writer = imageio.get_writer(output_video_path, fps=fps_out, codec='libx264', quality=8)
    
    for i, path in enumerate(frame_paths):
        writer.append_data(imageio.imread(path))
        print(f"  Progress: {i+1}/{len(frame_paths)} frames processed.", end='\r')

    writer.close()
    
    print("\n" + "=" * 60)
    print(f"Video saved successfully: {output_video_path}")
    print("=" * 60)

