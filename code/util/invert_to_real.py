"""
Relativistic Raytracer & Simulator - Schwarzschild Black Hole
Utility: Event Horizon Telescope (EHT) Resolution Emulation.
Processes simulated images to match the finite angular resolution (diffraction limit) 
observed by the EHT VLBI network.

@author: Alexandre Almeida

"""


import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from PIL import Image
import os


# -- PARAMETERS --

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
simulated_image_path = os.path.join(ROOT_DIR, 'results', 'bh_fbm_20d.png')
real_eht_image_path  = os.path.join(ROOT_DIR, 'textures', 'bh_real.jpg')

# Processing constants
blur_sigma = 40
rotation_angle_degrees = -90  


# -- IMAGE PROCESSING --

try:
    # Load simulated render
    img_rgb = np.array(Image.open(simulated_image_path)) / 255.0

    # Isolate color channels
    red_channel = img_rgb[:, :, 0]
    blue_channel = img_rgb[:, :, 2]
    
    isolated_disk = np.where(red_channel > blue_channel * 1.1, red_channel, 0.0)
    blurred_img = ndimage.gaussian_filter(isolated_disk, sigma=blur_sigma) 

    # Normalize brightness to standard range
    if blurred_img.max() > 0:
        blurred_img = blurred_img / blurred_img.max()

    # Contrast adjustment and scaling
    blurred_img = np.power(blurred_img, 2.1) * 0.8
    blurred_img = np.clip(blurred_img, 0.0, 1.0)

    # Rotate image to align with the EHT published orientation
    blurred_img = ndimage.rotate(blurred_img, rotation_angle_degrees, reshape=False, order=3, cval=0.0)
    blurred_img = np.clip(blurred_img, 0.0, 1.0)


    # -- PLOTTING & EXPORT --

    fig, ax = plt.subplots(2, 2, figsize=(14, 7))

    # 1. Original high-res simulation
    ax[0, 0].imshow(img_rgb)
    ax[0, 0].set_title("1. Original Simulation")
    ax[0, 0].axis('off')

    # 2. Isolated accretion disk (no background stars)
    ax[0, 1].imshow(isolated_disk, cmap='afmhot', origin='upper')
    ax[0, 1].set_title("2. Processing: Isolated Accretion Disk", fontsize=11)
    ax[0, 1].axis('off')

    # 3. Simulated EHT low-resolution result
    ax[1, 0].imshow(blurred_img, cmap='afmhot', origin='upper', vmin=0.0, vmax=1.0)
    ax[1, 0].set_title("3. Simulation under EHT Diffraction Limit")
    ax[1, 0].axis('off')

    # Color map reconstruction to export as 3-channel RGB image
    color_map = plt.get_cmap('afmhot')
    colored_img = color_map(blurred_img)
    colored_img = colored_img[:, :, :3]

    # Save output to disk
    output_processed_path = os.path.join(ROOT_DIR, 'results', 'bh_fbm_20d_processed.png')
    imageio.imwrite(output_processed_path, (colored_img * 255).astype(np.uint8))

    # 4. Actual real EHT observation of M87*
    ax[1, 1].imshow(Image.open(real_eht_image_path), cmap='afmhot', origin='upper')
    ax[1, 1].set_title("4. Actual EHT Observation")
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: Could not find '{simulated_image_path}' or the EHT reference asset.")
    print("Please make sure you have successfully run the simulation and saved the result image first.")