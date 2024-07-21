import os
import imageio

def create_gif(png_files, gif_path):
    images = []

    # Ensure all files exist and are png files
    for file in png_files:
        if not os.path.isfile(file) or not file.endswith('.png'):
            raise FileNotFoundError(f"File {file} does not exist or is not a PNG file.")

    # Read all images and append to the list
    for file in png_files:
        images.append(imageio.imread(file))

    # Save as gif
    imageio.mimsave(gif_path, images, duration=0.2)  # Adjust duration if needed
    print(f"GIF created and saved to {gif_path}")