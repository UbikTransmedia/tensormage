import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

class ImageProcessor:
    def __init__(self, image_shape, grid_size=(100, 100)):
        self.image_shape = image_shape
        self.grid_size = grid_size

    def load_images_from_folder(self, folder_path):
        images = []
        print(f"Loading images from folder: {folder_path}")
        for idx, filename in enumerate(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, filename)
            print(f" - Loading image {idx + 1}: {filename}")
            img = load_img(img_path, target_size=self.image_shape[:2])
            img_array = img_to_array(img)
            img_array = img_array.astype('float32') / 255.0  # Normalize
            images.append(img_array)
        print(f"Loaded {len(images)} images.")
        return np.array(images)

    def generate_morphed_frames(self, images, num_frames=25):
        num_images = len(images)
        frames = []
        print(f"Generating frames for GIF by morphing images...")

        # Generate forward frames
        for f in range(num_frames):
            t = f / (num_frames - 1)
            idx = int(t * (num_images - 1))
            next_idx = (idx + 1) % num_images
            alpha = t * (num_images - 1) - idx
            morphed_frame = (1 - alpha) * images[idx] + alpha * images[next_idx]
            frames.append(Image.fromarray((morphed_frame * 255).astype('uint8')))
            print(f" - Forward Frame {f + 1}/{num_frames} generated.")

        # Generate reverse frames (excluding the last frame to avoid duplication)
        for f in range(num_frames - 2, 0, -1):
            frames.append(frames[f])
            print(f" - Reverse Frame {num_frames - f}/{num_frames} generated.")

        print(f"Frame generation completed.")
        return frames

    def save_gif(self, frames, filename, duration=100):
        print(f"Saving GIF to {filename}...")
        frames[0].save(
            filename, save_all=True, append_images=frames[1:], loop=0, duration=duration
        )
        print(f"GIF saved successfully.")

# Usage of the class
if __name__ == "__main__":
    # Define parameters
    image_shape = (500, 500, 3)  # Adjust to the size of your images
    output_folder = '/home/guido/code/tensormage/img/'  # Change this to the folder where you want to save the GIF

    # Create an instance of the image processor
    processor = ImageProcessor(image_shape)

    # Load the images from a folder
    images = processor.load_images_from_folder('/home/guido/code/tensormage/img/solos2/')

    # Generate the morphed frames
    frames = processor.generate_morphed_frames(images)

    # Save the frames as a GIF
    processor.save_gif(frames, os.path.join(output_folder, 'morphed_animation_loop.gif'))

    # Display the first frame of the generated GIF
    frames[0].show()
