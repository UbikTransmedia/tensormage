import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageDraw

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

    def calculate_pixel_density(self, image):
        h, w = self.grid_size
        density = np.zeros((h, w))
        slice_height = self.image_shape[0] // h
        slice_width = self.image_shape[1] // w

        print(f"Calculating pixel density...")
        for i in range(h):
            for j in range(w):
                slice = image[i*slice_height:(i+1)*slice_height, j*slice_width:(j+1)*slice_width]
                density[i, j] = np.mean(slice)  # Average pixel value in the slice
        print(f"Pixel density calculation completed.")
        return density

    def generate_fractal_image(self, h, w, f, num_frames):
        """Generate a fractal image influenced by a sinusoidal function."""
        x = np.sin(2 * np.pi * f / num_frames)
        fractal = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                fractal[i, j] = np.sin(x * i * j / 100.0)

        # Normalize fractal to [0, 1] and avoid division by zero
        fractal_min = fractal.min()
        fractal_max = fractal.max()
        if fractal_max - fractal_min > 1e-8:  # Use a small epsilon to prevent division by zero
            fractal = (fractal - fractal_min) / (fractal_max - fractal_min)
        else:
            fractal.fill(0.5)  # If the range is too small, fill with a neutral value

        return fractal

    def generate_density_frames(self, images, num_frames=25):
        h, w = self.grid_size
        slice_height = self.image_shape[0] // h
        slice_width = self.image_shape[1] // w

        avg_density = np.zeros((h, w))
        print(f"Calculating average density across all images...")
        for idx, image in enumerate(images):
            print(f" - Processing image {idx + 1}/{len(images)}")
            density = self.calculate_pixel_density(image)
            avg_density += density
        avg_density /= len(images)  # Average density over all images
        print(f"Average density calculation completed.")

        frames = []
        print(f"Generating frames for GIF...")
        for f in range(num_frames):
            x = np.sin(2 * np.pi * f / num_frames)
            fractal = self.generate_fractal_image(h * slice_height, w * slice_width, f, num_frames)
            frame = np.ones((h * slice_height, w * slice_width, 3)) * 255  # Blank white frame

            for i in range(h):
                for j in range(w):
                    # Dynamic parameters
                    dynamic_radius = int(avg_density[i, j] * (slice_height // 2) * (1 + 0.5 * x))  # Vary radius
                    dynamic_color = int(255 * (1 - avg_density[i, j] * (0.5 + 0.5 * x)))  # Vary grayscale color
                    center = (j * slice_width + slice_width // 2, i * slice_height + slice_height // 2)

                    # Draw the circle using PIL for better GIF support
                    pil_frame = Image.fromarray(frame.astype('uint8'))
                    draw = ImageDraw.Draw(pil_frame)
                    draw.ellipse(
                        (center[0] - dynamic_radius, center[1] - dynamic_radius,
                         center[0] + dynamic_radius, center[1] + dynamic_radius),
                        fill=(dynamic_color, dynamic_color, dynamic_color)
                    )
                    frame = np.array(pil_frame)

            # Multiply the frame by the fractal image
            frame = frame * fractal[..., np.newaxis]

            # Replace any NaN values with zero after multiplication
            frame = np.nan_to_num(frame)

            frames.append(Image.fromarray(frame.astype('uint8')))
            print(f" - Frame {f + 1}/{num_frames} generated.")
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

    # Generate the density-based frames
    frames = processor.generate_density_frames(images)

    # Save the frames as a GIF
    processor.save_gif(frames, os.path.join(output_folder, 'density_animation.gif'))

    # Display the first frame of the generated GIF
    frames[0].show()
