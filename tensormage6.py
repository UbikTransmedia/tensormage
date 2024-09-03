import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class ImageProcessor:
    def __init__(self, image_shape, grid_size=(100, 100)):
        self.image_shape = image_shape
        self.grid_size = grid_size

    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=self.image_shape[:2])
            img_array = img_to_array(img)
            img_array = img_array.astype('float32') / 255.0  # Normalize
            images.append(img_array)
        return np.array(images)

    def calculate_pixel_density(self, image):
        h, w = self.grid_size
        density = np.zeros((h, w))
        slice_height = self.image_shape[0] // h
        slice_width = self.image_shape[1] // w

        for i in range(h):
            for j in range(w):
                slice = image[i*slice_height:(i+1)*slice_height, j*slice_width:(j+1)*slice_width]
                density[i, j] = np.mean(slice)  # Average pixel value in the slice

        return density

    def generate_density_image(self, images):
        num_images = len(images)
        h, w = self.grid_size
        avg_density = np.zeros((h, w))

        # Calculate slice dimensions based on image shape and grid size
        slice_height = self.image_shape[0] // h
        slice_width = self.image_shape[1] // w

        for image in images:
            density = self.calculate_pixel_density(image)
            avg_density += density

        avg_density /= num_images  # Calculate the average density

        # Generate the final image
        final_image = np.ones((h * slice_height, w * slice_width, 3))  # Create a blank white image
        for i in range(h):
            for j in range(w):
                radius = avg_density[i, j] * (slice_height // 1)  # Circle radius proportional to density
                circle_color = (1-avg_density[i, j], 1-avg_density[i, j], 1-avg_density[i, j])  # Grayscale
                center = (j * slice_width + slice_width // 2, i * slice_height + slice_height // 2)
                final_image = cv2.circle(final_image, center, int(radius), circle_color, -1)

        return final_image

    def save_image(self, image, filename):
        base, extension = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filename):
            filename = f"{base}_{counter}{extension}"
            counter += 1
        plt.imsave(filename, image)

# Usage of the class
if __name__ == "__main__":
    # Define parameters
    image_shape = (500, 500, 3)  # Adjust to the size of your images
    output_folder = '/home/guido/code/tensormage/img/'  # Change this to the folder where you want to save the images

    # Create an instance of the image processor
    processor = ImageProcessor(image_shape)

    # Load the images from a folder
    images = processor.load_images_from_folder('/home/guido/code/tensormage/img/solos2/')

    # Generate and save the density-based image
    density_image = processor.generate_density_image(images)
    processor.save_image(density_image, os.path.join(output_folder, 'density_image.png'))

    # Display the generated image
    plt.imshow(density_image)
    plt.title('Density-Based Image')
    plt.show()
