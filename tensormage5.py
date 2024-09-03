import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.spatial import Delaunay

class ImageProcessor:
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=self.image_shape[:2])
            img_array = img_to_array(img)
            img_array = img_array.astype('float32') / 255.0  # Normalize
            images.append(img_array)
        return np.array(images)

    def generate_triangle_similarity_image(self, images):
        points = self.get_delaunay_points(images[0].shape[:2])
        triangles = Delaunay(points).simplices
        merged_image = np.zeros_like(images[0])

        for idx, tri in enumerate(triangles):
            print(f"Processing triangle {idx+1}/{len(triangles)}")
            tri_coords = points[tri]
            for image_idx, image in enumerate(images):
                print(f" - Comparing triangle for image {image_idx+1}/{len(images)}")
                triangle = self.extract_triangle(image, tri_coords)
                similarity = self.compute_similarity(triangle, images)
                best_match = np.argmax(similarity)
                self.apply_triangle(merged_image, tri_coords, images[best_match])

        return merged_image


    def get_delaunay_points(self, shape):
        height, width = shape
        points = np.array([[0, 0], [0, height], [width, 0], [width, height]])

        for y in range(0, height, height // 10):
            for x in range(0, width, width // 10):
                points = np.append(points, [[x, y]], axis=0)

        return points

    def extract_triangle(self, image, tri_coords):
        # Ensure tri_coords are of the correct shape and type
        tri_coords = np.array(tri_coords, dtype=np.int32).reshape((-1, 2))

        # Create a mask with the same shape as the image, initialized to zeros
        mask = np.zeros(image.shape, dtype=np.uint8)

        # Fill the triangle on the mask
        cv2.fillConvexPoly(mask, tri_coords, (1, 1, 1))

        # Ensure the mask has the same data type as the image
        mask = mask.astype(image.dtype)

        # Apply the mask to the image using bitwise_and
        return cv2.bitwise_and(image, mask)

    def compute_similarity(self, triangle, images):
        similarities = []
        for img in images:
            img_triangle = self.extract_triangle(img, triangle)
            # Specify data_range=1.0 for normalized images
            sim = ssim(triangle, img_triangle, multichannel=True, win_size=3, channel_axis=-1, data_range=1.0)
            similarities.append(sim)
        return similarities

    def apply_triangle(self, image, tri_coords, triangle):
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri_coords), (1, 1, 1))
        image[mask > 0] = triangle[mask > 0]

    def save_image(self, image, filename):
        base, extension = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filename):
            filename = f"{base}_{counter}{extension}"
            counter += 1
        plt.imsave(filename, image)

# Uso de la clase
if __name__ == "__main__":
    # Definir los parámetros
    image_shape = (500, 500, 3)  # Cambia esto al tamaño de tus imágenes
    output_folder = '/home/guido/code/tensormage/img/'  # Cambia esto a la carpeta donde guardarás las imágenes

    # Crear instancia del procesador de imágenes
    processor = ImageProcessor(image_shape)

    # Cargar las imágenes de una carpeta
    images = processor.load_images_from_folder('/home/guido/code/tensormage/img/')

    # Generar y guardar la imagen basada en la similitud de triángulos
    triangle_similarity_image = processor.generate_triangle_similarity_image(images)
    processor.save_image(triangle_similarity_image, os.path.join(output_folder, 'triangle_similarity_image.png'))

    # Mostrar la imagen generada
    plt.imshow(triangle_similarity_image)
    plt.title('Triangle Similarity Image')
    plt.show()
