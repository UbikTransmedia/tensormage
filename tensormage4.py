import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

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

    def generate_average_image(self, images):
        averaged_image = np.mean(images, axis=0)
        return averaged_image

    def generate_weighted_sum_image(self, images, weights):
        weighted_sum_image = np.tensordot(weights, images, axes=1)
        weighted_sum_image = np.clip(weighted_sum_image, 0, 1)  # Ensure the image is within valid range
        return weighted_sum_image

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.image_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(np.prod(self.image_shape), activation='sigmoid'))
        model.add(layers.Reshape(self.image_shape))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, model, images, epochs=10):
        averaged_image = np.mean(images, axis=0)  # Average all images to create a combined target
        model.fit(images, np.array([averaged_image] * len(images)), epochs=epochs, shuffle=True)

    def generate_noise_influenced_image(self, model):
        random_input = np.random.normal(size=(1,) + self.image_shape)
        generated_image = model.predict(random_input)
        return np.squeeze(generated_image, axis=0)

    def generate_shape_similarity_image(self, images):
        merged_image = np.zeros_like(images[0])
        for i in range(merged_image.shape[0]):  # Iterate over rows
            for j in range(merged_image.shape[1]):  # Iterate over columns
                pixel_values = images[:, i, j, :]
                best_match = np.argmax(np.var(pixel_values, axis=0))
                merged_image[i, j, :] = pixel_values[best_match]
        return merged_image

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

    # Generar y guardar la imagen promedio
    average_image = processor.generate_average_image(images)
    processor.save_image(average_image, os.path.join(output_folder, 'average_image.png'))

    # Generar y guardar la imagen suma ponderada
    weights = np.linspace(0.1, 1.0, len(images))  # Crear pesos de ejemplo
    weighted_sum_image = processor.generate_weighted_sum_image(images, weights)
    processor.save_image(weighted_sum_image, os.path.join(output_folder, 'weighted_sum_image.png'))

    # Construir y entrenar el modelo de red neuronal
    model = processor.build_model()
    processor.train_model(model, images, epochs=50)  # Cambia el número de épocas según sea necesario

    # Generar y guardar la imagen influenciada por ruido
    noise_influenced_image = processor.generate_noise_influenced_image(model)
    processor.save_image(noise_influenced_image, os.path.join(output_folder, 'noise_influenced_image.png'))

    # Generar y guardar la imagen basada en la similitud de formas locales
    shape_similarity_image = processor.generate_shape_similarity_image(images)
    processor.save_image(shape_similarity_image, os.path.join(output_folder, 'shape_similarity_image.png'))

    # Mostrar las imágenes generadas
    #plt.imshow(average_image)
    #plt.title('Average Image')
    #plt.show()

    #plt.imshow(weighted_sum_image)
    #plt.title('Weighted Sum Image')
    #plt.show()

    #plt.imshow(noise_influenced_image)
    #plt.title('Noise Influenced Image')
    #plt.show()

    plt.imshow(shape_similarity_image)
    plt.title('Shape Similarity Image')
    plt.show()
