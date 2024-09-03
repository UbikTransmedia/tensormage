import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

class BasicImageNN:
    def __init__(self, image_shape, output_shape):
        self.image_shape = image_shape
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.image_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(np.prod(self.output_shape), activation='sigmoid'))
        model.add(layers.Reshape(self.output_shape))
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=self.image_shape[:2])
            img_array = img_to_array(img)
            img_array = img_array.astype('float32') / 255.0  # Normalize
            images.append(img_array)
        return np.array(images)

    def train(self, images, epochs=10):
        averaged_image = np.mean(images, axis=0)  # Average all images to create a combined target
        self.model.fit(images, np.array([averaged_image] * len(images)), epochs=epochs, shuffle=True)

    def generate_image(self):
        random_input = np.random.normal(size=(1,) + self.image_shape)
        generated_image = self.model.predict(random_input)
        return np.squeeze(generated_image, axis=0)

    def save_generated_image(self, image, filename):
        plt.imsave(filename, image)

# Uso de la clase
if __name__ == "__main__":
    # Definir los parámetros
    image_shape = (500, 500, 3)  # Cambia esto al tamaño de tus imágenes
    output_shape = (500, 500, 3)  # Salida con la misma dimensión que las imágenes de entrada

    # Crear instancia de la red neuronal
    network = BasicImageNN(image_shape, output_shape)

    # Cargar las imágenes de una carpeta
    images = network.load_images_from_folder('/home/guido/code/tensormage/img/')

    # Entrenar la red neuronal
    network.train(images, epochs=50)  # Cambia el número de épocas según sea necesario

    # Generar una nueva imagen combinada
    generated_image = network.generate_image()

    # Guardar la imagen generada en un archivo
    network.save_generated_image(generated_image, 'imagen_generada.png')

    # Mostrar la imagen generada
    plt.imshow(generated_image)
    plt.show()


# Uso de la clase
if __name__ == "__main__":
    # Definir los parámetros
    image_shape = (500, 500, 3)  # Cambia esto al tamaño de tus imágenes
    output_shape = (500, 500, 3)  # Salida con la misma dimensión que las imágenes de entrada

    # Crear instancia de la red neuronal
    network = BasicImageNN(image_shape, output_shape)

    # Cargar las imágenes de una carpeta
    images = network.load_images_from_folder('/home/guido/code/tensormage/img/')

    # Entrenar la red neuronal
    network.train(images, epochs=50)  # Cambia el número de épocas según sea necesario

    # Seleccionar una imagen semilla para generar una nueva imagen
    seed_image = images[0]  # Puedes seleccionar cualquier imagen de las cargadas

    # Generar una nueva imagen basada en la imagen semilla
    generated_image = network.generate_image(seed_image)

    # Guardar la imagen generada en un archivo
    network.save_generated_image(generated_image, 'imagen_generada.png')

    # Mostrar la imagen generada
    plt.imshow(generated_image)
    plt.show()
