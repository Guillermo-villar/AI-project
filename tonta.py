import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Cargar MNIST una sola vez para reutilizar

(x_train, y_train), (_, _) = mnist.load_data()
x_train_binary = (x_train > 128).astype(np.uint8) * 255
def show_image(image, label):
    """
    Muestra una imagen proporcionada y otra al azar del conjunto MNIST con sus etiquetas.

    Parameters:
        image (numpy array): Imagen a mostrar.
        label (int): Etiqueta correspondiente (predicción).
    """
    # Seleccionar una imagen y etiqueta aleatoria de MNIST
    random_index = np.random.randint(0, len(x_train))
    random_image = x_train_binary[random_index]
    random_label = y_train[random_index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Crear dos subgráficos

    # Mostrar la imagen proporcionada
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Predicción: {label}")
    axes[0].axis('off')

    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    examples = {}
    for image, label in zip(x_train_binary, y_train):
        if label not in examples:
            examples[label] = image
        if len(examples) == 10:  # Ya tenemos un ejemplo por clase
            break
    for i in range(10):
        axes[i].imshow(examples[i], cmap='gray')
        axes[i].set_title(f"Clase: {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
