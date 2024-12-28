import matplotlib.pyplot as plt
import numpy as np

def show_image(image, label):
    """
    Muestra una sola imagen y su etiqueta en una figura.

    Parameters:
        image (numpy array): Imagen a mostrar.
        label (int): Etiqueta correspondiente (predicción).
    """
    fig, ax = plt.subplots(figsize=(5, 5))  # Crear una figura para la imagen
    ax.imshow(image, cmap='gray')  # Mostrar la imagen en escala de grises
    ax.text(0.5, 0.5, f"Pred: {label}", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    ax.axis('off')  # Ocultar los ejes

    plt.tight_layout()
    plt.show()
