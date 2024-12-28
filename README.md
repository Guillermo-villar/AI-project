# Predicción de Dígitos con TensorFlow

Este proyecto utiliza TensorFlow para crear un modelo de Deep Learning capaz de predecir dígitos escritos a mano, utilizando una base de datos de imágenes de dígitos.

## Descripción

Este repositorio contiene un modelo de clasificación de dígitos basado en la base de datos **MNIST**. El modelo se entrena para reconocer dígitos escritos a mano del 0 al 9. La aplicación incluye una interfaz gráfica en Python que permite al usuario dibujar un dígito y predecir su valor utilizando el modelo entrenado.

## Características

- Modelo de clasificación de dígitos entrenado con TensorFlow.
- Interfaz gráfica en Python con **Tkinter** para dibujar dígitos a mano alzada.
- Predicción en tiempo real utilizando el modelo entrenado.
- Visualización de resultados con la predicción del dígito y la probabilidad.

## Instalación

Para ejecutar este proyecto en tu entorno local, sigue los pasos a continuación.

1. Clona el repositorio:

    ```bash
    git clone https://github.com/tu-usuario/ia-prediccion-digitos.git
    cd ia-prediccion-digitos
    ```

2. Crea y activa un entorno virtual:

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/Mac
    venv\Scripts\activate     # En Windows
    ```

3. Instala las dependencias necesarias:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Entrena el modelo de predicción ejecutando el siguiente comando:

    ```bash
    python train.py
    ```

2. Inicia la interfaz gráfica para dibujar y predecir un dígito:

    ```bash
    python app.py
    ```

3. La aplicación te permitirá dibujar un dígito y hacer una predicción basada en el modelo entrenado.

## Estructura del Proyecto

