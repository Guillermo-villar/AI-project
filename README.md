# Personal AI development proyect - Digit detection in Python

This proyect uses Tensorflow to create a Machine Learning pipeline able to detect hand written digits, learning from a MINST digit dataset. The graphical interface uses Tkinter.

## Description

This repository contains a digit classification model based on the MNIST dataset. The model is trained to recognize handwritten digits from 0 to 9. The application includes a graphical interface built with Python that allows the user to draw a digit and predict its value using the trained model.

## Characteristics

- Digit classification model trained with TensorFlow.
- Python graphical interface using Tkinter to draw handwritten digits.
- Real-time predictions using the trained model.
- Visualization of results, displaying the predicted digit and its probability.

## Instalation

To test and try this code on your own machine, follow along the instructions:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/digit-prediction-ia.git
    cd digit-prediction-ia
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    venv\Scripts\activate     # On Windows
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Train the prediction model by running the following command:

    ```bash
    python train.py
    ```

2. Launch the graphical interface to draw and predict a digit:

    ```bash
    python app.py
    ```

3. The app will allow you to draw a hand written digit and test the model with your own hands!

## Proyect structure

    **ML_pipeline.py** : This file will hold all Machine Learning logic, as well as the code for the training of the model and the function used to predict 
    **App.py** : This file will hold the graphical interface and house the actual running of my app. 
