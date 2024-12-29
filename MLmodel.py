import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
to_categorical = tf.keras.utils.to_categorical
Input = tf.keras.layers.Input

def train_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train = x_train / 255.0  # Normalize to [0, 1]
    x_test = x_test / 255.0  # Normalize to [0, 1]

    # Binarize the data (values below 0.5 become 0, above or equal 0.5 become 1), getting rid of gray gradients
    
    x_train = (x_train >= 0.5).astype(np.float32)
    x_test = (x_test >= 0.5).astype(np.float32)


    # One-hot encode the labels
    y_train_encoded = to_categorical(y_train, 10)
    y_test_encoded = to_categorical(y_test, 10)

    # Define the model using the Sequential API
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train_encoded, 
              validation_data=(x_test, y_test_encoded), 
              epochs=5, batch_size=32)

    print("Model trained successfully.")
    return model


def predict_digit(model, img_array):
    """
    Predicts the digit from an input image array.

    Parameters:
        model: Trained Keras model.
        img_array: Normalized numpy array of shape (28, 28).

    Returns:
        int: Predicted digit.
    """
    # Add batch dimension
    img_array = img_array.reshape(1, 28, 28)
    
    # Predict probabilities
    prediction = model.predict(img_array, verbose=0)
    
    # Return the class with highest probability
    return np.argmax(prediction)
