import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
from tonta import show_image
from MLmodel import train_model, predict_digit  # Import train and predict functions

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=10)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=10)

        self.image = Image.new("L", (28, 28), 255)  # Grayscale image
        self.draw_obj = ImageDraw.Draw(self.image)

        # Train the model when the app starts
        self.model = train_model()
        messagebox.showinfo("Info", "Model trained successfully!")

    def draw(self, event):
        x, y = event.x, event.y
        radius = 10  # Thickness of the drawing
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black")
        self.draw_obj.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 255)
        self.draw_obj = ImageDraw.Draw(self.image)

    def predict(self):
        # Convert canvas image to MNIST format
        img = self.image.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img, dtype=np.float32)
        
        # Invert colors (MNIST background is black, digits are white)
        img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Predict using the ML pipeline
        predicted_digit = predict_digit(self.model, img_array)
        # Almacenar la imagen y la predicción
        show_image(img_array, predicted_digit)
        # Show the result
        messagebox.showinfo("Prediction", f"The digit is: {predicted_digit}")



if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
