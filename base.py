import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
from tonta import show_image
from MLmodel import train_model, predict_digit

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        # Scaling factor for better visibility
        self.scale = 10  # Makes the 28x28 canvas 280x280 pixels on screen

        # Canvas with scaled resolution
        self.canvas = tk.Canvas(root, width=28 * self.scale, height=28 * self.scale, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=10)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=10)

        # Internal 28x28 grayscale image
        self.image = Image.new("L", (28, 28), 255)
        self.draw_obj = ImageDraw.Draw(self.image)

        # Train the model when the app starts
        self.model = train_model()
        messagebox.showinfo("Info", "Model trained successfully!")

    def draw(self, event):
        # Scale down the coordinates to match the 28x28 resolution
        x, y = event.x // self.scale, event.y // self.scale
        radius = 1  # Thickness in 28x28 resolution
        self.canvas.create_oval(
            (x - radius) * self.scale, (y - radius) * self.scale,
            (x + radius) * self.scale, (y + radius) * self.scale,
            fill="black"
        )
        self.draw_obj.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 255)
        self.draw_obj = ImageDraw.Draw(self.image)

    def predict(self):
        # Convert internal image to numpy array
        img_array = np.array(self.image, dtype=np.float32)

        # Normalize and invert colors (MNIST format: white digits on black background)
        img_array = (255 - img_array) / 255.0

        # Predict using the model
        predicted_digit = predict_digit(self.model, img_array.reshape(1, 28, 28))


        # Show the result
        messagebox.showinfo("Prediction", f"The digit is: {predicted_digit}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
