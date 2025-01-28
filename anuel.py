import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tkinter import messagebox

# Cargar el modelo entrenado
model = tf.keras.models.load_model('mnist_model.h5')

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Reconocimiento de dígitos")
        
        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.button_predict = tk.Button(master, text="Predecir", command=self.predict)
        self.button_predict.pack()
        
        self.button_clear = tk.Button(master, text="Limpiar", command=self.clear_canvas)
        self.button_clear.pack()
        
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black', outline='black')
        self.draw.line([x, y, x, y], fill='black', width=10)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Procesar la imagen para la predicción
        img = self.image.resize((28, 28))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=[0, -1])  # Añadir batch y canal
        
        # Realizar la predicción
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        
        # Mostrar el resultado
        tk.messagebox.showinfo("Predicción", f"El dígito es: {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()