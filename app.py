import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model("model/clasificador_basura_model.h5")  # Cambia por la ruta de tu modelo

# Definir las clases (ajusta según tu modelo)
classes = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Basura']

# Función para preprocesar la imagen
def preprocess_image_with_opencv(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No se encontraron objetos en la imagen.")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]
    resized_img = cv2.resize(cropped_img, (150, 150))
    normalized_img = resized_img / 255.0
    img_array = np.expand_dims(normalized_img, axis=0)

    return img_array

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    uploaded_image_path = None

    if request.method == "POST":
        # Obtener la imagen cargada
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file:
                # Verificar y crear la carpeta static si no existe
                if not os.path.exists("static"):
                    os.makedirs("static")

                # Guardar la imagen cargada temporalmente
                uploaded_image_path = os.path.join("static", image_file.filename)
                image_file.save(uploaded_image_path)

                # Preprocesar la imagen
                img = Image.open(uploaded_image_path)
                img_array = np.array(img)
                processed_image = preprocess_image_with_opencv(img_array)

                # Realizar la predicción
                predictions = model.predict(processed_image)
                prediction = classes[np.argmax(predictions)]
                confidence = np.max(predictions)

    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=uploaded_image_path)


# Ejecutar el servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    #app.run(debug=True)
