import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import io

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

# Ruta para la página principal
@app.route("/", methods=["GET"])
def index():
    return "¡Bienvenido al servidor de clasificación de basura!"

# Ruta para recibir la imagen del ESP32 y devolver la predicción
@app.route("/predict", methods=["POST"])
def predict():
    # Verificar si la solicitud contiene un archivo de imagen
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400
    
    image_file = request.files["image"]
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    # Leer la imagen desde el archivo recibido
    img = Image.open(image_file)
    img_array = np.array(img)

    # Preprocesar la imagen
    try:
        processed_image = preprocess_image_with_opencv(img_array)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Realizar la predicción
    predictions = model.predict(processed_image)
    prediction = classes[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Retornar la respuesta con la predicción y confianza
    return jsonify({
        "prediction": prediction,
        "confidence": float(confidence)
    })

# Ejecutar el servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    #app.run(debug=True)

