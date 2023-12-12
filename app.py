from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Guardar el modelo después de compilar
model.save('mnist_model.h5')

# Cargar el modelo
model = tf.keras.models.load_model('mnist_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    # Obtener datos de la imagen desde la solicitud POST
    image_data = request.form['imageData'].split(',')[1]
    
    # Decodificar la imagen en formato base64
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocesamiento de la imagen para que coincida con el formato del modelo
    img = img.resize((28, 28)).convert('L')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))

    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
