from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='butterfly_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class indices
class_indices = np.load('class_indices.npy', allow_pickle=True).item()
class_names = {v: k for k, v in class_indices.items()}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded.')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            # Process and classify butterfly image
            img = preprocess_image(file_path)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            pred_class = np.argmax(output_data[0])
            pred_label = class_names[pred_class]
            confidence = output_data[0][pred_class]

            return render_template('index.html', prediction=pred_label, confidence=f'{confidence:.2f}')
        except Exception as e:
            return render_template('index.html', error=f'Error processing image: {str(e)}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
