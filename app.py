from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='web')

# Load your trained model
model = load_model('best_model.h5', compile=False)

# Ensure the 'static' directory exists
os.makedirs('static', exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')  # Renders home.html

@app.route('/about')
def about():
    return render_template('about.html')  # Renders about.html

@app.route('/index')
def index():
    return render_template('index.html')  # Renders index.html

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Read the file content once
        file_content = file.read()
        
        # Save the uploaded file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join('static', filename)
        with open(filepath, 'wb') as f:
            f.write(file_content)

        # Read and process the image
        img = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "Invalid image format"
        
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        predicted_mask = (prediction[0] > 0.5).astype(np.uint8) * 255

        # Save the mask to a file
        mask_filename = 'predicted_mask.png'
        mask_filepath = os.path.join('static', mask_filename)
        success = cv2.imwrite(mask_filepath, predicted_mask)

        # Check if the file was saved successfully
        if not success:
            return "Error saving the mask image"

        # Return result.html with the correct mask URL
        return render_template('result.html', mask_url=url_for('static', filename='predicted_mask.png'))


if __name__ == '__main__':
    app.run(debug=True)

