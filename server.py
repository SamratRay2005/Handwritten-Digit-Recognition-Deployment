from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('handwritten.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(",")[1]

    # Decode and convert the image to grayscale
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')

    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert to NumPy array and invert the colors (if necessary)
    image = np.array(image)
    image = np.invert(image)  # Invert to ensure the digit is black on white background

    # Print the image data before normalization
    # print("Image data before normalization (range 0-255):")
    # print(image)

    # Normalize the image to the range [0, 1]
    image = image / 255.0

    # Print out the image data after scaling
    # print("Image data after scaling (range 0-1):")
    # print(image)

    # Reshape to match model input
    image = image.reshape(1, 28, 28, 1)

    # Save the processed image (for debugging)
    processed_image = (image.reshape(28, 28) * 255).astype(np.uint8)
    Image.fromarray(processed_image).save('processed_image.png')

    # Predict the digit
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return jsonify({'prediction': int(predicted_digit)})

