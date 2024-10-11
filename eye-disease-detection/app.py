from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Serve index.html
@app.route('/')
def index():
    return render_template('index.html')

# Load the Keras model
model = tf.keras.models.load_model('pediatric_eye_disease_model.h5')

# Preprocessing function for input images
def preprocess_image(image, target_size):
    # Convert the image to RGB to ensure it has 3 channels
    image = image.convert("RGB")  
    # Resize the image to the target size
    image = image.resize(target_size)  
    # Convert the image to a numpy array
    image = np.array(image)  
    # Normalize the image (scale pixel values to [0, 1])
    image = image.astype("float32") / 255.0  
    # Add batch dimension (model expects input shape (batch_size, height, width, channels))
    image = np.expand_dims(image, axis=0)  
    return image

# Define a route for predicting the eye disease and for GET request (model info)
@app.route("/predict", methods=["POST", "GET"])
def predict_or_get_info():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400

            # Get the image from the request
            file = request.files["file"]

            # Open the image file
            image = Image.open(io.BytesIO(file.read()))

            # Preprocess the image
            processed_image = preprocess_image(image, target_size=(224, 224))

            # Make the prediction
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Assuming you have the class labels, map the prediction to the class
            class_labels = ["Disease A", "Disease B", "Disease C", "Healthy"]  # Modify this based on your classes
            result = class_labels[predicted_class]

            # Return the prediction result as JSON
            return jsonify({"prediction": result}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif request.method == 'GET':
        try:
            # Assuming you want to return model details or other data
            model_info = {
                "model_name": "Pediatric Eye Disease Model",
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "num_parameters": model.count_params()
            }

            return jsonify(model_info), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
