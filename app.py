from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import h5py
from keras.models import load_model

app = Flask(__name__)
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Load your own model
def load_cifar10_model():
    model = load_model('model.h5')
    return model

cifar10_model = load_cifar10_model()

# Image preprocessing function
def preprocess_image(image):
    # Implement your image preprocessing logic here
    # For example, resize the image to match the input size of your model
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']

    if image.filename == '':
        return redirect(request.url)

    if image:
        # Read the image file
        img = Image.open(image)

        # Preprocess the image
        img_array = preprocess_image(img)

        img_path = 'static/temp_image.png'
        img.save(img_path)

        # Make a prediction
        prediction = cifar10_model.predict(np.expand_dims(img_array, axis=0))


        # Process the prediction results as needed

        return render_template('result.html', prediction=prediction, class_names=class_names,image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
