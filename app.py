from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os


app = Flask(__name__)
# Allow CORS for all domains on all routes
CORS(app)

model = load_model('C:/Users/pcadmin/Documents/Erangi/The Gem Spectrum/GemApp/my_gem_model.h5')
categories = ['Amertine','Emerald']
save_directory = 'C:/Users/pcadmin/Documents/Erangi/The Gem Spectrum/GemApp/my-gem-app/images'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join(save_directory, filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})
    else:
        return jsonify({'error': 'Invalid file format'}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, port=5000)
