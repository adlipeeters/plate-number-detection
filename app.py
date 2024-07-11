from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_modified import run_prediction
import base64
import os
from werkzeug.utils import secure_filename
from omegaconf import OmegaConf
from ultralytics.yolo.utils import DEFAULT_CONFIG
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'assets/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'svg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    return 'Home Page'


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    print(request.files)
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Open the image using PIL
        image = Image.open(file)
        # Resize the image
        width, height = image.size
        new_width = 400
        new_height = int((new_width / width) * height)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Save the resized image
        resized_image.save(file_path)

        detected_plates, saved_image_path = run_prediction(file_path)

        with open(saved_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = {
            'detected_plates': detected_plates,
            'saved_image_path': saved_image_path,
            'image_base64': encoded_image
        }

        os.remove(file_path)
        return jsonify(response)
    else:
        return jsonify({"error": "Invalid file type. Only images are allowed (png, jpg, jpeg, webp, svg)."}), 400


@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'World')
    return f'Hello, {name}!'


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=8080)
