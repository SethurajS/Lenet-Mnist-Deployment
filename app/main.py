# Importing the requirements

from flask import Flask, request, jsonify
from app.torch_utils import get_data_normalized, get_prediction, check_file_format

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def predict():

    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error' : 'File is not uploaded !'})
        if not check_file_format(file.filename):
            return jsonify({'error' : 'File format not supported !'})

        try:
            image_Bytes = file.read()
            image_tensor = get_data_normalized(image_Bytes)
            prediction = get_prediction(image_tensor)
            result = {'Prediction' : prediction.item(), 'Predicted Class' : str(prediction.item())}
        
            return jsonify(result)
        
        except:
            return jsonify({'error' : 'error during prediction'})