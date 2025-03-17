import os
import logging
import traceback
import joblib
import torch
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
from model import predict as predict_clinical, load_model

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load CNN model
class StrokeCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 26 * 37, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 26 * 37)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = StrokeCNN()
cnn_model.load_state_dict(torch.load('stroke_cnn (1).pth', map_location='cpu'), strict=False)
cnn_model.eval()

def predict_image(image_path):
    """Predict stroke probability from image using CNN model."""
    transform = transforms.Compose([
        transforms.Resize((215, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = cnn_model(tensor)
        prob = torch.sigmoid(output).item()
    
    return prob

def safe_float(value, default=0.0):
    try:
        return float(value) if value and value.strip() else default
    except ValueError:
        return default

@app.route('/upload', methods=['POST'])
def handle_upload():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        filename = secure_filename(image_file.filename)
        if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Invalid file type"}), 400
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        clinical_data = request.form.to_dict()
        logging.info(f"Received clinical data: {clinical_data}")
        
        hypertension = 1 if clinical_data.get('hypertension', 'false').lower() == 'true' else 0
        heart_disease = 1 if clinical_data.get('heart_disease', 'false').lower() == 'true' else 0

        clinical_features = [
            safe_float(clinical_data.get('age', '0')),
            hypertension,
            heart_disease,
            safe_float(clinical_data.get('avg_glucose_level', '0')),
            safe_float(clinical_data.get('bmi', '0'))
        ]

        img_prob = predict_image(image_path)
        clinical_model = load_model()
        clinical_prob = predict_clinical(clinical_features, clinical_model)
        combined = (img_prob * 0.6) + (clinical_prob * 0.4)

        os.remove(image_path)

        return jsonify({
            "probability": round(combined * 100, 2),
            "risk_level": "high" if combined > 0.7 else 
                         "medium" if combined > 0.4 else 
                         "low"
        })

    except Exception as e:
        logging.error(f"Prediction failed: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)