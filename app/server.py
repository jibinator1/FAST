import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model import load_model, predict  # Import your model functions


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication


# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load the CNN model once when the server starts
try:
    model = load_model()
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None  # Prevents crashes if the model fails to load


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_images():
    """Handles image uploads, runs predictions, and returns results."""
    try:
        logger.info("📥 Request received")


        # Check if face image is uploaded
        if "face" not in request.files:
            logger.error("❌ Missing face image!")
            return jsonify({"error": "Missing required face image"}), 400


        # Save the uploaded face image
        file = request.files["face"]
        if file and allowed_file(file.filename):
            # Validate file size
            if file.content_length > MAX_FILE_SIZE:
                logger.error(f"❌ File size exceeds {MAX_FILE_SIZE} bytes")
                return jsonify({"error": f"File size exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB"}), 400


            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            logger.info(f"✅ Face image saved at {filepath}")
        else:
            logger.error(f"❌ Invalid file format for face image: {file.filename}")
            return jsonify({"error": "Invalid file format for face image. Only PNG, JPG, JPEG allowed."}), 400


        # Ensure the model is loaded
        if model is None:
            logger.error("❌ Model failed to load!")
            return jsonify({"error": "Model failed to load. Try restarting the server."}), 500


        # Run prediction for face image
        logger.info("🔍 Running prediction for face...")
        face_result = predict(filepath, model)
        logger.info(f"✅ Face prediction: {face_result}")


        # Clean up uploaded face image
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️ Deleted {filepath}")


        return jsonify({"results": {"face": face_result}})


    except Exception as e:
        logger.error(f"❌ Server Error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server and model status."""
    if model is None:
        return jsonify({"status": "unhealthy", "error": "Model failed to load"}), 500
    return jsonify({"status": "healthy"}), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
