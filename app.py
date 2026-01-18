from flask import Flask, render_template, request, jsonify
import os
import sys

# Ensure the parent directory is in the system path to import EnhancedSMSFraudDetector
# This is important if app.py is in a subdirectory like 'web_app'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_sms_detector import EnhancedSMSFraudDetector

app = Flask(__name__)

# Initialize the detector and load the model
detector = None
MODEL_FILE = 'enhanced_sms_detector.pkl'

def load_model_if_not_loaded():
    global detector
    if detector is None:
        detector = EnhancedSMSFraudDetector()
        try:
            # Check if the model file exists in the current directory
            if os.path.exists(MODEL_FILE):
                detector.load_model(MODEL_FILE)
                print("Flask: Enhanced model loaded successfully!")
            else:
                print(f"Flask: Model file '{MODEL_FILE}' not found. "
                      "Please ensure you've run enhanced_sms_detector.py to train and save the model.")
                # Fallback or raise an error depending on desired behavior
                # For now, we'll continue with an uninitialized detector, and handle errors on classify
        except Exception as e:
            print(f"Flask: Error loading model: {e}")
            detector = None # Ensure detector is None if loading fails

@app.before_request
def before_request():
    load_model_if_not_loaded()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_sms():
    if detector is None:
        return jsonify({'error': 'Model not loaded. Please ensure the model file exists and is accessible.'}), 500

    data = request.get_json()
    message = data.get('message', '')
    model_name = data.get('model_name', 'ensemble') # Default to ensemble

    if not message:
        return jsonify({'error': 'No SMS message provided.'}), 400

    try:
        result = detector.predict_sms(message, model_name)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during classification: {e}'}), 500

if __name__ == '__main__':
    # When running directly, ensure the model is loaded before starting the app
    load_model_if_not_loaded()
    app.run(debug=True) # debug=True for development, set to False for production