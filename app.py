import os
import sys
from flask import Flask, render_template, request, jsonify

from preprocess import SYMPTOM_FEATURES, SYMPTOM_DISPLAY_NAMES
from predict import SymptomPredictor, DISCLAIMER_TEXT

app = Flask(__name__)

# Initialize predictor instance
predictor = None
try:
    predictor = SymptomPredictor()
    print("SymptomPredictor loaded successfully.")
except Exception as e:
    print(f"Warning: Could not initialize SymptomPredictor at startup: {e}")


@app.route('/')
def index():
    """Render main web page."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global predictor
    if not predictor or not predictor.loaded:
        try:
            predictor = SymptomPredictor()
        except Exception:
            pass
    model_status = predictor.loaded if predictor else False
    return jsonify({
        "status": "healthy",
        "model_loaded": model_status,
        "disclaimer": DISCLAIMER_TEXT
    })


@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return categorized symptom list for interactive UI tags/checklist."""
    categories = {
        "Respiratory": ["cough", "sore_throat", "shortness_of_breath", "runny_nose", "sneezing", "wheezing", "sinus_pressure"],
        "Digestive": ["vomiting", "stomach_pain", "diarrhea", "nausea", "abdominal_cramps", "loss_of_appetite", "acid_reflux", "heartburn", "constipation"],
        "Pain & General": ["fever", "fatigue", "headache", "dizziness", "muscle_ache", "chills", "joint_pain", "chest_pain", "stiff_neck", "sweating", "dehydration"],
        "Other Symptoms": ["loss_of_taste_smell", "skin_rash", "high_blood_pressure", "frequent_urination", "burning_urination", "swollen_glands", "confusion", "eye_redness"]
    }

    categorized = {}
    for cat_name, keys in categories.items():
        categorized[cat_name] = [
            {
                "key": key,
                "label": SYMPTOM_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
            }
            for key in keys if key in SYMPTOM_FEATURES
        ]

    return jsonify({
        "success": True,
        "categories": categorized
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    POST /predict
    Payload: { "symptoms": "fever, cough and fatigue" } OR { "symptoms": ["fever", "cough"] }
    Returns prediction JSON object.
    """
    global predictor
    
    if not predictor or not predictor.loaded:
        try:
            predictor = SymptomPredictor()
        except Exception as e:
            return jsonify({
                "success": False,
                "error": "Model Error",
                "message": f"ML model is not loaded. Please run 'python train_model.py' to generate the model artifact. Error: {str(e)}"
            }), 500

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid Request",
                "message": "Request body must be a valid JSON object containing a 'symptoms' field."
            }), 400

        user_symptoms = data.get('symptoms')
        if not user_symptoms:
            return jsonify({
                "success": False,
                "error": "Empty Input",
                "message": "Please enter or select at least one symptom to generate a prediction."
            }), 400

        # Run prediction pipeline
        result = predictor.predict(user_symptoms)

        if not result.get("success"):
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Server Error",
            "message": f"An internal error occurred during prediction processing: {str(e)}"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting AI Healthcare Diagnosis Assistant on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
