import os
import joblib
import numpy as np

from preprocess import (
    SYMPTOM_FEATURES,
    SYMPTOM_DISPLAY_NAMES,
    extract_symptoms_from_text,
    symptoms_to_feature_vector
)

DISCLAIMER_TEXT = (
    "⚠️ IMPORTANT MEDICAL DISCLAIMER: This application is developed strictly for educational and demonstration "
    "purposes. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice "
    "of a qualified healthcare provider with any questions regarding a medical condition. If you think you have a medical "
    "emergency, call your emergency services immediately."
)


class SymptomPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'model', 'healthcare_model.pkl')
        
        self.model_path = model_path
        self.loaded = False
        self.model = None
        self.feature_names = SYMPTOM_FEATURES
        self.classes = []
        self.disease_metadata = {}
        
        self._load_model()

    def _load_model(self):
        """Load trained scikit-learn model and artifact dictionary."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. Please run train_model.py first."
            )
        
        artifact = joblib.load(self.model_path)
        self.model = artifact['model']
        self.feature_names = artifact.get('feature_names', SYMPTOM_FEATURES)
        self.classes = artifact.get('classes', self.model.classes_.tolist())
        self.disease_metadata = artifact.get('disease_metadata', {})
        self.loaded = True

    def predict(self, user_input: str | list[str]) -> dict:
        """
        Perform prediction based on text input or list of symptom keywords.
        Returns a dictionary with prediction results, probabilities, metadata, and disclaimer.
        """
        if not self.loaded:
            self._load_model()

        # Extract symptoms from string or list
        if isinstance(user_input, list):
            raw_text = " ".join(user_input)
        else:
            raw_text = str(user_input or "")

        detected_symptoms = extract_symptoms_from_text(raw_text)

        if not detected_symptoms:
            return {
                "success": False,
                "error": "No recognized symptoms detected",
                "message": "We could not identify any symptoms from your input. Please select symptoms from the checklist or try rephrasing (e.g. 'I have fever, cough, and headache').",
                "disclaimer": DISCLAIMER_TEXT
            }

        # Convert to binary feature vector
        feature_vector = symptoms_to_feature_vector(detected_symptoms)
        X = np.array([feature_vector])

        # Get probabilities for all classes
        probabilities = self.model.predict_proba(X)[0]

        # Rank classes by probability descending
        class_prob_pairs = list(zip(self.classes, probabilities))
        class_prob_pairs.sort(key=lambda x: x[1], reverse=True)

        top_condition, top_prob = class_prob_pairs[0]
        confidence_percentage = round(float(top_prob) * 100, 1)

        # Get metadata for top condition
        meta = self.disease_metadata.get(top_condition, {
            "severity": "Unknown",
            "description": f"Condition characterized by specified symptoms.",
            "next_steps": [
                "Consult a licensed medical doctor for clinical diagnosis.",
                "Keep track of symptom duration and intensity changes."
            ]
        })

        # Format alternative diagnoses (top 2-4 candidates)
        alternatives = []
        for cond, prob in class_prob_pairs[1:4]:
            if prob > 0.05:  # filter negligible probabilities
                alternatives.append({
                    "condition": cond,
                    "confidence": round(float(prob) * 100, 1)
                })

        # Format matched symptom display names
        matched_display = [
            SYMPTOM_DISPLAY_NAMES.get(sym, sym.replace("_", " ").title())
            for sym in detected_symptoms
        ]

        return {
            "success": True,
            "condition": top_condition,
            "confidence": confidence_percentage,
            "severity": meta.get("severity", "Moderate"),
            "description": meta.get("description", ""),
            "next_steps": meta.get("next_steps", []),
            "alternatives": alternatives,
            "symptoms_matched": matched_display,
            "raw_symptoms": detected_symptoms,
            "disclaimer": DISCLAIMER_TEXT
        }
