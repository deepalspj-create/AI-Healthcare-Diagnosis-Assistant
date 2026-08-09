import unittest
import json
import os
import sys

from train_model import train_and_save_model
from predict import SymptomPredictor
from app import app


class TestAIHealthcareDiagnosisAssistant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Train model and prepare test environment."""
        print("\n--- Training Model for Verification Suite ---")
        train_and_save_model()
        cls.predictor = SymptomPredictor()
        
        # Attach predictor to Flask app module
        import app as app_module
        app_module.predictor = cls.predictor
        cls.flask_app = app_module.app.test_client()

    def test_01_model_file_exists(self):
        """Verify model pickle was generated."""
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'healthcare_model.pkl')
        self.assertTrue(os.path.exists(model_path), "healthcare_model.pkl must exist")

    def test_02_predict_fever_cough_fatigue(self):
        """Test case: fever + cough + fatigue."""
        res = self.predictor.predict("I have fever, cough and fatigue")
        self.assertTrue(res['success'], "Prediction should succeed")
        self.assertIn(res['condition'], ["Influenza", "COVID-19", "Pneumonia", "Common Cold"])
        self.assertGreater(res['confidence'], 0)
        print(f"\n[Test Result] 'fever, cough and fatigue' -> {res['condition']} ({res['confidence']}%)")

    def test_03_predict_fever_cough(self):
        """Test case: fever + cough."""
        res = self.predictor.predict("fever and cough")
        self.assertTrue(res['success'], "Prediction should succeed")
        self.assertIn(res['condition'], ["Common Cold", "Influenza", "COVID-19", "Bronchitis"])
        print(f"[Test Result] 'fever and cough' -> {res['condition']} ({res['confidence']}%)")

    def test_04_predict_headache_sore_throat(self):
        """Test case: headache + sore throat."""
        res = self.predictor.predict("headache and sore throat")
        self.assertTrue(res['success'], "Prediction should succeed")
        self.assertIn(res['condition'], ["Strep Throat", "Sinusitis", "Common Cold"])
        print(f"[Test Result] 'headache and sore throat' -> {res['condition']} ({res['confidence']}%)")

    def test_05_predict_vomiting_stomach_pain(self):
        """Test case: vomiting + stomach pain."""
        res = self.predictor.predict("vomiting and stomach pain")
        self.assertTrue(res['success'], "Prediction should succeed")
        self.assertIn(res['condition'], ["Gastroenteritis", "Food Poisoning", "Appendicitis"])
        print(f"[Test Result] 'vomiting and stomach pain' -> {res['condition']} ({res['confidence']}%)")

    def test_06_predict_diarrhea_stomach_pain(self):
        """Test case: diarrhea + stomach pain."""
        res = self.predictor.predict("diarrhea and stomach pain")
        self.assertTrue(res['success'], "Prediction should succeed")
        self.assertIn(res['condition'], ["Gastroenteritis", "Food Poisoning", "Typhoid"])
        print(f"[Test Result] 'diarrhea and stomach pain' -> {res['condition']} ({res['confidence']}%)")

    def test_07_predict_fatigue_fever(self):
        """Test case: fatigue + fever."""
        res = self.predictor.predict("fatigue and fever")
        self.assertTrue(res['success'], "Prediction should succeed")
        self.assertIn(res['condition'], ["Influenza", "COVID-19", "Malaria", "Dengue Fever", "Typhoid"])
        print(f"[Test Result] 'fatigue and fever' -> {res['condition']} ({res['confidence']}%)")

    def test_08_empty_input_handling(self):
        """Test empty/unknown symptoms input."""
        res = self.predictor.predict("xyz123 nonsensical query")
        self.assertFalse(res['success'])
        self.assertEqual(res['error'], "No recognized symptoms detected")

    def test_09_flask_health_endpoint(self):
        """Test Flask GET /health endpoint."""
        response = self.flask_app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], "healthy")
        self.assertTrue(data['model_loaded'])

    def test_10_flask_symptoms_endpoint(self):
        """Test Flask GET /symptoms endpoint."""
        response = self.flask_app.get('/symptoms')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('Respiratory', data['categories'])

    def test_11_flask_predict_api(self):
        """Test Flask POST /predict endpoint."""
        response = self.flask_app.post(
            '/predict',
            data=json.dumps({"symptoms": "I have fever, cough and fatigue"}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('condition', data)
        self.assertIn('confidence', data)
        self.assertIn('disclaimer', data)


if __name__ == '__main__':
    unittest.main()
