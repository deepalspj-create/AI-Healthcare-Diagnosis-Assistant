import urllib.request
import urllib.error
import json

BASE_URL = "http://127.0.0.1:5000"

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'PythonAuditTest'})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.getcode(), resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')

def post(url, data_dict):
    json_bytes = json.dumps(data_dict).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=json_bytes,
        headers={'Content-Type': 'application/json', 'User-Agent': 'PythonAuditTest'}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.getcode(), resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')

def test_endpoints():
    print("=== Starting Full Audit Test Suite (Standard Library) ===")

    # 1. Test GET /
    status, body = get(f"{BASE_URL}/")
    assert status == 200, f"GET / failed with {status}"
    assert "AI Healthcare Diagnosis Assistant" in body, "Index HTML missing title"
    print("[PASS] GET / HTML page loaded successfully.")

    # 2. Test GET /health
    status, body = get(f"{BASE_URL}/health")
    assert status == 200, f"GET /health failed with {status}"
    data = json.loads(body)
    assert data.get("status") == "healthy", "Health status not healthy"
    assert data.get("model_loaded") is True, "Model not loaded in health endpoint"
    print("[PASS] GET /health check verified.")

    # 3. Test GET /symptoms
    status, body = get(f"{BASE_URL}/symptoms")
    assert status == 200, f"GET /symptoms failed with {status}"
    data = json.loads(body)
    assert data.get("success") is True, "Symptoms response success false"
    assert "Respiratory" in data.get("categories", {}), "Respiratory category missing"
    assert "Digestive" in data.get("categories", {}), "Digestive category missing"
    print("[PASS] GET /symptoms categories endpoint verified.")

    # 4. Test Predict Endpoints with multiple combinations
    test_cases = [
        "fever, cough and fatigue",
        "fever and cough",
        "headache and sore throat",
        "vomiting and stomach pain",
        "diarrhea and stomach pain",
        "fatigue and fever",
        "dizziness and high blood pressure",
        "burning urination and frequent pee",
        "heartburn and acid in throat",
        "shortness of breath and wheezing"
    ]

    print("\n--- Testing Model Predictions across 10 Symptom Combinations ---")
    for symptoms in test_cases:
        status, body = post(f"{BASE_URL}/predict", {"symptoms": symptoms})
        assert status == 200, f"Predict failed for '{symptoms}' with status {status}"
        res = json.loads(body)
        assert res.get("success") is True, f"Prediction failed for '{symptoms}'"
        assert res.get("condition") is not None, "Condition missing"
        assert res.get("confidence") > 0, "Confidence score 0"
        assert res.get("disclaimer") is not None, "Disclaimer missing"
        print(f" -> Input: '{symptoms}' => Condition: '{res['condition']}' ({res['confidence']}%) | Severity: {res['severity']}")

    # 5. Test Invalid & Empty Inputs
    print("\n--- Testing Invalid & Empty Input Handlers ---")
    status, body = post(f"{BASE_URL}/predict", {"symptoms": ""})
    assert status == 400, f"Empty symptoms should return 400, got {status}"
    data = json.loads(body)
    assert data.get("success") is False, "Empty symptoms should return success=false"
    print("[PASS] Empty input gracefully rejected with 400 Bad Request.")

    status, body = post(f"{BASE_URL}/predict", {"symptoms": "random gibberish qwerty12345"})
    assert status == 400, f"Unknown symptoms should return 400, got {status}"
    data = json.loads(body)
    assert data.get("success") is False, "Unknown symptoms should return success=false"
    print("[PASS] Unknown symptoms gracefully rejected with 400 Bad Request.")

    print("\n=== ALL AUDIT TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_endpoints()
