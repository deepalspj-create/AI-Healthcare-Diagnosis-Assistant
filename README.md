# AI Healthcare Diagnosis Assistant

A full-stack educational web application that analyzes user-entered symptoms and predicts possible health conditions using a **Random Forest Classifier** Machine Learning model.

> ⚠️ **IMPORTANT MEDICAL DISCLAIMER**: This application is built strictly for **educational and demonstration purposes** and does NOT provide real medical diagnosis or medical advice. Always consult a qualified healthcare professional for medical concerns or emergencies.

---

## 🌟 Key Features

- **Natural Language & Checklist Symptom Entry**: Type symptoms in plain English (e.g., *"I have fever, cough and fatigue"*) or click quick-select category pills (Respiratory, Digestive, Pain & General, Other).
- **Bidirectional UI Synchronization**: Selecting checklist pills auto-populates the text box and typing symptoms highlights matching pills in real-time.
- **Random Forest Machine Learning Inference**: Uses an ensemble of 100 decision trees to estimate class probabilities across 20+ medical conditions.
- **Rich Diagnostic Breakdown**:
  - Primary predicted condition with percentage confidence gauge score.
  - Risk & Severity Level indicators (`Mild`, `Moderate`, `Severe`, `Urgent`).
  - Extracted matched symptoms breakdown.
  - General condition overview & description.
  - Actionable recommended next steps.
  - Alternative candidate conditions list.
  - Conspicuous medical disclaimer warning box.
- **Modern Healthcare Aesthetic**: Clean, responsive layout with glassmorphic elements, smooth micro-animations, loading state skeletons, and dark/light contrast typography.
- **Robust REST API Backend**: Flask Python backend with input validation, symptom canonicalization, and structured JSON responses.

---

## 🛠️ Technologies Used

### Frontend
- **HTML5**: Semantic markup, accessible layout, SVG icons.
- **CSS3**: Modern layout (Flexbox & CSS Grid), CSS custom properties, glassmorphism, responsive breakpoints.
- **JavaScript (ES6+)**: Asynchronous fetch API, state management, real-time input parsing.

### Backend
- **Python 3.10+**: Core backend runtime.
- **Flask**: Micro web framework handling HTTP routes and REST API endpoints.

### Machine Learning & Data
- **Scikit-learn**: `RandomForestClassifier` with balanced class weights and cross-validation evaluation.
- **Pandas & NumPy**: Data manipulation and feature vector matrix conversion.
- **Joblib**: Model serialization and artifact packaging (`healthcare_model.pkl`).

---

## 🔬 How the Machine Learning Model Works

1. **Symptom Extraction & Preprocessing (`preprocess.py`)**:
   - Free-text input is normalized (lowercased, special characters stripped).
   - Scanned against a curated medical synonym map (e.g. *"feverish"* $\rightarrow$ `fever`, *"tummy ache"* $\rightarrow$ `stomach_pain`, *"throwing up"* $\rightarrow$ `vomiting`).
   - Mapped into a 35-dimensional binary feature vector $\mathbf{x} \in \{0, 1\}^{35}$.

2. **Model Training (`train_model.py`)**:
   - Trained on a multi-symptom dataset (`data/dataset.csv`) mapping symptom co-occurrences across 21 distinct conditions.
   - Fits a `RandomForestClassifier(n_estimators=100, random_state=42)` model.
   - Evaluated via 5-fold cross-validation.

3. **Inference & Probability Ranking (`predict.py`)**:
   - Computes class probabilities $P(C_k \mid \mathbf{x})$ using `predict_proba`.
   - Ranks the primary condition and secondary alternatives by confidence score.

---

## 🏗️ Architecture Overview

```text
AI-Healthcare-Diagnosis-Assistant/
│
├── app.py                  # Flask Web Server & REST API Routes (/predict, /symptoms, /health)
├── train_model.py          # Script to build ML dataset, train Random Forest, & export pickle
├── predict.py              # Inference module loading pickle & executing model predictions
├── preprocess.py           # NLP text normalization, synonym mapping, & vectorizer
├── test_app.py             # Automated unit & integration verification test suite
├── requirements.txt        # Python dependency manifest
├── README.md               # Project documentation
├── model/
│   └── healthcare_model.pkl# Serialized Random Forest model & metadata dictionary
├── data/
│   └── dataset.csv         # Healthcare symptom-disease dataset
├── templates/
│   └── index.html          # Responsive HTML5 web template
└── static/
    ├── style.css           # Modern Healthcare CSS design system
    └── script.js           # Client-side dynamic interactivity & API connector
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Clone / Navigate to Project Directory
```bash
cd AI-Healthcare-Diagnosis-Assistant
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML Model
Run the model training script to generate `model/healthcare_model.pkl`:
```bash
python train_model.py
```

### 5. Run Verification Suite (Optional)
Run the unit test suite to verify predictions:
```bash
python test_app.py
```

### 6. Launch the Flask Server
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 💡 Example Inputs & Outputs

### Example 1
- **Input**: `"I have fever, cough and fatigue"`
- **Predicted Condition**: `Influenza`
- **Confidence**: `86.5%`
- **Severity**: `Moderate`
- **Matched Symptoms**: `Fever`, `Cough`, `Fatigue / Tiredness`

### Example 2
- **Input**: `"vomiting and stomach pain"`
- **Predicted Condition**: `Gastroenteritis`
- **Confidence**: `92.0%`
- **Severity**: `Moderate`
- **Matched Symptoms**: `Vomiting`, `Stomach Pain / Abdominal Ache`

---

## 📜 Medical Safety Disclaimer

> **Notice**: This application is strictly an educational tool and software engineering portfolio project. It is not intended to diagnose, treat, cure, or prevent any medical condition. Users should not rely on predictions provided by this tool for medical decisions and must consult a licensed medical doctor or emergency services for real health concerns.
