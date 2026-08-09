import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from preprocess import SYMPTOM_FEATURES

# Medical condition details database for rich frontend results
DISEASE_METADATA = {
    "Influenza": {
        "severity": "Moderate",
        "description": "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. Symptoms range from mild to severe and include fever, body aches, cough, and fatigue.",
        "next_steps": [
            "Rest adequately and stay well hydrated with water, herbal teas, or warm broths.",
            "Consider over-the-counter pain relievers or fever reducers after consulting a pharmacist.",
            "Isolate from vulnerable individuals to prevent spread.",
            "Seek immediate medical attention if you experience severe breathlessness or persistent chest pain."
        ]
    },
    "Common Cold": {
        "severity": "Mild",
        "description": "A viral infection of the upper respiratory tract causing runny nose, sore throat, mild cough, and sneezing. Usually resolves within 7 to 10 days.",
        "next_steps": [
            "Get plenty of rest and drink abundant fluids.",
            "Use warm saline gargles to soothe throat irritation.",
            "Over-the-counter decongestants may help relieve nasal blockage.",
            "Consult a doctor if symptoms last longer than 10-14 days or worsen significantly."
        ]
    },
    "COVID-19": {
        "severity": "Moderate to High",
        "description": "An infectious respiratory disease caused by SARS-CoV-2. Key symptoms include fever, dry cough, severe fatigue, loss of taste/smell, and breathlessness.",
        "next_steps": [
            "Isolate immediately to protect family and community members.",
            "Perform a rapid antigen or RT-PCR test to confirm diagnosis.",
            "Monitor blood oxygen saturation levels (SpO2) regularly with a pulse oximeter.",
            "Seek emergency care immediately if SpO2 drops below 94% or if you experience chest pressure."
        ]
    },
    "Gastroenteritis": {
        "severity": "Moderate",
        "description": "An inflammation of the stomach and intestines caused by viral or bacterial infection, commonly leading to diarrhea, vomiting, nausea, and abdominal pain.",
        "next_steps": [
            "Prevent dehydration by sipping Oral Rehydration Solution (ORS) or clear liquids in small quantities.",
            "Follow a gentle diet (BRAT diet: Bananas, Rice, Applesauce, Toast) once vomiting subsides.",
            "Avoid dairy, spicy foods, caffeine, and alcohol.",
            "Seek medical assistance if high fever, blood in stool, or inability to retain fluids persists."
        ]
    },
    "Strep Throat": {
        "severity": "Moderate",
        "description": "A bacterial throat infection caused by Group A Streptococcus resulting in sudden severe sore throat, fever, and painful swallowing.",
        "next_steps": [
            "Consult a healthcare professional for a rapid strep test or throat culture.",
            "If bacterial infection is confirmed, take the full course of prescribed antibiotics as directed.",
            "Gargle with warm salt water and rest your voice.",
            "Drink cool liquids and eat soft foods."
        ]
    },
    "Migraine": {
        "severity": "Moderate",
        "description": "A neurological condition characterized by intense, throbbing headache typically on one side of the head, often accompanied by nausea and sensitivity to light/sound.",
        "next_steps": [
            "Rest in a quiet, dark, cool room with your eyes closed.",
            "Apply a cold compress to your forehead or temples.",
            "Stay hydrated and avoid known migraine triggers (bright lights, specific foods, stress).",
            "Consult a doctor for specialized prescription migraine therapies if headaches recur."
        ]
    },
    "Sinusitis": {
        "severity": "Mild to Moderate",
        "description": "Inflammation or swelling of the tissue lining the sinuses, causing facial pain/pressure, nasal congestion, thick nasal discharge, and headache.",
        "next_steps": [
            "Inhale steam or use a saline nasal spray/irrigation to keep sinus passages moist.",
            "Apply a warm compress over your forehead and cheekbones.",
            "Drink plenty of fluids to thin out mucus.",
            "Consult a physician if symptoms worsen after 7-10 days or if severe facial pain develops."
        ]
    },
    "Bronchitis": {
        "severity": "Moderate",
        "description": "Inflammation of the bronchial tubes that carry air to your lungs, causing persistent cough with mucus, chest tightness, wheezing, and fatigue.",
        "next_steps": [
            "Avoid smoke, dust, air pollution, and cold environmental air.",
            "Use a room humidifier to loosen respiratory secretions.",
            "Stay well hydrated with warm liquids.",
            "See a healthcare provider if you cough up blood, have high fever, or experience shortness of breath."
        ]
    },
    "Food Poisoning": {
        "severity": "Moderate",
        "description": "Illness caused by eating contaminated food containing harmful pathogens, resulting in rapid-onset vomiting, stomach cramps, diarrhea, and fever.",
        "next_steps": [
            "Stop eating solid foods for a few hours to allow stomach rest.",
            "Sip electrolyte fluids or clear broth continuously to prevent severe dehydration.",
            "Do not take anti-diarrheal medications without doctor approval, as they can delay toxin clearance.",
            "Seek urgent emergency care if you notice signs of severe dehydration, confusion, or extreme weakness."
        ]
    },
    "Pneumonia": {
        "severity": "High / Urgent",
        "description": "An infection that inflames air sacs in one or both lungs, which may fill with fluid or pus, causing severe cough, fever, chills, and shortness of breath.",
        "next_steps": [
            "Seek urgent evaluation by a medical provider or pulmonologist.",
            "A chest X-ray or medical lab evaluation is recommended for definitive diagnosis.",
            "Follow prescribed medication regimens strictly (antibiotics/antivirals).",
            "Seek immediate emergency care if breathing becomes severely labored."
        ]
    },
    "Urinary Tract Infection": {
        "severity": "Moderate",
        "description": "An infection in any part of the urinary system, causing burning sensation during urination, frequent urge to urinate, and lower abdominal discomfort.",
        "next_steps": [
            "Consult a healthcare provider promptly for a urine analysis.",
            "Drink plenty of water to flush bacteria from the urinary tract.",
            "Complete any prescribed antibiotic regimen entirely.",
            "Seek urgent care if you develop high fever, back/flank pain, or chills (possible kidney involvement)."
        ]
    },
    "GERD / Acid Reflux": {
        "severity": "Mild to Moderate",
        "description": "Gastroesophageal reflux disease occurs when stomach acid frequently flows back into the tube connecting your mouth and stomach, causing heartburn and regurgitation.",
        "next_steps": [
            "Eat smaller, more frequent meals and avoid lying down for 3 hours after eating.",
            "Avoid trigger foods such as fatty foods, spicy dishes, citrus, chocolate, and caffeine.",
            "Elevate the head of your bed during sleep.",
            "Consult a doctor if heartburn occurs more than twice a week."
        ]
    },
    "Asthma": {
        "severity": "Moderate to High",
        "description": "A chronic condition in which airways narrow and swell, producing extra mucus, leading to wheezing, chest tightness, shortness of breath, and coughing.",
        "next_steps": [
            "Use your prescribed rescue inhaler (such as Albuterol) as directed.",
            "Move away from known triggers (dust, allergens, smoke, exercise in cold air).",
            "Sit upright and stay calm during a mild flare-up.",
            "Seek EMERGENCY medical care immediately if inhalers provide no relief or breathing is severely impaired."
        ]
    },
    "Allergic Rhinitis / Allergy": {
        "severity": "Mild",
        "description": "An allergic response causing sneezing, itchy eyes, runny nose, and sinus pressure when exposed to environmental allergens like pollen, dust, or pet dander.",
        "next_steps": [
            "Identify and minimize exposure to specific allergen triggers.",
            "Consider over-the-counter antihistamines or nasal corticosteroid sprays.",
            "Keep indoor air clean using HEPA filters and closing windows during high pollen counts.",
            "Consult an allergist for long-term allergy management."
        ]
    },
    "Malaria": {
        "severity": "High / Urgent",
        "description": "A mosquito-borne infectious disease caused by a parasite, leading to high fever, severe chills, sweating, headache, and muscle aches.",
        "next_steps": [
            "Seek IMMEDIATE medical evaluation at an emergency clinic or hospital.",
            "A blood smear test or rapid diagnostic test (RDT) is necessary.",
            "Complete antimalarial treatment as prescribed by a medical professional.",
            "Ensure rest and strict hydration."
        ]
    },
    "Dengue Fever": {
        "severity": "High / Urgent",
        "description": "A mosquito-borne viral infection causing high fever, intense joint/bone pain ('breakbone fever'), skin rash, and headache.",
        "next_steps": [
            "Seek immediate healthcare evaluation to monitor blood platelet counts.",
            "Maintain strict bed rest and drink electrolyte fluids constantly.",
            "Avoid NSAIDs (like aspirin or ibuprofen) as they increase bleeding risks; use paracetamol only if advised.",
            "Go to the emergency room immediately if warning signs appear (bleeding gums, persistent vomiting, severe belly pain)."
        ]
    },
    "Typhoid": {
        "severity": "High / Urgent",
        "description": "A bacterial infection caused by Salmonella typhi, presenting with prolonged high fever, fatigue, stomach pain, headache, and diarrhea or constipation.",
        "next_steps": [
            "Seek urgent medical evaluation and blood/stool culture tests.",
            "Take prescribed full antibiotic course under strict medical supervision.",
            "Drink safe, boiled or bottled water and eat warm, hygienic foods.",
            "Monitor body temperature and stay strictly hydrated."
        ]
    },
    "Appendicitis": {
        "severity": "Urgent / Emergency",
        "description": "An inflammation of the appendix causing severe lower right abdominal pain, nausea, vomiting, fever, and loss of appetite.",
        "next_steps": [
            "SEEK IMMEDIATE EMERGENCY MEDICAL ATTENTION (Go to Emergency Room).",
            "Do NOT eat, drink, or take laxatives/painkillers before being evaluated by a surgeon.",
            "Early medical surgical evaluation is critical to prevent appendix rupture."
        ]
    },
    "Hypertension": {
        "severity": "Moderate to High",
        "description": "Consistently high blood pressure against artery walls, often asymptomatic but may cause headache, dizziness, or chest tightness when elevated.",
        "next_steps": [
            "Monitor blood pressure readings using a validated cuff.",
            "Reduce dietary sodium intake and engage in regular low-impact exercise.",
            "Consult a physician for formal diagnosis and potential antihypertensive medication.",
            "Seek immediate emergency care if high BP is accompanied by severe headache, chest pain, or vision changes."
        ]
    },
    "Diabetes": {
        "severity": "Moderate to High",
        "description": "A metabolic disorder characterized by high blood glucose levels, causing frequent urination, excessive thirst, persistent fatigue, and blurred vision.",
        "next_steps": [
            "Consult an endocrinologist or primary care physician for fasting blood sugar and HbA1c testing.",
            "Follow a balanced low-glycemic dietary plan.",
            "Engage in regular physical activity and monitor blood glucose levels.",
            "Discuss medication or insulin management with your doctor."
        ]
    },
    "Dehydration": {
        "severity": "Mild to Moderate",
        "description": "Occurs when body water loss exceeds fluid intake, resulting in extreme thirst, dry mouth, dizziness, fatigue, and dark urine.",
        "next_steps": [
            "Drink water or oral rehydration fluids slowly and continuously.",
            "Rest in a cool, shaded environment.",
            "Avoid caffeinated or sugary beverages.",
            "Seek urgent medical attention if unable to keep fluids down or if experiencing confusion or fainting."
        ]
    }
}


def train_and_save_model():
    """Train Random Forest Classifier and save artifact dictionary."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'data', 'dataset.csv')
    model_dir = os.path.join(base_dir, 'model')
    model_path = os.path.join(model_dir, 'healthcare_model.pkl')

    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Features and target
    X = df[SYMPTOM_FEATURES]
    y = df['disease']

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. Unique diseases: {y.nunique()}")

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )

    rf_model.fit(X.values, y.values)

    # Perform dynamic cross-validation based on minimum class size
    min_class_samples = int(y.value_counts().min())
    cv_splits = max(2, min(5, min_class_samples))
    if min_class_samples >= 2:
        cv_scores = cross_val_score(rf_model, X.values, y.values, cv=cv_splits)
        print(f"{cv_splits}-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    else:
        print("Cross-validation skipped (some classes have fewer than 2 samples).")

    # Classification report
    y_pred = rf_model.predict(X.values)
    print("\nTraining Set Evaluation Report:")
    print(classification_report(y.values, y_pred, zero_division=0))

    # Package pipeline artifact
    model_artifact = {
        'model': rf_model,
        'feature_names': SYMPTOM_FEATURES,
        'classes': rf_model.classes_.tolist(),
        'disease_metadata': DISEASE_METADATA
    }

    joblib.dump(model_artifact, model_path)
    print(f"Model successfully trained and saved to: {model_path}")


if __name__ == '__main__':
    train_and_save_model()
