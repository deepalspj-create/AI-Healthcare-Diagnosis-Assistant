import re

# Comprehensive list of recognized feature symptoms in canonical order
SYMPTOM_FEATURES = [
    "fever",
    "cough",
    "fatigue",
    "headache",
    "sore_throat",
    "vomiting",
    "stomach_pain",
    "diarrhea",
    "shortness_of_breath",
    "chest_pain",
    "nausea",
    "dizziness",
    "muscle_ache",
    "chills",
    "runny_nose",
    "loss_of_taste_smell",
    "joint_pain",
    "skin_rash",
    "abdominal_cramps",
    "loss_of_appetite",
    "sneezing",
    "high_blood_pressure",
    "frequent_urination",
    "burning_urination",
    "wheezing",
    "acid_reflux",
    "heartburn",
    "swollen_glands",
    "stiff_neck",
    "confusion",
    "sweating",
    "dehydration",
    "eye_redness",
    "sinus_pressure",
    "constipation"
]

# Display names for UI
SYMPTOM_DISPLAY_NAMES = {
    "fever": "Fever",
    "cough": "Cough",
    "fatigue": "Fatigue / Tiredness",
    "headache": "Headache",
    "sore_throat": "Sore Throat",
    "vomiting": "Vomiting",
    "stomach_pain": "Stomach Pain / Abdominal Ache",
    "diarrhea": "Diarrhea",
    "shortness_of_breath": "Shortness of Breath",
    "chest_pain": "Chest Pain",
    "nausea": "Nausea",
    "dizziness": "Dizziness / Lightheadedness",
    "muscle_ache": "Muscle / Body Ache",
    "chills": "Chills / Shivering",
    "runny_nose": "Runny / Stuffy Nose",
    "loss_of_taste_smell": "Loss of Taste / Smell",
    "joint_pain": "Joint Pain",
    "skin_rash": "Skin Rash",
    "abdominal_cramps": "Abdominal Cramps",
    "loss_of_appetite": "Loss of Appetite",
    "sneezing": "Sneezing",
    "high_blood_pressure": "High Blood Pressure",
    "frequent_urination": "Frequent Urination",
    "burning_urination": "Burning Urination",
    "wheezing": "Wheezing",
    "acid_reflux": "Acid Reflux",
    "heartburn": "Heartburn",
    "swollen_glands": "Swollen Glands / Lymph Nodes",
    "stiff_neck": "Stiff Neck",
    "confusion": "Confusion / Brain Fog",
    "sweating": "Profuse Sweating",
    "dehydration": "Dehydration / Extreme Thirst",
    "eye_redness": "Eye Redness",
    "sinus_pressure": "Sinus Pressure / Pain",
    "constipation": "Constipation"
}

# Synonyms dictionary for text extraction
SYNONYM_MAP = {
    "fever": ["fever", "feverish", "high temperature", "temperature", "pyrexia", "hot body", "running temperature", "fevers"],
    "cough": ["cough", "coughing", "dry cough", "wet cough", "coughing up", "coughes"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhaustion", "exhausted", "weakness", "weak", "lethargy", "lethargic", "sluggish", "low energy"],
    "headache": ["headache", "head pain", "head ache", "headaches", "head aches", "migraine pain", "throbbing head"],
    "sore_throat": ["sore throat", "throat pain", "scratchy throat", "swallow pain", "throat sore", "inflamed throat"],
    "vomiting": ["vomiting", "vomit", "throwing up", "puking", "puke", "threw up", "emesis"],
    "stomach_pain": ["stomach pain", "stomach ache", "stomachache", "belly pain", "abdominal pain", "tummy ache", "tummy pain", "gut pain"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose stools", "loose motion", "watery stool", "running stomach", "frequent stools"],
    "shortness_of_breath": ["shortness of breath", "breathless", "breathlessness", "difficulty breathing", "hard to breathe", "dyspnea", "gasping"],
    "chest_pain": ["chest pain", "chest tightness", "chest pressure", "pain in chest"],
    "nausea": ["nausea", "nauseous", "feeling sick", "queasy", "sick to stomach"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "lightheadedness", "vertigo", "feeling faint", "giddy"],
    "muscle_ache": ["muscle ache", "muscle pain", "body ache", "body pain", "myalgia", "sore muscles", "aching body"],
    "chills": ["chills", "shivering", "chilly", "cold chills", "rigors"],
    "runny_nose": ["runny nose", "running nose", "stuffy nose", "nasal congestion", "blocked nose", "rhinorrhea", "congested nose"],
    "loss_of_taste_smell": ["loss of taste", "loss of smell", "cannot taste", "cannot smell", "no taste", "no smell", "anosmia"],
    "joint_pain": ["joint pain", "pain in joints", "arthralgia", "aching joints", "stiff joints"],
    "skin_rash": ["skin rash", "rash", "red spots", "hives", "skin bumps", "itchy rash"],
    "abdominal_cramps": ["abdominal cramps", "stomach cramps", "cramping", "cramps"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "not hungry", "poor appetite", "loss appetite"],
    "sneezing": ["sneezing", "sneeze", "sneezes"],
    "high_blood_pressure": ["high blood pressure", "hypertension", "high bp", "elevated bp"],
    "frequent_urination": ["frequent urination", "urinating often", "peeing a lot", "peeing often", "frequent pee"],
    "burning_urination": ["burning urination", "painful urination", "burning pee", "dysuria", "stinging urination"],
    "wheezing": ["wheezing", "wheeze", "whistling breath"],
    "acid_reflux": ["acid reflux", "acid regurgitation", "gastroesophageal reflux", "acid rising"],
    "heartburn": ["heartburn", "burning in chest", "acid in throat"],
    "swollen_glands": ["swollen glands", "swollen lymph nodes", "swollen neck", "buboes"],
    "stiff_neck": ["stiff neck", "neck stiffness", "painful neck"],
    "confusion": ["confusion", "disoriented", "mental fog", "brain fog", "confused"],
    "sweating": ["sweating", "profuse sweating", "night sweats", "sweats"],
    "dehydration": ["dehydration", "dry mouth", "extreme thirst", "excessive thirst", "thirsty"],
    "eye_redness": ["eye redness", "red eyes", "pink eye", "red eye", "bloodshot eyes"],
    "sinus_pressure": ["sinus pressure", "sinus pain", "facial pain", "sinus headache"],
    "constipation": ["constipation", "hard stool", "cannot pass stool", "straining stool"]
}


def normalize_text(text: str) -> str:
    """Clean text by lowercasing and normalizing punctuation."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s,.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_symptoms_from_text(input_text: str) -> list[str]:
    """
    Extract recognized symptom keys from natural language text or comma-separated lists.
    Uses string matching against synonym list.
    """
    if not input_text:
        return []
    
    cleaned = normalize_text(input_text)
    detected_symptoms = set()

    for feature_key, synonyms in SYNONYM_MAP.items():
        for synonym in synonyms:
            # Match whole phrase or word in cleaned text
            pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(pattern, cleaned):
                detected_symptoms.add(feature_key)
                break

    return sorted(list(detected_symptoms))


def symptoms_to_feature_vector(detected_symptoms: list[str]) -> list[int]:
    """
    Convert a list of detected symptom keys into a binary feature vector matching SYMPTOM_FEATURES order.
    """
    symptom_set = set(detected_symptoms)
    return [1 if feature in symptom_set else 0 for feature in SYMPTOM_FEATURES]
