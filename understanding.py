import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# LOAD DATA
# -----------------------------
train_df = pd.read_csv("data/reflections_train.csv")
test_df = pd.read_csv("data/reflections_test.csv")

# -----------------------------
# CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

for df in [train_df, test_df]:
    df["journal_text"] = df["journal_text"].apply(clean_text)
    
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].mean())
    df["previous_day_mood"] = df["previous_day_mood"].fillna("neutral")
    df["face_emotion_hint"] = df["face_emotion_hint"].fillna("unknown")

# -----------------------------
# TEXT FEATURES
# -----------------------------
vectorizer = TfidfVectorizer(max_features=300)
X_text_train = vectorizer.fit_transform(train_df["journal_text"])
X_text_test = vectorizer.transform(test_df["journal_text"])

# -----------------------------
# METADATA FEATURES
# -----------------------------
meta_cols = ["stress_level", "energy_level", "duration_min", "sleep_hours"]

X_meta_train = train_df[meta_cols].values
X_meta_test = test_df[meta_cols].values

# -----------------------------
# COMBINE FEATURES
# -----------------------------
from scipy.sparse import hstack

X_train = hstack([X_text_train, X_meta_train])
X_test = hstack([X_text_test, X_meta_test])

# -----------------------------
# LABEL ENCODING
# -----------------------------
le = LabelEncoder()
y_train = le.fit_transform(train_df["emotional_state"])

# -----------------------------
# MODEL TRAINING
# -----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, train_df["intensity"])

# -----------------------------
# PREDICTIONS
# -----------------------------
state_preds = clf.predict(X_test)
state_labels = le.inverse_transform(state_preds)

intensity_preds = reg.predict(X_test)
intensity_preds = np.clip(intensity_preds, 1, 5)

# -----------------------------
# CONFIDENCE + UNCERTAINTY
# -----------------------------
probs = clf.predict_proba(X_test)
confidence = np.max(probs, axis=1)
uncertain_flag = (confidence < 0.4).astype(int)

# -----------------------------
# DECISION ENGINE
# -----------------------------
def decide_action(state, stress, energy, time_of_day):
    
    if state == "overwhelmed":
        return "breathing", "now"
    
    elif state == "restless":
        return "grounding", "within_15_min"
    
    elif state == "calm":
        return "deep_work", "now"
    
    elif state == "focused":
        return "deep_work", "now"
    
    elif state == "mixed":
        return "journaling", "later_today"
    
    else:
        return "light_planning", "later_today"

# -----------------------------
# SUPPORTIVE MESSAGE (BONUS)
# -----------------------------
def generate_message(state, intensity):
    
    if state == "restless":
        return "You seem a bit restless. Try a short breathing exercise to slow things down."
    
    elif state == "overwhelmed":
        return "It looks like you're feeling overwhelmed. Take a pause and try grounding yourself."
    
    elif state == "calm":
        return "You seem calm. This is a great time to focus on meaningful work."
    
    elif state == "focused":
        return "You're in a focused state. Consider doing deep work now."
    
    elif state == "mixed":
        return "Your emotions seem mixed. Try journaling to clear your thoughts."
    
    else:
        return "You seem neutral. A light activity or planning could help."

# -----------------------------
# APPLY DECISIONS
# -----------------------------
actions = []
timings = []

for i in range(len(test_df)):
    state = state_labels[i]
    stress = test_df.iloc[i]["stress_level"]
    energy = test_df.iloc[i]["energy_level"]
    time_of_day = test_df.iloc[i]["time_of_day"]
    
    action, timing = decide_action(state, stress, energy, time_of_day)
    
    actions.append(action)
    timings.append(timing)

# -----------------------------
# CREATE OUTPUT
# -----------------------------
predictions_df = pd.DataFrame({
    "id": test_df["id"],
    "predicted_state": state_labels,
    "predicted_intensity": intensity_preds,
    "confidence": confidence,
    "uncertain_flag": uncertain_flag,
    "what_to_do": actions,
    "when_to_do": timings
})

# ADD SUPPORTIVE MESSAGE
predictions_df["support_message"] = [
    generate_message(s, i) for s, i in zip(state_labels, intensity_preds)
]

# SAVE FILE
predictions_df.to_csv("predictions.csv", index=False)

print("\n✅ predictions.csv generated!")

# -----------------------------
# ERROR ANALYSIS (PRINT)
# -----------------------------
analysis_df = test_df.copy()
analysis_df["predicted_state"] = state_labels
analysis_df["confidence"] = confidence

analysis_df = analysis_df.sort_values(by="confidence")

print("\n🔍 TOP 10 MOST UNCERTAIN CASES:\n")

for i in range(10):
    row = analysis_df.iloc[i]
    
    print(f"\nCase {i+1}")
    print("Text:", row["journal_text"])
    print("Predicted:", row["predicted_state"])
    print("Confidence:", round(row["confidence"], 3))
    print("Stress:", row["stress_level"], "Energy:", row["energy_level"])