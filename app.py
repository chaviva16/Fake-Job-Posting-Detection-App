import streamlit as st
import joblib
import re
import string
import numpy as np
from scipy.sparse import hstack

# Load model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit layout
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♀️")
st.title("🕵️‍♀️ Fake Job Posting Detection")
st.write("Fill in the job details below and get an AI prediction:")

# Input text fields
title = st.text_input("Job Title")
company = st.text_area("Company Profile", height=100)
description = st.text_area("Job Description", height=150)
requirements = st.text_area("Requirements", height=100)
benefits = st.text_area("Benefits", height=100)

# Clear YES/NO toggle-style checkboxes for structured features
st.markdown("### Structured Job Details (Toggle YES/NO):")
telecommuting = st.checkbox("📡 Is this a remote job? (YES = Checked, NO = Unchecked)")
has_logo = st.checkbox("🏢 Does the job have a company logo? (YES = Checked, NO = Unchecked)")
has_questions = st.checkbox("❓ Are there screening questions? (YES = Checked, NO = Unchecked)")

# Predict button
if st.button("Detect Fraud"):
    combined = f"{title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined)
    tfidf_input = vectorizer.transform([cleaned])

    # Structured features as 2D array
    structured_input = np.array([[int(telecommuting), int(has_logo), int(has_questions)]])
    full_input = hstack([tfidf_input, structured_input])

    # Make prediction
    prediction = model.predict(full_input)[0]
    proba = model.predict_proba(full_input)[0][prediction]

    if prediction == 0:
        st.success(f"✅ This job posting looks **REAL** ({proba*100:.2f}% confidence).")
    else:
        st.error(f"⚠️ Warning! This job posting appears to be **FAKE** ({proba*100:.2f}% confidence).")
