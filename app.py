import streamlit as st
import joblib
import re
import string
import numpy as np
from scipy.sparse import hstack

# -----------------------------
# Load model + vectorizer
# -----------------------------
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♀️")
st.title("🕵️‍♀️ Fake Job Posting Detection")
st.write("Fill in the job details below and get an AI prediction:")

# Input fields
title = st.text_input("Job Title")
company = st.text_area("Company Profile", height=100)
description = st.text_area("Job Description", height=150)
requirements = st.text_area("Requirements", height=100)
benefits = st.text_area("Benefits", height=100)

# Structured features
st.markdown("### Structured Job Details (Toggle YES/NO):")
telecommuting = st.checkbox("📡 Remote Job?")
has_logo = st.checkbox("🏢 Company Logo?")
has_questions = st.checkbox("❓ Screening Questions?")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Detect Fraud"):

    # Combine all text inputs
    combined = f"{title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined)

    # ---------------------------------------------------
    # 1. Required fields check
    # ---------------------------------------------------
    if not title or not description:
        st.error("❌ Job Title and Job Description are required.")
        st.stop()

    # ---------------------------------------------------
    # 2. Minimum meaningful text check
    # ---------------------------------------------------
    if len(cleaned.split()) < 5:
        st.error("❌ Too little information provided. Please write full job details.")
        st.stop()

    # ---------------------------------------------------
    # 3. Nonsense-text detection (single random words, short garbage)
    # ---------------------------------------------------
    if re.fullmatch(r"[a-zA-Z]{1,4}", cleaned):
        st.error("❌ The text appears too short or meaningless. Please enter valid job details.")
        st.stop()

    # ---------------------------------------------------
    # 4. Transform text
    # ---------------------------------------------------
    tfidf_input = vectorizer.transform([cleaned])

    # structured inputs
    structured_input = np.array([[int(telecommuting), int(has_logo), int(has_questions)]])
    full_input = hstack([tfidf_input, structured_input])

    # ---------------------------------------------------
    # 5. Make prediction
    # ---------------------------------------------------
    prediction = model.predict(full_input)[0]
    proba = model.predict_proba(full_input)[0][prediction]

    # ---------------------------------------------------
    # 6. Output result
    # ---------------------------------------------------
    if prediction == 0:
        st.success(f"✅ This job posting looks **REAL** ({proba*100:.2f}% confidence).")
    else:
        st.error(f"⚠️ This job posting appears to be **FAKE** ({proba*100:.2f}% confidence).")
