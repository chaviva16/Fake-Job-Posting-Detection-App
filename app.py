import streamlit as st
import joblib
import re
import string
import numpy as np
from scipy.sparse import hstack

# -----------------------------
# 1. Load model & vectorizer
# -----------------------------
model = joblib.load("best_model_ensemble.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# 2. Offline stopwords
# -----------------------------
stop_words = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into
is it its itself just me more most my myself no nor not of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up very was
we were what when where which while who whom why will with you your yours yourself
yourselves
""".split())

# -----------------------------
# 3. Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♀️")
st.title("🕵️‍♀️ Fake Job Posting Detection")
st.write("Fill in job details and see if the posting is suspicious:")

# Inputs
title = st.text_input("Job Title")
company = st.text_area("Company Profile", height=100)
description = st.text_area("Job Description", height=150)
requirements = st.text_area("Requirements", height=100)
benefits = st.text_area("Benefits", height=100)

st.markdown("### Structured Details:")
telecommuting = st.checkbox("📡 Remote Job?")
has_logo = st.checkbox("🏢 Company Logo?")
has_questions = st.checkbox("❓ Screening Questions?")

# -----------------------------
# 5. Detect Fraud button
# -----------------------------
if st.button("Detect Fraud"):

    combined = f"{title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined)

    # Required fields
    if not title or not description:
        st.error("❌ Job Title and Job Description are required.")
        st.stop()

    # Short text rejection
    text_len = len(cleaned.split())
    if text_len < 5:
        st.error("❌ Text too short or meaningless.")
        st.stop()

    # Transform text
    tfidf_input = vectorizer.transform([cleaned])
    if tfidf_input.nnz == 0:
        st.error("❌ Text does not contain meaningful words.")
        st.stop()

    # New textual features
    num_words = len(description.split())
    num_caps = sum(1 for w in description.split() if w.isupper())
    suspicious_phrases = ["urgent hire", "quick money", "work from home", "make money fast", "no experience required"]
    suspicious_phrase = int(any(p in description.lower() for p in suspicious_phrases))

    structured_input = np.array([[int(telecommuting), int(has_logo), int(has_questions),
                                  text_len, num_words, num_caps, suspicious_phrase]])
    full_input = hstack([tfidf_input, structured_input])

    # Prediction with threshold
    proba = model.predict_proba(full_input)[0][1]  # probability of class 1 (fake)
    threshold = 0.45
    prediction = 1 if proba >= threshold else 0

    # -----------------------------
    # Suspicious Score Meter
    # -----------------------------
    st.markdown("### Suspicious Score Meter")
    st.progress(int(proba*100))

    # -----------------------------
    # Display result
    # -----------------------------
    if prediction == 0:
        st.success(f"✅ This job looks **REAL** ({(1-proba)*100:.2f}% confidence).")
    else:
        st.error(f"⚠️ Warning! This job appears to be **FAKE** ({proba*100:.2f}% confidence).")
