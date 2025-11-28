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
# 3. Clean text
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♀️", layout="wide")

st.title("🕵️‍♀️ Fake Job Posting Detection")
st.write("Fill in the job details below to check if the job posting looks **real or fraudulent**.")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Job Title")
    company = st.text_area("Company Profile", height=120)
    requirements = st.text_area("Job Requirements", height=120)

with col2:
    description = st.text_area("Job Description", height=200)
    benefits = st.text_area("Job Benefits", height=120)

st.markdown("### 📊 Structured Fields")
sc1, sc2, sc3 = st.columns(3)

with sc1:
    telecommuting = st.checkbox("📡 Remote Job?")
with sc2:
    has_logo = st.checkbox("🏢 Company Has Logo?")
with sc3:
    has_questions = st.checkbox("❓ Requires Screening Questions?")

# -----------------------------
# 5. Detect Fraud
# -----------------------------
if st.button("🔍 Detect Fraud", use_container_width=True):

    # Validate input
    if not title or not description:
        st.error("❌ Job Title and Job Description are required.")
        st.stop()

    combined = f"{title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined)

    if len(cleaned.split()) < 5:
        st.error("❌ The text is too short to analyze.")
        st.stop()

    # Vectorize
    tfidf_input = vectorizer.transform([cleaned])

    if tfidf_input.nnz == 0:
        st.error("❌ Not enough meaningful words to analyze.")
        st.stop()

    # Additional text features
    num_words = len(description.split())
    num_caps = sum(1 for w in description.split() if w.isupper())

    suspicious_phrases = [
        "urgent hire", "quick money", "work from home",
        "make money fast", "no experience required",
        "instant payment", "easy earnings", "training bonus"
    ]

    suspicious_phrase = int(any(p in description.lower() for p in suspicious_phrases))

    structured_input = np.array([[
        int(telecommuting),
        int(has_logo),
        int(has_questions),
        len(cleaned.split()),
        num_words,
        num_caps,
        suspicious_phrase
    ]])

    full_input = hstack([tfidf_input, structured_input])

    # Predict
    proba = model.predict_proba(full_input)[0][1]
    threshold = 0.45
    prediction = 1 if proba >= threshold else 0

    # -----------------------------
    # Suspicious Score Meter
    # -----------------------------
    st.markdown("### 🔥 Suspicious Score")
    st.progress(int(proba * 100))

    # -----------------------------
    # Show results
    # -----------------------------
    if prediction == 0:
        st.success(f"✅ This job looks **REAL** ({(1 - proba) * 100:.2f}% confidence)")
    else:
        st.error(f"⚠️ This job appears **FAKE or Suspicious** ({proba * 100:.2f}% confidence)")

    # -----------------------------
    # Warning / Explanation
    # -----------------------------
    st.markdown("### 🧪 Analysis Breakdown")

    reasons = []

    if suspicious_phrase:
        reasons.append("Contains suspicious phrases (e.g., 'quick money', 'no experience required').")

    if telecommuting and not has_logo:
        reasons.append("Remote job **without** company logo (common scam pattern).")

    if not benefits.strip():
        reasons.append("Missing benefits, which many fake listings skip.")

    if num_caps > 5:
        reasons.append("Too many ALL-CAPS words in job description.")

    if num_words < 50:
        reasons.append("Job description is unusually short.")

    if reasons:
        st.warning("❗ **Possible red flags detected:**")
        for r in reasons:
            st.write("- " + r)
    else:
        st.info("No major red flags detected in content.")

