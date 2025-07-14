## 🕵️‍♀️ Fake Job Posting Detection App

A machine learning Streamlit web application that detects whether a job posting is **real** or **fake** using Natural Language Processing and structured metadata.
This app was built to help users identify potentially fraudulent job offers and avoid scams online.

🔗 **Live App:** [https://fake-job-posting-detection-app.streamlit.app/](https://fake-job-posting-detection-app.streamlit.app/)

---

## 📌 Project Motivation

Online job fraud has become increasingly common, targeting vulnerable job seekers. Fake postings often lure applicants with promises of easy work-from-home jobs, high pay, or no requirements. This project aims to:

- Help users identify fake job postings.
- Showcase how machine learning can be applied in real world safety tools.
- Build a clean and accessible app for non technical users.

---

## 🧠 How the App Works

1. **User inputs** job details — title, description, company profile, requirements, and benefits.
2. The input is **cleaned and preprocessed** using custom NLP logic.
3. The text is **vectorized using TF-IDF**, and combined with:
   - `telecommuting` status
   - `has_company_logo`
   - `has_questions` flag
4. A trained **XGBoost model** then predicts whether the job is **REAL** or **FAKE**.
5. The app returns a clear result with a **confidence score**.

---

## 🎯 Features

✅ Detects job fraud based on:
- Text features (title, description, etc.)
- Structured metadata (remote job, logo, questions)

📊 Model insights:
- Trained on over 17,000 labeled job posts
- Handles imbalanced classes using **SMOTE**
- F1-score of 0.81 on fake job detection

⚡ Clean UI with:
- Real-time prediction
- Confidence percentage
- Clear success/error messages

---

## 📊 Dataset

This project uses the **Fake Job Postings** dataset from Kaggle, which contains over 17,000 job listings with a binary label for `fraudulent`.

🔗 [Fake Job Postings Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

---

## 🚀 Deployment

The app is deployed using **Streamlit Cloud** for ease of access and live interaction.

## 🧪 Model Performance
Metric	Value
Accuracy	98%
Precision	89%
Recall	73%
F1-Score	81%
🔍 The model was evaluated on a test set using a confusion matrix, and showed strong ability to catch fake jobs while minimizing false alarms.

## 📌 Example Use Cases
🧑‍💼 Job seekers who want to verify the legitimacy of a post

🕵️‍♀️ HR platforms checking job authenticity before approval

💻 Educational demo for NLP + classification pipelines

## 👩‍💻 Author
Chaviva Otabor – Data Scientist & AI Enthusiast

## 📜 License
This project is licensed under the MIT License.
Feel free to fork, reuse, and enhance it for learning or production purposes.

💡 Want to Contribute?
Feel free to open issues or submit pull requests with ideas, bug fixes, or improvements.


