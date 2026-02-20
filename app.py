import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Healthcare Sentiment AI", layout="centered")

st.markdown("""
<style>
.title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#4CAF50;
}
.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏥 Healthcare Review Sentiment AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">SVM-based Intelligent Sentiment Classification System</div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

review = st.text_area("Enter Patient Review", height=150)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        vector = vectorizer.transform([review])
        prediction = model.predict(vector)[0]

        # Confidence for multi-class SVM
        confidence_values = model.decision_function(vector)
        confidence_score = round(max(confidence_values[0]), 3)

        # Probability (only works if trained with probability=True)
        try:
            probs = model.predict_proba(vector)[0]
            probability = round(max(probs) * 100, 2)
        except:
            probability = None

        if prediction == "positive":
            st.success(f"Predicted Sentiment: POSITIVE 😊")
        elif prediction == "negative":
            st.error(f"Predicted Sentiment: NEGATIVE 😡")
        else:
            st.info(f"Predicted Sentiment: NEUTRAL 😐")

        st.write(f"Confidence Score: {confidence_score}")

        if probability is not None:
            st.write(f"Probability: {probability}%")

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Review": review,
            "Prediction": prediction
        })

if st.session_state.history:
    st.markdown("### 📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
