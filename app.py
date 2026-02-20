import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page title
st.set_page_config(page_title="Healthcare Sentiment Analyzer")

st.title("🏥 Healthcare Review Sentiment Analyzer (SVM)")
st.write("Enter a healthcare review below to predict its sentiment.")

# Input box
user_input = st.text_area("Patient Review")

# Prediction button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            st.success(f"Predicted Sentiment: {prediction} 😊")
        elif prediction == "negative":
            st.error(f"Predicted Sentiment: {prediction} 😡")
        else:
            st.info(f"Predicted Sentiment: {prediction} 😐")
