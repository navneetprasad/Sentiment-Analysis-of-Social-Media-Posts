import streamlit as st
import joblib
import re

# 1. Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 2. Recreate the exact cleaning function used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# 3. Build the Streamlit Interface
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("Social Media Sentiment Analyzer")
st.write("Enter a tweet or customer review below to predict whether its sentiment is Positive, Negative, or Neutral.")

# Text input area for the user
user_input = st.text_area("Enter your text here:", height=150, placeholder="e.g., The flight was delayed by 3 hours, absolute nightmare...")

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        # Step A: Preprocess the raw input using our function
        cleaned_input = clean_text(user_input)
        
        # Step B: Convert the text into numbers using the exact vocabulary from training
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Step C: Make the prediction
        prediction = model.predict(vectorized_input)[0]
        
        # Step D: Display the result dynamically
        if prediction == 'positive':
            st.success("🟢 Positive Sentiment Detected!")
        elif prediction == 'negative':
            st.error("🔴 Negative Sentiment Detected!")
        else:
            st.info("⚪ Neutral Sentiment Detected!")
    else:
        st.warning("Please enter some text to analyze.")