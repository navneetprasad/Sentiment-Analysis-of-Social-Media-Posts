# Social Media Sentiment Analyzer

An NLP-based machine learning project that classifies social media posts into Positive, Negative, or Neutral sentiments. Built with Scikit-learn and deployed via an interactive Streamlit web dashboard.

---

## Features

- **Text Preprocessing**: Cleans raw social media data by removing URLs, user mentions (`@`), numbers, and punctuation using Regular Expressions.
- **NLP Vectorization**: Utilizes a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to extract the top 5,000 most important words and convert text into numerical feature matrices.
- **Machine Learning**: Employs a Logistic Regression classification model, achieving an 81% overall accuracy with a highly sensitive 94% recall for identifying negative customer feedback.
- **Interactive UI**: A real-time Streamlit dashboard allowing users to input custom text and instantly view predicted sentiment labels.

---

## Project Structure

```text
├── Tweets.csv               # US Airline Sentiment dataset (Kaggle)
├── setup.py                 # Data preprocessing and model training script
├── sentiment_model.pkl      # Serialized Logistic Regression model
├── tfidf_vectorizer.pkl     # Serialized TF-IDF vectorizer vocabulary
├── app.py                   # Streamlit web application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
