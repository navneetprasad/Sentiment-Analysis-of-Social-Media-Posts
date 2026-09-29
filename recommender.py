from pdb import run

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the dataset
# Ensure 'Tweets.csv' is in the same folder as this script
df  = pd.read_csv('Tweets.csv')

#Isloate the columns we need
df = df[['text', 'airline_sentiment']]

# 2.Text Cleaning Function
def clean_text(text):
    text = str(text).lower() # Convert to lowercase
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove user mentions (e.g., @VirginAmerica)
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    return text.strip()

# 3. Apply the cleaning function to create a new column 
df['cleaned_text'] = df['text'].apply(clean_text)

# Define features(X) and target variable(y)
X = df['cleaned_text']
y = df['airline_sentiment']

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vectorize the text(Convert text to numerical features)
#TF-IDF scores words based on their frequency in a tweet vs rarity across the whole dataset
vectorizer = TfidfVectorizer(max_features=5000) # Limit to top 5000 features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)#Note: We only transform the test set, not fit it again

# 6. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000) # Increase max_iter to ensure convergence
model.fit(X_train_tfidf, y_train)   

# 7. Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# 8. Save the trained model and vectorizer for future use
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully.")
