from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re


import os  # Import the 'os' module

# ...

# Load your trained Logistic Regression model (logistic_model.pkl)
model_path = os.path.join("models", "logistic_model.pkl")  # Specify the correct path
model = joblib.load(model_path)
#model = joblib.load('c:\Users\mailt\logistic_model.pkl')


# Load the TfidfVectorizer for text preprocessing
vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")  # Specify the correct path
vectorizer = joblib.load(vectorizer_path)

# ...


app = Flask(__name__)

# Load your trained Logistic Regression model (logistic_model.pkl)
#model = joblib.load('logistic_model.pkl')

# Load the TfidfVectorizer for text preprocessing
#vectorizer = joblib.load('tfidf_vectorizer.pkl')

def preprocess_text(text):
    # Perform text preprocessing (e.g., lowercasing, removing special characters)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        news_text = request.form['news_text']
        preprocessed_text = preprocess_text(news_text)
        # Transform the preprocessed text using the TfidfVectorizer
        text_vector = vectorizer.transform([preprocessed_text])
        # Perform sentiment analysis using the model
        sentiment_label = model.predict(text_vector)[0]

        return render_template('index.html', sentiment=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
