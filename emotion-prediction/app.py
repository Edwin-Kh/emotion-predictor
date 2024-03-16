from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

# Load the trained Naive Bayes model
naive_bayes_model = joblib.load('trained_model.joblib')

# Initialize TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Function to preprocess text
def preprocess_text(text):

    # Return the preprocessed text
    return text

# Function to predict emotion
def predict_emotion(message):
    # Preprocess the input message
    preprocessed_message = preprocess_text(message)

    # Vectorize the preprocessed message
    vectorized_message = tfidf_vectorizer.transform([preprocessed_message])

    # Make prediction using the trained model
    prediction = naive_bayes_model.predict(vectorized_message)[0]

    # Map numeric label to emotion name
    emotion_mapping = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
    predicted_emotion = emotion_mapping[prediction]

    return predicted_emotion

# Route for rendering the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting emotion based on message
@app.route('/predict', methods=['POST'])
def predict():
    # Get message from request
    message = request.form['message']

    # Predict emotion
    predicted_emotion = predict_emotion(message)

    # Pass message and prediction result to template
    return render_template('index.html', message=message, prediction_result=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)
