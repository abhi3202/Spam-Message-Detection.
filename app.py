import pickle
import string
import re
from flask import Flask, render_template, request

# NLTK imports and downloads
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    tf_idf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    print("Please ensure these files are in the same directory as app.py.")
    exit()

# Initialize stemmer and stopwords
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Text preprocessing function (similar to your notebook's clean_text)
def clean_text(text):
    text = text.lower() # 1. Lowercase
    text = nltk.word_tokenize(text) # 2. Tokenize

    y = []
    for i in text:
        if i.isalnum(): # 3. Remove non-alphanumeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords_set and i not in string.punctuation: # 4. Remove stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i)) # 5. Stemming

    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        transformed_message = clean_text(message)
        vector_input = tf_idf.transform([transformed_message])
        result = model.predict(vector_input)[0]

        if result == 1:
            prediction_text = "Spam"
        else:
            prediction_text = "Not Spam"

        return render_template('index.html', prediction=prediction_text, original_message=message)

if __name__ == '__main__':
    app.run(debug=True)