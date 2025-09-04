# Spam-Message-Detection.
This mini project, titled “Spam Message Detection: A Machine Learning Approach,” focuses on the application of machine learning algorithms to distinguish between spam and legitimate (ham) messages. By training models on historical datasets of labelled messages, the system can learn the distinguishing characteristics of spam content.


📩 Spam Message Detection - A Machine Learning Approach
🔍 Overview

Spam messages are one of the biggest nuisances in digital communication, often containing phishing links, fraudulent schemes, or unwanted promotions. This project demonstrates how Machine Learning (ML) can be applied to classify text messages as either Spam or Ham (legitimate).

Using a combination of Natural Language Processing (NLP) techniques and supervised learning models, the system learns from historical SMS datasets and predicts whether a new message is spam.

🧾 Abstract

Unsolicited spam messages pose significant challenges in communication systems, leading to inconvenience, privacy issues, and financial risks. In this project, we implement and compare various ML algorithms — including Naïve Bayes, Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests — to build an efficient spam detection model.

Key highlights:

Preprocessing using tokenization, stop-word removal, stemming/lemmatization

Feature extraction using TF-IDF

Comparative analysis of multiple algorithms

Logistic Regression achieved the best performance with ~98% accuracy

⚙️ Features

📊 Exploratory Data Analysis (EDA): Understands ham vs spam distribution.

🧹 Text Preprocessing: Cleaning, stemming, and vectorization with TF-IDF.

🤖 Multiple Models: Naïve Bayes, Logistic Regression, KNN, Decision Trees, Random Forests, XGBoost, etc.

📈 Model Evaluation: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

💾 Model Persistence: Trained Logistic Regression model and vectorizer are saved with pickle for reuse.

🌐 Flask Web App: User-friendly interface (app.py) to input messages and check predictions in real-time.


🛠️ Tech Stack

Languages: Python

Frameworks & Libraries:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn

NLP: nltk, re, string

Machine Learning: scikit-learn, xgboost

Web App: flask

Serialization: pickle

📂 Project Structure
├── app.py                          # Flask web application for real-time spam detection
├── Sapm_predictor.ipynb            # Jupyter Notebook with full ML pipeline
├── model.pkl                       # Trained Logistic Regression model
├── vectorizer.pkl                  # Saved TF-IDF vectorizer
├── spam.docx                       # Abstract & summary report
├── Spam Message Detection Documentation.docx  # Full project documentation
└── README.md                       # Project documentation


🚀 Getting Started

1️⃣ Clone the Repository
git clone https://github.com/yourusername/spam-message-detection.git
cd spam-message-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook
Open Sapm_predictor.ipynb in Jupyter Notebook or JupyterLab and run all cells.

4️⃣ Run the Web Application
python app.py

Then open http://127.0.0.1:5000/ in your browser.
Type a message and get instant predictions (Spam / Not Spam).

📊 Results

Logistic Regression achieved the highest accuracy (98%) and precision, making it the most reliable model for this task.

The web app interface ensures easy testing without diving into code.



📌 Future Enhancements

Incorporating deep learning models (RNNs, LSTMs, Transformers) for better context understanding.

Real-time spam detection pipeline with APIs.

Handling non-textual spam (image/voice-based).

Deployment on Heroku, AWS, or Docker for public use.

👨‍💻 Authors

B. Abhinay

M. Bala

N. Sujith

Sk. Abdul

Under the guidance of Mr. P. Yugandhar Reddy
Department of AI & ML, Acharya Nagarjuna University

📜 License

This project is for educational and research purposes. Feel free to fork and experiment!
