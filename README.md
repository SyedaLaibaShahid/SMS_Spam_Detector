# SMS Spam Detection using Machine Learning (TF-IDF + Classifiers)

A Machine Learning–based system to classify SMS messages as Spam or Ham (Not Spam).
This project uses TF-IDF feature extraction, 10 ML models, hyperparameter tuning, and a Streamlit web app for deployment.

## 🚀 Features
Text preprocessing (cleaning, lowercasing, URL removal, punctuation removal)
TF-IDF vectorization (uni + bi-grams)
10 Machine Learning models tested:
Logistic Regression
Linear SVM
RBF SVM
Naive Bayes
Random Forest
Gradient Boosting
AdaBoost
Decision Tree
KNN
SGD Classifier

GridSearchCV for hyperparameter tuning
Selection of best performing model
Hybrid detection (keyword + ML) to improve tricky spam messages
Streamlit web interface for live prediction
Saved model (best_model.pkl) + vectorizer (tfidf.pkl)


## 🛠️ Technologies Used
Python
Scikit-learn
Pandas / NumPy
TF-IDF Vectorizer
Streamlit
GridSearchCV

## 📊 Dataset
Dataset: Spam SMS Dataset (UCI / Kaggle version)
Instances: ~5,500 messages
Labels:
ham = normal messages
spam = promotional / phishing messages

## 🧠 Model Training Workflow
Load and clean dataset
Preprocess SMS text
Convert text into TF-IDF features
Train 10 ML models
Perform 5-fold cross-validation
Hyperparameter tuning (Logistic Regression, SVM, Naive Bayes)
Select best model based on F1-score
Save final model + vectorizer
