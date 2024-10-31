# SMS Spam Detector
This project focuses on creating a spam detection model using traditional machine learning techniques. Various models are tested to determine the most accurate one for deployment, ultimately selecting the best-performing algorithm.

# Table of Contents
1. Project Overview
2. Dataset
3. Data Preprocessing
4. Model Selection
5. Technologies Used
6. Deployment

# Project Overview
The aim of this project is to build a machine learning model that accurately classifies SMS messages as either "spam" or "ham" (non-spam). Using exploratory data analysis (EDA), preprocessing, model building, and evaluation, we identify the optimal model for deployment in a simple user interface.

# Dataset
The dataset consists of 5,572 SMS messages, out of which 673 are labeled as spam, and the remaining messages are labeled as ham. This dataset is stored in a CSV file format.

# Data Preprocessing
The following steps were taken to preprocess the data:
 * Label Encoding: The target column is encoded, with spam messages labeled as 1 and ham messages as 0.
 * Text Processing:
 * Tokenization, stemming, and removal of stopwords and punctuation are applied to the text data.
 * The processed text is then transformed into numerical features using TF-IDF Vectorization.

# Model Selection
The data is split into training and testing sets, after which we apply several machine learning models:
 * Naive Bayes Classifiers: Given the binary classification nature of this project, we use Naive Bayes models, including:
    - Multinomial Naive Bayes
    - Gaussian Naive Bayes
    - Bernoulli Naive Bayes
 * Model Performance: After evaluation, Multinomial Naive Bayes is selected as the best-performing model for its high accuracy in classifying SMS messages.

# Technologies Used
 * Programming Language: Python
 * Data Manipulation and Preprocessing: NumPy, Pandas, NLTK
 * Text Vectorization: TF-IDF Vectorizer from Scikit-Learn
 * Machine Learning Models: Scikit-Learn for Naive Bayes classifiers
 * Deployment: Streamlit

# Deployment
The chosen model, along with the TF-IDF vectorizer, is saved for deployment. A simple frontend interface is created using Streamlit, where users can input text messages to be classified as either spam or ham.

