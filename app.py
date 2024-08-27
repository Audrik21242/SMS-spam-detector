import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('Multinomial_Classifier.pkl', 'rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area('Enter the message')


def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing the connectors
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


if st.button("Predict spam or not"):
    text_transform = transform_text(input_sms)
    vector_input = tfidf.transform([text_transform])
    model_output = model.predict(vector_input)[0]
    if model_output == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
