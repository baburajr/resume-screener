
import pandas as pd
import streamlit as st
import pickle
import json
import re
import string
from nltk.tokenize import word_tokenize
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('punkt')

# setting of english stop words
setofStopWords = set(stopwords.words('english')+['``', "''"])

# function to clean user resume text


def cleanResume(resumeData):
    resumeData = re.sub('http\S+\s*', ' ', resumeData)  # remove URLs
    resumeData = re.sub('RT|cc', ' ', resumeData)  # remove RT and cc
    resumeData = re.sub('#\S+', '', resumeData)  # remove hashtags
    resumeData = re.sub('@\S+', '  ', resumeData)  # remove mentions
    resumeData = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeData)  # remove punctuations
    resumeData = re.sub(r'[^\x00-\x7f]', r' ', resumeData)
    resumeData = re.sub('\s+', ' ', resumeData)  # remove extra whitespace
    resumeData = resumeData.lower()  # convert to lowercase
    resumeDataTokens = word_tokenize(resumeData)  # tokenize
    # remove stopwords
    filteredData = [w for w in resumeDataTokens if not w in setofStopWords]
    return ' '.join(filteredData)


st.markdown('''
# Resume Screening App using Python, Tensorflow and Streamlit

### A simple resume scanner to check the job probability of a CV based on skills.

**Credits**
- App built by [Baburaj R](https://linkedin.com/in/baburajr88/)
- Built in `Python` using `Tensorflow`, `Keras`, `NLTK` and `Streamlit`
''')

user_input = st.text_area("Paste your resume below")
# cleaning the user input
cleaned_input = cleanResume(user_input)
# user_input = st.empty()
submit = st.button('Submit')
# settings for padding sequence
max_length = 3000
trunc_type = 'post'
padding_type = 'post'

# getting feature text tokenizer for model training
with open('assets/feature_token/feature_tokenizer.pickle', 'rb') as handle:
    feature_tokenizer = pickle.load(handle)

# getting label encoding dictionary from model training
with open('assets/dict/dictionary.pickle', 'rb') as handle:
    encoding_to_label = pickle.load(handle)

# handling unknown label case and load original labels from json file
encoding_to_label[0] = 'unknown'
with open("assets/labels.json", "r") as read_file:
    original_labels = json.load(read_file)

# converting user input to padded sequence
predict_sequences = feature_tokenizer.texts_to_sequences([cleaned_input])
predict_padded = pad_sequences(
    predict_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
predict_padded = np.array(predict_padded)

# loading model and make prediction
model = keras.models.load_model('assets/model/')
prediction = model.predict(predict_padded)

encodings = np.argpartition(prediction[0], -5)[-5:]
encodings = encodings[np.argsort(prediction[0][encodings])]
encodings = reversed(encodings)

# function to predict the probability


def scanning():
    predict = []
    for encoding in encodings:
        label = encoding_to_label[encoding]
        probability = prediction[0][encoding] * 100
        probability = round(probability, 2)
        predict.append('{} - {}%'.format(original_labels[label], probability))
    return predict


pred = scanning()

if submit:
    for i in range(len(pred)):
        st.write(pred[i])
    user_input = " "
