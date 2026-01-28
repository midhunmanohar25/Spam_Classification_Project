import streamlit as st
import pickle
import spacy
import contractions
from bs4 import BeautifulSoup
import re


model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))


nlp = spacy.load("en_core_web_sm")

def normilization_pipeline(text):
    """
    A more realistic pipeline to clean scraped data
    """
    
    # --- Step 1: Structural Cleaning ---
    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Removing special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    
    
    # --- Step 2: Linguistic Normalization (Pre-tokenization) ---
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    
    
    # --- Step 3: Token-based Normalization using SpaCy ---
    doc = nlp(text)
    
    clean_tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    
    clean_tokens = " ".join(clean_tokens) # To convet list into str
    # print(text)
    return clean_tokens



# st.write("tfidf fitted:", hasattr(tfidf, "idf_"))
# st.write("Model fitted:", hasattr(model, "class_count_"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button("Predict"):
    
    # Steps to follow:
    # 1. preprocess
    transformed_sms = normilization_pipeline(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")