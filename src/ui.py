import streamlit as st
import requests

API_URL = "http://api:8000/predict"

st.title("Twitter Sentiment Analysis")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            st.success(f"Sentiment: {data['label']}")
            st.info(f"Confidence: {data['confidence']}")
        else:
            st.error("API Error")
