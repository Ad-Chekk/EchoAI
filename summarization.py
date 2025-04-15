from transformers import pipeline
import streamlit as st

@st.cache_resource
def get_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.error(f"Error loading summarizer model: {e}")
        return None
