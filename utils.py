import os
import streamlit as st
from PyPDF2 import PdfReader

def get_api_key(key_name):
    """Try to get API key from Streamlit secrets, then environment variables."""
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.getenv(key_name)

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
