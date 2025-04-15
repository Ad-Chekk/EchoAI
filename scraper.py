import requests
from bs4 import BeautifulSoup
import streamlit as st


def extract_text_from_url(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'footer', 'nav', 'header', 'aside', 'iframe']):
                element.decompose()
                
            # Extract text with paragraph breaks
            text = soup.get_text(separator="\n", strip=True)
            return text
        else:
            st.error(f"Request failed with status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""