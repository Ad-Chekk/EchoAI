import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

# --------- TEXT SCRAPER ----------
def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator="\n", strip=True)
            return text
        else:
            return f"[Error] Request failed with status code: {response.status_code}"
    except Exception as e:
        return f"[Error] Exception: {e}"

# --------- SUMMARIZER PIPELINE ----------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --------- NER PIPELINE ----------
@st.cache_resource
def load_ner():
    return pipeline("ner", grouped_entities=True)

# --------- Chatting-style text generation (basic) ----------
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

# --------- UI ----------
st.title("🕸️ Webpage Text Analyzer with Transformers")
url_input = st.text_input("🔗 Enter a webpage URL:", "https://www.geeksforgeeks.org/introduction-to-hashing-2/")

if st.button("Extract and Process"):
    with st.spinner("Fetching and analyzing content..."):
        raw_text = extract_text_from_url(url_input)

        if "[Error]" in raw_text:
            st.error(raw_text)
        else:
            st.subheader("📄 Extracted Text")
            st.text_area("Webpage Text", raw_text[:5000] + ("..." if len(raw_text) > 5000 else ""), height=300)

            option = st.radio("What do you want to do with the text?", 
                              ["Summarize", "Extract Keywords & Entities", "Chat with the Text"])

            if option == "Summarize":
                summarizer = load_summarizer()
                chunks = [raw_text[i:i+1000] for i in range(0, min(len(raw_text), 3000), 1000)]
                summary = ""
                for chunk in chunks:
                    out = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                    summary += out[0]['summary_text'] + "\n"
                st.subheader("🧠 Summary")
                st.write(summary)

            elif option == "Extract Keywords & Entities":
                ner = load_ner()
                entities = ner(raw_text[:2000])  # avoid long text timeout
                st.subheader("🔍 Named Entities & Keywords")
                for ent in entities:
                    st.write(f"• {ent['word']} ({ent['entity_group']})")

            elif option == "Chat with the Text":
                st.subheader("💬 Chat Simulation (Prompt Completion)")
                user_question = st.text_input("Ask something about the content:")
                if user_question:
                    gen = load_generator()
                    prompt = f"Based on this article:\n{raw_text[:1000]}\nQ: {user_question}\nA:"
                    response = gen(prompt, max_length=250, do_sample=True)[0]['generated_text']
                    answer = response.split("A:")[-1].strip()
                    st.markdown(f"**🤖 Answer:** {answer}")

