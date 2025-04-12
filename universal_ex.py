# # Required installations:
# # pip install requests trafilatura transformers keybert spacy
# # python -m spacy download en_core_web_sm

# import requests
# import trafilatura
# from transformers import pipeline
# from keybert import KeyBERT
# import spacy

# # === NLP Models ===
# # Summarizer (small model for efficiency)
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# # Keyword extractor
# kw_model = KeyBERT('all-MiniLM-L6-v2')

# # Named Entity Recognizer
# nlp = spacy.load("en_core_web_sm")


# # === Step 1: Extract text from URL ===
# def extract_text_from_url(url: str) -> str:
#     try:
#         response = requests.get(url, timeout=10)
#         if response.status_code == 200:
#             downloaded = trafilatura.extract(response.text)
#             return downloaded if downloaded else ""
#         else:
#             print(f"Request failed with status code: {response.status_code}")
#             return ""
#     except Exception as e:
#         print(f"Error fetching URL: {e}")
#         return ""


# # === Step 2: Summarize text ===
# def summarize_text(text: str, max_tokens: int = 1024) -> str:
#     try:
#         return summarizer(text[:max_tokens])[0]['summary_text']
#     except Exception as e:
#         print(f"Error during summarization: {e}")
#         return "Summary not available."


# # === Step 3: Extract Keywords ===
# def extract_keywords(text: str, top_n: int = 5):
#     try:
#         return kw_model.extract_keywords(text, top_n=top_n)
#     except Exception as e:
#         print(f"Error extracting keywords: {e}")
#         return []


# # === Step 4: Named Entity Recognition ===
# def extract_entities(text: str):
#     try:
#         doc = nlp(text)
#         return [(ent.text, ent.label_) for ent in doc.ents]
#     except Exception as e:
#         print(f"Error in NER: {e}")
#         return []


# # === Pipeline Function ===
# def analyze_webpage(url: str) -> dict:
#     print(f"Analyzing: {url}")
#     raw_text = extract_text_from_url(url)
#     if not raw_text:
#         return {"error": "Failed to extract text from the URL."}

#     summary = summarize_text(raw_text)
#     keywords = extract_keywords(raw_text)
#     entities = extract_entities(raw_text)

#     return {
#         "url": url,
#         "summary": summary,
#         "keywords": keywords,
#         "entities": entities,
#         "raw_text": raw_text,
#     }

# #https://en.wikipedia.org/wiki/Natural_language_processing
# # === Example Usage ===
# if __name__ == "__main__":
#     url = "https://timesofindia.indiatimes.com/"
#     result = analyze_webpage(url)

#     print("\n=== Summary ===\n", result["summary"])
#     print("\n=== Keywords ===\n", result["keywords"])
#     print("\n=== Named Entities ===\n", result["entities"])
#     # Uncomment if you want full text:
#     # print("\n=== Raw Text ===\n", result["raw_text"])
