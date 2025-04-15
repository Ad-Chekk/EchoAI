# Web Content Analyzer with LLMs 🧠🌐

A powerful, intelligent web content analysis framework built with advanced **Large Language Models (LLMs)** and traditional **Machine Learning pipelines**. This tool can intelligently scrape, analyze, and visualize web data for  topics, summarization, Q and A qith the web based data and more.
possible use cases include--\
education - understant topics from web based sites instantly
news based checking - analyze news and identify working entities (useful for cybersecurity and terrorism based news analysis)
financial news analysys - analyse key economic entities and summarize financial news quickly to act on stocks and other investments

## 🔍 Features

- 🧠 LLM-powered content understanding (Summarization, NER, etc.)
- 🕷️ Custom web scraping (Scrapy-based)
- 📊 NLP pipelines: Sentiment analysis, POS tagging, keyword extraction
- ⚡ Streamlit dashboard for real-time visualization
- 🧱 Modular design with reusable ML pipelines
- 📦 Docker-ready setup

---


## 🧠 LLM & ML Models Used

### 🔹 LLMs
- **HuggingFace Transformers**
  - `facebook/bart-large-cnn` – Text Summarization
  - `dslim/bert-base-NER` – Named Entity Recognition
  -  `eepset/roberta-base-squad2` - Q and A based system

### 🔹 Classical ML/NLP Models
- **Sentiment Analysis** – Logistic Regression + TF-IDF (custom-trained)
- **Keyword Extraction** – RAKE (Rapid Automatic Keyword Extraction)
- **POS Tagging** – spaCy’s `en_core_web_sm` pipeline

---

## ⟳ ML Pipelines

| Pipeline | Description |
|---------|-------------|
| `text_preprocessing`|punctuation removal, stopword filtering |
| `summarization.py` | BART-based summarizer (LLM) |
| `keyword_entity`|for keyword and entity extraction
| `QandAmodule.py` | Question and answering model |
| `scraper.py` | simple scraping purpose |

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/web-content-analyzer
cd web-content-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the UI
streamlit run app.py


## 📊 Output Examples

- Summarized text
- ![WhatsApp Image 2025-04-13 at 14 25 00_982b5f6a](https://github.com/user-attachments/assets/28d0f7cf-f056-46ab-81ff-a6ca39a64cc2)

- Extracted keywords and entities
![WhatsApp Image 2025-04-13 at 14 26 30_91d6a342](https://github.com/user-attachments/assets/b3babd7b-37d8-48b1-a2c0-95a9eb4a8a20)


- Q and A with the processed data
- ![WhatsApp Image 2025-04-13 at 18 47 35_55dec7fd](https://github.com/user-attachments/assets/722794a1-ef07-41de-b50d-1e9d96229f96)


---


## 📜 License

MIT License — feel free to for
