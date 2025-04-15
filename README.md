# Web Content Analyzer with LLMs 🧠🌐

A powerful, intelligent web content analysis framework built with advanced **Large Language Models (LLMs)** and traditional **Machine Learning pipelines**. This tool can intelligently scrape, analyze, and visualize web data for sentiment, topics, summarization, and more.

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
- Extracted entities
- Sentiment polarity (positive/negative/neutral)
- Top keywords
- Visualized content charts

---


## 📜 License

MIT License — feel free to for