EchoSift : Data Extraction framework

A powerful, intelligent web content analysis framework that combines **advanced Large Language Models (LLMs)** with **traditional Machine Learning pipelines** and **security-aware scraping practices**. EchoSift intelligently scrapes, analyzes, and visualizes web data for summarization, sentiment, metadata, and more — through a real-time, interactive dashboard.

---

##  Key Features

- ⚙️ **LLM-powered Content Understanding**  
  - Summarization, Named Entity Recognition, QA over scraped content.
  
- 🕷️ **Custom Web Scraping**  
  - Built with **Scrapy + BeautifulSoup**, designed to bypass clutter and extract clean content from structured and unstructured sites.

- 🔐 **Security & Metadata-Driven Architecture**  
  - Server-side metadata tracking includes:
    - IP address resolution
    - Response time logging
    - Content size and header analysis
    - Depth control (anti-infinite loops)
    - `robots.txt` compliance and anomaly flagging  
  - Enables **performance benchmarking**, **access filtering**, and **scraper behavior modeling**.

- 📊 **NLP Pipelines**  
  - Sentiment analysis, POS tagging, keyword/entity extraction, summarization

- ⚡ **Interactive Visualization via Streamlit Dashboard**  
  - Real-time content stats, keyword charts, success rate meters, global web distribution maps, and new panels (see below)

- 🧱 **Modular Design**  
  - Reusable ML pipelines that can scale horizontally
  - Easily extendable with other transformer-based tools or APIs

- 📦 **Docker-Ready**  
  - Fully containerized for local or cloud-based deployments

---

##  Models Used

###  LLMs

| Model | Purpose |
|-------|---------|
| `facebook/bart-large-cnn` | Abstractive Summarization |
| `dslim/bert-base-NER`     | Named Entity Recognition  |
| `deepset/roberta-base-squad2` | Question Answering |

###  Classical ML/NLP

| Task               | Model                            |
|--------------------|----------------------------------|
| Sentiment Analysis | Logistic Regression + TF-IDF     |
| Keyword Extraction | RAKE (Rapid Automatic KE)        |
| POS Tagging        | spaCy `en_core_web_sm`           |

---

##  ML Pipelines

| Pipeline           | Description                                      |
|-------------------|--------------------------------------------------|
| `text_preprocessing.py` | Punctuation removal, stopword filtering       |
| `summarization.py`      | Summarization using BART                     |
| `keyword_entity.py`     | Entity & keyword scoring and filtering       |
| `QandAmodule.py`        | Question Answering over dynamic text chunks |
| `scraper.py`            | Web page scraper with IP, timing, and headers |

---

## 🚨 Security & Metadata Layer

EchoSift integrates a lightweight metadata-driven layer to ensure compliant and secure scraping:

- Logs:
  - Source IP, Host IP
  - Response time and latency
  - Content length and encoding
- Anomaly Checks:
  - Infinite redirection loops
  - Incomplete HTML structure
  - Broken domains or blocked headers
- Future Additions:
  - Rate-limit detection
  - Bot detection fingerprinting
  - IP-based geolocation analytics

> This security architecture not only improves efficiency but also allows for meaningful benchmarking and comparative performance across scraping tasks.

---

## 📊 Output Examples

| Module        | Output                                  |
|---------------|------------------------------------------|
| Content Stats | % Content Retained, % Removed, Text Type |
| NLP Insights  | Top Keywords, Named Entities, Sentiment |
![WhatsApp Image 2025-04-21 at 00 02 08_603cf4ed](https://github.com/user-attachments/assets/93020e07-ae5c-43a3-a95a-e1a0f50adf56)

| QA Interface  | Real-time Q&A with transformer context   |


| Metadata Logs | Response time, Content Length, Depth     |
![WhatsApp Image 2025-04-21 at 00 07 03_820a43ba](https://github.com/user-attachments/assets/5413a7c7-c580-451f-b098-17fa951e2916)


---

###  New Output Panels (Coming Soon)
futurework (suggestions welcomed)
- ✅ **Time-Series Graph** of Scraper Performance by Hour/Day
- ✅ **Security Violation Heatmap** (e.g., unusual redirects, flagged pages)
- ✅ **Scraping Quality Score** based on metadata + content structure
- ✅ **Dynamic Filter Panel**: By domain, latency, country, or failure type

---

## 🚀 Getting Started

1. **Clone the Repo**
```bash
git clone https://github.com/yourusername/EchoAI
cd EchoAI
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit UI**
```bash
streamlit run app.py
```

---

## 📜 License

MIT License — Feel free to fork, improve, and build on it.
