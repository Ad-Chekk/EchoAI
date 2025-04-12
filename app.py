import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state for text
if "text_data" not in st.session_state:
    st.session_state.text_data = ""
if "raw_text_data" not in st.session_state:
    st.session_state.raw_text_data = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []

# Simple sentence tokenizer that doesn't rely on NLTK
def simple_sent_tokenize(text):
    # Replace common abbreviations to avoid splitting at them
    text = re.sub(r'(?<=[Mm]r)\.', '##DOT##', text)
    text = re.sub(r'(?<=[Mm]rs)\.', '##DOT##', text)
    text = re.sub(r'(?<=[Dd]r)\.', '##DOT##', text)
    text = re.sub(r'(?<=[Pp]rof)\.', '##DOT##', text)
    text = re.sub(r'(?<=[Ii]\.e)\.', '##DOT##', text)
    text = re.sub(r'(?<=[e]\.g)\.', '##DOT##', text)
    text = re.sub(r'(?<=[A-Z])\.(?=[A-Z]\.)', '##DOT##', text)  # Initials like U.S.A.
    
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore the dots
    sentences = [s.replace('##DOT##', '.') for s in sentences]
    
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]

# Basic English stopwords list
english_stopwords = {
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing'
}

# Function to extract text from a URL
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

# Function to preprocess and clean the extracted text
def preprocess_text(text: str) -> str:
    # Break into sentences using our simple tokenizer
    sentences = simple_sent_tokenize(text)
    
    # Common boilerplate patterns to remove
    boilerplate_patterns = [
        r'cookie[s]? policy',
        r'privacy policy',
        r'terms (of|and) (use|service)',
        r'all rights reserved',
        r'copyright ©',
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # dates like mm/dd/yyyy
        r'subscribe (to|for) our newsletter',
        r'follow us on',
        r'share this',
        r'comments',
        r'related articles',
        r'click here',
        r'accept cookies',
        r'advertisement',
        r'contact us',
        r'sign up',
        r'log in',
    ]
    
    # Create a combined pattern
    combined_pattern = '|'.join(boilerplate_patterns)
    
    # Filter out sentences with boilerplate content or that are too short
    filtered_sentences = []
    for sentence in sentences:
        # Skip very short sentences (likely not content)
        if len(sentence.split()) < 4:
            continue
            
        # Skip sentences matching boilerplate patterns
        if re.search(combined_pattern, sentence.lower()):
            continue
            
        # Skip sentences that are mostly stopwords
        words = sentence.split()
        if len(words) > 0:
            stopword_count = sum(1 for word in words if word.lower() in english_stopwords)
            stopword_ratio = stopword_count / len(words) if len(words) > 0 else 0
            if stopword_ratio > 0.8:  # Skip if more than 80% of words are stopwords
                continue
                
        filtered_sentences.append(sentence)
    
    # Combine the filtered sentences
    processed_text = " ".join(filtered_sentences)
    
    # Remove extra whitespace and normalize
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

# Chunk the text into meaningful segments for better QA
def chunk_text(text, chunk_size=200, overlap=50):
    sentences = simple_sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        
        if current_length + len(words) > chunk_size and current_length > 0:
            # Complete the current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start a new chunk with overlap
            overlap_start = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start:]
            current_length = len(current_chunk)
        
        current_chunk.extend(words)
        current_length += len(words)
    
    # Add the last chunk if it's not empty
    if current_length > 0:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

# Function to create paragraph chunks from text
def create_paragraph_chunks(text):
    # Split by paragraph markers
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean up paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # For very long paragraphs, further split them
    result = []
    for p in paragraphs:
        if len(p.split()) > 150:  # If paragraph is very long
            sentences = simple_sent_tokenize(p)
            # Group sentences into smaller chunks
            for i in range(0, len(sentences), 5):
                chunk = " ".join(sentences[i:i+5])
                if chunk.strip():
                    result.append(chunk)
        else:
            result.append(p)
            
    # If we have very few chunks, duplicate to ensure we have enough context
    if len(result) == 1:
        result.append(result[0])
    
    return result

# Enhanced QA function with better context retrieval
def answer_question(question, chunks, qa_pipeline):
    # If we have very few chunks, just use all of them
    if len(chunks) <= 3:
        context = " ".join(chunks)
        result = qa_pipeline(question=question, context=context)
        return result["answer"], result["score"], context
    
    # For many chunks, use TF-IDF to find most relevant ones
    try:
        # Create vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Create document term matrix
        tfidf_matrix = vectorizer.fit_transform(chunks)
        
        # Transform question to vector
        question_vector = vectorizer.transform([question])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(question_vector, tfidf_matrix)[0]
        
        # Get top 3 most relevant chunks
        top_indices = similarity_scores.argsort()[-3:][::-1]
        
        # Combine relevant chunks
        relevant_context = " ".join([chunks[i] for i in top_indices])
        
        # Run QA on relevant context
        result = qa_pipeline(question=question, context=relevant_context)
        
        return result["answer"], result["score"], relevant_context
    except Exception as e:
        # Fallback to using first few chunks if TF-IDF fails
        context = " ".join(chunks[:3])
        result = qa_pipeline(question=question, context=context)
        return result["answer"], result["score"], context

# Enhance the answer with additional context if confidence is low
def enhance_answer(answer, question, context, confidence, summarizer):
    # If confidence is high, just return the answer but make sure it's a complete sentence
    if confidence > 0.7:
        if not answer.endswith(('.', '!', '?')):
            answer = answer.strip() + "."
        if not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        return answer
    
    # For lower confidence, try to provide a more comprehensive response
    try:
        # For "what is X" questions, look for definitions
        if question.lower().startswith("what is") or question.lower().startswith("define"):
            term = question.lower().replace("what is", "").replace("define", "").strip().rstrip("?")
            pattern = rf"\b{re.escape(term)}\s+(?:is|are|refers to|means)\b"
            matches = re.findall(f"[^.]*{pattern}[^.]*\.", context, re.IGNORECASE)
            
            if matches:
                # Use the definition sentences
                return " ".join(matches[:2]).strip()
        
        # For medium confidence, add sentences containing keywords from the answer
        if confidence > 0.3:
            # Try to find sentences containing important words from the answer
            answer_words = [w.lower() for w in re.findall(r'\b\w+\b', answer) if len(w) > 4 and w.lower() not in english_stopwords]
            
            if answer_words:
                sentences = simple_sent_tokenize(context)
                relevant_sentences = []
                
                for sentence in sentences:
                    for word in answer_words:
                        if word in sentence.lower():
                            relevant_sentences.append(sentence)
                            break
                
                if relevant_sentences:
                    # Limit to 3 sentences max
                    return " ".join(relevant_sentences[:3]).strip()
        
        # For very low confidence, try summarizing the context
        if summarizer and confidence < 0.2:
            try:
                summary = summarizer(context[:1000], max_length=100, min_length=30)[0]["summary_text"]
                if len(summary) > len(answer):
                    return summary.strip()
            except:
                pass
        
        # Ensure the answer is properly formatted
        if not answer.endswith(('.', '!', '?')):
            answer = answer.strip() + "."
        if not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
            
        return answer
    except:
        # If enhancement fails, return original answer with basic formatting
        if not answer.endswith(('.', '!', '?')):
            answer = answer.strip() + "."
        if not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        return answer

# Load summarizer with error handling
@st.cache_resource
def get_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.error(f"Error loading summarizer model: {e}")
        return None

# Load keyword/entity extractor (improved)
def extract_keywords_entities(text):
    # Extract potential keywords (longer words, excluding stopwords)
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if len(w) > 5 and w not in english_stopwords]
    
    # Count frequency
    keyword_freq = {}
    for word in keywords:
        if word in keyword_freq:
            keyword_freq[word] += 1
        else:
            keyword_freq[word] = 1
    
    # Sort by frequency
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [word for word, freq in sorted_keywords[:15]]
    
    # Extract named entities (simple approach - capitalized words)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities = list(set(entities))[:15]
    
    return top_keywords, entities

# Load QA model for chatbot with error handling
@st.cache_resource
def get_qa_pipeline():
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        return None

# -------------------- Streamlit UI --------------------

st.title("🧠 Web Content Analyzer with LLM")

url = st.text_input("Enter a URL to extract text from:")

# Load models with progress indicators
with st.spinner("Loading models..."):
    summarizer = get_summarizer()
    qa_pipeline = get_qa_pipeline()

if st.button("Extract Text"):
    if url:
        with st.spinner("Extracting and processing text..."):
            st.session_state.raw_text_data = extract_text_from_url(url)
            st.session_state.text_data = preprocess_text(st.session_state.raw_text_data)
            
            # Create text chunks for improved QA - using paragraph-based chunking
            st.session_state.text_chunks = create_paragraph_chunks(st.session_state.text_data)
            
            chars_removed = len(st.session_state.raw_text_data) - len(st.session_state.text_data)
            percent_removed = (chars_removed / len(st.session_state.raw_text_data) * 100) if len(st.session_state.raw_text_data) > 0 else 0
            
            st.success(f"Text extracted and preprocessed. Removed approximately {chars_removed} characters ({percent_removed:.1f}%) of boilerplate content.")
    else:
        st.warning("Please enter a URL.")

if st.session_state.text_data:
    st.subheader("📄 Extracted Text:")
    
    tab1, tab2 = st.tabs(["Processed Text", "Raw Text"])
    
    with tab1:
        st.text_area("Processed Text (Boilerplate Removed)", st.session_state.text_data[:10000], height=300)
    
    with tab2:
        st.text_area("Raw Text", st.session_state.raw_text_data[:10000], height=300)
    
    st.subheader("🔍 Choose an Action:")
    action = st.radio("What would you like to do?", ["Summarize", "Extract Keywords and Entities", "Chat with the Text"])
    
    if action == "Summarize":
        if summarizer and len(st.session_state.text_data) > 200:
            with st.spinner("Generating summary..."):
                try:
                    # Split into chunks if text is very long
                    max_length = 1000
                    if len(st.session_state.text_data) > max_length:
                        chunks = [st.session_state.text_data[i:i+max_length] for i in range(0, min(3000, len(st.session_state.text_data)), max_length)]
                        summaries = []
                        for chunk in chunks[:3]:  # Limit to first 3 chunks to avoid excessive processing
                            summary = summarizer(chunk, max_length=100, min_length=30)[0]["summary_text"]
                            summaries.append(summary)
                        final_summary = " ".join(summaries)
                    else:
                        final_summary = summarizer(st.session_state.text_data, max_length=150, min_length=40)[0]["summary_text"]
                        
                    st.success("Summary:")
                    st.write(final_summary)
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
        else:
            st.warning("Text too short to summarize or summarizer model not available.")
    
    elif action == "Extract Keywords and Entities":
        with st.spinner("Extracting keywords and entities..."):
            try:
                keywords, entities = extract_keywords_entities(st.session_state.text_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔑 Keywords:**")
                    for kw in keywords:
                        st.write(f"• {kw}")
                
                with col2:
                    st.markdown("**🏷️ Entities:**")
                    for ent in entities:
                        st.write(f"• {ent}")
            except Exception as e:
                st.error(f"Error extracting keywords and entities: {e}")
    
    elif action == "Chat with the Text":
        st.markdown("💬 **Ask a question about the extracted content:**")
        
        # Add examples of good questions
        with st.expander("Question suggestions"):
            st.markdown("""
            Try questions like:
            * What is the main topic of this text?
            * Explain the concept of [term] mentioned in the text.
            * What are the key points about [topic]?
            * How does [concept] work according to this text?
            * What are the advantages of [topic] mentioned?
            * Compare [concept A] and [concept B] from the text.
            """)
        
        user_question = st.text_input("Your Question")
        
        if user_question and qa_pipeline:
            with st.spinner("Analyzing text and generating answer..."):
                try:
                    # Use our enhanced QA function
                    answer, confidence, context = answer_question(
                        user_question, 
                        st.session_state.text_chunks if st.session_state.text_chunks else [st.session_state.text_data], 
                        qa_pipeline
                    )
                    
                    # Enhance the answer
                    enhanced_answer = enhance_answer(answer, user_question, context, confidence, summarizer)
                    
                    # Save to chat history
                    st.session_state.chat_history.append((user_question, enhanced_answer, confidence))
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.session_state.chat_history.append((user_question, f"Sorry, I couldn't process that question: {str(e)}", 0.0))
        
        # Better styling for chat history
        st.markdown("### Chat History")
        for i, (q, a, conf) in enumerate(st.session_state.chat_history):
            # Alternate background colors for better readability
            bg_color = "#f0f2f6" if i % 2 == 0 else "#e6e9ef"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4b6fff;">
                <p style="font-weight: 600; color: #333;">Question: {q}</p>
                <p style="padding: 10px 0;">{a}</p>
                <p style="font-size: 0.8em; color: #666;"><em>Confidence: {conf:.2f}</em></p>
            </div>
            """, unsafe_allow_html=True)