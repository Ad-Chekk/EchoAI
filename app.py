import streamlit as st
from transformers import pipeline
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###file management 
from scraper import extract_text_from_url
from text_preprocessing import simple_sent_tokenize, preprocess_text, create_paragraph_chunks, english_stopwords
from summarization import st, get_summarizer
from keyword_entity import extract_keywords_entities
from QandAmodule import answer_question, get_qa_pipeline, enhance_answer



st.set_page_config(layout="wide")


# Initialize session state for text
if "text_data" not in st.session_state:
    st.session_state.text_data = ""
if "raw_text_data" not in st.session_state:
    st.session_state.raw_text_data = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []


# -------------------- Streamlit UI from here--------------------

st.title("🧠 Web Content Analyzer with LLM")

url = st.text_input("Enter a URL to extract text from:")

# Load models with progress indicators
with st.spinner("Loading models..."):
    summarizer = get_summarizer()
    qa_pipeline = get_qa_pipeline()

if st.button("Extract Text"):    #actual process flow
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