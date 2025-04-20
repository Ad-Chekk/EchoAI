import streamlit as st
from transformers import pipeline
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import nltk
from collections import Counter

###file management 
from scraper import extract_text_from_url
from text_preprocessing import simple_sent_tokenize, preprocess_text, create_paragraph_chunks, english_stopwords
from summarization import st, get_summarizer
from keyword_entity import extract_keywords_entities
from QandAmodule import answer_question, get_qa_pipeline, enhance_answer


# page_bg_img = """
# <style>
# [data-testid="stAppViewContainer"] {
# background-image: url("https://images.unsplash.com/photo-1601333924581-7b48591a926c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
# background-size: cover;
# }
# </style>
# """
# st.markdown(page_bg_img,unsafe_allow_html=True)


# Set page configuration with dark theme
st.set_page_config(layout="wide", page_title="Web Content Analyzer", page_icon="🔍")
with st.sidebar.container():
    st.markdown(
        """
        <div style="padding:0px; border:1px solid #444; border-radius:8px; background-color:#252525;">
            <h4 style="color:white;">Top Keywords by Relevance</h4>
        </div>
        """, 
        unsafe_allow_html=True
    )
# Custom CSS for dark theme - 
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #000000;
        color: white;
    }
    .stButton button {
        background-color: #4b6fff;
        color: white;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #3d3d3d;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: white !important;
    }
    .stRadio label, .stCheckbox label {
        color: white !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    /* Custom components */
    .keyword-tag {
        display: inline-block;
        margin: 4px;
        padding: 6px 8px;
        background-color: rgba(100, 100, 100, 0.3);
        border-radius: 4px;
        font-size: 0.9em;
    }
    
    .keyword-container {
        background-color: #2d2d2d;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 10px;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .chat-message-user {
        background-color: #2d2d2d;
        border-left: 4px solid #4b6fff;
    }
    .chat-message-ai {
        background-color: #383838;
        border-left: 4px solid #45a29e;
    }
    
    /* Tabs and expanders */
    .stExpander {
        background-color: #2d2d2d !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d2d2d;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b6fff !important;
    }
    
    /* Header styling */
    .app-header {
        padding: 10px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid #333;
    }
    
    /* Box containers */
    .content-box {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #333;
    }
    
    /* Card headers */
    .card-header {
        font-weight: bold;
        border-bottom: 1px solid #333;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    
    /* Status tags */
    .status-tag {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        margin-left: 10px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
            
#             [data-testid="stAppViewContainer"] {
# background-image: url("https://img.freepik.com/free-vector/background-template-wave-gradient-style_483537-5016.jpg");
# background-size: cover;
# }
</style>
""", unsafe_allow_html=True)

# Initialize session state for text
if "text_data" not in st.session_state:
    st.session_state.text_data = ""
if "raw_text_data" not in st.session_state:
    st.session_state.raw_text_data = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "content_stats" not in st.session_state:
    st.session_state.content_stats = {}
if "removed_content_types" not in st.session_state:
    st.session_state.removed_content_types = {}
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "entities" not in st.session_state:
    st.session_state.entities = []

# Function to analyze removed content
def analyze_removed_content(raw_text, processed_text):
    # Content type categories
    removed_content = {}
    
    # Identify HTML or code-like content
    html_tags = len(re.findall(r'<[^>]+>', raw_text))
    removed_content['HTML/Code'] = html_tags
    
    # Identify URLs
    urls = len(re.findall(r'https?://\S+', raw_text)) - len(re.findall(r'https?://\S+', processed_text))
    removed_content['URLs'] = urls
    
    # Identify special characters
    special_chars = sum(1 for c in raw_text if c in string.punctuation) - sum(1 for c in processed_text if c in string.punctuation)
    removed_content['Special Characters'] = special_chars
    
    # Identify whitespace
    whitespace = sum(1 for c in raw_text if c.isspace()) - sum(1 for c in processed_text if c.isspace())
    removed_content['Whitespace'] = whitespace
    
    # Calculate remaining as "Other"
    total_removed = len(raw_text) - len(processed_text)
    accounted_for = sum(removed_content.values())
    removed_content['Other Text'] = max(0, total_removed - accounted_for)
    
    return removed_content

# Function to render keywords and entities in the new style
def render_keywords_entities(keywords, entities):

    with st.sidebar:
        # Keywords section
        st.markdown('<div class="card-header">🔑 Keywords</div>', unsafe_allow_html=True)
        #st.markdown('<div class="keyword-container">', unsafe_allow_html=True)
        keywords_html = ""
        for kw in keywords[:30]:  # Limit to top 30 keywords
            keywords_html += f'<span class="keyword-tag">{kw}</span>'
        st.markdown(keywords_html + '</div>', unsafe_allow_html=True)
        
        # Entities section
        st.markdown('<div class="card-header">🏷️ Named Entities</div>', unsafe_allow_html=True)
        #st.markdown('<div class="keyword-container">', unsafe_allow_html=True)
        entities_html = ""
        for entity in entities[:30]:  # Limit to top 30 entities
            entities_html += f'<span class="keyword-tag">{entity}</span>'
        st.markdown(entities_html + '</div>', unsafe_allow_html=True)


# -------------------- Streamlit UI from here--------------------










# App header 
st.markdown("""
<div class="app-header">
    <h1>🔍 Web Content Analyzer</h1>
    <p>Extract, analyze, and interact with web content using AI</p>
</div>
""", unsafe_allow_html=True)

# Create a layout with 2 columns - left for content, right for analytics
left_column, right_column = st.columns([6, 3])

with left_column:
  






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
                
                # Calculate content stats
                chars_removed = len(st.session_state.raw_text_data) - len(st.session_state.text_data)
                percent_removed = (chars_removed / len(st.session_state.raw_text_data) * 100) if len(st.session_state.raw_text_data) > 0 else 0
                
                st.session_state.content_stats = {
                    "Original Length": len(st.session_state.raw_text_data),
                    "Processed Length": len(st.session_state.text_data),
                    "Characters Removed": chars_removed,
                    "Percent Removed": percent_removed
                }
                
                # Analyze removed content types
                st.session_state.removed_content_types = analyze_removed_content(
                    st.session_state.raw_text_data, 
                    st.session_state.text_data
                )
                
                # Also extract keywords and entities right away
                st.session_state.keywords, st.session_state.entities = extract_keywords_entities(st.session_state.text_data)
                
                st.success(f"Text extracted and preprocessed. Removed approximately {chars_removed} characters ({percent_removed:.1f}%) of boilerplate content.")
        else:
            st.warning("Please enter a URL.")
    
    if st.session_state.text_data:
        # Text content display
        st.markdown('<div class="card-header">📄 Extracted Content</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Processed Text", "Raw Text"])
        
        with tab1:
            st.text_area("Processed Text (Boilerplate Removed)", st.session_state.text_data[:10000], height=300)
        
        with tab2:
            st.text_area("Raw Text", st.session_state.raw_text_data[:10000], height=300)
        
        st.markdown('<div class="card-header">🔍 Analysis Tools</div>', unsafe_allow_html=True)
        action = st.radio("What would you like to do?", ["Summarize", "Chat with the Text"])
        
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
                        
                        # Calculate summarization stats
                        original_words = len(st.session_state.text_data.split())
                        summary_words = len(final_summary.split())
                        compression_ratio = (original_words - summary_words) / original_words * 100 if original_words > 0 else 0
                        
                        # Display summary
                        st.markdown('<div class="content-box">', unsafe_allow_html=True)
                        st.markdown(f"<h3>Summary:</h3><p>{final_summary}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            else:
                st.warning("Text too short to summarize or summarizer model not available.")
        
        elif action == "Chat with the Text":
            st.markdown('<div class="card-header">💬 Ask Questions About The Content</div>', unsafe_allow_html=True)
            
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
            if st.session_state.chat_history:
                st.markdown('<div class="card-header">Chat History</div>', unsafe_allow_html=True)
                for i, (q, a, conf) in enumerate(st.session_state.chat_history):
                    # User message
                    st.markdown(f"""
                    <div class="chat-message chat-message-user">
                        <p style="font-weight: 600; color: #ffffff;">You: {q}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI response
                    st.markdown(f"""
                    <div class="chat-message chat-message-ai">
                        <p style="font-weight: 600; color: #66fcf1;">Assistant:</p>
                        <p>{a}</p>
                        <p style="font-size: 0.8em; color: #c5c6c7;"><em>Confidence: {conf:.2f}</em></p>
                    </div>
                    """, unsafe_allow_html=True)

# Right sidebar for analytics, keywords and entities
with right_column:
    if st.session_state.text_data:
        # Content analysis section
        st.markdown('<div class="card-header">📊 Content Analysis</div>', unsafe_allow_html=True)
        
        # Text processing gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 100 - st.session_state.content_stats.get("Percent Removed", 0),
            title = {'text': "Content Retained (%)", 'font': {'color': 'white', 'size': 14}},
            delta = {'reference': 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                'bar': {'color': "#4b6fff"},
                'bgcolor': "gray",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 30], 'color': '#FF6B6B'},
                    {'range': [30, 70], 'color': '#FFD166'},
                    {'range': [70, 100], 'color': '#06D6A0'}
                ]
            }
        ))
        fig.update_layout(
            paper_bgcolor='#1e1e1e', 
            font={'color': 'white', 'family': 'Arial'},
            height=200,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Content removed by type
        removed_content = st.session_state.removed_content_types
        labels = list(removed_content.keys())
        values = list(removed_content.values())
        
        # Filter out zero values
        filtered_data = [(label, value) for label, value in zip(labels, values) if value > 0]
        if filtered_data:
            labels, values = zip(*filtered_data)
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                hole=.3,
                textinfo='label+percent',
                marker=dict(colors=['#4b6fff', '#45a29e', '#66fcf1', '#c5c6c7', '#e85d04'])
            )])
            fig.update_layout(
                title="Content Removed by Type",
                title_font_color='white',
                title_font_size=14,
                paper_bgcolor='#1e1e1e',
                font={'color': 'white', 'family': 'Arial', 'size': 11},
                height=300,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Keywords and entities in the new style
        if st.session_state.keywords and st.session_state.entities:
            render_keywords_entities(st.session_state.keywords, st.session_state.entities)

            # Add a keyword visualization (top 10 only)
            if len(st.session_state.keywords) > 5:
                top_keywords = st.session_state.keywords[:10]
                
                             # Create a simple horizontal bar
                keyword_df = pd.DataFrame({
                    'Keyword': top_keywords,
                    'Score': range(len(top_keywords), 0, -1)  # Descending score for visualization
                })
                
                fig = px.bar(
                    keyword_df, 
                    y='Keyword', 
                    x='Score',
                    orientation='h',
                    color='Score',
                    color_continuous_scale=['#4b6fff', '#66fcf1'],
                    title="Top Keywords by Relevance"
                )
                fig.update_layout(
                    paper_bgcolor='#1e1e1e',
                    plot_bgcolor='#2d2d2d',
                    font={'color': 'white'},
                    title_font_size=14,
                    height=250,
                    margin=dict(l=10, r=10, t=40, b=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
                st.sidebar.plotly_chart(fig, use_container_width=True)
               
        
        # If we have summary data, show the compression ratio
        if action == "Summarize" and 'final_summary' in locals():
            # Compression gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = compression_ratio,
                title = {'text': "Compression Ratio (%)", 'font': {'color': 'white', 'size': 14}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#4b6fff"},
                    'bgcolor': "gray",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': '#FFD166'},
                        {'range': [50, 100], 'color': '#06D6A0'}
                    ]
                }
            ))
            fig.update_layout(
                paper_bgcolor='#1e1e1e', 
                font={'color': 'white'},
                height=150,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # If we have chat history, show confidence trend
        if action == "Chat with the Text" and len(st.session_state.chat_history) > 1:
            st.markdown('<div class="card-header">📈 Answer Confidence</div>', unsafe_allow_html=True)
            confidence_data = [{'Question': f"Q{i+1}", 'Confidence': conf} 
                            for i, (_, _, conf) in enumerate(st.session_state.chat_history)]
            df = pd.DataFrame(confidence_data)
            
            fig = px.line(
                df, 
                x='Question', 
                y='Confidence',
                markers=True,
                line_shape='linear'
            )
            fig.update_traces(line=dict(color='#4b6fff', width=3), marker=dict(size=10))
            fig.update_layout(
                title='Answer Confidence Trend',
                title_font_color='white',
                title_font_size=14,
                xaxis_title='Questions',
                yaxis_title='Score',
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#2d2d2d',
                font={'color': 'white', 'size': 11},
                yaxis=dict(range=[0, 1]),
                height=250,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.sidebar.plotly_chart(fig, use_container_width=True)
    else:
        # Empty state for right sidebar
        st.markdown("""
        <div class="content-box" style="text-align: center; color: #888;">
            <h3>Analytics Dashboard</h3>
            <p>Extract content from a URL to see analytics, keywords, and entities.</p>
        </div>
        """, unsafe_allow_html=True)