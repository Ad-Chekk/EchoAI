import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import pipeline

#from file
from text_preprocessing import simple_sent_tokenize, english_stopwords

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

# Load keyword/entity extractor (improved)

# Load QA model for chatbot with error handling
@st.cache_resource
def get_qa_pipeline():
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        return None