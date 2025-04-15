import re


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


# Function to create paragraph chunks from text
#Breaks preprocessed text into paragraph-based chunks
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








#####use this function only for novels, whitepapers
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


