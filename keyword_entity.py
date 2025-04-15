import re
from text_preprocessing import english_stopwords


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
