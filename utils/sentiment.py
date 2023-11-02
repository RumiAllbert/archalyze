from transformers import pipeline

# Load sentiment analysis pipeline with BERT
sentiment_pipeline = pipeline("sentiment-analysis")

def compute_sentiments(text, max_length=512):
    # Tokenize text into words
    words = text.split()
    
    # Create chunks ensuring each chunk has no more than `max_length` words
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) < max_length:
            current_chunk.append(word)
            current_length += len(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))  # Add the last chunk
    
    # Compute sentiment scores for each chunk
    scores = [sentiment_pipeline(chunk)[0] for chunk in chunks]
    
    # Convert 'LABEL' to a numerical score: 'NEGATIVE' -> -1, 'POSITIVE' -> 1
    numerical_scores = [-1 if score['label'] == 'NEGATIVE' else 1 for score in scores]
    return numerical_scores
