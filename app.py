import streamlit as st
from collections import Counter
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from utils.network import visualize_character_network
from utils.sentiment import compute_sentiments

# Download stopwords and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

def word_frequency_analysis(text):
    st.subheader("Word Frequency Analysis")
    
    # Tokenize and filter out stopwords and non-alphabetic words
    tokens = [word for word in word_tokenize(text.lower()) if word.isalpha()]
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Count frequencies
    word_freq = Counter(filtered_tokens)
    
    # Convert to DataFrame for visualization
    df_word_freq = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
    
    # Display bar chart of word frequencies
    df_word_freq = df_word_freq.sort_values('Frequency', ascending=False)
    st.bar_chart(df_word_freq.set_index('Word'))

def sentiment_analysis(text):
    st.subheader("Sentiment Analysis")
    
    # Analyze sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Convert sentiment to DataFrame for visualization
    df_sentiment = pd.DataFrame({
        'Metric': ['Polarity', 'Subjectivity'],
        'Value': [sentiment.polarity, sentiment.subjectivity]
    })
    
    # Display bar chart of sentiment
    st.bar_chart(df_sentiment.set_index('Metric'))

def visual_story_arc(text):
    st.subheader("Visual Story Arc")
    sentiments = compute_sentiments(text)
    st.line_chart(sentiments)

# Streamlit App
def main():
    st.title("Archalyze - Analyze Your Story")
    
    uploaded_file = st.file_uploader("Upload your document", type=['txt'])
    
    if uploaded_file is not None:
        # Read the uploaded file
        text = uploaded_file.read().decode('utf-8')
        
       # Page navigation
        page = st.sidebar.selectbox(
            "Choose an analysis type",
            ("Word Frequency Analysis", "Sentiment Analysis", "Visual Story Arc", "Character Network")
        )
        
        if page == "Word Frequency Analysis":
            word_frequency_analysis(text)
        elif page == "Sentiment Analysis":
            sentiment_analysis(text)
        elif page == "Visual Story Arc":
            visual_story_arc(text)
        elif page == "Character Network":
            window_size = st.sidebar.slider("Window Size for Character Co-occurrence", 50, 500, 150)
            visualize_character_network(text, window_size)

if __name__ == "__main__":
    main()
