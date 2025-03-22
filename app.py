# news_sentiment_analyzer.py
import asyncio
import sys
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from newspaper import Article
from transformers import pipeline
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
import streamlit as st
import plotly.express as px
from collections import Counter
from textblob import TextBlob
import urllib.parse

# Configure event loop policy for Streamlit Cloud
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class NewsSentimentAnalyzer:
    def __init__(self, topic="technology"):
        self.topic = topic
        self.articles = []
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.stopwords = set(stopwords.words('english'))
        self.additional_stopwords = {'said', 'also', 'would', 'could', 'one', 'two', 
                                    'three', 'new', 'year', 'today', 'says'}
        self.stopwords.update(self.additional_stopwords)
        
    def _sanitize_query(self, query):
        return urllib.parse.quote_plus(query)
        
    def fetch_news_from_google_rss(self, num_articles=100):
        with st.spinner(f"Fetching news articles about '{self.topic}'..."):
            sanitized_topic = self._sanitize_query(self.topic)
            rss_url = f"https://news.google.com/rss/search?q={sanitized_topic}&hl=en-US&gl=US&ceid=US:en"
            
            try:
                news_feed = feedparser.parse(rss_url)
                articles_to_fetch = min(num_articles, len(news_feed.entries))
                
                for i, entry in enumerate(news_feed.entries[:articles_to_fetch]):
                    try:
                        article_data = {
                            "title": entry.title,
                            "link": entry.link,
                            "published": entry.published,
                            "source": entry.source.title if hasattr(entry, 'source') else "Unknown"
                        }

                        article = Article(entry.link)
                        article.download()
                        article.parse()
                        article_data["text"] = article.text
                        article.nlp()
                        article_data["keywords"] = article.keywords
                        self.articles.append(article_data)

                    except Exception as e:
                        st.warning(f"Error processing article {i+1}: {str(e)}")
                        continue
                        
                return self.articles
                
            except Exception as e:
                st.error(f"Failed to fetch news feed: {str(e)}")
                return []

    # Keep other class methods the same as previous version...

def about_us():
    st.sidebar.title("About Us")
    st.sidebar.markdown("""
    **AI-Powered News Sentiment Analyzer**
    
    ### Core Team:
    - **Ayodeji Adesegun**: Chief Mathematical Officer (CMO)
    - **AI Team**: NLP & Machine Learning Experts
    
    ### Contact:
    [contact@newsanalyzer.com](mailto:contact@newsanalyzer.com)
    """)

# Rest of the functions remain the same as previous version...

def main():
    st.set_page_config(
        page_title="News Sentiment Analyzer",
        page_icon="📰",
        layout="wide",
        menu_items={
            'Get Help': 'https://example.com/help',
            'Report a bug': "https://example.com/bug",
            'About': "AI-powered news sentiment analysis tool"
        }
    )
    
    st.title("📰 AI News Sentiment Analyzer")
    about_us()

    with st.sidebar:
        topic = st.text_input("Enter news topic:", "artificial intelligence")
        num_articles = st.slider("Number of articles:", 10, 500, 100)
        analyze_button = st.button("Analyze News", type="primary")

    # Rest of main function remains the same...

if __name__ == "__main__":
    main()
