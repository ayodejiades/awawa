import feedparser
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from newspaper import Article
from transformers import pipeline
import pandas as pd
from datetime import datetime
import json
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from textblob import TextBlob
import urllib.parse

# Download required NLTK data
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
        self.additional_stopwords = {'said', 'also', 'would', 'could', 'one', 'two', 'three', 'new', 'year', 'today', 'says'}
        self.stopwords.update(self.additional_stopwords)
        
    def _sanitize_query(self, query):
        return urllib.parse.quote_plus(query)
        
    def fetch_news_from_google_rss(self, num_articles=1000):
        with st.spinner(f"Fetching news articles about '{self.topic}'..."):
            sanitized_topic = self._sanitize_query(self.topic)
            rss_url = f"https://news.google.com/rss/search?q={sanitized_topic}&hl=en-US&gl=US&ceid=US:en"
            news_feed = feedparser.parse(rss_url)

            for i, entry in enumerate(news_feed.entries[:num_articles]):
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
                    st.warning(f"Error fetching article {i + 1}: {str(e)}")
                    continue
            return self.articles
    
    # Keep other class methods the same as before...

def about_us():
    st.sidebar.title("About Us")
    st.sidebar.markdown("""
    Welcome to the **WanNkan: AI-Powered News Sentiment Analyzer**!
    ### Meet the Team:
    - **Ayodeji Adesegun**: Chief Mathematical Officer (CMO)
    """)

def display_analysis_summary(analyzer, topic):
    positive_count = sum(1 for article in analyzer.articles if article["sentiment"] == "Positive")
    negative_count = sum(1 for article in analyzer.articles if article["sentiment"] == "Negative")
    neutral_count = sum(1 for article in analyzer.articles if article["sentiment"] == "Neutral")
    total_articles = len(analyzer.articles)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Positive Articles", f"{positive_count} ({positive_count/total_articles*100:.1f}%)")
    with col2:
        st.metric("Neutral Articles", f"{neutral_count} ({neutral_count/total_articles*100:.1f}%)")
    with col3:
        st.metric("Negative Articles", f"{negative_count} ({negative_count/total_articles*100:.1f}%)")
    
    st.plotly_chart(px.pie(
        names=['Positive', 'Neutral', 'Negative'],
        values=[positive_count, neutral_count, negative_count],
        title='Sentiment Distribution',
        color=['Positive', 'Neutral', 'Negative'],
        color_discrete_map={'Positive': '#66bb6a', 'Neutral': '#ffb74d', 'Negative': '#ef5350'}
    ))

def display_paginated_articles(articles, page_size=10):
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    
    total_pages = len(articles) // page_size + (1 if len(articles) % page_size else 0)
    col1, col2, col3 = st.columns([2,4,2])
    
    with col1:
        if st.button("Previous") and st.session_state.page_number > 0:
            st.session_state.page_number -= 1
    with col3:
        if st.button("Next") and st.session_state.page_number < total_pages - 1:
            st.session_state.page_number += 1
    
    start_idx = st.session_state.page_number * page_size
    end_idx = start_idx + page_size
    
    for article in articles[start_idx:end_idx]:
        with st.expander(f"{article['title']}"):
            st.markdown(f"**Source:** {article['source']}")
            st.markdown(f"**Sentiment:** {article['sentiment']} ({article['sentiment_score']:.2f})")
            st.markdown(f"**Summary:** {article['ai_summary']}")
            st.markdown(f"[Read Full Article]({article['link']})")

def main():
    st.set_page_config(page_title="News Analyzer", page_icon="📰", layout="wide")
    st.title("📰 AI-Powered News Sentiment Analyzer")
    about_us()

    with st.sidebar:
        topic = st.text_input("Enter news topic:", "climate change")
        num_articles = st.slider("Number of articles:", 10, 1000, 100)
        analyze_button = st.button("Analyze News", type="primary")

    if analyze_button:
        analyzer = NewsSentimentAnalyzer(topic)
        with st.spinner("Analyzing news articles..."):
            analyzer.fetch_news_from_google_rss(num_articles)
            analyzer.analyze_sentiment()
            analyzer.generate_summaries()
        
        st.session_state['analyzer'] = analyzer
        st.session_state['current_page'] = 'summary'

    if 'analyzer' in st.session_state:
        analyzer = st.session_state['analyzer']
        
        page_options = ["Summary", "Visualizations", "Articles", "Data Export"]
        page = st.sidebar.radio("Navigation", page_options)
        
        if page == "Summary":
            st.header("Analysis Summary")
            display_analysis_summary(analyzer, topic)
            
        elif page == "Visualizations":
            st.header("Data Visualizations")
            wordcloud_img, _ = analyzer.generate_wordcloud()
            if wordcloud_img:
                st.image(wordcloud_img.to_array(), use_column_width=True)
            
            st.plotly_chart(px.bar(
                pd.DataFrame(analyzer.get_trending_keywords().items(), columns=['Keyword', 'Frequency']),
                x='Keyword', y='Frequency', title='Trending Keywords'
            ))
            
        elif page == "Articles":
            st.header("Analyzed Articles")
            display_paginated_articles(analyzer.articles)
            
        elif page == "Data Export":
            st.header("Export Data")
            st.download_button(
                label="Download Full Analysis (JSON)",
                data=analyzer.export_results(),
                file_name=f"{topic}_analysis.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
