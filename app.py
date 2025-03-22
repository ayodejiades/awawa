import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from newspaper import Article
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from textblob import TextBlob
import urllib.parse
import re

# Configure NLTK
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class NewsAnalyzer:
    def __init__(self, topic="technology"):
        self.topic = topic
        self.articles = []
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english')).union(
            {'said', 'would', 'could', 'new', 'year', 'today'})
        
    def _clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return ' '.join([word for word in word_tokenize(text) 
                        if word not in self.stop_words and len(word) > 2])

    def fetch_news(self, num_articles=50):
        try:
            query = urllib.parse.quote_plus(self.topic)
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en")
            
            for entry in feed.entries[:num_articles]:
                try:
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    self.articles.append({
                        'title': entry.title,
                        'text': self._clean_text(article.text),
                        'source': entry.source.title if hasattr(entry, 'source') else "Unknown",
                        'link': entry.link,
                        'published': entry.published
                    })
                except Exception as e:
                    st.warning(f"Skipping article: {str(e)}")
                    continue
            return True
        except Exception as e:
            st.error(f"News fetch failed: {str(e)}")
            return False

    def analyze_sentiment(self):
        for article in self.articles:
            vs = self.analyzer.polarity_scores(article['text'])
            article.update({
                'sentiment': 'Positive' if vs['compound'] >= 0.05 else 
                            'Negative' if vs['compound'] <= -0.05 else 'Neutral',
                'sentiment_score': vs['compound'],
                'polarity': TextBlob(article['text']).sentiment.polarity,
                'subjectivity': TextBlob(article['text']).sentiment.subjectivity
            })

# Streamlit App
def main():
    st.set_page_config(page_title="News Analyzer", layout="wide")
    st.title("📰 AI News Sentiment Analysis")
    
    with st.sidebar:
        topic = st.text_input("Enter topic:", "AI")
        num_articles = st.slider("Articles to analyze:", 10, 100, 30)
        if st.button("Analyze News"):
            with st.spinner("Processing..."):
                analyzer = NewsAnalyzer(topic)
                if analyzer.fetch_news(num_articles):
                    analyzer.analyze_sentiment()
                    st.session_state.analyzer = analyzer
                    st.success("Analysis complete!")
                else:
                    st.error("Failed to fetch news")

    if 'analyzer' in st.session_state:
        analyzer = st.session_state.analyzer
        
        # Sentiment Summary
        st.header("Sentiment Overview")
        sentiment_counts = Counter([a['sentiment'] for a in analyzer.articles])
        fig = px.pie(values=sentiment_counts.values(), 
                    names=sentiment_counts.keys(),
                    color=sentiment_counts.keys(),
                    color_discrete_map={'Positive':'green', 'Neutral':'orange', 'Negative':'red'})
        st.plotly_chart(fig, use_container_width=True)

        # Articles Display
        st.header("Analyzed Articles")
        for article in analyzer.articles:
            with st.expander(article['title']):
                st.markdown(f"**{article['sentiment']}** ({article['sentiment_score']:.2f})")
                st.caption(f"Source: {article['source']} | Published: {article['published']}")
                st.write(article['text'][:500] + "...")
                st.markdown(f"[Read full article]({article['link']})")

if __name__ == "__main__":
    main()
