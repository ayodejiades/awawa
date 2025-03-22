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

class NewsSentimentAnalyzer:
    def __init__(self, topic="technology"):
        """Initialize the News Sentiment Analyzer with a specific topic."""
        self.topic = topic
        self.articles = []
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.stopwords = set(stopwords.words('english'))
        self.additional_stopwords = {'said', 'also', 'would', 'could', 'one', 'two', 'three', 'new', 'year', 'today', 'says'}
        self.stopwords.update(self.additional_stopwords)
        
    def _sanitize_query(self, query):
        """Sanitize the query for safe search."""
        # URL encode the query to handle special characters and spaces
        return urllib.parse.quote_plus(query)
        
    def about_us():
        st.sidebar.title("About Us")
        st.sidebar.markdown("""
        Welcome to the **WanNkan: AI-Powered News Sentiment Analyzer**! This app is designed to help you stay informed by analyzing the sentiment of recent news articles and providing a quick summary.
    
        ### How It Works:
        1. Enter a topic of interest (e.g., technology, politics, sports).  
        2. The app fetches the latest news articles from Google News RSS.  
        3. It analyzes the sentiment of each article (Positive, Negative, or Neutral).  
        4. A summary and a word cloud are generated to give you insights at a glance.  
    
        ### Meet the Team:
        - **Ayodeji Adesegun**: Chief Mathematical Officer (CMO).  
         
    
        ### Contact Us:
        Have questions or feedback? Reach out to us at [contact@newsanalyzer.com](mailto:contact@newsanalyzer.com).
        """)
   
    def fetch_news_from_google_rss(self, num_articles=10):
        """Fetch news articles from Google News RSS feed."""
        with st.spinner(f"Fetching news articles about '{self.topic}'..."):
            # Sanitize the topic for the URL
            sanitized_topic = self._sanitize_query(self.topic)
            
            # Format the Google News RSS URL with the sanitized topic
            rss_url = f"https://news.google.com/rss/search?q={sanitized_topic}&hl=en-US&gl=US&ceid=US:en"
            
            # Parse the RSS feed
            news_feed = feedparser.parse(rss_url)
            
            # Process each entry in the feed
            for i, entry in enumerate(news_feed.entries[:num_articles]):
                try:
                    article_data = {
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.published,
                        "published_parsed": entry.published_parsed,
                        "source": entry.source.title if hasattr(entry, 'source') else "Unknown"
                    }
                    
                    # Extract the full text content using newspaper3k
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    article_data["text"] = article.text
                    article_data["authors"] = article.authors
                    article_data["publish_date"] = article.publish_date
                    article_data["top_image"] = article.top_image if hasattr(article, 'top_image') else ""
                    
                    # Extract keywords
                    article.nlp()
                    article_data["keywords"] = article.keywords
                    
                    self.articles.append(article_data)
                    
                except Exception as e:
                    st.warning(f"Error fetching article: {str(e)}")
                    continue
                    
            return self.articles
    
    def analyze_sentiment(self):
        """Analyze the sentiment of collected articles using multiple methods."""
        with st.spinner("Analyzing sentiment of articles..."):
            for article in self.articles:
                # Use VADER for sentiment analysis
                vader_scores = self.sentiment_analyzer.polarity_scores(article["text"])
                
                # Use TextBlob for additional sentiment analysis
                blob = TextBlob(article["text"])
                textblob_polarity = blob.sentiment.polarity
                textblob_subjectivity = blob.sentiment.subjectivity
                
                # Determine overall sentiment from VADER
                compound_score = vader_scores["compound"]
                if compound_score >= 0.05:
                    sentiment = "Positive"
                elif compound_score <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                    
                # Add sentiment data to the article
                article["sentiment"] = sentiment
                article["sentiment_score"] = compound_score
                article["vader_scores"] = vader_scores
                article["textblob_polarity"] = textblob_polarity
                article["textblob_subjectivity"] = textblob_subjectivity
                
            return self.articles
    
    def generate_summaries(self, max_length=150):
        """Generate AI summaries of the articles."""
        with st.spinner("Generating article summaries..."):
            for article in self.articles:
                try:
                    # Prepare text for summarization (handle long articles)
                    text = article["text"]
                    if len(text.split()) > 1024:  # Truncate long articles for BART model
                        text = " ".join(text.split()[:1024])
                    
                    # Generate summary using BART
                    if text.strip():  # Ensure there's text to summarize
                        summary = self.summarizer(text, max_length=max_length, min_length=30, 
                                                do_sample=False)[0]['summary_text']
                        article["ai_summary"] = summary
                    else:
                        article["ai_summary"] = "No text available for summarization."
                    
                except Exception as e:
                    st.warning(f"Error generating summary: {str(e)}")
                    article["ai_summary"] = "Summary generation failed."
                    
            return self.articles
    
    def generate_wordcloud(self):
        """Generate a word cloud from all articles."""
        with st.spinner("Generating word cloud..."):
            # Combine all article texts
            all_text = ' '.join([article["text"] for article in self.articles])
            
            # Clean and tokenize the text
            all_text = re.sub(r'[^\w\s]', '', all_text.lower())
            tokens = word_tokenize(all_text)
            
            # Remove stopwords
            filtered_tokens = [word for word in tokens if word not in self.stopwords and len(word) > 2]
            
            # Create word frequency dictionary
            word_freq = Counter(filtered_tokens)
            
            # Create and configure the WordCloud object
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                max_words=100,
                colormap='viridis',
                contour_width=1, 
                contour_color='steelblue'
            ).generate_from_frequencies(word_freq)
            
            # Create a BytesIO object to store the image
            img_bytes = BytesIO()
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(img_bytes, format='png')
            plt.close()
            img_bytes.seek(0)
            
            # Convert to base64 for downloading in the web app
            wordcloud_b64 = base64.b64encode(img_bytes.read()).decode()
            
            return wordcloud, wordcloud_b64
    
    def extract_entities(self):
        """Extract named entities from the articles."""
        try:
            from nltk import ne_chunk
            from nltk.tag import pos_tag
            
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            
            entity_counts = Counter()
            
            for article in self.articles:
                text = article["text"]
                tokens = word_tokenize(text)
                tagged = pos_tag(tokens)
                entities = ne_chunk(tagged)
                
                for chunk in entities:
                    if hasattr(chunk, 'label'):
                        entity_name = ' '.join([c[0] for c in chunk])
                        entity_counts[entity_name] += 1
                        
            return dict(entity_counts.most_common(20))
        except Exception as e:
            st.warning(f"Error extracting entities: {str(e)}")
            return {}
    
    def get_trending_keywords(self):
        """Get trending keywords from all articles."""
        all_keywords = []
        for article in self.articles:
            if "keywords" in article and article["keywords"]:
                all_keywords.extend(article["keywords"])
        
        # Count keyword occurrences
        keyword_counts = Counter(all_keywords)
        return dict(keyword_counts.most_common(15))
            
    def export_results(self):
        """Export the analysis results to a JSON file."""
        output_file = f"{self.topic}_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create a simplified version for export
        export_data = []
        for article in self.articles:
            export_item = {
                "headline": article["title"],
                "source": article["source"],
                "published": article["published"],
                "sentiment": article["sentiment"],
                "sentiment_score": article["sentiment_score"],
                "summary": article["ai_summary"],
                "link": article["link"],
                "keywords": article.get("keywords", [])
            }
            export_data.append(export_item)
            
        return json.dumps(export_data, indent=2, ensure_ascii=False)

# Streamlit web application
def main():
    st.set_page_config(
        page_title="AI-Powered News Sentiment Analyzer",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 AI-Powered News Sentiment Analyzer")
    
    st.markdown("""
    Analyze the sentiment of recent news articles on any topic using AI. This tool will:
    - Fetch recent news articles from Google News
    - Analyze the sentiment (positive, negative, or neutral)
    - Generate concise summaries
    - Create visualizations to help understand the news landscape
    """)

    # Add About Us page to the sidebar
    about_us()
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("Search Parameters")
        topic = st.text_input("Enter a news topic to analyze:", "climate change")
        
        # Handle potentially sensitive search queries
        st.caption("Note: Search queries are automatically sanitized for safety.")
        
        num_articles = st.slider("Number of articles to analyze:", 
                                min_value=3, max_value=20, value=7,
                                help="More articles will take longer to process")
        
        summary_length = st.slider("Summary length (words):", 
                                min_value=50, max_value=250, value=150)
        
        st.header("Analysis Options")
        generate_wordcloud = st.checkbox("Generate word cloud", value=True)
        extract_entities_option = st.checkbox("Extract named entities", value=True)
        
        analyze_button = st.button("Analyze News", type="primary")
    
    # Main content area placeholders
    progress_bar = st.empty()
    error_message = st.empty()
    sentiment_summary = st.empty()
    charts_container = st.container()
    articles_container = st.container()
    
    # When analyze button is clicked
    if analyze_button:
        try:
            # Initialize analyzer and fetch articles
            analyzer = NewsSentimentAnalyzer(topic=topic)
            
            # Progress bar
            progress_state = 0
            progress_bar_obj = progress_bar.progress(progress_state, "Initializing...")
            
            # Fetch articles
            articles = analyzer.fetch_news_from_google_rss(num_articles=num_articles)
            progress_state += 0.2
            progress_bar_obj.progress(progress_state, "Fetching articles...")
            
            if not articles:
                error_message.error(f"No articles found for topic: '{topic}'. Please try a different search term.")
                progress_bar.empty()
                return
                
            # Analyze sentiment
            analyzer.analyze_sentiment()
            progress_state += 0.2
            progress_bar_obj.progress(progress_state, "Analyzing sentiment...")
            
            # Generate summaries
            analyzer.generate_summaries(max_length=summary_length)
            progress_state += 0.2
            progress_bar_obj.progress(progress_state, "Generating summaries...")
            
            # Generate word cloud if selected
            wordcloud_img = None
            wordcloud_b64 = None
            if generate_wordcloud:
                wordcloud_img, wordcloud_b64 = analyzer.generate_wordcloud()
                progress_state += 0.2
                progress_bar_obj.progress(progress_state, "Creating visualizations...")
            
            # Extract entities if selected
            entities = None
            if extract_entities_option:
                entities = analyzer.extract_entities()
                progress_state += 0.2
                progress_bar_obj.progress(progress_state, "Extracting entities...")
            
            # Get trending keywords
            trending_keywords = analyzer.get_trending_keywords()
            
            # Export results
            json_export = analyzer.export_results()
            progress_bar_obj.progress(1.0, "Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            # Display sentiment summary
            positive_count = sum(1 for article in analyzer.articles if article["sentiment"] == "Positive")
            negative_count = sum(1 for article in analyzer.articles if article["sentiment"] == "Negative")
            neutral_count = sum(1 for article in analyzer.articles if article["sentiment"] == "Neutral")
            
            total_articles = len(analyzer.articles)
            sentiment_summary.markdown(f"""
            ### Analysis Results for "{topic}"
            
            Found **{total_articles}** articles with the following sentiment distribution:
            - 😃 Positive: {positive_count} articles ({positive_count/total_articles*100:.1f}%)
            - 😐 Neutral: {neutral_count} articles ({neutral_count/total_articles*100:.1f}%)  
            - 😞 Negative: {negative_count} articles ({negative_count/total_articles*100:.1f}%)
            """)
            
            # Create charts in the chart container
            with charts_container:
                st.subheader("Visualizations")
                
                col1, col2 = st.columns(2)
                
                # Sentiment Distribution Chart
                with col1:
                    sentiment_data = {
                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                        'Count': [positive_count, neutral_count, negative_count]
                    }
                    fig_sentiment = px.pie(
                        sentiment_data, 
                        values='Count', 
                        names='Sentiment',
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': '#66bb6a',
                            'Neutral': '#ffb74d',
                            'Negative': '#ef5350'
                        },
                        title='Sentiment Distribution'
                    )
                    st.plotly_chart(fig_sentiment)
                
                # Word Cloud
                with col2:
                    if wordcloud_img and wordcloud_b64:
                        st.subheader("Word Cloud")
                        st.image(wordcloud_img.to_array(), use_column_width=True)
                        
                        # Download button for word cloud
                        st.download_button(
                            label="Download Word Cloud",
                            data=base64.b64decode(wordcloud_b64),
                            file_name=f"{topic}_wordcloud.png",
                            mime="image/png"
                        )
                
                # Trending Keywords Chart
                if trending_keywords:
                    keywords_df = pd.DataFrame({
                        'Keyword': list(trending_keywords.keys()),
                        'Frequency': list(trending_keywords.values())
                    }).sort_values('Frequency', ascending=False)
                    
                    fig_keywords = px.bar(
                        keywords_df,
                        x='Keyword',
                        y='Frequency',
                        title='Trending Keywords',
                        color='Frequency',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_keywords)
                
                # Named Entities Chart
                if entities:
                    entities_df = pd.DataFrame({
                        'Entity': list(entities.keys()),
                        'Mentions': list(entities.values())
                    }).sort_values('Mentions', ascending=False).head(15)
                    
                    fig_entities = px.bar(
                        entities_df,
                        x='Entity',
                        y='Mentions',
                        title='Top Named Entities',
                        color='Mentions',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_entities)
                
                # Download button for JSON export
                st.download_button(
                    label="Download Full Analysis (JSON)",
                    data=json_export,
                    file_name=f"{topic}_news_analysis.json",
                    mime="application/json"
                )
            
            # Display individual articles
            with articles_container:
                st.subheader("Analyzed Articles")
                
                for i, article in enumerate(analyzer.articles):
                    with st.expander(f"{i+1}. {article['title']}"):
                        # Format the article content
                        st.markdown(f"**Source:** {article['source']}")
                        st.markdown(f"**Published:** {article['published']}")
                        
                        # Sentiment with appropriate color
                        sentiment_color = {
                            "Positive": "green",
                            "Neutral": "orange",
                            "Negative": "red"
                        }.get(article["sentiment"], "blue")
                        
                        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};font-weight:bold'>{article['sentiment']}</span> (Score: {article['sentiment_score']:.2f})", unsafe_allow_html=True)
                        
                        st.markdown("**AI Summary:**")
                        st.markdown(f"> {article['ai_summary']}")
                        
                        # Show TextBlob sentiment metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Polarity:** {article['textblob_polarity']:.2f} (-1 to +1)")
                        with col2:
                            st.markdown(f"**Subjectivity:** {article['textblob_subjectivity']:.2f} (0 to 1)")
                        
                        # Keywords
                        if "keywords" in article and article["keywords"]:
                            st.markdown("**Keywords:**")
                            st.write(", ".join(article["keywords"]))
                        
                        # Link to original article
                        st.markdown(f"[Read Original Article]({article['link']})")
                
        except Exception as e:
            error_message.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please try again with a different topic or fewer articles.")
    
    # Show instructions if no analysis has been run
    if not analyze_button:
        st.info("👈 Enter a topic in the sidebar and click 'Analyze News' to get started!")
        
        with st.expander("Sample Topics"):
            st.markdown("""
            Here are some sample topics to try:
            - climate change
            - artificial intelligence
            - space exploration
            - cryptocurrency
            - global economy
            - healthcare innovation
            - renewable energy
            
            You can also try more specific searches like:
            - Tesla electric vehicles
            - COVID-19 vaccine research
            - Olympic games preparations
            """)
            
        with st.expander("How it Works"):
            st.markdown("""
            This tool uses several AI and NLP techniques:
            
            1. **News Collection**: Fetches recent articles from Google News RSS feeds
            2. **Content Extraction**: Uses newspaper3k to extract full article text
            3. **Sentiment Analysis**: 
               - VADER (Valence Aware Dictionary and sEntiment Reasoner) for rule-based sentiment scoring
               - TextBlob for additional polarity and subjectivity analysis
            4. **Summarization**: Uses BART-large-CNN transformer model to generate concise summaries
            5. **Visualization**: Creates word clouds, entity charts, and sentiment distributions
            """)
            
        with st.expander("Tips for Best Results"):
            st.markdown("""
            - Use specific topics for more focused results
            - Try comparing similar topics to see sentiment differences
            - Analyze 5-10 articles for best performance balance
            - Download the JSON export for further analysis in other tools
            """)

if __name__ == "__main__":
    main()
