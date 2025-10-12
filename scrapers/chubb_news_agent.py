"""
News Agent for Chubb Insurance Company
Scrapes news headlines and performs sentiment analysis
"""

import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import json
from textblob import TextBlob
import time


class ChubbNewsAgent:
    def __init__(self):
        self.company_name = "Chubb insurance"
        self.search_terms = ["Chubb insurance", "Chubb Limited", "Chubb Corp"]
        self.news_data = []
        
    def fetch_google_news(self, days_back: int = 10) -> List[Dict]:
        """
        Fetch news from Google News RSS feed
        """
        print(f"📰 Fetching Google News for {self.company_name}...")
        news_items = []
        
        try:
            # Google News RSS feed URL
            search_query = "Chubb+insurance"
            url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            
            items = soup.find_all('item')
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for item in items:
                try:
                    title = item.title.text if item.title else ""
                    link = item.link.text if item.link else ""
                    pub_date_str = item.pubDate.text if item.pubDate else ""
                    
                    # Parse date
                    pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                    
                    # Only include articles within the time window
                    if pub_date >= cutoff_date:
                        news_items.append({
                            'title': title,
                            'url': link,
                            'date': pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                            'source': 'Google News'
                        })
                except Exception as e:
                    print(f"⚠️  Error parsing item: {e}")
                    continue
                    
            print(f"✅ Found {len(news_items)} articles from Google News")
            
        except Exception as e:
            print(f"❌ Error fetching Google News: {e}")
            
        return news_items
    
    def fetch_bing_news(self, days_back: int = 10) -> List[Dict]:
        """
        Fetch news from Bing News Search (requires API key)
        Note: This requires BING_NEWS_API_KEY environment variable
        """
        print(f"📰 Fetching Bing News for {self.company_name}...")
        news_items = []
        
        try:
            import os
            api_key = os.environ.get('BING_NEWS_API_KEY')
            
            if not api_key:
                print("⚠️  Bing News API key not found. Skipping Bing News...")
                return news_items
            
            endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {
                "q": "Chubb insurance",
                "count": 50,
                "freshness": "Week",
                "mkt": "en-US"
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
            data = response.json()
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for article in data.get('value', []):
                try:
                    pub_date = datetime.strptime(article['datePublished'][:19], "%Y-%m-%dT%H:%M:%S")
                    
                    if pub_date >= cutoff_date:
                        news_items.append({
                            'title': article.get('name', ''),
                            'url': article.get('url', ''),
                            'date': pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                            'source': 'Bing News'
                        })
                except Exception as e:
                    print(f"⚠️  Error parsing Bing article: {e}")
                    continue
                    
            print(f"✅ Found {len(news_items)} articles from Bing News")
            
        except Exception as e:
            print(f"❌ Error fetching Bing News: {e}")
            
        return news_items
    
    def fetch_newsapi(self, days_back: int = 10) -> List[Dict]:
        """
        Fetch news from NewsAPI.org (requires API key)
        Note: This requires NEWS_API_KEY environment variable
        """
        print(f"📰 Fetching NewsAPI.org for {self.company_name}...")
        news_items = []
        
        try:
            import os
            api_key = os.environ.get('NEWS_API_KEY')
            
            if not api_key:
                print("⚠️  NewsAPI key not found. Skipping NewsAPI...")
                return news_items
            
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "Chubb insurance",
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": api_key,
                "pageSize": 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            for article in data.get('articles', []):
                try:
                    pub_date = datetime.strptime(article['publishedAt'][:19], "%Y-%m-%dT%H:%M:%S")
                    
                    news_items.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'date': pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                        'source': f"NewsAPI - {article.get('source', {}).get('name', 'Unknown')}"
                    })
                except Exception as e:
                    print(f"⚠️  Error parsing NewsAPI article: {e}")
                    continue
                    
            print(f"✅ Found {len(news_items)} articles from NewsAPI")
            
        except Exception as e:
            print(f"❌ Error fetching NewsAPI: {e}")
            
        return news_items
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using TextBlob
        Returns polarity (-1 to 1) and subjectivity (0 to 1)
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return {
                'sentiment': sentiment,
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3)
            }
        except Exception as e:
            print(f"⚠️  Error analyzing sentiment: {e}")
            return {
                'sentiment': 'Unknown',
                'polarity': 0.0,
                'subjectivity': 0.0
            }
    
    def collect_news(self, days_back: int = 10) -> pd.DataFrame:
        """
        Collect news from all sources
        """
        print(f"\n🔍 Starting news collection for past {days_back} days...")
        print("="*70)
        
        all_news = []
        
        # Fetch from different sources
        all_news.extend(self.fetch_google_news(days_back))
        time.sleep(1)  # Be respectful with requests
        
        all_news.extend(self.fetch_bing_news(days_back))
        time.sleep(1)
        
        all_news.extend(self.fetch_newsapi(days_back))
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_news = []
        
        for item in all_news:
            title_lower = item['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_news.append(item)
        
        print(f"\n📊 Total unique articles found: {len(unique_news)}")
        print("="*70)
        
        # Convert to DataFrame
        if unique_news:
            df = pd.DataFrame(unique_news)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            self.news_data = df
            return df
        else:
            print("⚠️  No news articles found!")
            return pd.DataFrame()
    
    def apply_sentiment_analysis(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply sentiment analysis to collected news
        """
        if df is None:
            df = self.news_data
        
        if df.empty:
            print("⚠️  No data to analyze!")
            return df
        
        print("\n🧠 Applying sentiment analysis...")
        print("="*70)
        
        sentiments = []
        for idx, row in df.iterrows():
            sentiment_data = self.analyze_sentiment(row['title'])
            sentiments.append(sentiment_data)
        
        # Add sentiment columns
        df['sentiment'] = [s['sentiment'] for s in sentiments]
        df['polarity'] = [s['polarity'] for s in sentiments]
        df['subjectivity'] = [s['subjectivity'] for s in sentiments]
        
        return df
    
    def generate_report(self, df: pd.DataFrame = None):
        """
        Generate a sentiment analysis report
        """
        if df is None:
            df = self.news_data
        
        if df.empty:
            print("⚠️  No data to report!")
            return
        
        print("\n" + "="*70)
        print("📈 SENTIMENT ANALYSIS REPORT - CHUBB INSURANCE COMPANY")
        print("="*70)
        
        # Overall statistics
        print(f"\n📅 Analysis Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"📰 Total Articles Analyzed: {len(df)}")
        
        # Sentiment breakdown
        print("\n🎭 Sentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment}: {count} articles ({percentage:.1f}%)")
        
        # Average scores
        print("\n📊 Average Sentiment Scores:")
        print(f"   Polarity: {df['polarity'].mean():.3f} (range: -1 to 1)")
        print(f"   Subjectivity: {df['subjectivity'].mean():.3f} (range: 0 to 1)")
        
        # Recent headlines
        print("\n📰 Recent Headlines (Top 10):")
        print("-"*70)
        for idx, row in df.head(10).iterrows():
            emoji = "🟢" if row['sentiment'] == "Positive" else "🔴" if row['sentiment'] == "Negative" else "⚪"
            print(f"\n{emoji} [{row['date'].strftime('%Y-%m-%d')}] {row['sentiment']} (Score: {row['polarity']:.2f})")
            print(f"   {row['title'][:100]}...")
            print(f"   Source: {row['source']}")
        
        # Most positive and negative
        print("\n"+"="*70)
        print("🌟 Most Positive Headline:")
        most_positive = df.loc[df['polarity'].idxmax()]
        print(f"   [{most_positive['date'].strftime('%Y-%m-%d')}] Score: {most_positive['polarity']:.3f}")
        print(f"   {most_positive['title']}")
        
        print("\n⚠️  Most Negative Headline:")
        most_negative = df.loc[df['polarity'].idxmin()]
        print(f"   [{most_negative['date'].strftime('%Y-%m-%d')}] Score: {most_negative['polarity']:.3f}")
        print(f"   {most_negative['title']}")
        
        print("\n" + "="*70)
    
    def save_results(self, filename: str = "chubb_news_sentiment.csv"):
        """
        Save results to CSV file
        """
        if self.news_data.empty:
            print("⚠️  No data to save!")
            return
        
        try:
            self.news_data.to_csv(filename, index=False)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")
    
    def run_analysis(self, days_back: int = 10):
        """
        Run the complete analysis pipeline
        """
        # Collect news
        df = self.collect_news(days_back)
        
        if df.empty:
            return None
        
        # Apply sentiment analysis
        df = self.apply_sentiment_analysis(df)
        self.news_data = df
        
        # Generate report
        self.generate_report(df)
        
        # Save results
        self.save_results()
        
        return df


def main():
    """
    Main function to run the news agent
    """
    print("\n" + "🏢"*35)
    print("   CHUBB INSURANCE COMPANY - NEWS SENTIMENT ANALYZER")
    print("🏢"*35 + "\n")
    
    # Initialize agent
    agent = ChubbNewsAgent()
    
    # Run analysis for past 10 days
    df = agent.run_analysis(days_back=10)
    
    if df is not None:
        print("\n✅ Analysis complete!")
        print(f"📊 View detailed results in: chubb_news_sentiment.csv")
    else:
        print("\n❌ Analysis failed - no news found!")


if __name__ == "__main__":
    main()
