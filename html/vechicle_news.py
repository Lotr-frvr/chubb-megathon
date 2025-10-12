"""
News Agent for Tesla Model 3
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


class VehicleNewsAgent:
    def __init__(self, vehicle_name="Tesla Model 3"):
        self.vehicle_name = vehicle_name
        self.search_terms = [
            vehicle_name,
            f"{vehicle_name} reviews",
            f"{vehicle_name} launch",
            f"{vehicle_name} performance",
        ]
        self.news_data = []
        
    def fetch_google_news(self, days_back: int = 10) -> List[Dict]:
        print(f"📰 Fetching Google News for {self.vehicle_name}...")
        news_items = []
        
        try:
            search_query = "+".join(self.vehicle_name.split())
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
                    pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                    
                    if pub_date >= cutoff_date:
                        news_items.append({
                            'title': title,
                            'url': link,
                            'date': pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                            'source': 'Google News'
                        })
                except Exception as e:
                    print(f"⚠️ Error parsing item: {e}")
                    continue
                    
            print(f"✅ Found {len(news_items)} articles from Google News")
            
        except Exception as e:
            print(f"❌ Error fetching Google News: {e}")
            
        return news_items
    
    def fetch_newsapi(self, days_back: int = 10) -> List[Dict]:
        print(f"📰 Fetching NewsAPI.org for {self.vehicle_name}...")
        news_items = []
        
        try:
            import os
            api_key = os.environ.get('NEWS_API_KEY')
            if not api_key:
                print("⚠️ NewsAPI key not found. Skipping NewsAPI...")
                return news_items
            
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": self.vehicle_name,
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
                    print(f"⚠️ Error parsing NewsAPI article: {e}")
                    continue
                    
            print(f"✅ Found {len(news_items)} articles from NewsAPI")
            
        except Exception as e:
            print(f"❌ Error fetching NewsAPI: {e}")
            
        return news_items
    
    def analyze_sentiment(self, text: str) -> Dict:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
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
            print(f"⚠️ Error analyzing sentiment: {e}")
            return {'sentiment': 'Unknown', 'polarity': 0.0, 'subjectivity': 0.0}
    
    def collect_news(self, days_back: int = 10) -> pd.DataFrame:
        print(f"\n🔍 Collecting news for {self.vehicle_name} (past {days_back} days)...")
        print("="*70)
        all_news = []
        all_news.extend(self.fetch_google_news(days_back))
        time.sleep(1)
        all_news.extend(self.fetch_newsapi(days_back))
        
        # Deduplicate
        seen_titles = set()
        unique_news = []
        for item in all_news:
            title_lower = item['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_news.append(item)
        
        print(f"\n📊 Total unique articles: {len(unique_news)}")
        print("="*70)
        
        if unique_news:
            df = pd.DataFrame(unique_news)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            self.news_data = df
            return df
        else:
            print("⚠️ No news articles found!")
            return pd.DataFrame()
    
    def apply_sentiment_analysis(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.news_data
        if df.empty:
            print("⚠️ No data to analyze!")
            return df
        
        print("\n🧠 Applying sentiment analysis...")
        print("="*70)
        sentiments = [self.analyze_sentiment(row['title']) for _, row in df.iterrows()]
        df['sentiment'] = [s['sentiment'] for s in sentiments]
        df['polarity'] = [s['polarity'] for s in sentiments]
        df['subjectivity'] = [s['subjectivity'] for s in sentiments]
        return df
    
    def generate_report(self, df: pd.DataFrame = None):
        if df is None:
            df = self.news_data
        if df.empty:
            print("⚠️ No data to report!")
            return
        
        print("\n" + "="*70)
        print(f"📈 SENTIMENT REPORT - {self.vehicle_name.upper()}")
        print("="*70)
        print(f"\n📅 Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"📰 Articles: {len(df)}")
        
        sentiment_counts = df['sentiment'].value_counts()
        print("\n🎭 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"   {sentiment}: {count} ({(count/len(df))*100:.1f}%)")
        
        print("\n📊 Avg Scores:")
        print(f"   Polarity: {df['polarity'].mean():.3f}")
        print(f"   Subjectivity: {df['subjectivity'].mean():.3f}")
        
        print("\n📰 Recent Headlines:")
        print("-"*70)
        for _, row in df.head(10).iterrows():
            emoji = "🟢" if row['sentiment'] == "Positive" else "🔴" if row['sentiment'] == "Negative" else "⚪"
            print(f"{emoji} [{row['date'].strftime('%Y-%m-%d')}] {row['title'][:100]}")
        
        print("\n🌟 Most Positive:")
        pos = df.loc[df['polarity'].idxmax()]
        print(f"   {pos['title']} ({pos['polarity']:.3f})")
        
        print("\n⚠️ Most Negative:")
        neg = df.loc[df['polarity'].idxmin()]
        print(f"   {neg['title']} ({neg['polarity']:.3f})")
        
        print("="*70)
    
    def save_results(self, filename: str = None):
        if filename is None:
            filename = f"{self.vehicle_name.replace(' ', '_').lower()}_news_sentiment.csv"
        if self.news_data.empty:
            print("⚠️ No data to save!")
            return
        self.news_data.to_csv(filename, index=False)
        print(f"\n💾 Saved to: {filename}")
    
    def run_analysis(self, days_back: int = 10):
        df = self.collect_news(days_back)
        if df.empty:
            return None
        df = self.apply_sentiment_analysis(df)
        self.generate_report(df)
        self.save_results()
        return df


def main():
    print("\n" + "🚗"*25)
    print("   VEHICLE NEWS SENTIMENT ANALYZER")
    print("🚗"*25 + "\n")
    
    # Example: analyze Tesla Model 3
    agent = VehicleNewsAgent("Tesla Model 3")
    df = agent.run_analysis(days_back=10)
    
    if df is not None:
        print("\n✅ Analysis complete!")
    else:
        print("\n❌ No results found!")


if __name__ == "__main__":
    main()
