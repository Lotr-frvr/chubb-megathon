"""
Insurance Industry News Sentiment Analyzer
Compares Chubb Insurance with major competitors
"""

from datetime import datetime
import pandas as pd
import time
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests


class InsuranceSentimentAgent:
    def __init__(self):
        # Company names to analyze
        self.companies = {
            "American International Group": "AIG insurance",
            "The Travelers Companies": "Travelers insurance",
            "Allstate Corporation": "Allstate insurance",
            "Progressive Corporation": "Progressive insurance"
        }
        self.results = pd.DataFrame()
    
    def fetch_google_news(self, search_query: str, days_back: int = 10):
        """
        Fetch Google News headlines for a given company
        """
        print(f"📰 Fetching Google News for '{search_query}'...")
        url = f"https://news.google.com/rss/search?q={search_query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        news_items = []
        cutoff_date = datetime.now() - pd.Timedelta(days=days_back)

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")

            for item in items:
                title = item.title.text if item.title else ""
                link = item.link.text if item.link else ""
                pub_date_str = item.pubDate.text if item.pubDate else ""
                try:
                    pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                    if pub_date >= cutoff_date:
                        news_items.append({
                            "title": title,
                            "url": link,
                            "date": pub_date.strftime("%Y-%m-%d"),
                        })
                except Exception:
                    continue

        except Exception as e:
            print(f"⚠️  Error fetching Google News for {search_query}: {e}")

        print(f"✅ Found {len(news_items)} articles for {search_query}")
        return news_items
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using TextBlob
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return sentiment, round(polarity, 3)
    
    def collect_and_analyze(self, days_back=10):
        """
        Collect news for all companies and analyze sentiment
        """
        print(f"\n🔍 Collecting news for Chubb and competitors (last {days_back} days)...")
        print("=" * 70)

        all_data = []

        for company, term in self.companies.items():
            items = self.fetch_google_news(term, days_back)
            time.sleep(1)

            for item in items:
                sentiment, polarity = self.analyze_sentiment(item["title"])
                all_data.append({
                    "company": company,
                    "title": item["title"],
                    "date": item["date"],
                    "sentiment": sentiment,
                    "polarity": polarity,
                    "url": item["url"]
                })
        
        df = pd.DataFrame(all_data)
        if df.empty:
            print("⚠️  No news data found!")
            return pd.DataFrame()
        
        self.results = df
        return df
    
    def generate_comparison_report(self, df=None):
        """
        Generate sentiment comparison across companies
        """
        if df is None:
            df = self.results
        if df.empty:
            print("⚠️  No data to report!")
            return

        print("\n" + "=" * 70)
        print("🏢 INSURANCE COMPANY SENTIMENT COMPARISON REPORT")
        print("=" * 70)

        for company in self.companies.keys():
            subset = df[df["company"] == company]
            if subset.empty:
                continue
            total = len(subset)
            avg_polarity = subset["polarity"].mean()
            sentiment_counts = subset["sentiment"].value_counts()
            pos = sentiment_counts.get("Positive", 0)
            neg = sentiment_counts.get("Negative", 0)
            neu = sentiment_counts.get("Neutral", 0)

            print(f"\n📊 {company}")
            print(f"   Total Articles: {total}")
            print(f"   Avg Polarity: {avg_polarity:.3f}")
            print(f"   Sentiment Breakdown → Positive: {pos}, Neutral: {neu}, Negative: {neg}")

        # Overall comparison
        print("\n" + "=" * 70)
        summary = df.groupby("company")["polarity"].mean().sort_values(ascending=False)
        print("🏆 Average Sentiment Polarity Ranking:")
        for rank, (comp, score) in enumerate(summary.items(), 1):
            emoji = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
            print(f"   {rank}. {comp:<35} {emoji} ({score:.3f})")
        print("=" * 70)

    def save_results(self, filename="insurance_sentiment_comparison.csv"):
        """
        Save the sentiment data to CSV
        """
        if self.results.empty:
            print("⚠️  No results to save.")
            return
        self.results.to_csv(filename, index=False)
        print(f"\n💾 Saved detailed results to {filename}")

    def run(self, days_back=10):
        """
        Run full pipeline
        """
        df = self.collect_and_analyze(days_back)
        if df.empty:
            return
        self.generate_comparison_report(df)
        self.save_results()


def main():
    print("\n" + "🏦"*35)
    print("   INSURANCE INDUSTRY - SENTIMENT ANALYSIS COMPARISON")
    print("🏦"*35 + "\n")

    agent = InsuranceSentimentAgent()
    agent.run(days_back=10)

    print("\n✅ Analysis complete! Compare sentiment of Chubb vs. competitors.")
    print("📊 Results saved to: insurance_sentiment_comparison.csv")


if __name__ == "__main__":
    main()
