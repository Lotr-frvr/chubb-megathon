"""
Competitor Stock Agent for Insurance Industry
Analyzes stock performance and technical indicators for major insurers
(Excludes Chubb Limited)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict


class CompetitorStockAgent:
    def __init__(self):
        # Major competitors (excluding Chubb)
        self.competitors = {
            "AIG": "American International Group",
            "TRV": "The Travelers Companies",
            "ALL": "Allstate Corporation",
            "PGR": "Progressive Corporation",
            "CNA": "CNA Financial Corporation"
        }
        self.all_results = {}

    def fetch_yahoo_finance(self, ticker: str, days_back: int = 10) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance API
        """
        print(f"📈 Fetching Yahoo Finance data for {ticker}...")
        try:
            end_date = int(time.time())
            start_date = int((datetime.now() - timedelta(days=days_back)).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                "period1": start_date,
                "period2": end_date,
                "interval": "1d",
                "includePrePost": "false",
                "events": "div,splits"
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "date": [datetime.fromtimestamp(ts) for ts in timestamps],
                "open": quotes["open"],
                "high": quotes["high"],
                "low": quotes["low"],
                "close": quotes["close"],
                "volume": quotes["volume"]
            }).dropna()

            print(f"✅ {ticker}: Retrieved {len(df)} trading days")
            return df

        except Exception as e:
            print(f"❌ Error fetching Yahoo Finance data for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic technical indicators
        """
        if df.empty:
            return df

        df["daily_return"] = df["close"].pct_change() * 100
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["EMA_5"] = df["close"].ewm(span=5, adjust=False).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        return df

    def analyze_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze stock trend summary
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        first = df.iloc[0]

        change_pct = ((latest["close"] - first["close"]) / first["close"]) * 100
        volatility = df["daily_return"].std()

        trend = "Sideways ➡️"
        if change_pct > 5:
            trend = "Uptrend 📈"
        elif change_pct < -5:
            trend = "Downtrend 📉"

        rsi_signal = "Neutral"
        if "RSI" in df.columns and not pd.isna(latest["RSI"]):
            if latest["RSI"] > 70:
                rsi_signal = "Overbought ⚠️"
            elif latest["RSI"] < 30:
                rsi_signal = "Oversold 💡"

        macd_signal = "Bearish 🐻"
        if "MACD" in df.columns:
            if latest["MACD"] > latest["MACD_signal"]:
                macd_signal = "Bullish 🐂"

        return {
            "latest_price": latest["close"],
            "period_change_pct": change_pct,
            "volatility": volatility,
            "rsi": latest.get("RSI"),
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "trend": trend,
        }

    def analyze_competitors(self, days_back: int = 10):
        """
        Analyze all competitor stocks
        """
        print(f"\n🔍 Collecting and analyzing competitor stocks (last {days_back} days)...")
        print("=" * 70)

        for ticker, name in self.competitors.items():
            df = self.fetch_yahoo_finance(ticker, days_back)
            if df.empty:
                continue

            df = self.calculate_technical_indicators(df)
            analysis = self.analyze_trends(df)
            self.all_results[ticker] = {
                "company": name,
                **analysis,
            }

        return self.all_results

    def generate_comparison_report(self):
        """
        Print comparative stock performance report
        """
        if not self.all_results:
            print("⚠️ No competitor data to report!")
            return

        print("\n" + "=" * 70)
        print("🏢 INSURANCE INDUSTRY - COMPETITOR STOCK COMPARISON")
        print("=" * 70)

        summary = []
        for ticker, data in self.all_results.items():
            print(f"\n📊 {data['company']} ({ticker})")
            print(f"   Latest Price: ${data['latest_price']:.2f}")
            print(f"   Trend: {data['trend']}")
            print(f"   Period Change: {data['period_change_pct']:+.2f}%")
            print(f"   Volatility: {data['volatility']:.2f}%")
            print(f"   RSI: {data['rsi']:.2f} ({data['rsi_signal']})")
            print(f"   MACD: {data['macd_signal']}")

            summary.append((ticker, data["company"], data["period_change_pct"]))

        print("\n" + "=" * 70)
        print("🏆 PERFORMANCE RANKING (by % change):")
        for rank, (ticker, name, change) in enumerate(sorted(summary, key=lambda x: x[2], reverse=True), 1):
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            print(f"   {rank}. {name:<35} {emoji} ({change:+.2f}%)")
        print("=" * 70)

    def save_results(self, filename="insurance_competitor_stocks.csv"):
        """
        Save summary results to CSV
        """
        if not self.all_results:
            print("⚠️ No data to save.")
            return

        df = pd.DataFrame.from_dict(self.all_results, orient="index")
        df.to_csv(filename)
        print(f"\n💾 Saved competitor stock results to {filename}")

    def run(self, days_back: int = 10):
        """
        Full run pipeline
        """
        self.analyze_competitors(days_back)
        self.generate_comparison_report()
        self.save_results()


def main():
    print("\n" + "💹" * 35)
    print("   INSURANCE COMPETITOR STOCK ANALYSIS (Excluding CHUBB)")
    print("💹" * 35 + "\n")

    agent = CompetitorStockAgent()
    agent.run(days_back=10)

    print("\n✅ Competitor stock analysis complete!")
    print("📊 Results saved to: insurance_competitor_stocks.csv")


if __name__ == "__main__":
    main()
