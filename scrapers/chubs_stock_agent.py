"""
Stock Agent for Chubb Insurance Company (Ticker: CB)
Fetches stock prices and performs technical analysis
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple
import json
import time


class ChubbStockAgent:
    def __init__(self):
        self.ticker = "CB"  # Chubb Limited ticker symbol
        self.company_name = "Chubb Limited"
        self.stock_data = None
        
    def fetch_yahoo_finance(self, days_back: int = 10) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance API
        """
        print(f"📈 Fetching Yahoo Finance data for {self.ticker}...")
        
        try:
            # Calculate timestamps
            end_date = int(time.time())
            start_date = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}"
            params = {
                "period1": start_date,
                "period2": end_date,
                "interval": "1d",
                "includePrePost": "false",
                "events": "div,splits"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            # Parse response
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            # Clean data (remove None values)
            df = df.dropna()
            
            print(f"✅ Retrieved {len(df)} trading days of data")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    def fetch_alpha_vantage(self, days_back: int = 10) -> pd.DataFrame:
        """
        Fetch stock data from Alpha Vantage API (requires API key)
        Note: This requires ALPHA_VANTAGE_API_KEY environment variable
        """
        print(f"📈 Fetching Alpha Vantage data for {self.ticker}...")
        
        try:
            import os
            api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            
            if not api_key:
                print("⚠️  Alpha Vantage API key not found. Skipping...")
                return pd.DataFrame()
            
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": self.ticker,
                "outputsize": "compact",
                "apikey": api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                print("⚠️  No data received from Alpha Vantage")
                return pd.DataFrame()
            
            # Parse data
            time_series = data["Time Series (Daily)"]
            records = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for date_str, values in time_series.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date >= cutoff_date:
                    records.append({
                        'date': date,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['5. volume'])
                    })
            
            df = pd.DataFrame(records)
            df = df.sort_values('date')
            
            print(f"✅ Retrieved {len(df)} trading days of data")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        """
        if df.empty:
            return df
        
        print("\n📊 Calculating technical indicators...")
        
        # Daily returns
        df['daily_return'] = df['close'].pct_change() * 100
        
        # Simple Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=min(10, len(df))).mean()
        
        # Exponential Moving Average
        df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
        
        # Bollinger Bands
        window = min(20, len(df))
        if window >= 2:
            df['BB_middle'] = df['close'].rolling(window=window).mean()
            bb_std = df['close'].rolling(window=window).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Volume indicators
        df['volume_MA'] = df['volume'].rolling(window=5).mean()
        
        print("✅ Technical indicators calculated")
        return df
    
    def analyze_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze price trends and patterns
        """
        if df.empty or len(df) < 2:
            return {}
        
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        first = df.iloc[0]
        
        # Price change metrics
        day_change = latest['close'] - previous['close']
        day_change_pct = (day_change / previous['close']) * 100
        
        period_change = latest['close'] - first['close']
        period_change_pct = (period_change / first['close']) * 100
        
        # Volatility
        volatility = df['daily_return'].std()
        
        # Trend analysis
        if period_change_pct > 5:
            trend = "Strong Uptrend 📈"
        elif period_change_pct > 0:
            trend = "Mild Uptrend ↗️"
        elif period_change_pct < -5:
            trend = "Strong Downtrend 📉"
        elif period_change_pct < 0:
            trend = "Mild Downtrend ↘️"
        else:
            trend = "Sideways ➡️"
        
        # RSI analysis
        rsi_signal = "N/A"
        if 'RSI' in df.columns and pd.notna(latest.get('RSI')):
            rsi = latest['RSI']
            if rsi > 70:
                rsi_signal = "Overbought ⚠️"
            elif rsi < 30:
                rsi_signal = "Oversold 💡"
            else:
                rsi_signal = "Neutral ➖"
        
        # MACD analysis
        macd_signal = "N/A"
        if 'MACD' in df.columns and pd.notna(latest.get('MACD')):
            if latest['MACD'] > latest['MACD_signal']:
                macd_signal = "Bullish 🐂"
            else:
                macd_signal = "Bearish 🐻"
        
        return {
            'latest_price': latest['close'],
            'day_change': day_change,
            'day_change_pct': day_change_pct,
            'period_change': period_change,
            'period_change_pct': period_change_pct,
            'period_high': df['high'].max(),
            'period_low': df['low'].min(),
            'avg_volume': df['volume'].mean(),
            'volatility': volatility,
            'trend': trend,
            'rsi': latest.get('RSI', None),
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal
        }
    
    def collect_stock_data(self, days_back: int = 10) -> pd.DataFrame:
        """
        Collect stock data from available sources
        """
        print(f"\n🔍 Starting stock data collection for {self.ticker} - past {days_back} days...")
        print("="*70)
        
        # Try Yahoo Finance first (no API key needed)
        df = self.fetch_yahoo_finance(days_back)
        
        # If Yahoo fails, try Alpha Vantage
        if df.empty:
            time.sleep(1)
            df = self.fetch_alpha_vantage(days_back)
        
        if df.empty:
            print("❌ Failed to fetch stock data from any source!")
            return df
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        self.stock_data = df
        return df
    
    def generate_report(self, df: pd.DataFrame = None):
        """
        Generate a comprehensive stock analysis report
        """
        if df is None:
            df = self.stock_data
        
        if df is None or df.empty:
            print("⚠️  No data to report!")
            return
        
        # Analyze trends
        analysis = self.analyze_trends(df)
        
        print("\n" + "="*70)
        print(f"📊 STOCK ANALYSIS REPORT - {self.company_name} ({self.ticker})")
        print("="*70)
        
        # Period information
        print(f"\n📅 Analysis Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"📈 Trading Days: {len(df)}")
        
        # Current price and changes
        print("\n💰 PRICE INFORMATION:")
        print(f"   Current Price: ${analysis['latest_price']:.2f}")
        
        change_emoji = "🟢" if analysis['day_change'] > 0 else "🔴" if analysis['day_change'] < 0 else "⚪"
        print(f"   {change_emoji} Day Change: ${analysis['day_change']:.2f} ({analysis['day_change_pct']:.2f}%)")
        
        period_emoji = "🟢" if analysis['period_change'] > 0 else "🔴" if analysis['period_change'] < 0 else "⚪"
        print(f"   {period_emoji} Period Change: ${analysis['period_change']:.2f} ({analysis['period_change_pct']:.2f}%)")
        
        print(f"   Period High: ${analysis['period_high']:.2f}")
        print(f"   Period Low: ${analysis['period_low']:.2f}")
        print(f"   Range: ${analysis['period_high'] - analysis['period_low']:.2f}")
        
        # Volume
        print(f"\n📊 VOLUME:")
        print(f"   Average Daily Volume: {analysis['avg_volume']:,.0f}")
        latest_vol = df.iloc[-1]['volume']
        vol_diff = ((latest_vol - analysis['avg_volume']) / analysis['avg_volume']) * 100
        vol_emoji = "🔥" if vol_diff > 20 else "❄️" if vol_diff < -20 else "➖"
        print(f"   {vol_emoji} Latest Volume: {latest_vol:,.0f} ({vol_diff:+.1f}% vs avg)")
        
        # Technical Analysis
        print(f"\n🔧 TECHNICAL ANALYSIS:")
        print(f"   Trend: {analysis['trend']}")
        print(f"   Volatility: {analysis['volatility']:.2f}%")
        
        if analysis['rsi'] is not None and pd.notna(analysis['rsi']):
            print(f"   RSI (14): {analysis['rsi']:.2f} - {analysis['rsi_signal']}")
        
        print(f"   MACD Signal: {analysis['macd_signal']}")
        
        # Moving Averages
        latest = df.iloc[-1]
        if 'SMA_5' in df.columns and pd.notna(latest.get('SMA_5')):
            print(f"\n📈 MOVING AVERAGES:")
            print(f"   5-Day SMA: ${latest['SMA_5']:.2f}")
            if pd.notna(latest.get('SMA_10')):
                print(f"   10-Day SMA: ${latest['SMA_10']:.2f}")
            print(f"   5-Day EMA: ${latest['EMA_5']:.2f}")
            
            # Price vs MA analysis
            if latest['close'] > latest['SMA_5']:
                print(f"   📊 Price is ABOVE 5-day average (Bullish)")
            else:
                print(f"   📊 Price is BELOW 5-day average (Bearish)")
        
        # Recent price action
        print(f"\n📅 RECENT PRICE ACTION (Last 5 Days):")
        print("-"*70)
        recent = df.tail(5)
        for idx, row in recent.iterrows():
            change = row.get('daily_return', 0)
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"{emoji} {date_str}: ${row['close']:7.2f} | Change: {change:+6.2f}% | Vol: {row['volume']:>12,}")
        
        # Trading signals
        print(f"\n🎯 TRADING SIGNALS:")
        signals = []
        
        if analysis['rsi'] is not None and pd.notna(analysis['rsi']):
            if analysis['rsi'] < 30:
                signals.append("🟢 RSI indicates oversold - potential buy opportunity")
            elif analysis['rsi'] > 70:
                signals.append("🔴 RSI indicates overbought - consider taking profits")
        
        if 'MACD' in df.columns:
            if latest['MACD'] > latest['MACD_signal'] and df.iloc[-2]['MACD'] <= df.iloc[-2]['MACD_signal']:
                signals.append("🟢 MACD bullish crossover - buy signal")
            elif latest['MACD'] < latest['MACD_signal'] and df.iloc[-2]['MACD'] >= df.iloc[-2]['MACD_signal']:
                signals.append("🔴 MACD bearish crossover - sell signal")
        
        if analysis['period_change_pct'] > 10:
            signals.append("⚠️  Strong upward momentum - watch for resistance")
        elif analysis['period_change_pct'] < -10:
            signals.append("⚠️  Strong downward momentum - watch for support")
        
        if signals:
            for signal in signals:
                print(f"   {signal}")
        else:
            print("   ➖ No strong signals detected - hold position")
        
        print("\n" + "="*70)
    
    def save_results(self, filename: str = "chubb_stock_analysis.csv"):
        """
        Save results to CSV file
        """
        if self.stock_data is None or self.stock_data.empty:
            print("⚠️  No data to save!")
            return
        
        try:
            self.stock_data.to_csv(filename, index=False)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")
    
    def run_analysis(self, days_back: int = 10):
        """
        Run the complete stock analysis pipeline
        """
        # Collect stock data
        df = self.collect_stock_data(days_back)
        
        if df.empty:
            return None
        
        # Generate report
        self.generate_report(df)
        
        # Save results
        self.save_results()
        
        return df


def main():
    """
    Main function to run the stock agent
    """
    print("\n" + "💹"*35)
    print("   CHUBB LIMITED (CB) - STOCK ANALYSIS TOOL")
    print("💹"*35 + "\n")
    
    # Initialize agent
    agent = ChubbStockAgent()
    
    # Run analysis for past 10 days
    df = agent.run_analysis(days_back=10)
    
    if df is not None:
        print("\n✅ Analysis complete!")
        print(f"📊 View detailed results in: chubb_stock_analysis.csv")
    else:
        print("\n❌ Analysis failed - no stock data found!")


if __name__ == "__main__":
    main()
