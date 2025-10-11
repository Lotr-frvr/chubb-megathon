import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime, timedelta
import warnings
import multiprocessing as mp
from functools import partial
import os
warnings.filterwarnings('ignore')

# Configure for multi-GPU and multi-CPU
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['MKL_NUM_THREADS'] = '40'
os.environ['NUMEXPR_NUM_THREADS'] = '40'

import torch
if torch.cuda.is_available():
    print(f"CUDA Available: {torch.cuda.device_count()} GPUs detected")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available, using CPU only")


class DatasetAnalyzer:
    """Analyzes original dataset and extracts statistical properties"""
    
    def __init__(self, df, n_jobs=40):
        self.df = df
        self.n_jobs = n_jobs
        self.stats = {}
        self.correlations = {}
        
    def _analyze_single_numerical(self, col):
        """Analyze a single numerical column"""
        return col, {
            'mean': self.df[col].mean(),
            'std': self.df[col].std(),
            'min': self.df[col].min(),
            'max': self.df[col].max(),
            'median': self.df[col].median(),
            'q25': self.df[col].quantile(0.25),
            'q75': self.df[col].quantile(0.75),
            'skewness': self.df[col].skew(),
            'kurtosis': self.df[col].kurtosis()
        }
        
    def analyze_numerical(self):
        """Analyze numerical columns in parallel"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"Analyzing {len(numerical_cols)} numerical columns...")
        
        with mp.Pool(processes=min(self.n_jobs, len(numerical_cols))) as pool:
            results = pool.map(self._analyze_single_numerical, numerical_cols)
        
        for col, stats in results:
            self.stats[col] = stats
            
    def analyze_categorical(self):
        """Analyze categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"Analyzing {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            self.stats[col] = {
                'unique_values': self.df[col].unique().tolist(),
                'value_counts': self.df[col].value_counts(normalize=True).to_dict()
            }
            
    def analyze_correlations(self):
        """Analyze correlations between numerical features"""
        print("Computing correlation matrix...")
        numerical_df = self.df.select_dtypes(include=[np.number])
        self.correlations = numerical_df.corr().to_dict()
        
    def analyze_churn_patterns(self):
        """Analyze patterns related to churn"""
        if 'Churn' in self.df.columns:
            print("Analyzing churn patterns...")
            churn_stats = {}
            for col in self.df.columns:
                if col != 'Churn':
                    if self.df[col].dtype in [np.float64, np.int64]:
                        churn_stats[col] = {
                            'churn_mean': self.df[self.df['Churn']==1][col].mean(),
                            'no_churn_mean': self.df[self.df['Churn']==0][col].mean()
                        }
            self.stats['churn_patterns'] = churn_stats
            
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n" + "="*60)
        print("Statistical Analysis of Original Dataset")
        print("="*60)
        self.analyze_numerical()
        self.analyze_categorical()
        self.analyze_correlations()
        self.analyze_churn_patterns()
        print("Analysis complete!")
        return self.stats, self.correlations


class ExternalDataSimulator:
    """
    Simulates EXTERNAL independent data sources that exist in real world:
    - Company news sentiment (from news APIs)
    - Stock market data (from financial markets)
    - Competitor sentiment (from competitor news)
    - Vehicle model news (from automotive industry)
    - Economic indicators (from market data)
    """
    
    def __init__(self, df):
        self.df = df
        self.start_date = pd.to_datetime('2018-01-01')
        self.end_date = pd.to_datetime('2023-12-31')
        
    def generate_time_series_data(self):
        """Generate time-series external data that changes over time"""
        
        # Create daily time series for the date range
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        n_days = len(date_range)
        
        print("\n" + "="*60)
        print("Generating External Market Data (Independent of Customers)")
        print("="*60)
        
        # ==================================================================
        # RULE 1: COMPANY NEWS SENTIMENT (Independent time series)
        # ==================================================================
        print("\n[RULE 1] Company News Sentiment")
        print("  - Simulates daily news sentiment from media sources")
        print("  - Random walk with mean reversion")
        print("  - Range: 0.3 to 0.9 (0=very negative, 1=very positive)")
        
        company_sentiment_ts = self._generate_sentiment_time_series(
            n_days, 
            mean=0.65, 
            volatility=0.05,
            start_value=0.6
        )
        
        # ==================================================================
        # RULE 2: COMPANY STOCK PRICE (Correlated with sentiment)
        # ==================================================================
        print("\n[RULE 2] Company Stock Performance")
        print("  - Simulates stock market data from exchanges")
        print("  - 70% correlated with company sentiment")
        print("  - Daily returns: -3% to +3%")
        
        company_stock_ts = self._generate_stock_returns(
            company_sentiment_ts,
            correlation=0.7,
            volatility=0.015
        )
        
        # ==================================================================
        # RULE 3: COMPETITOR SENTIMENT (Inverse relationship)
        # ==================================================================
        print("\n[RULE 3] Competitor News Sentiment")
        print("  - Simulates competitor news from media")
        print("  - 50% inverse correlation with our company")
        print("  - When we do well, they often do worse")
        
        competitor_sentiment_ts = self._generate_competitor_sentiment(
            company_sentiment_ts,
            inverse_correlation=0.5
        )
        
        # ==================================================================
        # RULE 4: COMPETITOR STOCK PRICE
        # ==================================================================
        print("\n[RULE 4] Competitor Stock Performance")
        print("  - Competitor stock from market data")
        print("  - 65% correlated with their sentiment")
        
        competitor_stock_ts = self._generate_stock_returns(
            competitor_sentiment_ts,
            correlation=0.65,
            volatility=0.018
        )
        
        # ==================================================================
        # RULE 5: VEHICLE MODEL NEWS SENTIMENT
        # ==================================================================
        print("\n[RULE 5] Vehicle Model News Sentiment")
        print("  - Simulates automotive industry news")
        print("  - Different models have different sentiment patterns")
        print("  - Based on recalls, reviews, safety ratings")
        
        vehicle_sentiment_ts = self._generate_vehicle_sentiment(n_days)
        
        # ==================================================================
        # RULE 6: ECONOMIC INDICATOR (GDP Growth Rate)
        # ==================================================================
        print("\n[RULE 6] Economic Indicator (GDP Growth)")
        print("  - Quarterly GDP growth rate")
        print("  - Affects overall market sentiment")
        print("  - Range: -2% to +4%")
        
        gdp_growth_ts = self._generate_economic_indicator(n_days)
        
        # ==================================================================
        # RULE 7: INDUSTRY TREND INDEX
        # ==================================================================
        print("\n[RULE 7] Auto Insurance Industry Trend Index")
        print("  - Overall industry health score (0-100)")
        print("  - Based on claim rates, regulations, competition")
        
        industry_index_ts = self._generate_industry_index(n_days)
        
        # Create external data DataFrame
        external_data = pd.DataFrame({
            'date': date_range,
            'company_sentiment': company_sentiment_ts,
            'company_stock_return': company_stock_ts,
            'competitor_sentiment': competitor_sentiment_ts,
            'competitor_stock_return': competitor_stock_ts,
            'vehicle_news_sentiment': vehicle_sentiment_ts,
            'gdp_growth_rate': gdp_growth_ts,
            'industry_trend_index': industry_index_ts
        })
        
        return external_data
    
    def _generate_sentiment_time_series(self, n_days, mean, volatility, start_value):
        """Generate sentiment using random walk with mean reversion"""
        sentiment = np.zeros(n_days)
        sentiment[0] = start_value
        
        for i in range(1, n_days):
            # Mean reversion: sentiment drifts back to mean
            drift = (mean - sentiment[i-1]) * 0.1
            shock = np.random.normal(0, volatility)
            sentiment[i] = sentiment[i-1] + drift + shock
            
        # Clip to valid range
        sentiment = np.clip(sentiment, 0.3, 0.9)
        return sentiment
    
    def _generate_stock_returns(self, sentiment_ts, correlation, volatility):
        """Generate stock returns correlated with sentiment"""
        n_days = len(sentiment_ts)
        
        # Normalize sentiment to returns (-2% to +2%)
        base_returns = (sentiment_ts - 0.6) * 0.05
        
        # Add correlated noise
        uncorrelated_noise = np.random.normal(0, volatility, n_days)
        correlated_noise = (np.random.normal(0, volatility, n_days) * correlation + 
                           uncorrelated_noise * (1 - correlation))
        
        stock_returns = base_returns + correlated_noise
        return stock_returns
    
    def _generate_competitor_sentiment(self, company_sentiment, inverse_correlation):
        """Generate competitor sentiment with inverse correlation"""
        n_days = len(company_sentiment)
        
        # Inverse relationship
        base_competitor = 1 - company_sentiment + 0.6  # Shift back to 0.3-0.9 range
        
        # Add some independence
        independent_noise = np.random.normal(0, 0.08, n_days)
        competitor_sentiment = base_competitor * inverse_correlation + independent_noise
        
        return np.clip(competitor_sentiment, 0.3, 0.9)
    
    def _generate_vehicle_sentiment(self, n_days):
        """Generate vehicle model sentiment (cycles and events)"""
        # Simulate product cycles and recall events
        base_sentiment = 0.7
        
        # Seasonal cycle (new model releases)
        t = np.arange(n_days)
        seasonal = 0.1 * np.sin(2 * np.pi * t / 365)
        
        # Random events (recalls, awards)
        events = np.random.choice([0, -0.2, 0.1], size=n_days, p=[0.95, 0.03, 0.02])
        events_smoothed = pd.Series(events).rolling(30, min_periods=1).mean().values
        
        sentiment = base_sentiment + seasonal + events_smoothed + np.random.normal(0, 0.05, n_days)
        return np.clip(sentiment, 0.4, 0.95)
    
    def _generate_economic_indicator(self, n_days):
        """Generate GDP growth rate (quarterly changes)"""
        n_quarters = n_days // 90 + 1
        quarterly_growth = np.random.normal(0.02, 0.015, n_quarters)
        quarterly_growth = np.clip(quarterly_growth, -0.02, 0.04)
        
        # Expand to daily (constant within quarter)
        daily_growth = np.repeat(quarterly_growth, 90)[:n_days]
        return daily_growth
    
    def _generate_industry_index(self, n_days):
        """Generate auto insurance industry health index"""
        # Long-term trend
        trend = np.linspace(65, 75, n_days)
        
        # Business cycles
        cycle = 8 * np.sin(2 * np.pi * np.arange(n_days) / 730)  # 2-year cycle
        
        # Random shocks
        shocks = np.random.normal(0, 2, n_days)
        
        index = trend + cycle + shocks
        return np.clip(index, 40, 95)
    
    def merge_with_customer_data(self, external_data):
        """
        RULE 8: Merge external data with customer data based on dates
        Each customer record gets the external market conditions from their date
        """
        print("\n" + "="*60)
        print("[RULE 8] Merging External Data with Customer Records")
        print("="*60)
        print("  - Each customer gets market conditions from their 'cust_orig_date'")
        print("  - Like looking up stock price on the day they joined")
        
        df_merged = self.df.copy()
        
        # Convert customer date to datetime
        if 'cust_orig_date' in df_merged.columns:
            df_merged['cust_orig_date'] = pd.to_datetime(df_merged['cust_orig_date'])
            
            # Merge external data based on date
            df_merged = df_merged.merge(
                external_data,
                left_on='cust_orig_date',
                right_on='date',
                how='left'
            )
            
            # Fill any missing dates with nearby values
            for col in external_data.columns:
                if col != 'date' and col in df_merged.columns:
                    df_merged[col].fillna(method='ffill', inplace=True)
                    df_merged[col].fillna(method='bfill', inplace=True)
        
        print(f"  - Merged {len(external_data.columns)-1} external features")
        
        return df_merged


class MultiGPUSyntheticDataGenerator:
    """Generates synthetic data using CTGAN with multi-GPU support"""
    
    def __init__(self, df, gpu_ids=[0, 1, 2, 3]):
        self.df = df
        self.gpu_ids = gpu_ids
        self.models = []
        
    def prepare_data(self):
        """Prepare data for CTGAN"""
        print("\n" + "="*60)
        print("Preparing Data for CTGAN Training")
        print("="*60)
        df_prepared = self.df.copy()
        
        # Handle date columns
        date_cols = df_prepared.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df_prepared[col] = df_prepared[col].astype(str)
            
        # Fill missing values
        for col in df_prepared.columns:
            if df_prepared[col].dtype == 'object':
                df_prepared[col].fillna('Unknown', inplace=True)
            else:
                df_prepared[col].fillna(df_prepared[col].median(), inplace=True)
        
        print(f"  - Dataset shape: {df_prepared.shape}")
        print(f"  - Features: {df_prepared.shape[1]}")
        
        return df_prepared
    
    def train_ctgan_single_gpu(self, gpu_id, data, epochs):
        """Train CTGAN on a single GPU"""
        try:
            from ctgan import CTGAN
            
            device = f'cuda:{gpu_id}'
            print(f"\n[GPU {gpu_id}] Starting training...")
            
            model = CTGAN(
                epochs=epochs,
                batch_size=1000,
                generator_dim=(512, 512, 512),
                discriminator_dim=(512, 512, 512),
                pac=10,
                cuda=device
            )
            
            model.fit(data)
            print(f"[GPU {gpu_id}] Training complete!")
            
            return model
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {e}")
            return None
    
    def train_multi_gpu(self, epochs=300):
        """Train multiple CTGAN models in parallel across GPUs"""
        try:
            from ctgan import CTGAN
            
            print("\n" + "="*60)
            print(f"Training {len(self.gpu_ids)} CTGAN Models on {len(self.gpu_ids)} GPUs")
            print("="*60)
            
            df_prepared = self.prepare_data()
            
            from concurrent.futures import ProcessPoolExecutor
            
            with ProcessPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
                futures = []
                for gpu_id in self.gpu_ids:
                    future = executor.submit(
                        self.train_ctgan_single_gpu, 
                        gpu_id, 
                        df_prepared, 
                        epochs
                    )
                    futures.append((gpu_id, future))
                
                for gpu_id, future in futures:
                    try:
                        model = future.result()
                        if model is not None:
                            self.models.append(model)
                    except Exception as e:
                        print(f"[GPU {gpu_id}] Failed: {e}")
            
            print(f"\n✓ Successfully trained {len(self.models)} models!")
            
        except ImportError:
            print("ERROR: CTGAN not installed")
            print("Install: pip install ctgan torch")
            return None
    
    def generate_synthetic_data(self, n_samples):
        """Generate synthetic data using all trained models"""
        if not self.models:
            print("ERROR: No models trained")
            return None
        
        print("\n" + "="*60)
        print(f"Generating {n_samples:,} Synthetic Samples")
        print("="*60)
        
        samples_per_model = n_samples // len(self.models)
        remainder = n_samples % len(self.models)
        
        all_synthetic_data = []
        
        for i, model in enumerate(self.models):
            n_samples_this_model = samples_per_model + (1 if i < remainder else 0)
            print(f"  Model {i+1}: Generating {n_samples_this_model:,} samples...")
            
            synthetic_data = model.sample(n_samples_this_model)
            all_synthetic_data.append(synthetic_data)
        
        combined_synthetic_data = pd.concat(all_synthetic_data, ignore_index=True)
        
        print(f"\n✓ Total generated: {len(combined_synthetic_data):,} samples")
        return combined_synthetic_data


def main():
    """Main execution workflow"""
    
    print("=" * 80)
    print("SYNTHETIC DATASET GENERATOR WITH EXTERNAL MARKET DATA")
    print("Hardware: 4 GPUs | 40 CPU Cores")
    print("=" * 80)
    
    # Load your data
    df = pd.read_csv('autoinsurance_churn_cleaned.csv')
    print(f"\nOriginal dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Step 1: Analyze original dataset
    analyzer = DatasetAnalyzer(df, n_jobs=40)
    stats, correlations = analyzer.run_full_analysis()
    
    # Step 2: Generate EXTERNAL independent data
    external_sim = ExternalDataSimulator(df)
    external_data = external_sim.generate_time_series_data()
    
    print(f"\nExternal data generated: {len(external_data):,} days of market data")
    print("\nExternal features:")
    for col in external_data.columns:
        if col != 'date':
            print(f"  - {col}")
    
    # Step 3: Merge external data with customer records
    df_enhanced = external_sim.merge_with_customer_data(external_data)
    
    print(f"\nEnhanced dataset: {df_enhanced.shape[0]:,} rows, {df_enhanced.shape[1]} columns")
    print(f"New external features added: {df_enhanced.shape[1] - df.shape[1]}")
    
    # Display sample
    print("\n" + "="*60)
    print("Sample of Enhanced Data (with external market data)")
    print("="*60)
    display_cols = ['cust_orig_date', 'company_sentiment', 'company_stock_return', 
                    'competitor_sentiment', 'gdp_growth_rate', 'Churn']
    available_cols = [col for col in display_cols if col in df_enhanced.columns]
    print(df_enhanced[available_cols].head(10))
    
    # Step 4: Train CTGAN and generate synthetic data
    print("\n" + "="*80)
    print("CTGAN SYNTHETIC DATA GENERATION")
    print("="*80)
    
    gpu_ids = [0, 1, 2, 3]
    synth_gen = MultiGPUSyntheticDataGenerator(df_enhanced, gpu_ids=gpu_ids)
    
    synth_gen.train_multi_gpu(epochs=300)
    
    n_synthetic_samples = len(df) * 100
    synthetic_df = synth_gen.generate_synthetic_data(n_synthetic_samples)
    
    if synthetic_df is not None:
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Original: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Enhanced: {df_enhanced.shape[0]:,} rows, {df_enhanced.shape[1]} columns")
        print(f"Synthetic: {synthetic_df.shape[0]:,} rows, {synthetic_df.shape[1]} columns")
        print(f"Multiplier: {len(synthetic_df) / len(df):.0f}x")
        
        # Save outputs
        print("\nSaving files...")
        df_enhanced.to_csv('enhanced_with_external_data.csv', index=False)
        synthetic_df.to_csv('synthetic_data_large.csv', index=False)
        external_data.to_csv('external_market_data.csv', index=False)
        
        import json
        with open('dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("FILES SAVED:")
        print("  1. enhanced_with_external_data.csv - Original + external market data")
        print("  2. synthetic_data_large.csv - Large synthetic dataset")
        print("  3. external_market_data.csv - Time series of market conditions")
        print("  4. dataset_statistics.json - Statistical analysis")
        print("="*80)
    
    return df_enhanced, synthetic_df, external_data


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    enhanced_data, synthetic_data, external_data = main()