import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set environment variables for multi-threading
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['MKL_NUM_THREADS'] = '40'

# ===============================
# Configuration Class
# ===============================
class Config:
    """Centralized configuration for the data augmentation pipeline."""
    TARGET_COLUMN: str = 'Churn'
    RANDOM_STATE: int = 42
    SIGNAL_START_DATE: str = '2009-01-01'  # Extended to cover your data range
    SIGNAL_END_DATE: str = '2023-12-31'
    CHUBB_SENTIMENT_MEAN: float = 0.65
    CHUBB_SENTIMENT_VOLATILITY: float = 0.05
    CHUBB_SENTIMENT_START: float = 0.6
    SENTIMENT_REVERSION_RATE: float = 0.15
    MARKET_RETURN_MEAN: float = 0.0005
    MARKET_RETURN_VOLATILITY: float = 0.01
    GDP_INITIAL_GROWTH: float = 0.02
    GDP_GROWTH_VOLATILITY: float = 0.0005
    CUSTOMER_ENGAGEMENT_BETA_A: float = 2
    CUSTOMER_ENGAGEMENT_BETA_B: float = 5
    OUTPUT_AUGMENTED_CSV: str = 'final_augmented_data.csv'
    PLOT_DIR: str = 'plots'

# Ensure plot directory exists
os.makedirs(Config.PLOT_DIR, exist_ok=True)

# ===============================
# Non-Linear Signal Generator
# ===============================
class NonLinearSignalGenerator:
    """
    Generates non-linear, correlated synthetic signals to augment dataset.
    """
    def __init__(self, start_date: str = Config.SIGNAL_START_DATE, 
                 end_date: str = Config.SIGNAL_END_DATE):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates a DataFrame of synthetic time-series signals.
        """
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        n_days = len(date_range)
        print(f"Generating {n_days} days of non-linear signals between {self.start_date.date()} and {self.end_date.date()}...")

        # 1️⃣ Base Sentiment (Chubb) - Mean-reverting random walk
        chubb_sentiment = self._mean_reverting_random_walk(
            n_days,
            mean=Config.CHUBB_SENTIMENT_MEAN,
            volatility=Config.CHUBB_SENTIMENT_VOLATILITY,
            start=Config.CHUBB_SENTIMENT_START,
            reversion_rate=Config.SENTIMENT_REVERSION_RATE
        )

        # 2️⃣ Competitor Sentiment
        competitor_base = self._mean_reverting_random_walk(
            n_days,
            mean=0.52,
            volatility=0.10,
            start=0.48,
            reversion_rate=0.08
        )
        competitor_shocks = np.random.choice([0, -0.05, 0.05], size=n_days, p=[0.9, 0.05, 0.05])
        competitor_sentiment = competitor_base * 0.92 + (1 - chubb_sentiment) * 0.08 + competitor_shocks
        competitor_sentiment = np.clip(competitor_sentiment, 0.3, 0.9)

        # 3️⃣ Stock Returns
        market_index_return = np.random.normal(Config.MARKET_RETURN_MEAN, Config.MARKET_RETURN_VOLATILITY, n_days)
        
        chubb_idiosyncratic = np.random.normal(0, 0.012, n_days)
        chubb_momentum = np.random.choice([-1, 0, 1], size=n_days, p=[0.3, 0.4, 0.3]) * 0.003
        chubb_stock_return = (0.15 * market_index_return + 
                             0.10 * np.tanh((chubb_sentiment - 0.6) * 2) * 0.015 + 
                             0.65 * chubb_idiosyncratic +
                             0.15 * chubb_momentum)
        
        competitor_idiosyncratic = np.random.normal(0, 0.015, n_days)
        competitor_sector = np.random.normal(0, 0.008, n_days)
        competitor_momentum = np.random.choice([-1, 0, 1], size=n_days, p=[0.35, 0.3, 0.35]) * 0.004
        competitor_stock_return = (0.25 * market_index_return + 
                                  0.08 * np.tanh((competitor_sentiment - 0.55) * 2) * 0.012 + 
                                  0.50 * competitor_idiosyncratic +
                                  0.12 * competitor_sector +
                                  0.05 * competitor_momentum)

        # 4️⃣ Economic signal (GDP growth)
        gdp_growth = Config.GDP_INITIAL_GROWTH + np.cumsum(np.random.normal(0, Config.GDP_GROWTH_VOLATILITY, n_days))
        gdp_growth = np.clip(gdp_growth, 0.0, 0.04)

        # 5️⃣ Customer engagement
        engagement_base = np.random.beta(Config.CUSTOMER_ENGAGEMENT_BETA_A, Config.CUSTOMER_ENGAGEMENT_BETA_B, n_days)
        engagement = engagement_base * 0.75 + chubb_sentiment * 0.15 + np.random.normal(0, 0.08, n_days)
        engagement = np.clip(engagement, 0.0, 1.0)

        # Combine into DataFrame
        signals_df = pd.DataFrame({
            'date': date_range,
            'chubb_sentiment': chubb_sentiment,
            'competitor_sentiment': competitor_sentiment,
            'chubb_stock_return': chubb_stock_return,
            'competitor_stock_return': competitor_stock_return,
            'market_index_return': market_index_return,
            'gdp_growth': gdp_growth,
            'customer_engagement': engagement
        })

        return signals_df

    def _mean_reverting_random_walk(self, n: int, mean: float, volatility: float, start: float, reversion_rate: float) -> np.ndarray:
        """
        Generates a mean-reverting random walk time series.
        """
        series = np.zeros(n)
        series[0] = start
        for i in range(1, n):
            series[i] = series[i-1] + (mean - series[i-1]) * reversion_rate + np.random.normal(0, volatility)
        return np.clip(series, mean - 3*volatility, mean + 3*volatility)

    def merge_with_customers(self, customer_df: pd.DataFrame, signals_df: pd.DataFrame, 
                             date_col: str = 'cust_orig_date') -> pd.DataFrame:
        """
        Merges generated signals with customer data based on each customer's date.
        """
        df_merged = customer_df.copy()
        
        if date_col not in df_merged.columns:
            raise ValueError(f"Date column '{date_col}' not found in customer_df.")

        # Convert customer dates to datetime
        df_merged[date_col] = pd.to_datetime(df_merged[date_col], errors='coerce')
        
        # Handle any missing dates by filling with the end date
        df_merged[date_col].fillna(pd.to_datetime(Config.SIGNAL_END_DATE), inplace=True)

        # Merge signals based on customer's original date
        signals_df_copy = signals_df.copy()
        signals_df_copy.rename(columns={'date': 'signal_date'}, inplace=True)

        df_merged = df_merged.merge(
            signals_df_copy, 
            left_on=date_col, 
            right_on='signal_date', 
            how='left'
        )

        # Drop the temporary signal_date column
        df_merged.drop(columns=['signal_date'], inplace=True, errors='ignore')

        # Fill any missing signals (for dates outside the range)
        signal_cols = [col for col in signals_df_copy.columns if col != 'signal_date']
        
        for col in signal_cols:
            if df_merged[col].isnull().any():
                # Forward fill, then backward fill, then use mean as last resort
                df_merged[col].fillna(method='ffill', inplace=True)
                df_merged[col].fillna(method='bfill', inplace=True)
                if df_merged[col].isnull().any():
                    df_merged[col].fillna(signals_df[col.replace('_signal', '')].mean(), inplace=True)
        
        print(f"Successfully merged {len(signal_cols)} signals with {len(df_merged)} customer records.")
        return df_merged


# ===============================
# Main Execution
# ===============================
def main():
    print("=== DATA AUGMENTATION WITH INDIVIDUAL DATE-BASED SIGNALS ===\n")
    
    # Load the original CSV
    try:
        df = pd.read_csv('autoinsurance_churn.csv')
        original_rows = len(df)
        original_cols = len(df.columns)
        print(f"Loaded {original_rows:,} rows and {original_cols} columns from 'autoinsurance_churn.csv'")
    except FileNotFoundError:
        print("Error: 'autoinsurance_churn.csv' not found.")
        return None, None

    # Generate time-series signals
    print("\n--- Generating synthetic signals ---")
    generator = NonLinearSignalGenerator(
        start_date=Config.SIGNAL_START_DATE,
        end_date=Config.SIGNAL_END_DATE
    )
    signals_df = generator.generate_signals()
    print(f"Generated {len(signals_df)} days of signals with {len(signals_df.columns)-1} features")

    # Merge signals with customer data based on cust_orig_date
    print("\n--- Merging signals with customer data ---")
    df_augmented = generator.merge_with_customers(df, signals_df, date_col='cust_orig_date')
    
    # Verify row count preservation
    final_rows = len(df_augmented)
    final_cols = len(df_augmented.columns)
    print(f"\n✅ Augmentation complete:")
    print(f"   Original: {original_rows:,} rows × {original_cols} columns")
    print(f"   Augmented: {final_rows:,} rows × {final_cols} columns")
    print(f"   Added {final_cols - original_cols} new signal columns")
    
    if original_rows != final_rows:
        print(f"   ⚠️ WARNING: Row count changed by {final_rows - original_rows}")
    else:
        print(f"   ✓ Row count preserved perfectly!")

    # Generate correlation matrix
    print("\n--- Generating correlation matrix ---")
    numeric_cols = df_augmented.select_dtypes(include=np.number).columns.tolist()
    correlation_matrix = df_augmented[numeric_cols].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix - Augmented Data with Individual Date Signals", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOT_DIR, 'correlation_matrix_individual_dates.png'), dpi=300)
    plt.close()
    print(f"Correlation matrix saved to {Config.PLOT_DIR}/correlation_matrix_individual_dates.png")

    # Save augmented data
    df_augmented.to_csv(Config.OUTPUT_AUGMENTED_CSV, index=False)
    print(f"\n✅ Final augmented dataset saved to {Config.OUTPUT_AUGMENTED_CSV}")
    
    # Display sample of new columns
    print("\n--- Sample of augmented data (first 3 rows, showing new columns) ---")
    new_cols = [col for col in df_augmented.columns if col not in df.columns]
    if new_cols:
        print(df_augmented[['individual_id', 'cust_orig_date'] + new_cols].head(3).to_string())
    
    return df_augmented, signals_df


if __name__ == "__main__":
    augmented_data, signals = main()