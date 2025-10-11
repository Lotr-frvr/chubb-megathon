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

import torch
if torch.cuda.is_available():
    print(f"✓ CUDA: {torch.cuda.device_count()} GPUs detected")
else:
    print("⚠ CPU only mode")

# ===============================
# Configuration Class
# ===============================
class Config:
    """Centralized configuration for the data augmentation pipeline."""
    N_SAMPLES_BALANCED: int = 500
    TARGET_COLUMN: str = 'Churn'
    RANDOM_STATE: int = 42
    SIGNAL_START_DATE: str = '2018-01-01'
    SIGNAL_END_DATE: str = '2023-12-31'
    CHUBB_SENTIMENT_MEAN: float = 0.65
    CHUBB_SENTIMENT_VOLATILITY: float = 0.05
    CHUBB_SENTIMENT_START: float = 0.6
    SENTIMENT_REVERSION_RATE: float = 0.15 # Rate at which sentiment reverts to mean
    MARKET_RETURN_MEAN: float = 0.0005
    MARKET_RETURN_VOLATILITY: float = 0.01
    GDP_INITIAL_GROWTH: float = 0.02
    GDP_GROWTH_VOLATILITY: float = 0.0005
    CUSTOMER_ENGAGEMENT_BETA_A: float = 2
    CUSTOMER_ENGAGEMENT_BETA_B: float = 5
    OUTPUT_BALANCED_CSV: str = 'balanced_sample_500.csv'
    OUTPUT_AUGMENTED_CSV: str = 'augmented_with_signals.csv'
    OUTPUT_SIGNALS_CSV: str = 'synthetic_time_series_signals.csv'
    PLOT_DIR: str = 'plots' # Directory for saving plots

# Ensure plot directory exists
os.makedirs(Config.PLOT_DIR, exist_ok=True)

# ===============================
# 1️⃣ Balanced Sampler
# ===============================
class BalancedDataSampler:
    """Balances churn vs non-churn customers using undersampling."""
    def __init__(self, df: pd.DataFrame, target_col: str = Config.TARGET_COLUMN):
        self.df = df
        self.target_col = target_col

    def get_balanced_sample(self, n_samples: int = Config.N_SAMPLES_BALANCED) -> pd.DataFrame:
        """
        Generates a balanced sample of the DataFrame by undersampling.

        Args:
            n_samples: The total number of samples desired in the balanced DataFrame.
                       It will be split equally between the two target classes.

        Returns:
            A new DataFrame with an approximately equal number of samples for each target class.
        """
        churn_counts = self.df[self.target_col].value_counts()
        
        # Determine samples per class, ensuring we don't request more than available
        samples_per_class = n_samples // 2
        
        df_churn_0 = self.df[self.df[self.target_col] == 0]
        df_churn_1 = self.df[self.df[self.target_col] == 1]

        n_0 = min(samples_per_class, len(df_churn_0))
        n_1 = min(samples_per_class, len(df_churn_1))
        
        if n_0 < samples_per_class or n_1 < samples_per_class:
            print(f"Warning: Could not get {samples_per_class} samples for each class. "
                  f"Got {n_0} for class 0 and {n_1} for class 1.")
            
        df_churn_0_sampled = df_churn_0.sample(n=n_0, random_state=Config.RANDOM_STATE)
        df_churn_1_sampled = df_churn_1.sample(n=n_1, random_state=Config.RANDOM_STATE)
        
        df_balanced = pd.concat([df_churn_0_sampled, df_churn_1_sampled]).sample(
            frac=1, random_state=Config.RANDOM_STATE
        ).reset_index(drop=True)
        
        print(f"Balanced sample created with {len(df_balanced)} rows. "
              f"Class 0: {len(df_churn_0_sampled)}, Class 1: {len(df_churn_1_sampled)}")
        return df_balanced


# ===============================
# 2️⃣ Non-Linear Signal Generator
# ===============================
class NonLinearSignalGenerator:
    """
    Generates non-linear, correlated synthetic signals to augment dataset.
    These signals are not learned from real time-series data but are
    mathematically generated to introduce new, potentially relevant features.
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

        # 2️⃣ Competitor Sentiment (highly independent with different dynamics)
        # Use separate mean-reverting process with very different parameters
        competitor_base = self._mean_reverting_random_walk(
            n_days,
            mean=0.52,
            volatility=0.10,
            start=0.48,
            reversion_rate=0.08
        )
        # Add random spikes/dips to make it more distinct
        competitor_shocks = np.random.choice([0, -0.05, 0.05], size=n_days, p=[0.9, 0.05, 0.05])
        # Very weak inverse relationship to Chubb (correlation ~0.05-0.15)
        competitor_sentiment = competitor_base * 0.92 + (1 - chubb_sentiment) * 0.08 + competitor_shocks
        competitor_sentiment = np.clip(competitor_sentiment, 0.3, 0.9)

        # 3️⃣ Stock Returns (more randomized, reduced correlation with each other)
        market_index_return = np.random.normal(Config.MARKET_RETURN_MEAN, Config.MARKET_RETURN_VOLATILITY, n_days)
        
        # Chubb stock: Diverse sources of randomness (correlation with market ~0.30-0.40)
        chubb_idiosyncratic = np.random.normal(0, 0.012, n_days)  # Company-specific shocks
        chubb_momentum = np.random.choice([-1, 0, 1], size=n_days, p=[0.3, 0.4, 0.3]) * 0.003  # Momentum effects
        chubb_stock_return = (0.15 * market_index_return + 
                             0.10 * np.tanh((chubb_sentiment - 0.6) * 2) * 0.015 + 
                             0.65 * chubb_idiosyncratic +
                             0.15 * chubb_momentum)
        
        # Competitor stock: Different random process (correlation with market ~0.25-0.35, with Chubb ~0.15-0.25)
        competitor_idiosyncratic = np.random.normal(0, 0.015, n_days)  # Different company-specific variance
        competitor_sector = np.random.normal(0, 0.008, n_days)  # Sector-specific noise
        competitor_momentum = np.random.choice([-1, 0, 1], size=n_days, p=[0.35, 0.3, 0.35]) * 0.004  # Different momentum
        competitor_stock_return = (0.25 * market_index_return + 
                                  0.08 * np.tanh((competitor_sentiment - 0.55) * 2) * 0.012 + 
                                  0.50 * competitor_idiosyncratic +
                                  0.12 * competitor_sector +
                                  0.05 * competitor_momentum)

        # 4️⃣ Economic signal (GDP growth, smooth trend + noise)
        gdp_growth = Config.GDP_INITIAL_GROWTH + np.cumsum(np.random.normal(0, Config.GDP_GROWTH_VOLATILITY, n_days))
        gdp_growth = np.clip(gdp_growth, 0.0, 0.04) # Keep GDP growth realistic

        # 5️⃣ Customer engagement (more independent, less correlated with sentiment)
        # Mostly based on beta distribution with weak sentiment influence (correlation ~0.2-0.3)
        engagement_base = np.random.beta(Config.CUSTOMER_ENGAGEMENT_BETA_A, Config.CUSTOMER_ENGAGEMENT_BETA_B, n_days)
        engagement = engagement_base * 0.75 + chubb_sentiment * 0.15 + np.random.normal(0, 0.08, n_days)
        engagement = np.clip(engagement, 0.0, 1.0) # Scale engagement between 0 and 1

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
            # The series is pulled towards the mean, plus some random noise
            series[i] = series[i-1] + (mean - series[i-1]) * reversion_rate + np.random.normal(0, volatility)
        return np.clip(series, mean - 3*volatility, mean + 3*volatility) # Clip to reasonable bounds

    def merge_with_customers(self, customer_df: pd.DataFrame, signals_df: pd.DataFrame, 
                             date_col: str = 'cust_orig_date') -> pd.DataFrame:
        """
        Merges generated signals with customer data based on a date column.
        The external signals are associated with the customer's original date.
        
        Args:
            customer_df: The DataFrame containing customer information.
            signals_df: The DataFrame containing generated time-series signals with a 'date' column.
            date_col: The name of the date column in customer_df to use for merging.

        Returns:
            The customer DataFrame augmented with the merged signals.
        """
        df_merged = customer_df.copy()
        
        if date_col not in df_merged.columns:
            raise ValueError(f"Date column '{date_col}' not found in customer_df.")
            
        df_merged[date_col] = pd.to_datetime(df_merged[date_col])
        
        # Perform a left merge to add signals to customer data
        df_merged = df_merged.merge(
            signals_df, left_on=date_col, right_on='date', how='left'
        )
        
        # Fill missing signals (for cust_orig_date outside the signals date range)
        signal_cols = signals_df.columns.drop('date').tolist()
        
        for col in signal_cols:
            if df_merged[col].isnull().any():
                df_merged[col].fillna(method='ffill', inplace=True)
                df_merged[col].fillna(method='bfill', inplace=True)
                if df_merged[col].isnull().any(): # If still missing (e.g., all NaNs in a column)
                    df_merged[col].fillna(df_merged[col].mean(), inplace=True) # Fallback to mean

        df_merged.drop('date', axis=1, inplace=True) # Drop the merge key from signals_df
        
        print(f"Successfully merged {len(signal_cols)} signals with customer data.")
        return df_merged


# ===============================
# 3️⃣ Synthetic Data Analyzer
# ===============================
class SyntheticDataAnalyzer:
    """
    Analyzes the quality of synthetic signals by comparing statistical properties
    and impact on feature importance.
    """
    def __init__(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                 target_col: str = Config.TARGET_COLUMN):
        self.real_df = real_df
        self.synthetic_df = synthetic_df
        self.target_col = target_col
        self.common_numeric_cols = self._get_common_numeric_cols()
        self.new_signal_cols = self._get_new_signal_cols()

    def _get_numeric_cols(self, df: pd.DataFrame) -> list[str]:
        """Helper to get numeric columns from a DataFrame."""
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def _get_common_numeric_cols(self) -> list[str]:
        """Identifies numeric columns present in both real_df and synthetic_df."""
        real_numeric = set(self._get_numeric_cols(self.real_df))
        synth_numeric = set(self._get_numeric_cols(self.synthetic_df))
        return sorted(list(real_numeric.intersection(synth_numeric)))

    def _get_new_signal_cols(self) -> list[str]:
        """Identifies newly added numeric signal columns in synthetic_df."""
        real_cols = set(self.real_df.columns)
        synth_numeric_cols = set(self._get_numeric_cols(self.synthetic_df))
        new_cols = synth_numeric_cols.difference(real_cols)
        return sorted(list(new_cols))

    def analyze(self):
        print("\n--- SYNTHETIC DATA ANALYSIS ---")
        self._compare_common_distributions()
        self._analyze_new_signal_distributions()
        self._compare_correlations()
        self._feature_importance_preservation()

    def _compare_common_distributions(self):
        """
        Compares distributions of common numeric columns between real and synthetic data.
        These should ideally be identical if the original data was just augmented.
        """
        print("\n1. Distribution Comparison (Common Numeric Columns):")
        if not self.common_numeric_cols:
            print("No common numeric columns found to compare distributions.")
            return

        for col in self.common_numeric_cols:
            # Skip the target column for KS/KL as it's typically categorical and exact match is expected
            if col == self.target_col:
                continue 
            
            # Ensure no NaNs as KS-test requires no missing values
            real_data = self.real_df[col].dropna()
            synth_data = self.synthetic_df[col].dropna()
            
            if not real_data.empty and not synth_data.empty:
                ks_stat, ks_p = ks_2samp(real_data, synth_data)
                
                # For KL divergence, create histograms. Add small epsilon to avoid log(0)
                hist_real, _ = np.histogram(real_data, bins=50, density=True)
                hist_synth, _ = np.histogram(synth_data, bins=50, density=True)
                kl_div = entropy(hist_real + 1e-8, hist_synth + 1e-8)
                print(f"  {col:25} KS_Stat={ks_stat:.4f} (p={ks_p:.4f}) KL_Div={kl_div:.4f}")
            else:
                print(f"  {col:25} Skipped (contains empty data after dropping NaNs).")
                
        print("  (Expected KS_Stat ~0 and KL_Div ~0 for common columns as original data is preserved)")

    def _analyze_new_signal_distributions(self):
        """
        Analyzes and visualizes the distributions of the newly generated synthetic signals.
        """
        print("\n2. Analysis of New Synthetic Signal Distributions:")
        if not self.new_signal_cols:
            print("No new signal columns detected in the synthetic DataFrame.")
            return

        plt.figure(figsize=(15, 5 * (len(self.new_signal_cols) // 3 + 1)))
        for i, col in enumerate(self.new_signal_cols):
            print(f"  {col:25} Mean={self.synthetic_df[col].mean():.4f}, "
                  f"Std={self.synthetic_df[col].std():.4f}, "
                  f"Min={self.synthetic_df[col].min():.4f}, "
                  f"Max={self.synthetic_df[col].max():.4f}")
            
            plt.subplot(len(self.new_signal_cols) // 3 + 1, 3, i + 1)
            sns.histplot(self.synthetic_df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOT_DIR, 'new_signal_distributions.png'))
        plt.close()
        print(f"  Histogram plots saved to {Config.PLOT_DIR}/new_signal_distributions.png")

    def _compare_correlations(self):
        """
        Compares correlation matrices and highlights correlations involving new signals.
        """
        print("\n3. Correlation Analysis:")
        
        # Correlation for original features
        real_corr = self.real_df[self.common_numeric_cols].corr()
        print(f"\n  Original Feature Correlation Matrix (first 5x5):\n{real_corr.head(5).iloc[:, :5]}")

        # Correlation for augmented data (all numeric features)
        all_numeric_augmented_cols = self._get_numeric_cols(self.synthetic_df)
        synth_full_corr = self.synthetic_df[all_numeric_augmented_cols].corr()
        print(f"\n  Augmented Data Full Correlation Matrix (first 5x5 including new signals):\n{synth_full_corr.head(5).iloc[:, :5]}")
        
        # If new signals exist, show their correlations
        if self.new_signal_cols:
            print("\n  Correlations of New Signals with Churn and other Signals:")
            # Correlations of new signals with the target
            if self.target_col in synth_full_corr.index:
                new_signal_target_corr = synth_full_corr.loc[self.new_signal_cols, self.target_col].sort_values(ascending=False)
                print(f"    New Signal vs. {self.target_col}:\n{new_signal_target_corr}")

            # Correlations among new signals
            new_signals_only_corr = self.synthetic_df[self.new_signal_cols].corr()
            print(f"\n    Correlation among New Signals (first 5x5):\n{new_signals_only_corr.head(5).iloc[:, :5]}")
            
            # Visualize full augmented correlation matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(synth_full_corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Correlation Matrix of Augmented Data')
            plt.savefig(os.path.join(Config.PLOT_DIR, 'augmented_correlation_matrix.png'))
            plt.close()
            print(f"  Full correlation heatmap saved to {Config.PLOT_DIR}/augmented_correlation_matrix.png")

    def _feature_importance_preservation(self):
        """
        Compares feature importances for 'Churn' prediction using RandomForest
        on the original balanced data versus the augmented data.
        """
        print("\n4. Feature Importance Analysis (RandomForest for Churn Prediction):")
        if self.target_col not in self.real_df.columns or self.target_col not in self.synthetic_df.columns:
            print(f"  Target column '{self.target_col}' missing, skipping Feature Importance check.")
            return

        # Prepare data for RF models
        X_real = self.real_df[self.common_numeric_cols].drop(columns=[self.target_col], errors='ignore')
        y_real = self.real_df[self.target_col]
        
        X_synth = self.synthetic_df[self._get_numeric_cols(self.synthetic_df)].drop(columns=[self.target_col], errors='ignore')
        y_synth = self.synthetic_df[self.target_col]

        # Ensure no non-numeric columns remain, and handle potential NaNs
        X_real = X_real.select_dtypes(include=np.number).fillna(X_real.mean())
        X_synth = X_synth.select_dtypes(include=np.number).fillna(X_synth.mean())

        if X_real.empty or X_synth.empty or y_real.empty or y_synth.empty:
            print("  Insufficient numeric data or target for Feature Importance analysis.")
            return

        # Train on real (balanced) data
        rf_real = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=-1)
        rf_real.fit(X_real, y_real)
        importances_real = pd.Series(rf_real.feature_importances_, index=X_real.columns).sort_values(ascending=False)
        
        # Train on synthetic (augmented) data
        rf_synth = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=-1)
        rf_synth.fit(X_synth, y_synth)
        importances_synth = pd.Series(rf_synth.feature_importances_, index=X_synth.columns).sort_values(ascending=False)

        print("\n  Top features from original (balanced) data:")
        print(importances_real.head(10).to_string())

        print("\n  Top features from augmented data (including new signals):")
        print(importances_synth.head(10).to_string())
        
        # Compare and plot
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=importances_real.head(10).values, y=importances_real.head(10).index)
        plt.title('Feature Importances (Original Balanced Data)')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=importances_synth.head(10).values, y=importances_synth.head(10).index)
        plt.title('Feature Importances (Augmented Data with Signals)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOT_DIR, 'feature_importances_comparison.png'))
        plt.close()
        print(f"  Feature importance plots saved to {Config.PLOT_DIR}/feature_importances_comparison.png")


# ===============================
# 4️⃣ Main Execution
# ===============================
def main():
    print("=== SMART NON-LINEAR DATA AUGMENTATION (Optimized) ===")
    
    try:
        df = pd.read_csv('autoinsurance_churn_cleaned.csv')
        print(f"Loaded {len(df):,} rows from 'autoinsurance_churn_cleaned.csv'")
    except FileNotFoundError:
        print("Error: 'autoinsurance_churn_cleaned.csv' not found. Please ensure the file is in the same directory.")
        return

    # Convert 'cust_orig_date' to datetime upfront if it exists
    if 'cust_orig_date' in df.columns:
        df['cust_orig_date'] = pd.to_datetime(df['cust_orig_date'], errors='coerce')
        # Drop rows where date conversion failed if necessary, or fill with a sensible value
        df.dropna(subset=['cust_orig_date'], inplace=True)
        print(f"Converted 'cust_orig_date' to datetime. {len(df):,} rows remaining after dropping NaNs in date.")
    else:
        print("Warning: 'cust_orig_date' column not found. Signal merging might be affected.")

    # Step 1: Balanced sample
    print("\n--- Step 1: Balancing the dataset ---")
    sampler = BalancedDataSampler(df, target_col=Config.TARGET_COLUMN)
    df_balanced = sampler.get_balanced_sample(n_samples=Config.N_SAMPLES_BALANCED)

    # Step 2: Generate non-linear signals
    print("\n--- Step 2: Generating non-linear synthetic signals ---")
    generator = NonLinearSignalGenerator(
        start_date=Config.SIGNAL_START_DATE,
        end_date=Config.SIGNAL_END_DATE
    )
    signals = generator.generate_signals()

    # Step 3: Merge with customer data
    print("\n--- Step 3: Merging signals with customer data ---")
    df_augmented = generator.merge_with_customers(df_balanced, signals, date_col='cust_orig_date')

    # Step 4: Analyze signals
    print("\n--- Step 4: Analyzing synthetic data quality ---")
    analyzer = SyntheticDataAnalyzer(df_balanced, df_augmented, target_col=Config.TARGET_COLUMN)
    analyzer.analyze()

    # Step 5: Save files
    print("\n--- Step 5: Saving processed data files ---")
    df_balanced.to_csv(Config.OUTPUT_BALANCED_CSV, index=False)
    df_augmented.to_csv(Config.OUTPUT_AUGMENTED_CSV, index=False)
    signals.to_csv(Config.OUTPUT_SIGNALS_CSV, index=False)
    print(f"  Balanced sample saved to '{Config.OUTPUT_BALANCED_CSV}'")
    print(f"  Augmented data saved to '{Config.OUTPUT_AUGMENTED_CSV}'")
    print(f"  Synthetic time-series signals saved to '{Config.OUTPUT_SIGNALS_CSV}'")
    print("\nProcessing complete!")

    return df_balanced, df_augmented, signals


if __name__ == "__main__":
    balanced_data, augmented_data, signals = main()