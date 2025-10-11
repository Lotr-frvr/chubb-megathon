"""
Multi-GPU Training Script for Auto Insurance Churn Prediction
Uses distributed computing across 4 GPUs for data processing and model training

QUICK START - CONTROLLING MODEL PARAMETERS:
============================================

All hyperparameters are in the main() function around line 400.

TO CONTROL EPOCHS (Training Time):
  - MAX_ROUNDS: Number of boosting iterations
    * Quick test: 500-1000
    * Default: 5000
    * Full training: 10000+
  
  - EARLY_STOPPING_ROUNDS: Stop if no improvement
    * Impatient: 50
    * Default: 100
    * Patient: 200-500

TO CONTROL MODEL SIZE/COMPLEXITY (Number of Parameters):
  - INTERACTIONS: Feature interactions (BIGGEST impact on parameter count)
    * None: 0 (smallest)
    * Few: 3-5
    * Default: 8
    * Many: 15-20 (largest)
  
  - MAX_BINS: Feature discretization bins
    * Small: 64-128
    * Default: 256
    * Large: 512-1024
  
  - MAX_LEAVES: Tree complexity per feature
    * Simple: 2
    * Default: 3
    * Complex: 4-5

EXAMPLE CONFIGURATIONS:
  1. Fast/Simple:   MAX_ROUNDS=1000, INTERACTIONS=3, MAX_BINS=128
  2. Balanced:      MAX_ROUNDS=5000, INTERACTIONS=8, MAX_BINS=256 (default)
  3. Large/Complex: MAX_ROUNDS=10000, INTERACTIONS=15, MAX_BINS=512
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from interpret.glassbox import ExplainableBoostingClassifier
import joblib
from multiprocessing import cpu_count
import time

# Set environment variables for optimal multi-GPU usage
os.environ['OMP_NUM_THREADS'] = str(cpu_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Use all 4 GPUs


def load_and_preprocess_data(filepath, sample_fraction=None, use_gpu=False):
    """
    Load and preprocess the auto insurance churn dataset.
    
    Args:
        filepath: Path to the CSV file
        sample_fraction: Fraction of data to use (None for all data)
        use_gpu: Whether to use GPU-accelerated cuDF (requires RAPIDS)
    """
    print("Loading data...")
    start_time = time.time()
    
    if use_gpu:
        try:
            import cudf
            df = cudf.read_csv(filepath)
            print("Using GPU-accelerated cuDF for data processing")
        except ImportError:
            print("cuDF not available, falling back to pandas")
            df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    
    # Sample data if requested
    if sample_fraction is not None and sample_fraction < 1.0:
        original_size = len(df)
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"Sampled {len(df)} rows from {original_size} ({sample_fraction*100:.1f}%)")
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    
    # Convert back to pandas if using cuDF
    if use_gpu and 'cudf' in str(type(df)):
        df = df.to_pandas()
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Drop ID columns and date columns that won't be useful for prediction
    columns_to_drop = ['individual_id', 'address_id', 'cust_orig_date', 
                       'date_of_birth', 'acct_suspd_date']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    # Handle categorical variables
    categorical_columns = ['city', 'state', 'county', 'marital_status', 'home_market_value']
    
    # Ensure numeric columns are proper numeric types
    numeric_columns = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'latitude', 
                       'longitude', 'income', 'has_children', 'length_of_residence',
                       'home_owner', 'college_degree', 'good_credit',
                       'chubb_sentiment', 'competitor_sentiment', 'chubb_stock_return',
                       'competitor_stock_return', 'market_index_return', 'gdp_growth',
                       'customer_engagement']
    
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Handle missing values
    # For numeric columns, fill with median
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # For categorical columns, fill with mode or 'Unknown' BEFORE converting to category
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
            # Convert to category type after filling missing values
            df_processed[col] = df_processed[col].astype('category')
    
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print(f"Missing values after preprocessing: {df_processed.isnull().sum().sum()}")
    
    return df_processed


def split_data(df, target_column='Churn', test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    print(f"\nSplitting data with test_size={test_size}...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Training set churn rate: {y_train.mean():.3f}")
    print(f"Test set churn rate: {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test


def train_ebm_model_distributed(X_train, y_train, n_jobs=-1, 
                                max_rounds=5000, early_stopping_rounds=100,
                                max_bins=256, max_leaves=3, interactions=8,
                                max_interaction_bins=16, learning_rate=0.01,
                                min_samples_leaf=5):
    """
    Train an Explainable Boosting Machine (EBM) classifier with distributed computing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_jobs: Number of parallel jobs (-1 uses all available CPUs)
        
        === EPOCH/ITERATION CONTROL ===
        max_rounds: Number of boosting rounds (like epochs) - higher = longer training
        early_stopping_rounds: Stop if no improvement for this many rounds
        
        === MODEL COMPLEXITY CONTROL ===
        max_bins: Number of bins for numeric features (64-512, higher = more complex)
        max_leaves: Leaves per tree (2-5, higher = more complex)
        interactions: Number of feature interactions (0-20, higher = more parameters)
        max_interaction_bins: Bins for interactions (8-32)
        
        === OTHER PARAMETERS ===
        learning_rate: Learning rate (0.001-0.1)
        min_samples_leaf: Minimum samples per leaf (higher = simpler, faster)
    """
    print("\nTraining EBM model with distributed computing...")
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    print(f"\nModel Configuration:")
    print(f"  Max Rounds (epochs): {max_rounds}")
    print(f"  Early Stopping: {early_stopping_rounds}")
    print(f"  Max Bins: {max_bins}")
    print(f"  Max Leaves: {max_leaves}")
    print(f"  Interactions: {interactions}")
    print(f"  Learning Rate: {learning_rate}")
    
    start_time = time.time()
    
    # Initialize the EBM classifier with configurable parameters
    ebm = ExplainableBoostingClassifier(
        max_bins=max_bins,                      # Bins for numeric features
        max_interaction_bins=max_interaction_bins,  # Bins for interactions
        interactions=interactions,              # Number of interaction terms
        outer_bags=8,                          # Number of outer bags for bagging
        inner_bags=0,                          # Disable inner bagging for speed
        learning_rate=learning_rate,           # Learning rate
        min_samples_leaf=min_samples_leaf,     # Minimum samples per leaf
        max_leaves=max_leaves,                 # Maximum leaves per tree
        max_rounds=max_rounds,                 # Maximum boosting rounds (epochs)
        early_stopping_rounds=early_stopping_rounds,  # Early stopping
        n_jobs=n_jobs,                         # Use all available cores
        random_state=42
    )
    
    # Train the model
    ebm.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Model training complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Number of features used: {len(ebm.feature_names_in_)}")
    
    return ebm


def train_xgboost_gpu(X_train, y_train, X_test, y_test):
    """
    Train XGBoost with GPU acceleration for comparison.
    XGBoost can utilize multiple GPUs for faster training.
    """
    try:
        import xgboost as xgb
        print("\nTraining XGBoost with GPU acceleration for comparison...")
        start_time = time.time()
        
        # Convert categorical features to numeric
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        for col in X_train_encoded.select_dtypes(include=['category', 'object']).columns:
            X_train_encoded[col] = X_train_encoded[col].astype('category').cat.codes
            X_test_encoded[col] = X_test_encoded[col].astype('category').cat.codes
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # XGBoost GPU parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist',  # GPU accelerated training
            'gpu_id': 0,                # Primary GPU
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        xgb_model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        xgb_model.fit(
            X_train_encoded, y_train,
            eval_set=[(X_test_encoded, y_test)],
            verbose=50
        )
        
        training_time = time.time() - start_time
        print(f"XGBoost training complete in {training_time:.2f} seconds")
        
        return xgb_model, X_test_encoded
        
    except ImportError:
        print("XGBoost not installed. Skipping GPU-accelerated XGBoost training.")
        return None, None


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """Evaluate the model on both train and test sets."""
    print(f"\nEvaluating {model_name} performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities for AUC
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_pred_proba)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
    }
    
    # Print results
    print("\n" + "="*50)
    print(f"{model_name.upper()} - TRAINING SET PERFORMANCE")
    print("="*50)
    for metric, value in train_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\n" + "="*50)
    print(f"{model_name.upper()} - TEST SET PERFORMANCE")
    print("="*50)
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Classification report
    print("\n" + "="*50)
    print(f"DETAILED CLASSIFICATION REPORT - {model_name} (Test Set)")
    print("="*50)
    print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix - {model_name} (Test Set):")
    print(cm)
    
    return train_metrics, test_metrics, cm, y_test_pred, y_test_pred_proba


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def plot_feature_importance(model, top_n=20, save_path='feature_importance.png'):
    """Plot feature importance from the EBM model."""
    # Get global explanation
    ebm_global = model.explain_global()
    
    # Extract feature names and importance scores
    feature_names = ebm_global.data()['names']
    importance_scores = ebm_global.data()['scores']
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance - EBM Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()
    
    return importance_df


def main():
    """
    Main function to run the multi-GPU training pipeline.
    
    KEY HYPERPARAMETERS TO TUNE:
    
    1. EPOCHS/ITERATIONS (Training Duration):
       - MAX_ROUNDS: Number of boosting rounds (like epochs)
         * Default: 5000
         * Quick test: 500-1000
         * Full training: 3000-10000
       
       - EARLY_STOPPING_ROUNDS: Stop if no improvement
         * Default: 100
         * More patient: 200-500
         * Less patient: 50
    
    2. MODEL COMPLEXITY (Number of Parameters):
       - MAX_BINS: Feature discretization bins
         * Smaller model: 64-128
         * Default: 256
         * Larger model: 512-1024
       
       - MAX_LEAVES: Tree complexity
         * Simpler: 2
         * Default: 3
         * More complex: 4-5
       
       - INTERACTIONS: Feature interactions (biggest impact on parameters)
         * No interactions: 0
         * Few: 3-5
         * Default: 8
         * Many: 10-20
    
    3. EXAMPLES:
       - Fast/Simple model: MAX_ROUNDS=1000, MAX_BINS=128, INTERACTIONS=3
       - Balanced model: MAX_ROUNDS=5000, MAX_BINS=256, INTERACTIONS=8 (default)
       - Complex model: MAX_ROUNDS=10000, MAX_BINS=512, INTERACTIONS=15
    """
    
    print("="*60)
    print("MULTI-GPU TRAINING PIPELINE - AUTO INSURANCE CHURN")
    print("="*60)
    print(f"Available CPUs: {cpu_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Configuration
    DATA_PATH = 'final_data.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    OUTPUT_DIR = 'multi_gpu_results'
    
    # Data sampling - adjust based on your needs
    # Set to None for full dataset, or a value like 0.2 for 20% of data
    SAMPLE_FRACTION = None  # Using FULL dataset - set to 0.2 for 20% sampling
    
    # ========== MODEL HYPERPARAMETERS - EASY TO CONFIGURE ==========
    # Control training iterations (epochs)
    MAX_ROUNDS = 5000              # Number of boosting rounds (like epochs)
    EARLY_STOPPING_ROUNDS = 100    # Stop early if no improvement
    
    # Control model complexity (number of parameters)
    MAX_BINS = 256                 # Bins for features (64-512: higher = more complex)
    MAX_LEAVES = 3                 # Leaves per tree (2-5: higher = more complex)
    INTERACTIONS = 8               # Feature interactions (0-20: higher = more parameters)
    MAX_INTERACTION_BINS = 16      # Bins for interactions (8-32)
    
    # Other training parameters
    LEARNING_RATE = 0.01           # Learning rate (0.001-0.1)
    MIN_SAMPLES_LEAF = 5           # Min samples per leaf (higher = simpler)
    # ===============================================================
    
    # Try to use GPU for data loading (requires RAPIDS/cuDF)
    USE_GPU_DATA_LOADING = False  # Set to True if you have RAPIDS installed
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)
    df = load_and_preprocess_data(
        DATA_PATH, 
        sample_fraction=SAMPLE_FRACTION,
        use_gpu=USE_GPU_DATA_LOADING
    )
    
    # Step 2: Split data
    print("\n" + "="*60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*60)
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target_column='Churn',
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Step 3: Train EBM model with distributed computing
    print("\n" + "="*60)
    print("STEP 3: TRAINING EBM MODEL (CPU-DISTRIBUTED)")
    print("="*60)
    ebm_model = train_ebm_model_distributed(
        X_train, y_train, 
        n_jobs=-1,
        max_rounds=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        max_bins=MAX_BINS,
        max_leaves=MAX_LEAVES,
        interactions=INTERACTIONS,
        max_interaction_bins=MAX_INTERACTION_BINS,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF
    )
    
    # Step 4: Train XGBoost with GPU (optional comparison)
    print("\n" + "="*60)
    print("STEP 4: TRAINING XGBOOST (GPU-ACCELERATED) - OPTIONAL")
    print("="*60)
    xgb_model, X_test_encoded = train_xgboost_gpu(X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluate EBM model
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    ebm_train_metrics, ebm_test_metrics, ebm_cm, y_test_pred, y_test_pred_proba = evaluate_model(
        ebm_model, X_train, y_train, X_test, y_test, model_name="EBM"
    )
    
    # Evaluate XGBoost if available
    if xgb_model is not None:
        X_train_encoded = X_train.copy()
        for col in X_train_encoded.select_dtypes(include=['category', 'object']).columns:
            X_train_encoded[col] = X_train_encoded[col].astype('category').cat.codes
        
        xgb_train_metrics, xgb_test_metrics, xgb_cm, _, _ = evaluate_model(
            xgb_model, X_train_encoded, y_train, X_test_encoded, y_test, model_name="XGBoost-GPU"
        )
    
    # Step 6: Visualizations and saving results
    print("\n" + "="*60)
    print("STEP 6: SAVING RESULTS")
    print("="*60)
    
    # EBM Results
    plot_confusion_matrix(ebm_cm, "EBM", save_path=f'{OUTPUT_DIR}/ebm_confusion_matrix.png')
    importance_df = plot_feature_importance(
        ebm_model, 
        top_n=20, 
        save_path=f'{OUTPUT_DIR}/ebm_feature_importance.png'
    )
    
    # XGBoost Results
    if xgb_model is not None:
        plot_confusion_matrix(xgb_cm, "XGBoost-GPU", save_path=f'{OUTPUT_DIR}/xgboost_confusion_matrix.png')
    
    # Save models
    ebm_model_path = f'{OUTPUT_DIR}/ebm_model.pkl'
    joblib.dump(ebm_model, ebm_model_path)
    print(f"\nEBM model saved to: {ebm_model_path}")
    
    if xgb_model is not None:
        xgb_model_path = f'{OUTPUT_DIR}/xgboost_model.pkl'
        joblib.dump(xgb_model, xgb_model_path)
        print(f"XGBoost model saved to: {xgb_model_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'ebm_predicted': y_test_pred,
        'ebm_probability_churn': y_test_pred_proba
    })
    predictions_path = f'{OUTPUT_DIR}/test_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Test predictions saved to: {predictions_path}")
    
    # Save summary
    with open(f'{OUTPUT_DIR}/training_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("MULTI-GPU TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset Size: {len(df)} rows\n")
        f.write(f"Training Size: {len(X_train)} rows\n")
        f.write(f"Test Size: {len(X_test)} rows\n")
        f.write(f"Sample Fraction: {SAMPLE_FRACTION}\n\n")
        
        f.write("MODEL HYPERPARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Max Rounds (epochs): {MAX_ROUNDS}\n")
        f.write(f"Early Stopping Rounds: {EARLY_STOPPING_ROUNDS}\n")
        f.write(f"Max Bins: {MAX_BINS}\n")
        f.write(f"Max Leaves: {MAX_LEAVES}\n")
        f.write(f"Interactions: {INTERACTIONS}\n")
        f.write(f"Max Interaction Bins: {MAX_INTERACTION_BINS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Min Samples Leaf: {MIN_SAMPLES_LEAF}\n\n")
        
        f.write("EBM MODEL PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write("Test Metrics:\n")
        for metric, value in ebm_test_metrics.items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")
        f.write("\n")
        
        if xgb_model is not None:
            f.write("XGBOOST-GPU MODEL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write("Test Metrics:\n")
            for metric, value in xgb_test_metrics.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
            f.write("\n")
        
        f.write("TOP 10 FEATURES (EBM):\n")
        f.write("-" * 40 + "\n")
        for idx, row in importance_df.head(10).iterrows():
            f.write(f"{idx+1}. {row['Feature']}: {row['Importance']:.4f}\n")
    
    print(f"Training summary saved to: {OUTPUT_DIR}/training_summary.txt")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nAll results saved in: {OUTPUT_DIR}/")
    print("\nEBM Model Performance:")
    print(f"  Test Accuracy: {ebm_test_metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC: {ebm_test_metrics['roc_auc']:.4f}")
    print(f"  Test F1-Score: {ebm_test_metrics['f1']:.4f}")
    
    if xgb_model is not None:
        print("\nXGBoost-GPU Model Performance:")
        print(f"  Test Accuracy: {xgb_test_metrics['accuracy']:.4f}")
        print(f"  Test ROC-AUC: {xgb_test_metrics['roc_auc']:.4f}")
        print(f"  Test F1-Score: {xgb_test_metrics['f1']:.4f}")
    
    print(f"\nTop 5 most important features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")


if __name__ == "__main__":
    main()
