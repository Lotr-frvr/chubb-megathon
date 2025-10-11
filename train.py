"""
Train and test an Explainable Boosting Machine (EBM) model on auto insurance churn data
using Microsoft's InterpretML library.
"""

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
import matplotlib.pyplot as plt
import seaborn as sns
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show


def load_and_preprocess_data(filepath):
    """Load and preprocess the auto insurance churn dataset."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    
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
                       'home_owner', 'college_degree', 'good_credit']
    
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
    print(f"Missing values after preprocessing:\n{df_processed.isnull().sum().sum()}")
    
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


def train_ebm_model(X_train, y_train, feature_names=None):
    """Train an Explainable Boosting Machine (EBM) classifier."""
    print("\nTraining EBM model...")
    
    # Initialize the EBM classifier
    # EBM automatically handles both numeric and categorical features
    ebm = ExplainableBoostingClassifier(
        max_bins=256,           # Maximum number of bins for numeric features
        max_interaction_bins=32,  # Bins for interaction terms
        interactions=10,         # Number of pairwise interaction terms to detect
        learning_rate=0.01,      # Learning rate
        min_samples_leaf=2,      # Minimum samples per leaf
        max_leaves=3,            # Maximum leaves per tree
        n_jobs=-1,               # Use all CPU cores
        random_state=42
    )
    
    # Train the model
    ebm.fit(X_train, y_train)
    
    print("Model training complete!")
    print(f"Number of features used: {len(ebm.feature_names_in_)}")
    print(f"Feature names: {ebm.feature_names_in_}")
    
    return ebm


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the model on both train and test sets."""
    print("\nEvaluating model performance...")
    
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
    print("TRAINING SET PERFORMANCE")
    print("="*50)
    for metric, value in train_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\n" + "="*50)
    print("TEST SET PERFORMANCE")
    print("="*50)
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Classification report
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("="*50)
    print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(cm)
    
    return train_metrics, test_metrics, cm, y_test_pred, y_test_pred_proba


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix - EBM Model')
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


def save_model_summary(model, train_metrics, test_metrics, importance_df, 
                       save_path='model_summary.txt'):
    """Save a summary of the model and its performance."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EXPLAINABLE BOOSTING MACHINE (EBM) MODEL SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODEL PARAMETERS:\n")
        f.write(f"Max Bins: {model.max_bins}\n")
        f.write(f"Max Interaction Bins: {model.max_interaction_bins}\n")
        f.write(f"Interactions: {model.interactions}\n")
        f.write(f"Learning Rate: {model.learning_rate}\n")
        f.write(f"Max Leaves: {model.max_leaves}\n")
        f.write(f"Min Samples Leaf: {model.min_samples_leaf}\n\n")
        
        f.write("TRAINING PERFORMANCE:\n")
        for metric, value in train_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        f.write("\n")
        
        f.write("TEST PERFORMANCE:\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        f.write("\n")
        
        f.write("TOP FEATURES BY IMPORTANCE:\n")
        for idx, row in importance_df.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    
    print(f"Model summary saved to: {save_path}")


def main():
    """Main function to run the entire training pipeline."""
    # Configuration
    DATA_PATH = 'autoinsurance_churn_cleaned.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    OUTPUT_DIR = 'ebm_results'
    
    # For large datasets, optionally use a sample for faster training
    # Set to None to use full dataset, or a float between 0 and 1 for sampling
    SAMPLE_FRACTION = 0.1  # Use 10% of data for faster training
    
    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("EBM TRAINING PIPELINE - AUTO INSURANCE CHURN PREDICTION")
    print("="*60)
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(DATA_PATH)
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target_column='Churn',
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Step 3: Train EBM model
    ebm_model = train_ebm_model(X_train, y_train)
    
    # Step 4: Evaluate model
    train_metrics, test_metrics, cm, y_test_pred, y_test_pred_proba = evaluate_model(
        ebm_model, X_train, y_train, X_test, y_test
    )
    
    # Step 5: Visualizations and saving results
    plot_confusion_matrix(cm, save_path=f'{OUTPUT_DIR}/confusion_matrix.png')
    importance_df = plot_feature_importance(
        ebm_model, 
        top_n=20, 
        save_path=f'{OUTPUT_DIR}/feature_importance.png'
    )
    save_model_summary(
        ebm_model, 
        train_metrics, 
        test_metrics, 
        importance_df,
        save_path=f'{OUTPUT_DIR}/model_summary.txt'
    )
    
    # Step 6: Save the model
    import joblib
    model_path = f'{OUTPUT_DIR}/ebm_model.pkl'
    joblib.dump(ebm_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Step 7: Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_test_pred,
        'probability_churn': y_test_pred_proba
    })
    predictions_path = f'{OUTPUT_DIR}/test_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Test predictions saved to: {predictions_path}")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nAll results saved in: {OUTPUT_DIR}/")
    print("\nKey findings:")
    print(f"- Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"- Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"- Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"\nTop 5 most important features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")


if __name__ == "__main__":
    main()
