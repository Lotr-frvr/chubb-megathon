"""
Interactive Explainability Dashboard for EBM Model
Uses InterpretML's visualization capabilities to explain model predictions
"""

import pandas as pd
import numpy as np
import joblib
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression, ClassificationTree
import os
import sys


def load_trained_model(model_path='ebm_results_multi_gpu/ebm_model.pkl'):
    """Load the trained EBM model."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model


def load_data_splits(output_dir='ebm_results_multi_gpu'):
    """
    Load or recreate the train/test splits.
    Since we saved predictions, we can load the test data info.
    For full functionality, we'll recreate the splits.
    """
    from train_multi_gpu import load_and_preprocess_data, split_data
    
    print("Loading and preprocessing data...")
    DATA_PATH = 'autoinsurance_churn_cleaned.csv'
    SAMPLE_FRACTION = None  # Use full dataset
    
    df = load_and_preprocess_data(DATA_PATH, sample_fraction=SAMPLE_FRACTION)
    X_train, X_test, y_train, y_test = split_data(df, target_column='Churn', test_size=0.2, random_state=42)
    
    print(f"Data loaded: {len(X_train)} train samples, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


def create_global_explanations(ebm_model, output_dir='ebm_results_multi_gpu/explanations'):
    """
    Create global (model-level) explanations.
    Shows overall feature importance and behavior across the dataset.
    """
    print("\n" + "="*60)
    print("CREATING GLOBAL EXPLANATIONS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate global explanation
    print("Generating global explanation for EBM model...")
    ebm_global = ebm_model.explain_global()
    
    # Save interactive HTML - use file_name parameter correctly
    html_path = os.path.join(output_dir, 'ebm_global_explanation.html')
    from interpret import preserve
    preserve(ebm_global, file_name=html_path, auto_open=False)
    print(f"Global explanation saved to: {html_path}")
    
    # Display in notebook/browser (if available)
    print(f"To view interactively, open {html_path} in a web browser")
    
    return ebm_global


def create_local_explanations(ebm_model, X_test, y_test, num_samples=100, output_dir='ebm_results_multi_gpu/explanations'):
    """
    Create local (instance-level) explanations.
    Shows why the model made specific predictions for individual samples.
    """
    print("\n" + "="*60)
    print("CREATING LOCAL EXPLANATIONS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a subset of test samples for local explanation
    if len(X_test) > num_samples:
        print(f"Selecting {num_samples} samples from test set for local explanations...")
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
    else:
        X_sample = X_test
        y_sample = y_test
    
    # Generate local explanations
    print(f"Generating local explanations for {len(X_sample)} samples...")
    ebm_local = ebm_model.explain_local(X_sample, y_sample)
    
    # Save interactive HTML
    html_path = os.path.join(output_dir, 'ebm_local_explanation.html')
    preserve(ebm_local, html_path)
    print(f"Local explanation saved to: {html_path}")
    
    # Display in notebook/browser (if available)
    try:
        show(ebm_local)
        print("Local explanation displayed!")
    except Exception as e:
        print(f"Could not display interactively: {e}")
        print(f"Please open {html_path} in a web browser to view the interactive explanation.")
    
    return ebm_local


def create_specific_instance_explanation(ebm_model, X_test, y_test, instance_index=0, output_dir='ebm_results_multi_gpu/explanations'):
    """
    Create explanation for a specific instance.
    Useful for understanding individual predictions.
    """
    print("\n" + "="*60)
    print(f"CREATING EXPLANATION FOR INSTANCE {instance_index}")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get single instance
    X_instance = X_test.iloc[[instance_index]]
    y_instance = y_test.iloc[[instance_index]]
    
    # Make prediction
    prediction = ebm_model.predict(X_instance)[0]
    probability = ebm_model.predict_proba(X_instance)[0]
    
    print(f"\nInstance {instance_index} details:")
    print(f"Actual churn: {y_instance.values[0]}")
    print(f"Predicted churn: {prediction}")
    print(f"Probability of No Churn: {probability[0]:.4f}")
    print(f"Probability of Churn: {probability[1]:.4f}")
    
    # Generate explanation
    instance_explanation = ebm_model.explain_local(X_instance, y_instance)
    
    # Save interactive HTML
    html_path = os.path.join(output_dir, f'ebm_instance_{instance_index}_explanation.html')
    preserve(instance_explanation, html_path)
    print(f"Instance explanation saved to: {html_path}")
    
    # Display
    try:
        show(instance_explanation)
    except Exception as e:
        print(f"Could not display interactively: {e}")
        print(f"Please open {html_path} in a web browser.")
    
    return instance_explanation


def train_comparison_models(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression and Decision Tree for comparison.
    """
    print("\n" + "="*60)
    print("TRAINING COMPARISON MODELS")
    print("="*60)
    
    # Convert categorical variables for traditional models
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in X_train_encoded.select_dtypes(include=['category', 'object']).columns:
        X_train_encoded[col] = X_train_encoded[col].astype('category').cat.codes
        X_test_encoded[col] = X_test_encoded[col].astype('category').cat.codes
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_encoded, y_train)
    lr_score = lr_model.score(X_test_encoded, y_test)
    print(f"Logistic Regression Test Accuracy: {lr_score:.4f}")
    
    # Train Decision Tree
    print("\nTraining Decision Tree...")
    dt_model = ClassificationTree(random_state=42, max_depth=5)
    dt_model.fit(X_train_encoded, y_train)
    dt_score = dt_model.score(X_test_encoded, y_test)
    print(f"Decision Tree Test Accuracy: {dt_score:.4f}")
    
    return lr_model, dt_model, X_test_encoded


def create_comparison_explanations(ebm_global, lr_model, dt_model, output_dir='ebm_results_multi_gpu/explanations'):
    """
    Create side-by-side comparison of multiple models.
    """
    print("\n" + "="*60)
    print("CREATING MODEL COMPARISON")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate global explanations for each model
    print("Generating global explanation for Logistic Regression...")
    lr_global = lr_model.explain_global()
    
    print("Generating global explanation for Decision Tree...")
    dt_global = dt_model.explain_global()
    
    # Create comparison visualization
    print("\nCreating side-by-side comparison...")
    html_path = os.path.join(output_dir, 'model_comparison.html')
    preserve([ebm_global, lr_global, dt_global], html_path)
    print(f"Model comparison saved to: {html_path}")
    
    # Display comparison
    try:
        show([ebm_global, lr_global, dt_global])
        print("Model comparison displayed!")
    except Exception as e:
        print(f"Could not display interactively: {e}")
        print(f"Please open {html_path} in a web browser.")
    
    return lr_global, dt_global


def analyze_high_risk_customers(ebm_model, X_test, y_test, top_n=20, output_dir='ebm_results_multi_gpu/explanations'):
    """
    Identify and explain predictions for highest-risk customers.
    """
    print("\n" + "="*60)
    print(f"ANALYZING TOP {top_n} HIGH-RISK CUSTOMERS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get churn probabilities
    churn_probs = ebm_model.predict_proba(X_test)[:, 1]
    
    # Get top N highest risk customers
    high_risk_indices = np.argsort(churn_probs)[-top_n:][::-1]
    
    print(f"\nTop {top_n} customers with highest churn risk:")
    print("-" * 60)
    
    high_risk_data = []
    for rank, idx in enumerate(high_risk_indices, 1):
        actual = y_test.iloc[idx]
        prob = churn_probs[idx]
        prediction = ebm_model.predict(X_test.iloc[[idx]])[0]
        
        high_risk_data.append({
            'Rank': rank,
            'Index': idx,
            'Actual_Churn': actual,
            'Predicted_Churn': prediction,
            'Churn_Probability': prob,
            'Correct': actual == prediction
        })
        
        print(f"{rank}. Index {idx}: Probability={prob:.4f}, Actual={actual}, Predicted={prediction}")
    
    # Save to CSV
    high_risk_df = pd.DataFrame(high_risk_data)
    csv_path = os.path.join(output_dir, 'high_risk_customers.csv')
    high_risk_df.to_csv(csv_path, index=False)
    print(f"\nHigh-risk customers data saved to: {csv_path}")
    
    # Create local explanation for top 5 highest risk
    print(f"\nGenerating detailed explanations for top 5 highest-risk customers...")
    top_5_indices = high_risk_indices[:5]
    X_high_risk = X_test.iloc[top_5_indices]
    y_high_risk = y_test.iloc[top_5_indices]
    
    high_risk_explanation = ebm_model.explain_local(X_high_risk, y_high_risk)
    
    html_path = os.path.join(output_dir, 'high_risk_explanations.html')
    preserve(high_risk_explanation, html_path)
    print(f"High-risk explanations saved to: {html_path}")
    
    try:
        show(high_risk_explanation)
    except Exception as e:
        print(f"Could not display interactively: {e}")
        print(f"Please open {html_path} in a web browser.")
    
    return high_risk_df, high_risk_explanation


def main():
    """Main function to create all explanations."""
    
    print("="*60)
    print("INTERPRETABLE ML EXPLANATION DASHBOARD")
    print("="*60)
    
    OUTPUT_DIR = 'ebm_results_multi_gpu/explanations'
    
    # Step 1: Load trained EBM model
    print("\n" + "="*60)
    print("STEP 1: LOADING TRAINED MODEL")
    print("="*60)
    ebm_model = load_trained_model('ebm_results_multi_gpu/ebm_model.pkl')
    
    # Step 2: Load data
    print("\n" + "="*60)
    print("STEP 2: LOADING DATA")
    print("="*60)
    X_train, X_test, y_train, y_test = load_data_splits()
    
    # Step 3: Create global explanations for EBM
    ebm_global = create_global_explanations(ebm_model, OUTPUT_DIR)
    
    # Step 4: Create local explanations for EBM
    ebm_local = create_local_explanations(ebm_model, X_test, y_test, num_samples=100, output_dir=OUTPUT_DIR)
    
    # Step 5: Create explanation for specific instances
    print("\n" + "="*60)
    print("STEP 5: CREATING SPECIFIC INSTANCE EXPLANATIONS")
    print("="*60)
    
    # Explain a few specific instances
    for i in [0, 10, 100, 1000]:
        if i < len(X_test):
            create_specific_instance_explanation(ebm_model, X_test, y_test, instance_index=i, output_dir=OUTPUT_DIR)
    
    # Step 6: Train comparison models
    print("\n" + "="*60)
    print("STEP 6: TRAINING COMPARISON MODELS")
    print("="*60)
    lr_model, dt_model, X_test_encoded = train_comparison_models(X_train, y_train, X_test, y_test)
    
    # Step 7: Create model comparison
    lr_global, dt_global = create_comparison_explanations(ebm_global, lr_model, dt_model, OUTPUT_DIR)
    
    # Step 8: Analyze high-risk customers
    high_risk_df, high_risk_explanation = analyze_high_risk_customers(
        ebm_model, X_test, y_test, top_n=50, output_dir=OUTPUT_DIR
    )
    
    # Step 9: Save models
    print("\n" + "="*60)
    print("STEP 9: SAVING COMPARISON MODELS")
    print("="*60)
    
    lr_path = os.path.join(OUTPUT_DIR, 'logistic_regression_model.pkl')
    dt_path = os.path.join(OUTPUT_DIR, 'decision_tree_model.pkl')
    
    joblib.dump(lr_model, lr_path)
    joblib.dump(dt_model, dt_path)
    
    print(f"Logistic Regression model saved to: {lr_path}")
    print(f"Decision Tree model saved to: {dt_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("EXPLANATION GENERATION COMPLETE!")
    print("="*60)
    print(f"\nAll explanations saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. ebm_global_explanation.html - Overall model behavior")
    print("  2. ebm_local_explanation.html - Individual predictions (100 samples)")
    print("  3. ebm_instance_*_explanation.html - Specific instance explanations")
    print("  4. model_comparison.html - Side-by-side model comparison")
    print("  5. high_risk_explanations.html - Top 5 high-risk customer explanations")
    print("  6. high_risk_customers.csv - List of 50 highest-risk customers")
    print("\nTo view these explanations:")
    print("  - Open the .html files in a web browser")
    print("  - Or use 'show()' function in a Jupyter notebook")
    print("\nKey Insights:")
    print(f"  - Total test samples: {len(X_test)}")
    print(f"  - Churn rate: {y_test.mean():.2%}")
    print(f"  - High-risk customers identified: {len(high_risk_df)}")
    
    return {
        'ebm_global': ebm_global,
        'ebm_local': ebm_local,
        'lr_global': lr_global,
        'dt_global': dt_global,
        'high_risk_df': high_risk_df
    }


if __name__ == "__main__":
    results = main()
