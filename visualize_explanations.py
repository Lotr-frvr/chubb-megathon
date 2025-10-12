"""
Interactive Explainability Dashboard for EBM Model  
Uses InterpretML's show() function to create interactive visualizations
"""

import pandas as pd
import numpy as np
import joblib
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression, ClassificationTree
import os


def load_trained_model(model_path='ebm_results_multi_gpu/ebm_model.pkl'):
    """Load the trained EBM model."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model


def load_data_splits(output_dir='ebm_results_multi_gpu'):
    """Load or recreate the train/test splits."""
    from train_multi_gpu import load_and_preprocess_data, split_data
    
    print("Loading and preprocessing data...")
    DATA_PATH = 'autoinsurance_churn_cleaned.csv'
    SAMPLE_FRACTION = None  # Use full dataset
    
    df = load_and_preprocess_data(DATA_PATH, sample_fraction=SAMPLE_FRACTION)
    X_train, X_test, y_train, y_test = split_data(df, target_column='Churn', test_size=0.2, random_state=42)
    
    print(f"Data loaded: {len(X_train)} train samples, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


def save_explanation_html(explanation_obj, output_path):
    """
    Save explanation object to HTML file.
    Uses the internal rendering capabilities of InterpretML.
    """
    try:
        from interpret.provider._visualize import DashProvider
        from interpret.visual._plot import plot_bar, plot_line, plot_pairwise_heatmap
        
        # Get the explanation data
        data_dict = explanation_obj.data()
        
        # Create a simple HTML representation
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Explanation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Model Explanation</h1>
    <p>For interactive visualizations, please run this script in a Jupyter notebook and use show() function.</p>
    <p>Explanation data:</p>
    <pre>{str(data_dict)}</pre>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html_content)
        print(f"Explanation saved to: {output_path}")
        
    except Exception as e:
        print(f"Note: HTML export requires running in Jupyter environment. Error: {e}")
        print(f"Explanation object created successfully. Use show() to visualize.")


def main():
    """Main function to create all explanations."""
    
    print("="*60)
    print("INTERPRETABLE ML EXPLANATION DASHBOARD")
    print("="*60)
    
    OUTPUT_DIR = 'ebm_results_multi_gpu/explanations'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    
    # Step 3: Create GLOBAL explanation for EBM
    print("\n" + "="*60)
    print("STEP 3: CREATING GLOBAL EXPLANATION")
    print("="*60)
    print("Generating global explanation for EBM model...")
    ebm_global = ebm_model.explain_global()
    
    print("\nTo view global explanation interactively, use:")
    print("  from interpret import show")
    print("  show(ebm_global)")
    
    # Save feature importance
    global_data = ebm_global.data()
    if 'names' in global_data and 'scores' in global_data:
        importance_df = pd.DataFrame({
            'Feature': global_data['names'],
            'Importance': global_data['scores']
        }).sort_values('Importance', ascending=False)
        
        csv_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
        importance_df.to_csv(csv_path, index=False)
        print(f"\nFeature importance saved to: {csv_path}")
        
        print("\nTop 10 Most Important Features:")
        print("="*50)
        for idx, row in importance_df.head(10).iterrows():
            print(f"{idx+1:2d}. {row['Feature']:40s} {row['Importance']:>10.4f}")
    
    # Step 4: Create LOCAL explanations for EBM (sample)
    print("\n" + "="*60)
    print("STEP 4: CREATING LOCAL EXPLANATIONS (100 SAMPLES)")
    print("="*60)
    
    # Select 100 random samples
    sample_size = min(100, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    y_sample = y_test.iloc[sample_indices]
    
    print(f"Generating local explanations for {sample_size} samples...")
    ebm_local = ebm_model.explain_local(X_sample, y_sample)
    
    print("\nTo view local explanations interactively, use:")
    print("  from interpret import show")
    print("  show(ebm_local)")
    
    # Step 5: Train comparison models
    print("\n" + "="*60)
    print("STEP 5: TRAINING COMPARISON MODELS")
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
    print("\nTraining Classification Tree...")
    dt_model = ClassificationTree(random_state=42, max_depth=5)
    dt_model.fit(X_train_encoded, y_train)
    dt_score = dt_model.score(X_test_encoded, y_test)
    print(f"Decision Tree Test Accuracy: {dt_score:.4f}")
    
    # Step 6: Create comparison explanations
    print("\n" + "="*60)
    print("STEP 6: CREATING MODEL COMPARISON")
    print("="*60)
    
    print("Generating global explanation for Logistic Regression...")
    lr_global = lr_model.explain_global()
    
    print("Generating global explanation for Decision Tree...")
    dt_global = dt_model.explain_global()
    
    print("\nTo view side-by-side model comparison, use:")
    print("  from interpret import show")
    print("  show([ebm_global, lr_global, dt_global])")
    
    # Step 7: Analyze high-risk customers
    print("\n" + "="*60)
    print("STEP 7: ANALYZING HIGH-RISK CUSTOMERS")
    print("="*60)
    
    # Get churn probabilities
    churn_probs = ebm_model.predict_proba(X_test)[:, 1]
    
    # Get top 50 highest risk customers
    top_n = 50
    high_risk_indices = np.argsort(churn_probs)[-top_n:][::-1]
    
    print(f"\nTop {top_n} customers with highest churn risk:")
    print("-" * 80)
    
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
        
        if rank <= 20:  # Print first 20
            print(f"{rank:3d}. Index {idx:6d}: Prob={prob:.4f}, Actual={actual}, Predicted={prediction}, Correct={actual==prediction}")
    
    # Save to CSV
    high_risk_df = pd.DataFrame(high_risk_data)
    csv_path = os.path.join(OUTPUT_DIR, 'high_risk_customers.csv')
    high_risk_df.to_csv(csv_path, index=False)
    print(f"\nHigh-risk customers data saved to: {csv_path}")
    
    # Create local explanation for top 10 highest risk
    print(f"\nGenerating detailed explanations for top 10 highest-risk customers...")
    top_10_indices = high_risk_indices[:10]
    X_high_risk = X_test.iloc[top_10_indices]
    y_high_risk = y_test.iloc[top_10_indices]
    
    high_risk_explanation = ebm_model.explain_local(X_high_risk, y_high_risk)
    
    print("\nTo view high-risk customer explanations, use:")
    print("  from interpret import show")
    print("  show(high_risk_explanation)")
    
    # Step 8: Save models and summary
    print("\n" + "="*60)
    print("STEP 8: SAVING MODELS AND SUMMARY")
    print("="*60)
    
    lr_path = os.path.join(OUTPUT_DIR, 'logistic_regression_model.pkl')
    dt_path = os.path.join(OUTPUT_DIR, 'decision_tree_model.pkl')
    
    joblib.dump(lr_model, lr_path)
    joblib.dump(dt_model, dt_path)
    
    print(f"Logistic Regression model saved to: {lr_path}")
    print(f"Decision Tree model saved to: {dt_path}")
    
    # Create summary report
    summary_path = os.path.join(OUTPUT_DIR, 'explanation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EXPLANATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write(f"  Total samples: {len(X_train) + len(X_test)}\n")
        f.write(f"  Training samples: {len(X_train)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Overall churn rate: {y_test.mean():.2%}\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"  EBM Model - Test Accuracy: N/A (see full report)\n")
        f.write(f"  Logistic Regression - Test Accuracy: {lr_score:.4f}\n")
        f.write(f"  Decision Tree - Test Accuracy: {dt_score:.4f}\n\n")
        
        f.write("TOP 10 MOST IMPORTANT FEATURES:\n")
        for idx, row in importance_df.head(10).iterrows():
            f.write(f"  {idx+1:2d}. {row['Feature']:40s} {row['Importance']:>10.4f}\n")
        
        f.write("\nHIGH-RISK CUSTOMERS:\n")
        f.write(f"  Identified {top_n} highest-risk customers\n")
        f.write(f"  Average churn probability: {high_risk_df['Churn_Probability'].mean():.2%}\n")
        f.write(f"  Prediction accuracy on high-risk: {high_risk_df['Correct'].mean():.2%}\n")
        
        f.write("\n\nHOW TO USE EXPLANATIONS:\n")
        f.write("="*60 + "\n")
        f.write("1. In a Jupyter notebook or IPython, run:\n\n")
        f.write("   import joblib\n")
        f.write("   from interpret import show\n")
        f.write("   ebm = joblib.load('ebm_results_multi_gpu/ebm_model.pkl')\n")
        f.write("   ebm_global = ebm.explain_global()\n")
        f.write("   show(ebm_global)\n\n")
        f.write("2. For local explanations (individual predictions):\n\n")
        f.write("   ebm_local = ebm.explain_local(X_test, y_test)\n")
        f.write("   show(ebm_local)\n\n")
        f.write("3. For model comparison:\n\n")
        f.write("   show([ebm_global, lr_global, dt_global])\n")
    
    print(f"Explanation summary saved to: {summary_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("EXPLANATION GENERATION COMPLETE!")
    print("="*60)
    print(f"\nAll results saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print(f"  1. feature_importance.csv - Feature importance rankings")
    print(f"  2. high_risk_customers.csv - Top {top_n} high-risk customers")
    print(f"  3. logistic_regression_model.pkl - Trained LR model")
    print(f"  4. decision_tree_model.pkl - Trained DT model")
    print(f"  5. explanation_summary.txt - Summary report")
    
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATION INSTRUCTIONS:")
    print("="*60)
    print("\nTo view interactive explanations, use show() in Python/Jupyter:")
    print("\n  # Global explanation (overall model behavior)")
    print("  show(ebm_global)")
    print("\n  # Local explanation (individual predictions)")
    print("  show(ebm_local)")
    print("\n  # Model comparison (EBM vs LR vs DT)")
    print("  show([ebm_global, lr_global, dt_global])")
    print("\n  # High-risk customer explanations")
    print("  show(high_risk_explanation)")
    
    return {
        'ebm_global': ebm_global,
        'ebm_local': ebm_local,
        'lr_global': lr_global,
        'dt_global': dt_global,
        'high_risk_explanation': high_risk_explanation,
        'high_risk_df': high_risk_df,
        'importance_df': importance_df
    }


if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*60)
    print("VISUALIZATION AVAILABLE IN PYTHON SESSION")
    print("="*60)
    print("\nYou can now use these variables:")
    print("  - ebm_global: Global EBM explanation")
    print("  - ebm_local: Local EBM explanations (100 samples)")
    print("  - lr_global: Logistic Regression global explanation")
    print("  - dt_global: Decision Tree global explanation")
    print("  - high_risk_explanation: Top 10 high-risk customer explanations")
    print("\nExample: show(results['ebm_global'])")
