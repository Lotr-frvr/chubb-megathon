"""
Quick Demo: How to use InterpretML's show() function
for EBM model explanations

This script demonstrates the requested functionality:
1. ebm_global = ebm.explain_global()
2. show(ebm_global)  
3. ebm_local = ebm.explain_local(X_test, y_test)
4. show(ebm_local)
5. show([logistic_regression_global, decision_tree_global])
"""

import joblib
import pandas as pd
from interpret import show

def demo_show_function():
    """
    Demonstrates how to use the show() function with EBM models.
    
    NOTE: show() function works best in Jupyter notebooks or IPython.
    In a regular Python script, it will try to open a web browser.
    """
    
    print("="*60)
    print("INTERPRETABLE ML - show() FUNCTION DEMO")
    print("="*60)
    
    # Load the trained EBM model
    print("\n1. Loading trained EBM model...")
    ebm_model = joblib.load('ebm_results_multi_gpu/ebm_model.pkl')
    print("   ✓ Model loaded")
    
    # Load test predictions to get X_test and y_test
    print("\n2. Loading test data...")
    predictions_df = pd.read_csv('ebm_results_multi_gpu/test_predictions.csv')
    
    # For a quick demo, we need to recreate X_test and y_test
    # Since we have the full dataset, let's load a small sample
    from train_multi_gpu import load_and_preprocess_data, split_data
    
    print("   Loading small sample for demonstration...")
    df = load_and_preprocess_data('autoinsurance_churn_cleaned.csv', sample_fraction=0.01)
    X_train, X_test, y_train, y_test = split_data(df, target_column='Churn', test_size=0.2, random_state=42)
    print(f"   ✓ Sample data loaded: {len(X_test)} test samples")
    
    # ========================================================
    # EXAMPLE 1: Global Explanation
    # ========================================================
    print("\n" + "="*60)
    print("EXAMPLE 1: GLOBAL EXPLANATION")
    print("="*60)
    print("\nCode:")
    print("  ebm_global = ebm.explain_global()")
    print("  show(ebm_global)")
    print("\nGenerating...")
    
    ebm_global = ebm_model.explain_global()
    print("✓ Global explanation created!")
    print("\nThis shows:")
    print("  - Overall feature importance")
    print("  - Shape functions for each feature")
    print("  - Feature interactions")
    
    # Try to show it
    try:
        print("\nAttempting to display...")
        show(ebm_global)
        print("✓ Visualization opened in browser!")
    except Exception as e:
        print(f"Note: show() works best in Jupyter. Error: {e}")
        print("In Jupyter notebook, this would display an interactive dashboard.")
    
    # ========================================================
    # EXAMPLE 2: Local Explanation
    # ========================================================
    print("\n" + "="*60)
    print("EXAMPLE 2: LOCAL EXPLANATION")
    print("="*60)
    print("\nCode:")
    print("  ebm_local = ebm.explain_local(X_test, y_test)")
    print("  show(ebm_local)")
    print("\nGenerating for", min(50, len(X_test)), "samples...")
    
    # Use a smaller sample for demo
    sample_size = min(50, len(X_test))
    X_sample = X_test.head(sample_size)
    y_sample = y_test.head(sample_size)
    
    ebm_local = ebm_model.explain_local(X_sample, y_sample)
    print("✓ Local explanation created!")
    print("\nThis shows:")
    print("  - Individual prediction explanations")
    print("  - Feature contributions for each prediction")
    print("  - Why the model predicted churn/no-churn")
    
    try:
        print("\nAttempting to display...")
        show(ebm_local)
        print("✓ Visualization opened in browser!")
    except Exception as e:
        print(f"Note: show() works best in Jupyter. Error: {e}")
        print("In Jupyter notebook, this would display interactive instance explanations.")
    
    # ========================================================
    # EXAMPLE 3: Model Comparison
    # ========================================================
    print("\n" + "="*60)
    print("EXAMPLE 3: MODEL COMPARISON")
    print("="*60)
    print("\nCode:")
    print("  logistic_regression_global = lr.explain_global()")
    print("  decision_tree_global = dt.explain_global()")
    print("  show([ebm_global, logistic_regression_global, decision_tree_global])")
    
    # Train quick comparison models
    from interpret.glassbox import LogisticRegression, ClassificationTree
    
    print("\nTraining Logistic Regression...")
    # Encode categorical features
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    for col in X_train_enc.select_dtypes(include=['category', 'object']).columns:
        X_train_enc[col] = X_train_enc[col].astype('category').cat.codes
        X_test_enc[col] = X_test_enc[col].astype('category').cat.codes
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_enc, y_train)
    logistic_regression_global = lr.explain_global()
    print("✓ Logistic Regression trained and explained")
    
    print("\nTraining Decision Tree...")
    dt = ClassificationTree(random_state=42, max_depth=5)
    dt.fit(X_train_enc, y_train)
    decision_tree_global = dt.explain_global()
    print("✓ Decision Tree trained and explained")
    
    print("\nThis allows side-by-side comparison of:")
    print("  - Feature importance across models")
    print("  - Model behaviors and patterns")
    print("  - Strengths and weaknesses")
    
    try:
        print("\nAttempting to display comparison...")
        show([ebm_global, logistic_regression_global, decision_tree_global])
        print("✓ Comparison visualization opened in browser!")
    except Exception as e:
        print(f"Note: show() works best in Jupyter. Error: {e}")
        print("In Jupyter notebook, this would display side-by-side model comparison.")
    
    # ========================================================
    # SUMMARY
    # ========================================================
    print("\n" + "="*60)
    print("SUMMARY: HOW TO USE IN JUPYTER NOTEBOOK")
    print("="*60)
    
    jupyter_code = '''
# In Jupyter Notebook:
import joblib
from interpret import show

# Load model
ebm = joblib.load('ebm_results_multi_gpu/ebm_model.pkl')

# 1. Global explanation
ebm_global = ebm.explain_global()
show(ebm_global)

# 2. Local explanation  
ebm_local = ebm.explain_local(X_test, y_test)
show(ebm_local)

# 3. Model comparison
from interpret.glassbox import LogisticRegression, ClassificationTree

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_global = lr.explain_global()

dt = ClassificationTree()
dt.fit(X_train, y_train)
dt_global = dt.explain_global()

show([ebm_global, lr_global, dt_global])
'''
    
    print(jupyter_code)
    
    print("\n" + "="*60)
    print("INTERACTIVE DASHBOARD FEATURES")
    print("="*60)
    print("\nWhen using show() in Jupyter, you get:")
    print("  ✓ Interactive plots you can hover over")
    print("  ✓ Dropdown menus to select features")
    print("  ✓ Sliders to navigate between instances")
    print("  ✓ Tabs to compare different models")
    print("  ✓ Export options for charts")
    print("\nThis is the power of InterpretML's explainability!")
    
    return ebm_global, ebm_local, logistic_regression_global, decision_tree_global


if __name__ == "__main__":
    try:
        results = demo_show_function()
        print("\n✓ Demo completed successfully!")
        print("\nExplanation objects are available in 'results' tuple:")
        print("  results[0] = ebm_global")
        print("  results[1] = ebm_local")
        print("  results[2] = logistic_regression_global")
        print("  results[3] = decision_tree_global")
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
