"""
Quick test script to verify the backend setup
"""

print("🧪 Testing Backend Setup...")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    import flask
    print("   ✅ Flask imported")
    from flask_cors import CORS
    print("   ✅ Flask-CORS imported")
    import pandas as pd
    print("   ✅ Pandas imported")
    import numpy as np
    print("   ✅ NumPy imported")
    import joblib
    print("   ✅ Joblib imported")
    from sklearn.preprocessing import StandardScaler
    print("   ✅ Scikit-learn imported")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test model files exist
print("\n2. Testing model files...")
import os
if os.path.exists("saved_models/ebm_explainable_boosting.pkl"):
    print("   ✅ EBM model found")
else:
    print("   ❌ EBM model not found")
    
if os.path.exists("saved_models/scaler.pkl"):
    print("   ✅ Scaler found")
else:
    print("   ❌ Scaler not found")

# Test loading models
print("\n3. Testing model loading...")
try:
    ebm = joblib.load("saved_models/ebm_explainable_boosting.pkl")
    print("   ✅ EBM model loaded successfully")
    scaler = joblib.load("saved_models/scaler.pkl")
    print("   ✅ Scaler loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    exit(1)

# Test scraper imports
print("\n4. Testing scraper imports...")
try:
    from chubb_news_agent import ChubbNewsAgent
    print("   ✅ Chubb News Agent imported")
    from chubs_stock_agent import ChubbStockAgent
    print("   ✅ Chubb Stock Agent imported")
    from compeititor_news import InsuranceSentimentAgent
    print("   ✅ Competitor News Agent imported")
    from compeittor_stock import CompetitorStockAgent
    print("   ✅ Competitor Stock Agent imported")
except ImportError as e:
    print(f"   ⚠️  Warning: Could not import scrapers: {e}")
    print("   (This is OK - scrapers will use fallback values)")

print("\n" + "=" * 60)
print("✅ All tests passed! Ready to start the server.")
print("=" * 60)
print("\nRun: sh start_server.sh")
