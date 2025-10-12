"""
Flask Backend for Churn Prediction
Takes input from webpage, runs scrapers, generates random values, and performs inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

# Import the scraper agents
from chubb_news_agent import ChubbNewsAgent
from chubs_stock_agent import ChubbStockAgent
from compeititor_news import InsuranceSentimentAgent
from compeittor_stock import CompetitorStockAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model and scaler
print("Loading model and scaler...")
ebm_model = joblib.load("saved_models/ebm_explainable_boosting.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
print("✅ Model and scaler loaded successfully!")

# Feature columns expected by the model
FEATURE_COLS = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income',
                'has_children', 'length_of_residence', 'marital_status',
                'home_owner', 'college_degree', 'good_credit',
                'chubb_sentiment', 'competitor_sentiment', 'chubb_stock_return',
                'competitor_stock_return', 'market_index_return', 
                'customer_engagement']


def run_scrapers():
    """
    Run all scraper agents and extract relevant metrics
    Returns a dict with sentiment and stock return values
    """
    print("\n" + "="*70)
    print("🤖 Running Scraper Agents...")
    print("="*70)
    
    scraped_data = {}
    
    try:
        # 1. Chubb News Sentiment
        print("\n📰 Fetching Chubb News Sentiment...")
        chubb_news = ChubbNewsAgent()
        chubb_df = chubb_news.collect_news(days_back=10)
        if not chubb_df.empty:
            chubb_df = chubb_news.apply_sentiment_analysis(chubb_df)
            scraped_data['chubb_sentiment'] = chubb_df['polarity'].mean()
            print(f"✅ Chubb Sentiment: {scraped_data['chubb_sentiment']:.3f}")
        else:
            scraped_data['chubb_sentiment'] = np.random.uniform(-0.2, 0.3)
            print(f"⚠️  Using random Chubb sentiment: {scraped_data['chubb_sentiment']:.3f}")
    except Exception as e:
        print(f"❌ Error in Chubb news scraper: {e}")
        scraped_data['chubb_sentiment'] = np.random.uniform(-0.2, 0.3)
    
    try:
        # 2. Competitor News Sentiment
        print("\n📰 Fetching Competitor News Sentiment...")
        comp_news = InsuranceSentimentAgent()
        comp_df = comp_news.collect_and_analyze(days_back=10)
        if not comp_df.empty:
            scraped_data['competitor_sentiment'] = comp_df['polarity'].mean()
            print(f"✅ Competitor Sentiment: {scraped_data['competitor_sentiment']:.3f}")
        else:
            scraped_data['competitor_sentiment'] = np.random.uniform(-0.2, 0.3)
            print(f"⚠️  Using random Competitor sentiment: {scraped_data['competitor_sentiment']:.3f}")
    except Exception as e:
        print(f"❌ Error in Competitor news scraper: {e}")
        scraped_data['competitor_sentiment'] = np.random.uniform(-0.2, 0.3)
    
    try:
        # 3. Chubb Stock Return
        print("\n📈 Fetching Chubb Stock Data...")
        chubb_stock = ChubbStockAgent()
        chubb_stock_df = chubb_stock.collect_stock_data(days_back=10)
        if not chubb_stock_df.empty:
            analysis = chubb_stock.analyze_trends(chubb_stock_df)
            scraped_data['chubb_stock_return'] = analysis['period_change_pct'] / 100
            print(f"✅ Chubb Stock Return: {scraped_data['chubb_stock_return']:.3f}")
        else:
            scraped_data['chubb_stock_return'] = np.random.uniform(-0.05, 0.05)
            print(f"⚠️  Using random Chubb stock return: {scraped_data['chubb_stock_return']:.3f}")
    except Exception as e:
        print(f"❌ Error in Chubb stock scraper: {e}")
        scraped_data['chubb_stock_return'] = np.random.uniform(-0.05, 0.05)
    
    try:
        # 4. Competitor Stock Return
        print("\n📈 Fetching Competitor Stock Data...")
        comp_stock = CompetitorStockAgent()
        comp_stock.analyze_competitors(days_back=10)
        if comp_stock.all_results:
            # Average the period change percentage across all competitors
            returns = [data['period_change_pct'] for data in comp_stock.all_results.values()]
            scraped_data['competitor_stock_return'] = np.mean(returns) / 100
            print(f"✅ Competitor Stock Return: {scraped_data['competitor_stock_return']:.3f}")
        else:
            scraped_data['competitor_stock_return'] = np.random.uniform(-0.05, 0.05)
            print(f"⚠️  Using random Competitor stock return: {scraped_data['competitor_stock_return']:.3f}")
    except Exception as e:
        print(f"❌ Error in Competitor stock scraper: {e}")
        scraped_data['competitor_stock_return'] = np.random.uniform(-0.05, 0.05)
    
    print("\n" + "="*70)
    print("✅ Scraping Complete!")
    print("="*70)
    
    return scraped_data


def generate_random_features():
    """
    Generate random values for features not scraped
    """
    return {
        'market_index_return': np.random.uniform(-0.03, 0.04),
        # 'gdp_growth': np.random.uniform(0.01, 0.03),
        'customer_engagement': np.random.uniform(0, 1)
    }


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Receives form data, runs scrapers, performs inference
    """
    try:
        print("\n" + "🎯"*35)
        print("   NEW PREDICTION REQUEST")
        print("🎯"*35)
        
        # Get form data
        form_data = request.json
        print("\n📝 Received Form Data:")
        print(form_data)
        
        # Extract and convert form values
        user_features = {
            'curr_ann_amt': float(form_data.get('curr_ann_amt', 0)),
            'days_tenure': int(form_data.get('days_tenure', 0)),
            'age_in_years': int(form_data.get('age_in_years', 0)),
            'income': float(form_data.get('income', 0)),
            'has_children': int(form_data.get('has_children', 0)),
            'length_of_residence': int(form_data.get('length_of_residence', 0)),
            'marital_status': int(form_data.get('marital_status', 0)),
            'home_owner': int(form_data.get('home_owner', 0)),
            'college_degree': int(form_data.get('college_degree', 0)),
            'good_credit': int(form_data.get('good_credit', 0))
        }
        
        # Run scrapers to get real-time data
        scraped_features = run_scrapers()
        
        # Generate random features
        random_features = generate_random_features()
        
        # Combine all features
        all_features = {**user_features, **scraped_features, **random_features}
        
        print("\n📊 Combined Feature Vector:")
        for key, value in all_features.items():
            print(f"   {key}: {value}")
        
        # Create DataFrame with correct column order
        X_input = pd.DataFrame([all_features])[FEATURE_COLS]
        
        # Scale the input
        X_scaled = scaler.transform(X_input)
        
        # Perform prediction
        prediction = ebm_model.predict(X_scaled)[0]
        prediction_proba = ebm_model.predict_proba(X_scaled)[0]
        
        # Get feature importance for this prediction
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
        ebm_local = ebm_model.explain_local(X_scaled_df, [prediction])
        local_data = ebm_local.data(0)
        
        feature_contributions = {}
        if 'scores' in local_data:
            feature_contributions = {
                feature: float(score) 
                for feature, score in zip(FEATURE_COLS, local_data['scores'])
            }
            # Sort by absolute contribution
            feature_contributions = dict(
                sorted(feature_contributions.items(), 
                       key=lambda x: abs(x[1]), 
                       reverse=True)
            )
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'WILL CHURN' if prediction == 1 else 'WILL NOT CHURN',
            'churn_probability': float(prediction_proba[1]),
            'no_churn_probability': float(prediction_proba[0]),
            'confidence': float(max(prediction_proba)),
            'input_features': all_features,
            'feature_contributions': feature_contributions,
            'scraped_data': scraped_features,
            'random_data': random_features
        }
        
        print("\n" + "="*70)
        print("🎯 PREDICTION RESULT")
        print("="*70)
        print(f"Prediction: {response['prediction_label']}")
        print(f"Churn Probability: {response['churn_probability']:.2%}")
        print(f"Confidence: {response['confidence']:.2%}")
        print("="*70)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ebm_model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/impute', methods=['POST'])
def impute_missing_data():
    """
    Impute missing values using KNN-based imputation from kn_inference.py
    """
    try:
        print("\n" + "="*70)
        print("🔮 KNN IMPUTATION REQUEST")
        print("="*70)
        
        data = request.json
        print(f"Received data: {data}")
        
        # Import the ChurnPredictor class
        from kn_inference import ChurnPredictor
        
        # Initialize predictor with model and data
        predictor = ChurnPredictor(
            model_path="./saved_models/ebm_explainable_boosting.pkl",
            data_path="./final_augmented_data.csv"
        )
        
        # Convert input data to proper format (handle None/empty strings)
        input_sample = {}
        for key, value in data.items():
            if value is None or value == '' or value == 'null':
                input_sample[key] = None
            else:
                # Try to convert to appropriate type
                try:
                    # Check if it's a numeric field
                    if key in ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income',
                              'has_children', 'length_of_residence', 'home_owner', 
                              'college_degree', 'good_credit', 'chubb_sentiment',
                              'competitor_sentiment', 'chubb_stock_return',
                              'competitor_stock_return', 'market_index_return',
                              'customer_engagement']:
                        input_sample[key] = float(value) if value else None
                    else:
                        input_sample[key] = value
                except (ValueError, TypeError):
                    input_sample[key] = value
        
        print(f"Processed input: {input_sample}")
        
        # Get prediction with imputation (this also fills missing values)
        result = predictor.predict_with_imputation(input_sample)
        
        print(f"Imputation result: {result}")
        
        # The predict_with_imputation returns prediction, but we need the filled data
        # We need to modify the approach - let's get the imputed values directly
        
        # Encode categorical features
        encoded_input = predictor._encode_categorical_features(input_sample)
        
        # Impute missing values and get the filled dataframe
        imputed_df = predictor._impute_missing_values(encoded_input)
        
        # Convert back to dictionary
        imputed_data = {}
        for col in predictor.feature_columns:
            if col in imputed_df.columns:
                imputed_data[col] = float(imputed_df[col].iloc[0])
            elif col in input_sample and input_sample[col] is not None:
                imputed_data[col] = input_sample[col]
        
        print("\n" + "="*70)
        print("✅ IMPUTATION SUCCESSFUL")
        print("="*70)
        print(f"Imputed {sum(1 for v in input_sample.values() if v is None)} missing values")
        print("="*70)
        
        return jsonify({
            'success': True,
            'imputed_data': imputed_data,
            'message': 'Missing values imputed successfully using KNN'
        })
    
    except Exception as e:
        print(f"\n❌ IMPUTATION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to impute missing values'
        }), 500


if __name__ == '__main__':
    print("\n" + "🚀"*35)
    print("   CHURN PREDICTION SERVER STARTING")
    print("🚀"*35 + "\n")
    print("Server running on http://localhost:5001")
    print("Endpoints:")
    print("  - POST /predict : Make a churn prediction")
    print("  - POST /impute  : Impute missing values using KNN")
    print("  - GET  /health  : Check server health")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
