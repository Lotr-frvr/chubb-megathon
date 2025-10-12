# Churn Prediction Web Application

## Overview
This application takes customer data from a web form, runs real-time scrapers to gather market data, and uses machine learning to predict customer churn probability.

## Features
- 📝 Interactive web form for customer data input
- 🤖 Real-time data scraping:
  - Chubb Insurance news sentiment analysis
  - Competitor news sentiment analysis
  - Chubb stock performance
  - Competitor stock performance
- 🧠 Machine learning inference using Explainable Boosting Machine (EBM)
- 📊 Detailed prediction results with feature importance

## Quick Start

### 1. Install Dependencies
```bash
pip install flask flask-cors scikit-learn pandas numpy joblib beautifulsoup4 textblob requests interpret lxml
```

### 2. Start the Backend Server
```bash
sh start_server.sh
```

The server will start on `http://localhost:5000`

### 3. Open the Web Interface
Open `nig.html` in your web browser:
```bash
open nig.html
```

Or simply double-click the `nig.html` file.

## Usage

1. Fill out the customer information form:
   - Current Annual Amount
   - Days of Tenure
   - Age
   - Income
   - Personal details (children, residence, marital status, etc.)

2. Click "Submit Prediction"

3. The system will:
   - Send your data to the backend server
   - Run all scraper agents to collect real-time market data
   - Generate random values for remaining features
   - Perform inference using the trained EBM model
   - Display comprehensive prediction results

4. View Results:
   - Churn prediction (Will Churn / Will Not Churn)
   - Probability scores
   - Confidence level
   - Real-time scraped data values
   - Top feature contributions to the prediction

## Architecture

```
┌─────────────────┐
│   nig.html      │  Web Interface
│  (Frontend)     │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────────────┐
│ backend_prediction.py   │  Flask Server
│  • Receives form data   │
│  • Runs scrapers        │
│  • Generates features   │
│  • Performs inference   │
└────────┬────────────────┘
         │
         ├──► chubb_news_agent.py (News sentiment)
         ├──► compeititor_news.py (Competitor news)
         ├──► chubs_stock_agent.py (Stock data)
         ├──► compeittor_stock.py (Competitor stocks)
         │
         ├──► saved_models/ebm_explainable_boosting.pkl
         └──► saved_models/scaler.pkl
```

## API Endpoints

### POST /predict
Accepts customer data and returns churn prediction.

**Request Body:**
```json
{
  "curr_ann_amt": "5000",
  "days_tenure": "365",
  "age_in_years": "45",
  "income": "75000",
  "has_children": "1",
  "length_of_residence": "5",
  "marital_status": "1",
  "home_owner": "1",
  "college_degree": "1",
  "good_credit": "1"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 0,
  "prediction_label": "WILL NOT CHURN",
  "churn_probability": 0.23,
  "no_churn_probability": 0.77,
  "confidence": 0.77,
  "scraped_data": {
    "chubb_sentiment": 0.156,
    "competitor_sentiment": 0.089,
    "chubb_stock_return": 0.023,
    "competitor_stock_return": 0.015
  },
  "feature_contributions": {
    "income": 0.0234,
    "days_tenure": -0.0156,
    ...
  }
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

## Troubleshooting

### Server won't start
- Ensure Python 3 is installed: `python3 --version`
- Install dependencies: `pip3 install -r requirements.txt`
- Check that port 5000 is not in use

### CORS errors in browser
- Make sure the Flask server is running
- The server includes CORS headers for all origins

### Scrapers returning random data
- This is expected if scrapers fail to fetch real data
- Check internet connection
- Some data sources may have rate limits

### Model file not found
- Ensure `saved_models/ebm_explainable_boosting.pkl` exists
- Ensure `saved_models/scaler.pkl` exists
- Run the training script first if models don't exist

## Files

- `nig.html` - Web interface with form
- `backend_prediction.py` - Flask server with prediction logic
- `start_server.sh` - Server startup script
- `chubb_news_agent.py` - Chubb news scraper
- `compeititor_news.py` - Competitor news scraper
- `chubs_stock_agent.py` - Chubb stock scraper
- `compeittor_stock.py` - Competitor stock scraper
- `interpretable_model.py` - Model training script
- `saved_models/` - Trained model files

## Notes

- The scrapers run every time a prediction is requested (real-time data)
- If scrapers fail, the system uses random values within reasonable ranges
- The EBM model provides interpretable predictions with feature importance
- All results are displayed directly in the browser

## Support

For issues or questions, check the console output for detailed error messages.
