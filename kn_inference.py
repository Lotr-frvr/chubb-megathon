import joblib
from interpret.glassbox import ExplainableBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Optional, Union

class ChurnPredictor:
    """A class to handle churn prediction with missing value imputation."""
    
    def __init__(self, model_path: str, data_path: str):
        """Initialize the predictor by loading data and model."""
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.X_test_scaled_df = None
        self.X_train_bal = None
        
        self._load_data(data_path)
        self._load_model(model_path)
    
    def _load_data(self, data_path: str):
        """Load and preprocess the data."""
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Data cleaning
        columns_to_drop = ['individual_id', 'address_id', 'cust_orig_date', 
                          'date_of_birth', 'acct_suspd_date']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        df = df.dropna().drop_duplicates()
        
        # Feature engineering
        self.categorical_columns = df.select_dtypes(include='object').columns.tolist()
        for col in self.categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Define features and target
        self.feature_columns = [
            'curr_ann_amt', 'days_tenure', 'age_in_years', 'income',
            'has_children', 'length_of_residence', 'marital_status',
            'home_owner', 'college_degree', 'good_credit',
            'chubb_sentiment', 'competitor_sentiment', 'chubb_stock_return',
            'competitor_stock_return', 'market_index_return', 'gdp_growth',
            'customer_engagement'
        ]
        
        X = df[self.feature_columns]
        y = df['Churn']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Balance and scale
        train_df = pd.concat([X_train, y_train], axis=1)
        majority = train_df[train_df['Churn'] == 0]
        minority = train_df[train_df['Churn'] == 1]
        minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
        balanced_train_df = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
        
        self.X_train_bal = balanced_train_df.drop('Churn', axis=1).fillna(
            balanced_train_df.drop('Churn', axis=1).median()
        )
        y_train_bal = balanced_train_df['Churn']
        X_test_filled = X_test.fillna(X_test.median())
        
        self.scaler = StandardScaler()
        X_train_bal_scaled = self.scaler.fit_transform(self.X_train_bal)
        X_test_scaled = self.scaler.transform(X_test_filled)
        
        self.X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        self.X_test = X_test_filled
        self.y_test = y_test
        
    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def get_global_explanation(self):
        """Get global feature importance."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.explain_global()
    
    def explain_single_prediction(self, sample_idx: int):
        """Explain a single prediction from the test set."""
        if self.X_test_scaled_df is None:
            raise ValueError("Test data not available")
        
        single_sample = self.X_test_scaled_df.iloc[[sample_idx]]
        true_label = self.y_test.iloc[sample_idx]
        
        prediction = self.model.predict(single_sample)[0]
        prediction_proba = self.model.predict_proba(single_sample)[0]
        
        explanation = self.model.explain_local(single_sample, [true_label])
        local_data = explanation.data(0)
        
        feature_contributions = {}
        if 'scores' in local_data:
            for feature, score in zip(self.feature_columns, local_data['scores']):
                feature_contributions[feature] = score
        
        return {
            'sample_index': sample_idx,
            'true_label': true_label,
            'predicted_label': prediction,
            'prediction_probability': prediction_proba,
            'feature_contributions': feature_contributions
        }
    
    def _encode_categorical_features(self, input_dict: Dict) -> pd.Series:
        """Encode categorical features in the input dictionary."""
        input_series = pd.Series(input_dict).reindex(self.feature_columns)
        
        for col in self.categorical_columns:
            if col in input_series and not pd.isna(input_series[col]):
                value = input_series[col]
                if isinstance(value, str):
                    if value in self.label_encoders[col].classes_:
                        input_series[col] = self.label_encoders[col].transform([value])[0]
                    else:
                        print(f"Warning: Unseen category '{value}' in '{col}'. Treating as missing.")
                        input_series[col] = np.nan
        
        return input_series.replace({None: np.nan})
    
    def _impute_missing_values(self, input_series: pd.Series) -> pd.DataFrame:
        """Impute missing values using nearest neighbors."""
        missing_features = input_series[input_series.isna()].index.tolist()
        present_features = input_series[input_series.notna()].index.tolist()
        
        if not present_features:
            # All features missing - use training median
            print("All features missing. Using training data median.")
            median_input = self.X_train_bal.median().to_frame().T
            return pd.DataFrame(
                self.scaler.transform(median_input), 
                columns=self.feature_columns
            )
        
        if not missing_features:
            # No missing features - just scale
            input_df = pd.DataFrame([input_series], columns=self.feature_columns)
            return pd.DataFrame(
                self.scaler.transform(input_df), 
                columns=self.feature_columns
            )
        
        # Scale present features temporarily
        temp_input = input_series.fillna(0)
        temp_scaled = self.scaler.transform(
            pd.DataFrame([temp_input], columns=self.feature_columns)
        )[0]
        input_scaled = pd.Series(temp_scaled, index=self.feature_columns)
        input_scaled[missing_features] = np.nan
        
        # Find neighbors for imputation
        neighbors_pool = self.X_test_scaled_df.dropna(
            subset=missing_features + present_features
        )
        
        if neighbors_pool.empty:
            # Fallback to test set median
            print("No suitable neighbors found. Using test set median.")
            for feature in missing_features:
                input_scaled[feature] = self.X_test_scaled_df[feature].median()
        else:
            # Find nearest neighbors
            n_neighbors = min(1000, len(neighbors_pool))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn.fit(neighbors_pool[present_features])
            
            distances, indices = nn.kneighbors(
                input_scaled[present_features].to_frame().T
            )
            closest_neighbors = neighbors_pool.iloc[indices[0]]
            
            # Impute using neighbor medians
            for feature in missing_features:
                input_scaled[feature] = closest_neighbors[feature].median()
        
        return input_scaled.to_frame().T
    
    def predict_with_imputation(self, input_sample: Dict) -> Dict:
        """Predict churn with automatic missing value imputation."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Encode categorical features
        encoded_input = self._encode_categorical_features(input_sample)
        
        # Impute missing values
        final_input = self._impute_missing_values(encoded_input)
        
        # Make prediction
        prediction = self.model.predict(final_input)[0]
        prediction_proba = self.model.predict_proba(final_input)[0]
        
        # Get explanation
        explanation = self.model.explain_local(final_input, [prediction])
        local_data = explanation.data(0)
        
        feature_contributions = {}
        if 'scores' in local_data:
            for feature, score in zip(self.feature_columns, local_data['scores']):
                feature_contributions[feature] = score
        
        return {
            'predicted_label': int(prediction),
            'prediction_probabilities': {
                'not_churn': float(prediction_proba[0]),
                'churn': float(prediction_proba[1])
            },
            'feature_contributions': feature_contributions
        }
    
    def evaluate_model(self):
        """Evaluate model performance and plot confusion matrix."""
        y_pred = self.model.predict(self.X_test_scaled_df)
        accuracy = self.model.score(self.X_test_scaled_df, self.y_test)
        
        cm = confusion_matrix(self.y_test, y_pred, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        
        plt.figure(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
        plt.tight_layout()
        plt.savefig("ebm_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy

def main():
    """Main function to demonstrate the ChurnPredictor class."""
    # Initialize predictor
    try:
        predictor = ChurnPredictor(
            model_path="./lockin/saved_models/ebm_explainable_boosting.pkl",
            data_path="./lockin/final_augmented_data.csv"
        )
    except FileNotFoundError as e:
        print(f"Initialization error: {e}")
        return
    
    # Global explanation
    print("=" * 60)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 60)
    global_explanation = predictor.get_global_explanation()
    print(global_explanation.data())
    
    # Single prediction explanation
    print("\n" + "=" * 60)
    print("SINGLE PREDICTION EXPLANATION")
    print("=" * 60)
    try:
        single_result = predictor.explain_single_prediction(12345)
        print(f"Sample Index: {single_result['sample_index']}")
        print(f"True Label: {single_result['true_label']}")
        print(f"Predicted Label: {single_result['predicted_label']}")
        print(f"Prediction Probability: {single_result['prediction_probability']}")
        print("\nFeature Contributions:")
        for feature, score in single_result['feature_contributions'].items():
            print(f"  {feature}: {score:.4f}")
    except Exception as e:
        print(f"Error in single prediction: {e}")
    
    # Model evaluation
    print(f"\nModel Accuracy: {predictor.evaluate_model():.4f}")
    
    # Demo predictions with imputation
    demo_samples = [
        {
            'name': "Full input (no missing values)",
            'data': {
                'curr_ann_amt': 1200, 'days_tenure': 1500, 'age_in_years': 45, 'income': 75000,
                'has_children': 1, 'length_of_residence': 10, 'marital_status': 'Married',
                'home_owner': 1, 'college_degree': 1, 'good_credit': 1,
                'chubb_sentiment': 0.8, 'competitor_sentiment': 0.3, 'chubb_stock_return': 0.05,
                'competitor_stock_return': -0.02, 'market_index_return': 0.03, 'gdp_growth': 0.02,
                'customer_engagement': 0.7
            }
        },
        {
            'name': "Input with missing numerical values",
            'data': {
                'curr_ann_amt': 800, 'days_tenure': 500, 'age_in_years': None, 'income': None,
                'has_children': 0, 'length_of_residence': 3, 'marital_status': 'Single',
                'home_owner': 0, 'college_degree': 0, 'good_credit': 0,
                'chubb_sentiment': 0.2, 'competitor_sentiment': 0.9, 'chubb_stock_return': -0.03,
                'competitor_stock_return': 0.08, 'market_index_return': 0.01, 'gdp_growth': 0.01,
                'customer_engagement': 0.3
            }
        }
    ]
    
    print("\n" + "=" * 60)
    print("DEMO: PREDICTIONS WITH IMPUTATION")
    print("=" * 60)
    
    for sample in demo_samples:
        print(f"\n--- {sample['name']} ---")
        try:
            result = predictor.predict_with_imputation(sample['data'])
            print(f"Predicted Label: {result['predicted_label']}")
            print(f"Probabilities - Not Churn: {result['prediction_probabilities']['not_churn']:.4f}, "
                  f"Churn: {result['prediction_probabilities']['churn']:.4f}")
            
            # Show top 5 most influential features
            top_features = sorted(
                result['feature_contributions'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            print("Top 5 Feature Contributions:")
            for feature, score in top_features:
                print(f"  {feature}: {score:.4f}")
                
        except Exception as e:
            print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()