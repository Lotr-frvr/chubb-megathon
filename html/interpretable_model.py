import joblib
from interpret.glassbox import ExplainableBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Part 2 - Database
df = pd.read_csv("final_augmented_data.csv")

# Part 3 - Data Cleaning
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()
df_cleaned.drop(columns=['individual_id', 'address_id', 'cust_orig_date',
                        'date_of_birth', 'acct_suspd_date'], inplace=True)
df.drop(columns=['individual_id', 'address_id', 'cust_orig_date',
                'date_of_birth', 'acct_suspd_date'], inplace=True)

# Part 4 - Feature Engineering
cat_cols = df.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Part 5 - Features and Target
feat_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income',
             'has_children', 'length_of_residence', 'marital_status',
             'home_owner', 'college_degree', 'good_credit']
feat_cols.extend(['chubb_sentiment', 'competitor_sentiment', 'chubb_stock_return',
                 'competitor_stock_return', 'market_index_return', 
                 'customer_engagement'])
X = df[feat_cols]
y = df['Churn']

# Part 6 - Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# Part 7 - Balance and Scale
train_df = pd.concat([X_train, y_train], axis=1)
majority = train_df[train_df['Churn'] == 0]
minority = train_df[train_df['Churn'] == 1]
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
balanced_train_df = pd.concat([majority, minority_upsampled])
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42)

X_train_bal = balanced_train_df.drop('Churn', axis=1)
y_train_bal = balanced_train_df['Churn']
X_train_bal = X_train_bal.fillna(X_train_bal.median())
X_test = X_test.fillna(X_test.median())

scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Load the model
ebm = joblib.load("saved_models/ebm_explainable_boosting.pkl")

# IMPORTANT: Convert scaled data back to DataFrame with proper feature names
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feat_cols, index=X_test.index)

# Global explanation (overall feature importance)
print("=" * 60)
print("GLOBAL FEATURE IMPORTANCE")
print("=" * 60)
ebm_global = ebm.explain_global()
print(ebm_global.data())


# Explain specific prediction example
print("\n" + "=" * 60)
print("SINGLE PREDICTION EXPLANATION")
print("=" * 60)
sample_idx = 12345
single_sample = X_test_scaled_df.iloc[[sample_idx]]
single_label = y_test.iloc[sample_idx]
prediction = ebm.predict(single_sample)[0]
prediction_proba = ebm.predict_proba(single_sample)[0]

print(f"Sample Index: {sample_idx}")
print(f"True Label: {single_label}")
print(f"Predicted Label: {prediction}")
print(f"Prediction Probability (Not Churn, Churn): {prediction_proba}")
print("\nFeature Contributions:")

ebm_single = ebm.explain_local(single_sample, [single_label])
local_data = ebm_single.data(0)
if 'scores' in local_data:
    for feature, score in zip(feat_cols, local_data['scores']):
        print(f"  {feature}: {score:.4f}")


print(f"\nModel Accuracy on Test Set: {ebm.score(X_test_scaled, y_test):.4f}")


# plot the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
y_pred = ebm.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred, labels=ebm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=ebm.classes_)
disp.plot(cmap=plt.cm.Blues)
# save the plot
plt.savefig("ebm_confusion_matrix.png")