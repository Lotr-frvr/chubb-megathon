import joblib
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                            classification_report, recall_score, roc_curve, auc)

from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from collections import Counter

# Part 2 - Database
df = pd.read_csv("final_augmented_data.csv")

# Part 3 - Data Cleaning
# Check missing values
null_counts = df.isnull().sum()
null_columns = null_counts[null_counts > 0]
print("Columns with missing values:")
print(null_columns)

# Drop rows with missing values
df_cleaned = df.dropna()
print("\nDataset shape after removing rows with missing values:", df_cleaned.shape)

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()
print(f"Dataset shape after removing duplicates: {df_cleaned.shape}")

# Drop unnecessary columns
df_cleaned.drop(columns=['individual_id', 'address_id', 'cust_orig_date', 
                        'date_of_birth', 'acct_suspd_date'], inplace=True)

df.drop(columns=['individual_id', 'address_id', 'cust_orig_date', 
                'date_of_birth', 'acct_suspd_date'], inplace=True)

# Part 4 - Feature Engineering
# Label encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Part 5 - Training and testing division
feat_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 
             'has_children', 'length_of_residence', 'marital_status', 
             'home_owner', 'college_degree', 'good_credit','city','state']
feat_cols.extend(['chubb_sentiment','competitor_sentiment','chubb_stock_return','competitor_stock_return','market_index_return','customer_engagement'])
X = df[feat_cols]
y = df['Churn']

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# Part 6 - Model Training
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

# Part 7 - SMOTE (Upsampling)
# Combine training data
train_df = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority = train_df[train_df['Churn'] == 0]
minority = train_df[train_df['Churn'] == 1]

# Upsample minority class
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)

# Combine balanced data
balanced_train_df = pd.concat([majority, minority_upsampled])
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42)

# Separate features and target
X_train_bal = balanced_train_df.drop('Churn', axis=1)
y_train_bal = balanced_train_df['Churn']

# Fill missing values
X_train_bal = X_train_bal.fillna(X_train_bal.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print("\nBefore balancing:", Counter(y_train))
print("After balancing:", Counter(y_train_bal))

# Visualize class balance
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x=y_train, palette="Set2", ax=axes[0])
axes[0].set_title("Before Balancing")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")

sns.countplot(x=y_train_bal, palette="Set1", ax=axes[1])
axes[1].set_title("After Balancing")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# Part 8 - Model Machine Learning
models = {
      "XGBoost": XGBClassifier(
        tree_method='hist',
        device='cuda',
        use_label_encoder=False,
        eval_metric='logloss',
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300,
        random_state=42
    ),

}

# Train models with BALANCED data
print("\nTraining models with balanced data...")
for name, model in tqdm(models.items(), desc="Training models", total=len(models)):
    model.fit(X_train_bal_scaled, y_train_bal)
    score = model.score(X_test_scaled, y_test)
    tqdm.write(f"Model: {name:<25} | Accuracy: {score:.2%}")

# Part 9 - Metrics and Evaluations

# ROC Curves
colors = ['blue', 'green', 'red', 'orange']
plt.figure(figsize=(12, 8))
plt.title("ROC Curves for All Models (Trained with Balanced Data)", fontsize=16)

for idx, (name, model) in enumerate(models.items()):
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[idx], 
             label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion Matrices
def plot_confusion_seaborn(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Churn", "Churn"],
                yticklabels=["Not Churn", "Churn"])
    plt.title(f"Confusion Matrix - {model_name}", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

for name, model in models.items():
    plot_confusion_seaborn(model, X_test_scaled, y_test, name)

# Classification Reports
for name, model in models.items():
    print(f"\nClassification Report - {name}")
    preds = model.predict(X_test_scaled)
    print(classification_report(y_test, preds, 
                               target_names=["Not Churn", "Churn"]))

# Feature Importance
feature_names = X_train.columns
models_with_importance = ['XGBoost', ]

for name in models_with_importance:
    model = models[name]
    # Use balanced training data for consistency
    model.fit(X_train_bal_scaled, y_train_bal)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_imp, palette='cividis')
        plt.title(f'Feature Importance - {name}', fontsize=14)
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

# Part 10 - Save Models and Final Metrics
os.makedirs("saved_models", exist_ok=True)

for name, model in models.items():
    filename = f"saved_models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"Model saved: {filename}")

# Final Evaluation - Use BALANCED training data
results = []

for name, model in models.items():
    # Train with balanced data, test on original imbalanced test set
    model.fit(X_train_bal_scaled, y_train_bal)
    preds = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results.append({"Model": name, "Accuracy": acc, "Recall": rec, "F1-Score": f1})

# Display results
df_metrics = pd.DataFrame(results)
highlight = df_metrics[["Accuracy", "Recall", "F1-Score"]].apply(lambda col: col == col.max())
highlight_color = 'background-color: lightgreen'

styled_df = df_metrics.style.apply(
    lambda df: highlight.replace({True: highlight_color, False: ''}), axis=None
)

print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE (Trained on Balanced Data)")
print("="*60)
print(df_metrics.to_string(index=False))
print("="*60)

import joblib
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                            classification_report, recall_score, roc_curve, auc)

from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from collections import Counter

# Part 2 - Database
df = pd.read_csv("final_augmented_data.csv")

# Part 3 - Data Cleaning
# Check missing values
null_counts = df.isnull().sum()
null_columns = null_counts[null_counts > 0]
print("Columns with missing values:")
print(null_columns)

# Drop rows with missing values
df_cleaned = df.dropna()
print("\nDataset shape after removing rows with missing values:", df_cleaned.shape)

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()
print(f"Dataset shape after removing duplicates: {df_cleaned.shape}")

# Drop unnecessary columns
df_cleaned.drop(columns=['individual_id', 'address_id', 'cust_orig_date', 
                        'date_of_birth', 'acct_suspd_date'], inplace=True)

df.drop(columns=['individual_id', 'address_id', 'cust_orig_date', 
                'date_of_birth', 'acct_suspd_date'], inplace=True)

# Part 4 - Feature Engineering
# Label encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# Part 6 - Model Training
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

# Part 7 - SMOTE (Upsampling)
# Combine training data
train_df = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority = train_df[train_df['Churn'] == 0]
minority = train_df[train_df['Churn'] == 1]

# Upsample minority class
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)

# Combine balanced data
balanced_train_df = pd.concat([majority, minority_upsampled])
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42)

# Separate features and target
X_train_bal = balanced_train_df.drop('Churn', axis=1)
y_train_bal = balanced_train_df['Churn']

# Fill missing values
X_train_bal = X_train_bal.fillna(X_train_bal.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)


from interpret.glassbox import ExplainableBoostingClassifier

# Train EBM model separately with tqdm
print("\nTraining Explainable Boosting Machine (EBM)...")

# Initialize EBM model with similar parameters to other gradient boosting models
ebm_model = ExplainableBoostingClassifier(
    max_rounds=500,
    learning_rate=0.05,
    max_bins=512,
    early_stopping_rounds=10,
    max_interaction_bins=32,
    interactions=10,  # Number of automatic interactions to detect
    outer_bags=8,
    inner_bags=0,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Train with progress indicator
from tqdm import tqdm
import time

# Simulate training progress (EBM doesn't have built-in progress tracking)
print("Fitting EBM model...")
start_time = time.time()

# Fit the model
ebm_model.fit(X_train_bal_scaled, y_train_bal)

training_time = time.time() - start_time
print(f"EBM training completed in {training_time:.2f} seconds")

# Evaluate EBM model
ebm_score = ebm_model.score(X_test_scaled, y_test)
print(f"EBM Model Accuracy: {ebm_score:.2%}")

# Add EBM to the models dictionary for consistent evaluation
models["EBM (Explainable Boosting)"] = ebm_model

# save the model
os.makedirs("saved_models", exist_ok=True)
filename = f"saved_models/ebm_explainable_boosting.pkl"
joblib.dump(ebm_model, filename)
print(f"EBM Model saved: {filename}")