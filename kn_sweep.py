import joblib
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                            classification_report, recall_score, roc_curve, auc, make_scorer)

from tqdm import tqdm # tqdm imported
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier

# Part 2 - Database
df = pd.read_csv("./lockin/final_augmented_data.csv")

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

# Drop unnecessary columns (apply to df_cleaned directly)
df_cleaned.drop(columns=['individual_id', 'address_id', 'cust_orig_date',
                        'date_of_birth', 'acct_suspd_date'], inplace=True)

# --- NEW: Data Sampling (sample to 100 points after cleaning) ---
target_sample_size = 1000
print(f"\nOriginal dataset size after cleaning: {df_cleaned.shape[0]} rows")

if 'Churn' not in df_cleaned.columns:
    print("Error: 'Churn' column not found for stratified sampling. Performing random sampling.")
    df_sampled = df_cleaned.sample(n=min(target_sample_size, len(df_cleaned)), random_state=42)
else:
    # Perform stratified sampling
    churn_counts = df_cleaned['Churn'].value_counts()
    # Check if we have enough unique classes and at least one sample per class
    if len(churn_counts) < 2 or (churn_counts.min() == 0) or (churn_counts.sum() < target_sample_size):
        print(f"Warning: Not enough diverse samples or too few total samples ({churn_counts.sum()}) for stratified sampling to {target_sample_size} rows. Performing random sampling.")
        df_sampled = df_cleaned.sample(n=min(target_sample_size, len(df_cleaned)), random_state=42)
    else:
        # Calculate the fraction of the dataset to keep
        # Using train_test_split to get a stratified sample of the indices
        # We need to make sure the smallest class still gets at least 1 sample if possible.
        # test_size here represents the fraction of data to keep for our sample.
        current_len = len(df_cleaned)
        if target_sample_size >= current_len:
            df_sampled = df_cleaned.copy() # No need to sample if target is larger or equal
            print(f"Target sample size ({target_sample_size}) is greater than or equal to current size ({current_len}). No sampling performed.")
        else:
            # We want to keep a fraction of the data. Let's make the 'test' set our desired sample.
            # train_test_split splits into (X_train, X_test, y_train, y_test).
            # We want the 'test' part to be our sample.
            # If target_sample_size is small, and min_samples_per_class_for_stratify is > 0
            # then we might need to adjust test_size to be at least min_samples_per_class_for_stratify / current_len
            # However, with small target_sample_size, min_samples can be an issue.
            # A common approach for small samples is to oversample the minority class first before splitting,
            # but here we're reducing the overall dataset size.

            # Simple stratified split to get the desired sample size
            # If the original df_cleaned is very small, this might not work perfectly with 100.
            # Let's use it to get indices and then filter.
            _, sampled_indices, _, _ = train_test_split(
                df_cleaned.index,
                df_cleaned['Churn'],
                test_size=target_sample_size, # Directly specify the number of samples for the test set
                stratify=df_cleaned['Churn'],
                random_state=42
            )
            df_sampled = df_cleaned.loc[sampled_indices]

# Assign the sampled DataFrame back to 'df' for the rest of the pipeline
df = df_sampled
print(f"Dataset shape after sampling to {df.shape[0]} rows: {df.shape}")
# --- END NEW: Data Sampling ---

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
             'home_owner', 'college_degree', 'good_credit']
feat_cols.extend(['chubb_sentiment','competitor_sentiment','chubb_stock_return','competitor_stock_return','market_index_return','gdp_growth','customer_engagement'])
X = df[feat_cols]
y = df['Churn']

print(f"\nX shape (sampled): {X.shape}")
print(f"y shape (sampled): {y.shape}")

# Part 6 - Model Training
# Split into train and test sets
# Ensure there are enough samples for stratification in train_test_split
if len(y) < 2 or y.value_counts().min() < 2: # At least 2 samples per class for stratification
    print("Warning: Not enough samples for stratified train_test_split from sampled data. Using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)

# Part 7 - SMOTE (Upsampling)
# Combine training data
train_df = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority = train_df[train_df['Churn'] == 0]
minority = train_df[train_df['Churn'] == 1]

# --- MODIFIED SECTION START ---
# Check if both classes are present in training data after splitting
if len(minority) == 0:
    print("Warning: Minority class not present in training data after sampling/splitting. Skipping upsampling.")
    X_train_bal = X_train.copy()
    y_train_bal = y_train.copy()
elif len(majority) == 0:
    print("Warning: Majority class not present in training data after sampling/splitting. Skipping upsampling (only minority class present).")
    # If only the minority class is present, we cannot 'balance' by upsampling to majority size
    # We proceed with the original X_train and y_train as the 'balanced' set,
    # as there's no imbalance to correct in this split.
    X_train_bal = X_train.copy()
    y_train_bal = y_train.copy()
else: # Both classes are present, proceed with upsampling
    # Upsample minority class
    minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)

    # Combine balanced data
    balanced_train_df = pd.concat([majority, minority_upsampled])
    balanced_train_df = balanced_train_df.sample(frac=1, random_state=42)

    # Separate features and target
    X_train_bal = balanced_train_df.drop('Churn', axis=1)
    y_train_bal = balanced_train_df['Churn']
# --- MODIFIED SECTION END ---

# Fill missing values (using median for numerical features)
for col in X_train_bal.columns:
    if X_train_bal[col].isnull().any():
        median_val = X_train_bal[col].median()
        X_train_bal[col].fillna(median_val, inplace=True)
        # Apply the same median to the test set
        if col in X_test.columns: # Check if column exists in X_test before filling
            X_test[col].fillna(median_val, inplace=True)

# Scale features
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print("\nBefore balancing (from sampled data):", Counter(y_train))
print("After balancing (from sampled data):", Counter(y_train_bal))

# Visualize class balance
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x=y_train, palette="Set2", ax=axes[0])
axes[0].set_title("Before Balancing (Sampled Data)")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")

sns.countplot(x=y_train_bal, palette="Set1", ax=axes[1])
axes[1].set_title("After Balancing (Sampled Data)")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# Part 8 - KNN Model Training and K-tuning
print("\nFinding the best 'k' for K-Nearest Neighbors on Sampled Data...")

# Define the parameter grid for n_neighbors (k)
# Dynamically set max_k based on the size of the balanced training set.
# k should be less than the number of training samples.
max_k = min(50, len(X_train_bal_scaled) // 2 -1) # K should be less than half of training samples for stability
if max_k < 1: # Ensure at least k=1 is tested if dataset is very small
    param_grid = {'n_neighbors': [1]}
    print("Warning: Very small training set after balancing, testing only k=1.")
else:
    param_grid = {'n_neighbors': list(range(1, max_k + 1, 2))}
    if not param_grid['n_neighbors']: # If range ends up empty, e.g. max_k=0
         param_grid = {'n_neighbors': [1]} # Fallback to [1]

knn = KNeighborsClassifier()

# Use GridSearchCV to find the best 'k'
# 'verbose=2' will provide detailed output during the grid search, indicating progress.
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search.fit(X_train_bal_scaled, y_train_bal)

print(f"\nBest 'k' value: {grid_search.best_params_['n_neighbors']}")
print(f"Best F1-Score (weighted) during CV: {grid_search.best_score_:.4f}")

# Get the best KNN model
best_knn_model = grid_search.best_estimator_

# Plotting F1-scores for different K values
k_values = [item['n_neighbors'] for item in grid_search.cv_results_['params']]
f1_scores = grid_search.cv_results_['mean_test_score']

plt.figure(figsize=(12, 6))
plt.plot(k_values, f1_scores, marker='o', linestyle='-', color='skyblue')
plt.title('F1-Score (Weighted) vs. K Value for KNN (Sampled Data)', fontsize=16)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Mean F1-Score (Weighted) on CV Folds', fontsize=12)
# Ensure xticks don't fail if k_values is empty or has only one element
if k_values:
    plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=grid_search.best_params_['n_neighbors'], color='red', linestyle='--', label=f"Best K = {grid_search.best_params_['n_neighbors']}")
plt.legend()
plt.tight_layout()
plt.show()


# Part 9 - Metrics and Evaluations for the Best KNN Model

print(f"\nEvaluating Best KNN Model (k={best_knn_model.n_neighbors}) on Sampled Data...")

# Predictions
y_pred_knn = best_knn_model.predict(X_test_scaled)

# Check if the model supports probabilities for both classes
if len(best_knn_model.classes_) > 1:
    y_proba_knn = best_knn_model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class

    # ROC Curve
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label=f'KNN (AUC = {roc_auc_knn:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - K-Nearest Neighbors (Sampled Data)', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Model trained on a single class. ROC Curve cannot be generated.")

# Confusion Matrix
def plot_confusion_seaborn(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Churn", "Churn"],
                yticklabels=["Not Churn", "Churn"])
    plt.title(f"Confusion Matrix - {model_name} (Sampled Data)", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

plot_confusion_seaborn(best_knn_model, X_test_scaled, y_test, f"K-Nearest Neighbors (k={best_knn_model.n_neighbors})")

# Part 10 - Save Model and Final Metrics
os.makedirs("saved_models", exist_ok=True)
filename = f"saved_models/best_knn_model_k_{best_knn_model.n_neighbors}_sampled.pkl"
joblib.dump(best_knn_model, filename)
print(f"Best KNN Model saved: {filename}")

# Final Evaluation - Use BALANCED training data for training, test on original imbalanced test set
results = []
preds = best_knn_model.predict(X_test_scaled)

acc = accuracy_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds) # Default is 'binary' for positive label 1

results.append({"Model": f"K-Nearest Neighbors (k={best_knn_model.n_neighbors})",
                "Accuracy": acc, "Recall": rec, "F1-Score": f1})

# Display results
df_metrics = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE (K-Nearest Neighbors on Sampled Data)")
print("="*60)
print(df_metrics.to_string(index=False))
print("="*60)