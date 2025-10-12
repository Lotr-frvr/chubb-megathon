import joblib
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Importing libraries
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier,BaggingClassifier,HistGradientBoostingClassifier)
from sklearn.metrics import accuracy_score

# Metrics and evaluation tools
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import roc_curve, auc

#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm  # Para notebooks como Kaggle/Colab

#%% md
# Part 2 - Database
#%%
# Load the dataset
df = pd.read_csv("final_augmented_data.csv")
# Part 3 - Data Cleaning
#%%
# Check the total number of missing (null) values in each column
null_counts = df.isnull().sum()

# Filter only columns that contain at least one missing value
null_columns = null_counts[null_counts > 0]

# Print the columns with missing values
print("Columns with missing values:")
print(null_columns)
#%%
# Display missing values summary before cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())
#%%
# Drop rows with any missing values
# This will remove rows with NaNs in latitude, longitude, city, county, home_market_value or acct_suspd_date
df_cleaned = df.dropna()
#%%
# Display shape after cleaning
print("\nDataset shape after removing rows with missing values:", df_cleaned.shape)

# Display remaining missing values (should be zero)
print("\nMissing values after cleaning:")
print(df_cleaned.isnull().sum())
#%%
# Verificar duplicatas
print("Duplicated rows:", df_cleaned.duplicated().sum())

# Remover duplicatas
df_cleaned = df_cleaned.drop_duplicates()
#%%
# Drop 'RowNumber' and 'CustomerId' as they are not useful for analysis
df_cleaned.drop(columns=['individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 'acct_suspd_date'], inplace=True)

# Drop 'RowNumber' and 'CustomerId' as they are not useful for analysis
df.drop(columns=['individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 'acct_suspd_date'], inplace=True)


# Check for duplicate rows
duplicate_rows = df_cleaned[df_cleaned.duplicated()]

# Print the number of duplicate rows
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")

# Optionally, display the first few duplicate rows (if any)
if not duplicate_rows.empty:
    print("\nExample duplicate rows:")
    print(duplicate_rows.head())
#%%
# Remove duplicate rows (keeping the first occurrence)
df_cleaned.drop_duplicates(inplace=True)

# Confirm removal
print(f"New dataset shape after removing duplicates: {df_cleaned.shape}")
#%% md
# Part 4 - Feature Engineering
#%%
# Identify categorical columns (of type 'object')
cat_cols = df.select_dtypes(include='object').columns

# Apply LabelEncoder to all categorical columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # store the encoders in case inverse transformation is needed later

# Part 5 - Training and testing division
#%%
feat_cols = ['curr_ann_amt', 'days_tenure', 
             'age_in_years', 'income', 
             'has_children','length_of_residence', 
             'marital_status', 'home_owner', 
             'college_degree', 
             'good_credit']

X = df[feat_cols]
y = df['Churn']
#%%
X.shape
#%%
y.shape
#%% md
# Part 6 - Model Training
#%%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=42)
#%% md
# Part 7 - SMOTE
#%%
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Combine training data
train_df = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority = train_df[train_df['Churn'] == 0]
minority = train_df[train_df['Churn'] == 1]

# Upsample minority class by random sampling with replacement
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)

# Combine balanced data
balanced_train_df = pd.concat([majority, minority_upsampled])

# Shuffle
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42)

# Separate again
X_train_bal = balanced_train_df.drop('Churn', axis=1)
y_train_bal = balanced_train_df['Churn']

# Scale features
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Check new distribution
print("Before balancing:", Counter(y_train))
print("After balancing:", Counter(y_train_bal))
#%%
# Configura estilo
sns.set(style="whitegrid")

# Figura com subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico antes do balanceamento
sns.countplot(x=y_train, palette="Set2", ax=axes[0])
axes[0].set_title("Before Balancing")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")

# Gráfico depois do balanceamento
sns.countplot(x=y_train_bal, palette="Set1", ax=axes[1])
axes[1].set_title("After Balancing")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Count")

# Ajuste layout
plt.tight_layout()
plt.show()
#%%
# Fill missing values in training and test sets with median
X_train_bal = X_train_bal.fillna(X_train_bal.median())
X_test = X_test.fillna(X_test.median())

print("Total NaN in X_train_bal:", np.isnan(X_train_bal).sum().sum())
#%%
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print("NaNs in X_train_bal_scaled:", np.isnan(X_train_bal_scaled).sum())
print("NaNs in X_test_scaled:", np.isnan(X_test_scaled).sum())
#%% md
# Part 8 - Model Machine learning
#%%


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

# Dictionary of models
models = {
    
          # Gradient Boosting
          "Gradient Boosting": GradientBoostingClassifier(random_state=42),

          # Bagging
          "Bagging": BaggingClassifier(random_state=42),

          # XGBoost
          "XGBoost": XGBClassifier(tree_method='gpu_hist',        # GPU acceleration
                                   predictor='gpu_predictor',
                                   use_label_encoder=False,
                                   eval_metric='logloss',
                                   learning_rate=0.05,
                                   max_depth=6,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   n_estimators=300,
                                   random_state=42),

          # LightGBM
          "LightGBM": LGBMClassifier(device='gpu',                  # GPU support
                                     boosting_type='gbdt',
                                     learning_rate=0.05,
                                     max_depth=6,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     n_estimators=300,
                                     random_state=42)}

# Loop com tqdm + print de desempenho de cada modelo
for name, model in tqdm(models.items(), desc="Training models", total=len(models)):
    model.fit(X_train_bal_scaled, y_train_bal)  # Treinamento
    score = model.score(X_test_scaled, y_test)  # Acurácia no teste
    
    # Imprimir resultado com tqdm.write (não quebra a barra de progresso)
    tqdm.write(f"Model: {name:<25} | Accuracy: {score:.2%}")
#%% md
# Part 9 - Metrics and Evaluations
#%%
# Define a list of colors to distinguish each model in the plot
colors = ['blue', 'green', 'red', 'orange', 'purple', 
          'brown', 'cyan', 'magenta', 'teal', 'olive',
          'gold', 'darkblue', 'pink']

# Set up the plot size and title
plt.figure(figsize=(12, 8))
plt.title("ROC Curves for All Models (Trained with SMOTE)", fontsize=16)

# Loop through each trained model
for idx, (name, model) in enumerate(models.items()):
    
    # Predict probabilities for the positive class (Churn = 1)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Compute False Positive Rate, True Positive Rate, and thresholds
    fpr, tpr, _ = roc_curve(y_test, probs)

    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.plot(fpr, tpr, 
             color=colors[idx % len(colors)], 
             label=f"{name} (AUC = {roc_auc:.3f})", 
             linewidth=2)

# Plot diagonal line for reference (random guessing)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)

# Customize the axes and layout
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
# Function to plot a confusion matrix using Seaborn
def plot_confusion_seaborn(model, X_test, y_test, model_name):
    """
    Plots the confusion matrix of a given model using seaborn heatmap.
    
    Parameters:
    - model: trained classifier
    - X_test: feature test set
    - y_test: true labels
    - model_name: name of the model (string) for plot title
    """
    
    # Generate predictions from the model
    preds = model.predict(X_test)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, preds)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Churn", "Churn"],     # Adjust labels to your use case
                yticklabels=["Not Churn", "Churn"])
    
    plt.title(f"Confusion Matrix - {model_name}", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# Loop through all trained models and plot their confusion matrices
for name, model in models.items():
    plot_confusion_seaborn(model, 
                           X_test_scaled,  # Use scaled features
                           y_test, 
                           name)
#%%
# Loop through all trained models to print classification reports
for name, model in models.items():
    print(f"\n Classification Report - {name}")
    
    # Predict the class labels on the test set
    preds = model.predict(X_test_scaled)  # Use scaled features if used in training
    
    # Generate and print the classification report
    print(classification_report(y_test,                 # True labels
                                preds,                  # Predicted labels
                                target_names=["Not Churn", "Churn"]  # Class labels (adjust to your use case)
                               ))
#%%
# Ensure X_train is a DataFrame to extract column names
feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]

# List of models that support feature importance
models_with_importance = ['XGBoost', 'LightGBM', 'Bagging', 'Gradient Boosting']

# Loop through each model to extract and plot feature importances
for name in models_with_importance:
    model = models[name]  # Retrieve model from your trained dictionary
    model.fit(X_train, y_train)  # Refit on full training set (optional but common)

    # Check if the model supports feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        # Safety check: dimension must match number of features
        if len(importances) != len(feature_names):
            print(f"[Warning] Dimension mismatch in model {name}. Skipping plot.")
            continue

        # Create a DataFrame with features and their importance
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Plot the top features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_imp, palette='cividis')
        plt.title(f'Feature Importance - {name}', fontsize=14)
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
#%% md
# Part 10 - Models final
#%%
# Create a folder named 'saved_models' if it doesn't already exist
os.makedirs("saved_models", exist_ok=True)

# Loop through all trained models and save each one as a .pkl file
for name, model in models.items():
    # Format the filename: lowercase with underscores instead of spaces
    filename = f"saved_models/{name.replace(' ', '_').lower()}.pkl"
    
    # Save the model using joblib
    joblib.dump(model, filename)
    
    # Print confirmation
    print(f"Model saved: {filename}")
#%%
from sklearn.metrics import f1_score

# Ensure no NaN values in training/testing data
X_train_filled = X_train.fillna(X_train.median())
X_test_filled = X_test.fillna(X_test.median())

# List to store F1-score results for each model
f1_results = []

# Loop through all trained models
for name, model in models.items():
    model.fit(X_train_filled, y_train)       # Train the model
    preds = model.predict(X_test_filled)     # Make predictions on test set

    f1 = f1_score(y_test, preds)             # Compute F1-score

    # Store the result with model name and rounded F1-score
    f1_results.append({
        "Model": name,
        "F1-Score": round(f1, 4)
    })

# Convert the results list into a DataFrame and sort by F1-score
df_f1_scores = pd.DataFrame(f1_results).sort_values(by="F1-Score", ascending=False)

# Display the final F1-score comparison table
df_f1_scores
#%%
# Fill missing values using median before training
X_train_filled = X_train.fillna(X_train.median())
X_test_filled = X_test.fillna(X_test.median())

# List to store evaluation results
results = []

# Loop through each model and evaluate
for name, model in models.items():
    model.fit(X_train_filled, y_train)         # Train the model
    preds = model.predict(X_test_filled)       # Predict on test set

    # Compute evaluation metrics
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Save results
    results.append({"Model": name,
                    "Accuracy": acc,
                    "Recall": rec,
                    "F1-Score": f1})

# Create DataFrame from results
df_metrics = pd.DataFrame(results)

# Create boolean DataFrame: True where values are the best
highlight = df_metrics[["Accuracy", "Recall", "F1-Score"]].apply(lambda col: col == col.max())

# Define style for highlighted best scores
highlight_color = 'background-color: green'

# Apply highlighting
styled_df = df_metrics.style.apply(lambda df: highlight.replace({True: highlight_color, False: ''}), axis=None)

# Display styled DataFrame
print(styled_df)