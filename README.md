# Chubb Megathon 2025 - Churns by Chubbs

## Problem Statement
"Churns by Chubbs" focuses on analyzing and predicting customer churn for Chubb Insurance. The goal is to leverage data-driven insights to identify patterns and reduce churn rates effectively.

## Project Structure

### 1. **News Analysis**
- **File:** `scrapers/news_agent.py`
- **Description:** Scrapes news headlines related to Chubb Insurance and performs sentiment analysis to understand public perception.
- **Usage:**
  ```bash
  python scrrapers/news_agent.py
  ```

### 2. **Stock Analysis**
- **File:** `scrapers/stock_agent.py`
- **Description:** Fetches stock prices for Chubb Insurance (Ticker: CB) and performs technical analysis to identify trends and signals.
- **Usage:**
  ```bash
  python scrrapers/stock_agent.py
  ```

### 3. **Model Development**
- **Folder:** `model/`
- **Description:** Contains code for building and training models to predict churn.

#### Key Files:
- **`kn_sweep.py`**
  - **Purpose:** Selects the best value of `k` for K-Nearest Neighbors (KNN) model.
  - **Usage:**
    ```bash
    python model/kn_sweep.py
    ```

- **`ctgan_data.py`**
  - **Purpose:** Trains a Conditional Tabular GAN (CTGAN) model to generate scalable synthetic data for training.
  - **Usage:**
    ```bash
    python model/ctgan_data.py
    ```

## Features

### 1. **Company Sentiments Dataset**

- **Description:** A newly integrated dataset that captures company sentiments from various sources, enabling deeper insights into customer and market perceptions.

### 2. **KNN for Automatic Completion**

- **Description:** Implements K-Nearest Neighbors (KNN) for automatic data completion, ensuring robust handling of missing values and improving data quality.

### 3. **LLM Advices**

- **Description:** Leverages Large Language Models (LLMs) to provide actionable insights and recommendations based on data analysis and predictions.

### 4. **Feature Explanation**

- **Description:** Detailed feature explanations to enhance understanding of the data and its impact on model predictions.

### 5. **EBMs and Explainability**

- **Description:** Utilizes Explainable Boosting Machines (EBMs) for interpretable machine learning. EBMs are open-box models that provide transparency and insights into the decision-making process.

## Note on UI Code

The UI code for this project will be maintained in a separate branch named `UI`. Please switch to the `UI` branch to access and work on the user interface components.

