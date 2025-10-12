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