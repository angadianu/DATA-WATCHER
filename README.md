# DATA WATCH

## Overview

DATA WATCH is a Streamlit-based Machine Learning Analytics Dashboard for exploring customer datasets, checking data quality, visualizing patterns, training baseline models, clustering numerical profiles, and generating transparent predictions. It is designed as a practical BCA portfolio project that runs locally without API keys or paid services.

## Features

- Dashboard with dataset health metrics and an overview chart
- CSV upload with automatic fallback to `data/sample_data.csv`
- Dataset preview, data types, missing values, duplicates, and statistical summary
- Histogram, box plot, correlation heatmap, and selectable scatter plots
- Classification with Logistic Regression and Random Forest Classifier
- Regression with Linear Regression and Random Forest Regressor
- Measured model comparison using a held-out test split
- K-Means clustering with StandardScaler
- Prediction widgets generated from trained model features
- CSV downloads for processed, clustered, and prediction data
- Defensive validation for empty, incomplete, or unsuitable datasets

## Machine Learning Algorithms

- Logistic Regression
- Random Forest Classifier
- Linear Regression
- Random Forest Regressor
- K-Means clustering

Numeric features use median imputation where needed. Categorical features use most-frequent imputation and one-hot encoding. Models use `random_state=42` where supported.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

```text
InsightForge-AI/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── sample_data.csv
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## GitHub Deployment

1. Create a new repository on GitHub.
2. From the project folder, initialize Git and commit the files:

```bash
git init
git add .
git commit -m "Initial InsightForge AI dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/InsightForge-AI.git
git push -u origin main
```

GitHub Pages cannot directly run a Python Streamlit application because it serves static files only. The repository can be hosted on GitHub, while the Streamlit application should be deployed using a free Python-compatible hosting service such as Streamlit Community Cloud. Connect the GitHub repository there and set the main file to `app.py`.


run using :-    python -m streamlit run app.py