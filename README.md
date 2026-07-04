# Predictive Modeling Using Machine Learning - Sales Dashboard & Pipeline 🤖📊

An interactive, premium Streamlit dashboard for data preprocessing, cleaning, exploratory analysis, and **predictive machine learning modeling**. Train and evaluate classification and regression models on-the-fly and visualize performance using confusion matrices, ROC curves, actual vs. predicted plots, and residual plots.

## 🎯 Project Overview
This project simulates real-world data engineering, analysis, and machine learning tasks. It demonstrates how to clean inconsistent customer transaction records (addressing missing values, typos, duplicates, and outliers) and use the cleaned data to build predictive machine learning models to forecast outcomes (e.g. customer satisfaction and transaction spend).

---

## ⚙️ Core Pipeline Components
1. **Messy Data Generator** (`src/generator.py`): Generates a synthetic dataset of 1,200+ records with intentional flaws (e.g., negative prices/quantities, extreme ages, conflicting duplicates, and missing dates).
2. **Data Cleaner** (`src/cleaner.py`): Imputes missing numeric values using medians, standardizes categorical representations (e.g., normalising gender inputs and payment methods like `GPay`), recalculates erroneous spend totals, and filters records.
3. **Exploratory Data Analysis** (`src/analyzer.py`): Creates publication-quality static Seaborn charts to visualize customer demographics and sales performance.
4. **Machine Learning Pipeline** (`src/model.py`): Fits a `ColumnTransformer` with `StandardScaler` (numeric features) and `OneHotEncoder` (categorical features) followed by the selected estimator. Handles train-test splits before fit transformations to prevent data leakage.
5. **Interactive Streamlit Web Dashboard** (`app.py`): A premium dark-mode dashboard displaying KPI cards, cleaning controls, interactive distributions, and a dedicated **Machine Learning Center**.

---

## 🤖 Machine Learning Tasks & Models

### 1. Classification: Predict High Customer Satisfaction
* **Target**: Binary classification of satisfaction rating (Rating $\ge 4$ ➔ `High Satisfaction` vs. `Low/Medium Satisfaction`).
* **Features**: Customer Age, Customer Gender, Product Category, Quantity, Unit Price, Total Spent, Payment Method.
* **Algorithms**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier.
* **Evaluation & Visualizations**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.

### 2. Regression: Predict Total Spent
* **Target**: Continuous value of `Total_Spent` (with `Quantity` and `Unit_Price` dropped to prevent mathematical leakage).
* **Features**: Customer Age, Customer Gender, Product Category, Payment Method, Customer Satisfaction.
* **Algorithms**: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
* **Evaluation & Visualizations**: $R^2$, MAE, RMSE, Actual vs. Predicted scatterplot, and Residuals plot.

---

## 🔧 Data Issues Handled & Cleaned
* **Duplicate Records**: Dropped exact duplicate rows and resolved conflicting entries sharing the same `Transaction_ID`.
* **Broken Formats & Missing Dates**: Standardized multiple input formats (e.g., YYYY-MM-DD, DD/MM/YYYY, and long text dates) and filled unparseable fields.
* **Customer Demographics**: Cleaned gender variations (`M`, `male`, `Mlae` ➔ `Male`) and imputed missing ages while capping extreme age outliers.
* **Numerical Imputation & Outlier Capping**: Imputed missing/zero unit prices based on category-specific medians and capped quantity anomalies.
* **Recalculated Expenditures**: Validated and recalculated incorrect `Total_Spent` values (`Quantity * Unit_Price`).
* **Customer Satisfaction Rating**: Standardized ratings to a $1$ to $5$ scale and filled missing inputs.

---

## 📂 Project Structure
```
d:\Internship project\
├── data/                      # Data storage
│   ├── raw_sales_data.csv     # Messy dataset
│   └── cleaned_sales_data.csv # Cleaned dataset
├── plots/                     # Static EDA visualizations
├── src/                       # Source files
│   ├── __init__.py
│   ├── generator.py           # Synthetic dataset generator
│   ├── cleaner.py             # Normalization and cleaning
│   ├── analyzer.py            # Static plot builder
│   └── model.py               # Machine Learning preprocessing & pipelines
├── app.py                     # Streamlit application
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Mac/Linux)
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts
You can run each stage of the pipeline independently from the command line:
```bash
# Generate messy data
python src/generator.py

# Run cleaning script
python src/cleaner.py

# Generate static charts
python src/analyzer.py
```

### 4. Start the Interactive Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to view the interactive dashboard, preprocess data, and train models.

---

## 🛠️ Built With
* **Python** — Core language
* **Pandas & NumPy** — Data manipulation and math
* **Scikit-Learn** — Machine Learning pipelines and evaluation metrics
* **Matplotlib & Seaborn** — Data visualization
* **Streamlit** — Web application framework
