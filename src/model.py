import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def prepare_ml_data(df, task_type="classification", target_col=None, feature_cols=None):
    """
    Prepare feature matrix X and target vector y from the dataframe.
    """
    if task_type == "classification":
        # Target: High Customer Satisfaction (Rating >= 4)
        if target_col is None:
            target_col = "High_Satisfaction"
        df_ml = df.copy()
        df_ml["High_Satisfaction"] = (df_ml["Customer_Satisfaction"] >= 4).astype(int)
    else:
        # Target: Total Spent
        if target_col is None:
            target_col = "Total_Spent"
        df_ml = df.copy()

    # Determine features
    if feature_cols is None:
        if task_type == "classification":
            # Exclude Satisfaction ratings and non-predictive IDs / dates
            exclude = ["Transaction_ID", "Customer_ID", "Transaction_Date", "Month_Year", "Customer_Satisfaction", "High_Satisfaction", "Age_Group"]
        else:
            # Exclude Total_Spent, non-predictive features, and Quantity/Unit_Price to prevent leakage
            exclude = ["Transaction_ID", "Customer_ID", "Transaction_Date", "Month_Year", "Total_Spent", "Age_Group", "Quantity", "Unit_Price"]
        feature_cols = [col for col in df_ml.columns if col not in exclude]

    X = df_ml[feature_cols]
    y = df_ml[target_col]

    return X, y, feature_cols

def get_pipeline(model_name, task_type, numeric_features, categorical_features, hyperparams):
    """
    Create a scikit-learn Pipeline with preprocessing and the specified model.
    """
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Model definition
    if task_type == "classification":
        if model_name == "logistic_regression":
            model = LogisticRegression(C=hyperparams.get("C", 1.0), random_state=42, max_iter=1000)
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier(
                max_depth=hyperparams.get("max_depth", None),
                min_samples_split=hyperparams.get("min_samples_split", 2),
                random_state=42
            )
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=hyperparams.get("n_estimators", 100),
                max_depth=hyperparams.get("max_depth", None),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
    else: # regression
        if model_name == "linear_regression":
            model = LinearRegression()
        elif model_name == "decision_tree":
            model = DecisionTreeRegressor(
                max_depth=hyperparams.get("max_depth", None),
                min_samples_split=hyperparams.get("min_samples_split", 2),
                random_state=42
            )
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=hyperparams.get("n_estimators", 100),
                max_depth=hyperparams.get("max_depth", None),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown regression model: {model_name}")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

def train_and_evaluate(X, y, model_name, task_type, test_size=0.2, hyperparams=None):
    """
    Split the dataset, train the pipeline, and compute performance metrics.
    """
    if hyperparams is None:
        hyperparams = {}

    # Strict Featurization Ordering: Split BEFORE preprocessing fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Separate features by type
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    pipeline = get_pipeline(model_name, task_type, numeric_features, categorical_features, hyperparams)
    
    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    
    results = {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    # Evaluate
    if task_type == "classification":
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(y_test, y_pred, zero_division=0)
        results["recall"] = recall_score(y_test, y_pred, zero_division=0)
        results["f1"] = f1_score(y_test, y_pred, zero_division=0)
        
        # Probabilities for ROC curve if available
        if hasattr(pipeline.named_steps['model'], "predict_proba"):
            results["y_prob"] = pipeline.predict_proba(X_test)[:, 1]
    else: # regression
        results["r2"] = r2_score(y_test, y_pred)
        results["mae"] = mean_absolute_error(y_test, y_pred)
        results["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))

    return results

def get_confusion_matrix_plot(y_test, y_pred):
    """Generate confusion matrix figure."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Custom color scheme matching Streamlit UI (Teal/Dark)
    sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu', cbar=False,
                xticklabels=['Low Sat (<4)', 'High Sat (>=4)'],
                yticklabels=['Low Sat (<4)', 'High Sat (>=4)'], ax=ax)
    
    ax.set_title("Confusion Matrix", fontweight='bold', pad=10, color="#FFFFFF")
    ax.set_ylabel("Actual Label", color="#E2E8F0")
    ax.set_xlabel("Predicted Label", color="#E2E8F0")
    
    # Configure label styling
    ax.tick_params(colors='#E2E8F0')
    for text in ax.texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
        
    plt.tight_layout()
    return fig

def get_roc_curve_plot(y_test, y_prob):
    """Generate ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='#0D9488', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#E2E8F0', lw=1.5, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight='bold', pad=10, color="#FFFFFF")
    ax.set_xlabel("False Positive Rate", color="#E2E8F0")
    ax.set_ylabel("True Positive Rate", color="#E2E8F0")
    ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155")
    ax.tick_params(colors='#E2E8F0')
    
    plt.tight_layout()
    return fig

def get_actual_vs_predicted_plot(y_test, y_pred):
    """Generate actual vs predicted scatterplot for regression."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#38BDF8", ax=ax, edgecolor="#1E293B")
    
    # 45-degree reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='#F59E0B', linewidth=2, label="Ideal Fit (y=x)")
    
    ax.set_title("Actual vs. Predicted Total Spent", fontweight='bold', pad=10, color="#FFFFFF")
    ax.set_xlabel("Actual Total Spent (₹)", color="#E2E8F0")
    ax.set_ylabel("Predicted Total Spent (₹)", color="#E2E8F0")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
    ax.legend(facecolor="#1E293B", edgecolor="#334155")
    ax.tick_params(colors='#E2E8F0')
    
    plt.tight_layout()
    return fig

def get_residuals_plot(y_test, y_pred):
    """Generate residual plot for regression."""
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="#F59E0B", ax=ax, edgecolor="#1E293B")
    ax.axhline(y=0, color='#0D9488', linestyle='--', linewidth=2)
    
    ax.set_title("Residuals vs. Fitted Values", fontweight='bold', pad=10, color="#FFFFFF")
    ax.set_xlabel("Predicted Total Spent (₹)", color="#E2E8F0")
    ax.set_ylabel("Residuals (Actual - Predicted) (₹)", color="#E2E8F0")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
    ax.tick_params(colors='#E2E8F0')
    
    plt.tight_layout()
    return fig
