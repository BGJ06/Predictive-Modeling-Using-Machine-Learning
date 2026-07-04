import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Import project modules
from src.generator import generate_messy_data
from src.cleaner import DataCleaner
from src.model import (
    prepare_ml_data, train_and_evaluate,
    get_confusion_matrix_plot, get_roc_curve_plot,
    get_actual_vs_predicted_plot, get_residuals_plot
)

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Sales Data Pipeline & Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    /* Card/Block styling */
    div[data-testid="metric-container"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 1.5rem 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #0D9488;
    }
    /* KPI Metric styling labels */
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    /* Headers and titles */
    h1, h2, h3 {
        color: #38BDF8 !important;
        font-family: 'Inter', sans-serif;
    }
    /* Buttons */
    .stButton>button {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #0F766E !important;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.4);
    }
    /* Success, warning alerts custom styles */
    .stAlert {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, 'data', 'raw_sales_data.csv')
clean_path = os.path.join(script_dir, 'data', 'cleaned_sales_data.csv')

# Configure matplotlib/seaborn styles for dark dashboard environment
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1E293B",
    "figure.facecolor": "#0F172A",
    "grid.color": "#334155",
    "text.color": "#F8FAFC",
    "axes.labelcolor": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "axes.edgecolor": "#334155",
})

# Sidebar Navigation
st.sidebar.title("📊 Pipeline Control")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go To",
    ["🏠 Dashboard Overview", "⚙️ Data Cleaning Center", "📊 Interactive Analysis", "🤖 Machine Learning Center", "📂 Dataset Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **A Data Cleaning & Visualization Project**\nDesigned to handle missing values, outliers, duplicates, and visualize premium data stories.")

# Check for existence of data
raw_exists = os.path.exists(raw_path)
clean_exists = os.path.exists(clean_path)

if page == "⚙️ Data Cleaning Center":
    st.title("⚙️ Data Cleaning & Pipeline Center")
    st.write("Generate raw datasets, inspect dirty data issues, and run the pipeline to clean, standardize, and engineering new features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Raw Data Generation")
        st.write("Generate a synthetic customer sales dataset with simulated duplicates, missing dates, out-of-bounds outliers, and typing inconsistencies.")
        if st.button("Generate Messy Dataset"):
            with st.spinner("Generating raw data..."):
                generate_messy_data(raw_path, num_records=1200)
                st.success("Successfully generated `data/raw_sales_data.csv` with 1,200+ raw records!")
                st.rerun()
                
    with col2:
        st.subheader("2. Data Cleaning Pipeline")
        st.write("Execute the cleaning pipeline: standardizes categorical strings, parses dates, imputes missing values via statistics, and caps logical outliers.")
        if not os.path.exists(raw_path):
            st.warning("⚠️ Please generate the messy dataset first.")
        else:
            if st.button("Run Cleaning Pipeline"):
                with st.spinner("Executing clean operations..."):
                    cleaner = DataCleaner(raw_path, clean_path)
                    stats = cleaner.clean()
                    st.session_state['cleaning_stats'] = stats
                    st.success("Pipeline ran successfully! Saved `data/cleaned_sales_data.csv`")
                    st.rerun()

    st.markdown("---")
    
    if raw_exists and clean_exists:
        df_raw = pd.read_csv(raw_path)
        df_clean = pd.read_csv(clean_path)
        
        st.subheader("📈 Pipeline Cleaning Summary Metrics")
        
        # Display Stats Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows count", f"{len(df_raw):,}")
        c2.metric("Cleaned Rows count", f"{len(df_clean):,}")
        c3.metric("Duplicate Rows Removed", f"{len(df_raw) - len(df_clean):,}")
        
        # Calculate approximate missing rate
        raw_null_count = df_raw.isna().sum().sum()
        clean_null_count = df_clean.isna().sum().sum()
        c4.metric("Missing Values Imputed", f"{raw_null_count - clean_null_count:,}")
        
        st.markdown("### 🔍 Before vs. After Cleaning Inspection")
        col_select = st.selectbox("Select a column to visually inspect cleaning transformations:", ["Customer_Age", "Customer_Gender", "Quantity", "Unit_Price", "Customer_Satisfaction"])
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"#### 🛑 Raw Data: `{col_select}`")
            # Try to plot raw distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            
            if col_select in ["Customer_Age", "Quantity", "Unit_Price"]:
                # Parse numeric, skip NaNs for plotting
                series = pd.to_numeric(df_raw[col_select], errors='coerce').dropna()
                if not series.empty:
                    sns.histplot(series, kde=True, color="#F59E0B", ax=ax, bins=20)
                    ax.set_title(f"Raw {col_select} (Outliers/Missing present)")
                else:
                    st.write("No numeric data to display")
            else:
                # Categorical countplot
                series = df_raw[col_select].fillna("Missing").astype(str)
                sns.countplot(y=series, palette="Oranges", ax=ax, order=series.value_counts().index[:8], hue=series, legend=False)
                ax.set_title(f"Raw {col_select} Casing & Typos")
                
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_right:
            st.markdown(f"#### ✅ Cleaned Data: `{col_select}`")
            # Plot cleaned distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            
            if col_select in ["Customer_Age", "Quantity", "Unit_Price"]:
                series = df_clean[col_select]
                sns.histplot(series, kde=True, color="#0D9488", ax=ax, bins=20)
                ax.set_title(f"Cleaned {col_select} (Capped & Imputed)")
            else:
                series = df_clean[col_select].astype(str)
                sns.countplot(y=series, palette="Tealgrn", ax=ax, order=series.value_counts().index, hue=series, legend=False)
                ax.set_title(f"Standardized {col_select}")
                
            plt.tight_layout()
            st.pyplot(fig)
            
    else:
        st.info("💡 Please click on the buttons above to generate and clean the dataset to activate visualizations.")

elif page == "🏠 Dashboard Overview":
    st.title("🏠 Executive Sales & Insights Dashboard")
    
    if not (raw_exists and clean_exists):
        st.warning("⚠️ The dataset has not been initialized. Please head over to the **⚙️ Data Cleaning Center** page in the sidebar to generate and clean the data!")
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000", caption="Process data to view dashboards", use_container_width=True)
    else:
        df = pd.read_csv(clean_path)
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        
        # Dynamic KPIs
        total_rev = df['Total_Spent'].sum()
        total_trans = len(df)
        avg_spend = df['Total_Spent'].mean()
        median_age = df['Customer_Age'].median()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"₹{total_rev:,.2f}")
        col2.metric("Total Transactions", f"{total_trans:,}")
        col3.metric("Average Ticket Size", f"₹{avg_spend:,.2f}")
        col4.metric("Median Customer Age", f"{median_age:.0f} Yrs")
        
        st.markdown("---")
        
        # Charts section
        col_left, col_right = st.columns([6, 4])
        
        with col_left:
            st.markdown("### 📈 Monthly Sales Revenue Trend")
            monthly_df = df.copy()
            monthly_df['Month_Year'] = monthly_df['Transaction_Date'].dt.to_period('M').dt.to_timestamp()
            trend = monthly_df.groupby('Month_Year')['Total_Spent'].sum().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=trend, x='Month_Year', y='Total_Spent', marker='o', color="#38BDF8", linewidth=3, ax=ax)
            ax.set_xlabel("Transaction Month")
            ax.set_ylabel("Revenue (INR ₹)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_right:
            st.markdown("### 📊 Revenue by Product Category")
            rev_cat = df.groupby('Product_Category')['Total_Spent'].sum().reset_index().sort_values(by='Total_Spent', ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(data=rev_cat, x='Product_Category', y='Total_Spent', palette="Blues_r", ax=ax, hue='Product_Category', legend=False)
            ax.set_ylabel("Total Revenue (INR ₹)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Insights list
        st.markdown("### 💡 Data Story & Key Insights")
        st.markdown(f"""
        - **Sales Leadership**: **{rev_cat.iloc[0]['Product_Category']}** is our highest-yielding product division, generating **₹{rev_cat.iloc[0]['Total_Spent']:,.2f}** in revenue.
        - **Customer Base**: The core of our customer demographic resides at a median age of **{median_age:.0f} years**, suggesting strong purchasing power in mid-career professionals.
        - **Ticket Analysis**: The typical customer spend is **₹{avg_spend:,.2f}** per transaction, which can serve as a baseline for future bundling or cross-selling initiatives.
        - **Pipeline Integrity**: Running the data cleaning pipeline resolved invalid ages, outliers, and duplicates, ensuring that decisions are based on accurate data.
        """)

elif page == "📊 Interactive Analysis":
    st.title("📊 Deep-Dive Exploratory Data Analysis")
    
    if not (raw_exists and clean_exists):
        st.warning("⚠️ No data available. Please generate and clean data inside **⚙️ Data Cleaning Center**.")
    else:
        df = pd.read_csv(clean_path)
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        
        # Interactive Filters
        st.markdown("### 🎛️ Dashboard Filters")
        c1, c2, c3 = st.columns(3)
        
        categories = ['All'] + list(df['Product_Category'].unique())
        selected_cat = c1.selectbox("Product Category:", categories)
        
        genders = ['All'] + list(df['Customer_Gender'].unique())
        selected_gender = c2.selectbox("Customer Gender:", genders)
        
        payment_methods = ['All'] + list(df['Payment_Method'].unique())
        selected_pay = c3.selectbox("Payment Method:", payment_methods)
        
        # Apply filters
        df_filtered = df.copy()
        if selected_cat != 'All':
            df_filtered = df_filtered[df_filtered['Product_Category'] == selected_cat]
        if selected_gender != 'All':
            df_filtered = df_filtered[df_filtered['Customer_Gender'] == selected_gender]
        if selected_pay != 'All':
            df_filtered = df_filtered[df_filtered['Payment_Method'] == selected_pay]
            
        # Display KPIs for filtered data
        f_rev = df_filtered['Total_Spent'].sum()
        f_cnt = len(df_filtered)
        f_avg = df_filtered['Total_Spent'].mean() if f_cnt > 0 else 0.0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Filtered Revenue", f"₹{f_rev:,.2f}")
        col2.metric("Filtered Transactions", f"{f_cnt:,}")
        col3.metric("Filtered Avg Spend", f"₹{f_avg:,.2f}")
        
        st.markdown("---")
        
        if f_cnt == 0:
            st.error("No data matches the selected filters. Try loosening your selection!")
        else:
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.markdown("#### 🎯 Spend Distribution by Age Group")
                fig, ax = plt.subplots(figsize=(6, 4))
                # Order groups logically
                age_order = ['18-25', '26-35', '36-50', '51-65', '65+']
                sns.boxplot(data=df_filtered, x='Age_Group', y='Total_Spent', order=[g for g in age_order if g in df_filtered['Age_Group'].unique()], palette="GnBu", ax=ax, hue='Age_Group', legend=False)
                ax.set_ylabel("Transaction Spend (INR ₹)")
                ax.set_xlabel("Age Groups")
                plt.tight_layout()
                st.pyplot(fig)
                
            with col_r:
                st.markdown("#### ⭐ Customer Satisfaction vs. Average Spend")
                fig, ax = plt.subplots(figsize=(6, 4))
                avg_sat_spend = df_filtered.groupby('Customer_Satisfaction')['Total_Spent'].mean().reset_index()
                sns.barplot(data=avg_sat_spend, x='Customer_Satisfaction', y='Total_Spent', color="#0D9488", ax=ax)
                ax.set_ylabel("Average Spend (INR ₹)")
                ax.set_xlabel("Satisfaction Rating")
                plt.tight_layout()
                st.pyplot(fig)
                
            st.markdown("---")
            
            # Pivot table heatmap
            st.markdown("#### 🗺️ Category and Payment Method Matrix (Total Revenue)")
            pivot_df = df_filtered.pivot_table(
                index='Product_Category',
                columns='Payment_Method',
                values='Total_Spent',
                aggfunc='sum'
            ).fillna(0)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(pivot_df, annot=True, fmt=",.0f", cmap="YlGnBu", cbar=True, ax=ax)
            ax.set_ylabel("Product Category")
            ax.set_xlabel("Payment Method")
            plt.tight_layout()
            st.pyplot(fig)

elif page == "🤖 Machine Learning Center":
    st.title("🤖 Predictive Modeling & ML Center")
    st.write("Train and evaluate machine learning models in real time using the cleaned transaction dataset. Explore classification and regression tasks.")
    
    if not (raw_exists and clean_exists):
        st.warning("⚠️ No cleaned data is available. Please generate and clean data inside the **⚙️ Data Cleaning Center** first!")
    else:
        df_clean = pd.read_csv(clean_path)
        
        # Select Task
        task_choice = st.selectbox(
            "Select Machine Learning Task:",
            ["Classification: Predict Customer Satisfaction", "Regression: Predict Total Spent"]
        )
        
        # Configure model parameters depending on task
        if "Classification" in task_choice:
            task_type = "classification"
            model_options = {
                "Logistic Regression": "logistic_regression",
                "Decision Tree Classifier": "decision_tree",
                "Random Forest Classifier": "random_forest"
            }
            selected_model_label = st.selectbox("Select Classification Algorithm:", list(model_options.keys()))
            model_name = model_options[selected_model_label]
            
            # Hyperparams
            hyperparams = {}
            with st.expander("Model Hyperparameters (Optional)", expanded=True):
                if model_name == "logistic_regression":
                    hyperparams["C"] = st.slider("Regularization Strength (C):", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
                elif model_name == "decision_tree":
                    max_depth_val = st.slider("Max Depth of Tree (0 for Unlimited):", min_value=0, max_value=20, value=5, step=1)
                    hyperparams["max_depth"] = None if max_depth_val == 0 else max_depth_val
                    hyperparams["min_samples_split"] = st.slider("Min Samples Split:", min_value=2, max_value=10, value=2)
                elif model_name == "random_forest":
                    hyperparams["n_estimators"] = st.slider("Number of Trees:", min_value=10, max_value=200, value=100, step=10)
                    max_depth_val = st.slider("Max Depth of Trees (0 for Unlimited):", min_value=0, max_value=20, value=5, step=1)
                    hyperparams["max_depth"] = None if max_depth_val == 0 else max_depth_val
                    
        else: # Regression
            task_type = "regression"
            model_options = {
                "Linear Regression": "linear_regression",
                "Decision Tree Regressor": "decision_tree",
                "Random Forest Regressor": "random_forest"
            }
            selected_model_label = st.selectbox("Select Regression Algorithm:", list(model_options.keys()))
            model_name = model_options[selected_model_label]
            
            # Hyperparams
            hyperparams = {}
            if model_name != "linear_regression":
                with st.expander("Model Hyperparameters (Optional)", expanded=True):
                    if model_name == "decision_tree":
                        max_depth_val = st.slider("Max Depth of Tree (0 for Unlimited):", min_value=0, max_value=20, value=5, step=1)
                        hyperparams["max_depth"] = None if max_depth_val == 0 else max_depth_val
                        hyperparams["min_samples_split"] = st.slider("Min Samples Split:", min_value=2, max_value=10, value=2)
                    elif model_name == "random_forest":
                        hyperparams["n_estimators"] = st.slider("Number of Trees:", min_value=10, max_value=200, value=100, step=10)
                        max_depth_val = st.slider("Max Depth of Trees (0 for Unlimited):", min_value=0, max_value=20, value=5, step=1)
                        hyperparams["max_depth"] = None if max_depth_val == 0 else max_depth_val

        # Setup feature details dynamically
        st.markdown("### 🛠️ Feature Engineering & Variables")
        
        # Let's inspect features to be used
        X_preview, y_preview, feat_preview = prepare_ml_data(df_clean, task_type=task_type)
        
        st.info(f"**Target Variable ($y$):** `{y_preview.name}`  \n**Features ($X$):** {', '.join([f'`{col}`' for col in feat_preview])}")
        
        test_size = st.slider("Test Split Size (%):", min_value=10, max_value=50, value=20, step=5) / 100.0
        
        # Train Button
        if st.button("🚀 Train & Evaluate Model"):
            with st.spinner("Preparing data, running train-test split, and fitting pipeline..."):
                X, y, feature_names = prepare_ml_data(df_clean, task_type=task_type)
                
                # Run train/eval
                results = train_and_evaluate(
                    X=X, y=y, model_name=model_name, task_type=task_type,
                    test_size=test_size, hyperparams=hyperparams
                )
                
                st.success(f"Successfully trained {selected_model_label}!")
                
                # Display Metrics in Cards
                st.subheader("📊 Performance Metrics")
                
                if task_type == "classification":
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy Score", f"{results['accuracy']:.2%}")
                    c2.metric("Precision Score", f"{results['precision']:.2%}")
                    c3.metric("Recall Score", f"{results['recall']:.2%}")
                    c4.metric("F1-Score", f"{results['f1']:.2%}")
                    
                    st.markdown("---")
                    st.subheader("📈 Classification Visualizations")
                    
                    # Columns for plots
                    col_l, col_r = st.columns(2)
                    
                    with col_l:
                        fig_cm = get_confusion_matrix_plot(results['y_test'], results['y_pred'])
                        st.pyplot(fig_cm)
                        
                    with col_r:
                        if "y_prob" in results:
                            fig_roc = get_roc_curve_plot(results['y_test'], results['y_prob'])
                            st.pyplot(fig_roc)
                        else:
                            st.warning("ROC Curve could not be computed (model does not support probabilities).")
                            
                else: # regression
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² Score (Variance Explained)", f"{results['r2']:.4f}")
                    c2.metric("Mean Absolute Error (MAE)", f"₹{results['mae']:.2f}")
                    c3.metric("Root Mean Squared Error (RMSE)", f"₹{results['rmse']:.2f}")
                    
                    st.markdown("---")
                    st.subheader("📈 Regression Visualizations")
                    
                    col_l, col_r = st.columns(2)
                    
                    with col_l:
                        fig_fit = get_actual_vs_predicted_plot(results['y_test'], results['y_pred'])
                        st.pyplot(fig_fit)
                        
                    with col_r:
                        fig_res = get_residuals_plot(results['y_test'], results['y_pred'])
                        st.pyplot(fig_res)
                        
                # Explain Pipeline
                with st.expander("🔍 Pipeline Details & Architecture", expanded=False):
                    st.markdown(f"""
                    - **Model Class**: `{type(results['pipeline'].named_steps['model']).__name__}`
                    - **Numeric Scaler**: `StandardScaler()` applied to {results['numeric_features']}
                    - **Categorical Encoder**: `OneHotEncoder(handle_unknown='ignore')` applied to {results['categorical_features']}
                    - **Total Training Instances**: {len(results['y_train'])}
                    - **Total Testing Instances**: {len(results['y_test'])}
                    """)

elif page == "📂 Dataset Explorer":
    st.title("📂 Dataset Explorer & Export")
    
    if not (raw_exists and clean_exists):
        st.warning("⚠️ No data available. Please generate and clean data inside **⚙️ Data Cleaning Center**.")
    else:
        df_raw = pd.read_csv(raw_path)
        df_clean = pd.read_csv(clean_path)
        
        tab1, tab2 = st.tabs(["✅ Cleaned Dataset", "🛑 Raw Messy Dataset"])
        
        with tab1:
            st.write(f"Showing cleaned and processed records. Total Rows: **{len(df_clean)}**")
            st.dataframe(df_clean, use_container_width=True)
            
            # Download button
            csv = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="cleaned_sales_data.csv",
                mime="text/csv"
            )
            
        with tab2:
            st.write(f"Showing raw uncleaned records. Total Rows: **{len(df_raw)}**")
            st.dataframe(df_raw, use_container_width=True)
