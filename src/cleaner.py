import os
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, raw_path, clean_path):
        self.raw_path = raw_path
        self.clean_path = clean_path
        self.stats = {}

    def load_data(self):
        """Load the raw CSV dataset."""
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Raw data file not found at: {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        self.stats['initial_rows'] = len(df)
        self.stats['initial_cols'] = len(df.columns)
        return df

    def clean(self):
        """Run the full cleaning pipeline and return the cleaned DataFrame."""
        print("Starting data cleaning pipeline...")
        df = self.load_data()
        
        # 1. Handle Duplicates
        initial_len = len(df)
        # Drop exact duplicates
        df.drop_duplicates(inplace=True)
        exact_dupes_removed = initial_len - len(df)
        self.stats['exact_duplicates_removed'] = exact_dupes_removed
        
        # Drop conflicting duplicates based on Transaction_ID (keep first)
        pre_id_dup_len = len(df)
        df.drop_duplicates(subset=['Transaction_ID'], keep='first', inplace=True)
        id_dupes_removed = pre_id_dup_len - len(df)
        self.stats['id_duplicates_removed'] = id_dupes_removed
        self.stats['total_duplicates_removed'] = exact_dupes_removed + id_dupes_removed

        # 2. Standardize Categorical Data
        # Drop rows with missing or invalid Transaction_ID (should be non-null and unique)
        df = df[df['Transaction_ID'].notna()]
        
        # Clean Gender
        df['Customer_Gender'] = df['Customer_Gender'].astype(str).str.strip().str.lower()
        gender_map = {
            'm': 'Male', 'male': 'Male', 'mlae': 'Male',
            'f': 'Female', 'female': 'Female', 'femal': 'Female',
            'o': 'Other', 'other': 'Other'
        }
        df['Customer_Gender'] = df['Customer_Gender'].map(gender_map).fillna('Unknown')
        
        # Clean Product Category
        df['Product_Category'] = df['Product_Category'].astype(str).str.strip().str.lower()
        category_map = {
            'electronics': 'Electronics', 'elec': 'Electronics', 'electronic': 'Electronics',
            'clothing': 'Clothing', 'cloth': 'Clothing',
            'home & kitchen': 'Home & Kitchen', 'home': 'Home & Kitchen', 'home & kitch': 'Home & Kitchen',
            'books': 'Books', 'book': 'Books',
            'beauty': 'Beauty'
        }
        df['Product_Category'] = df['Product_Category'].map(category_map).fillna('Unknown')

        # Clean Payment Method
        df['Payment_Method'] = df['Payment_Method'].astype(str).str.strip().str.lower()
        payment_map = {
            'credit card': 'Credit Card', 'cc': 'Credit Card',
            'gpay': 'GPay', 'google pay': 'GPay',
            'debit card': 'Debit Card', 'dc': 'Debit Card',
            'cash': 'Cash'
        }
        df['Payment_Method'] = df['Payment_Method'].map(payment_map).fillna('Unknown')

        # 3. Clean and Unify Dates
        # Coerce errors to NaT, standardizing various formats
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce', format='mixed')
        
        # Count missing dates
        missing_dates = df['Transaction_Date'].isna().sum()
        self.stats['missing_dates_handled'] = int(missing_dates)
        
        # Impute missing dates using forward fill, then backward fill (or drop if preferred)
        df['Transaction_Date'] = df['Transaction_Date'].ffill().bfill()
        
        # Extract Month-Year for trends
        df['Month_Year'] = df['Transaction_Date'].dt.to_period('M')

        # 4. Clean Numerical Fields & Manage Outliers
        
        # Clean Customer Age
        df['Customer_Age'] = pd.to_numeric(df['Customer_Age'], errors='coerce')
        missing_ages = df['Customer_Age'].isna().sum()
        self.stats['missing_ages_imputed'] = int(missing_ages)
        
        # Median age calculations for imputation
        median_age = df['Customer_Age'].median()
        if pd.isna(median_age):
            median_age = 35.0  # fallback
        df['Customer_Age'] = df['Customer_Age'].fillna(median_age)
        
        # Handle outliers in age (age < 18 or age > 90)
        age_outliers = ((df['Customer_Age'] < 18) | (df['Customer_Age'] > 90)).sum()
        self.stats['age_outliers_corrected'] = int(age_outliers)
        df.loc[(df['Customer_Age'] < 18) | (df['Customer_Age'] > 90), 'Customer_Age'] = median_age

        # Clean Quantity
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        missing_qty = df['Quantity'].isna().sum()
        self.stats['missing_quantity_imputed'] = int(missing_qty)
        df['Quantity'] = df['Quantity'].fillna(1.0) # default quantity is 1
        
        # Handle quantity outliers (quantity <= 0 or quantity > 10)
        # Cap at 10 for extreme quantities (or cap at 99th percentile, which here is capped to a logical value)
        qty_outliers = ((df['Quantity'] <= 0) | (df['Quantity'] > 10)).sum()
        self.stats['quantity_outliers_corrected'] = int(qty_outliers)
        df.loc[df['Quantity'] <= 0, 'Quantity'] = 1.0
        df.loc[df['Quantity'] > 10, 'Quantity'] = 5.0 # Replace extreme quantity outliers with a typical value like 5

        # Clean Unit Price
        df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce')
        missing_prices = df['Unit_Price'].isna().sum()
        self.stats['missing_prices_imputed'] = int(missing_prices)
        
        # Impute missing price with the median price of the corresponding category
        category_medians = df.groupby('Product_Category')['Unit_Price'].median().to_dict()
        overall_median_price = df['Unit_Price'].median()
        
        def impute_price(row):
            if pd.isna(row['Unit_Price']) or row['Unit_Price'] <= 0 or row['Unit_Price'] > 5000.0:
                cat = row['Product_Category']
                return category_medians.get(cat, overall_median_price)
            return row['Unit_Price']
            
        # Count outliers in price (price <= 0 or extreme outlier > 5000)
        price_outliers = ((df['Unit_Price'] <= 0) | (df['Unit_Price'] > 5000.0)).sum()
        self.stats['price_outliers_corrected'] = int(price_outliers)
        
        df['Unit_Price'] = df.apply(impute_price, axis=1)

        # 5. Recalculate Total Spent (mathematical check)
        df['Total_Spent'] = pd.to_numeric(df['Total_Spent'], errors='coerce')
        df['Calculated_Total'] = round(df['Quantity'] * df['Unit_Price'], 2)
        
        # Calculate how many values were incorrect or missing
        incorrect_totals = (~np.isclose(df['Total_Spent'].fillna(-1), df['Calculated_Total'])).sum()
        self.stats['incorrect_totals_corrected'] = int(incorrect_totals)
        
        # Set Total_Spent to Calculated_Total
        df['Total_Spent'] = df['Calculated_Total']
        df.drop(columns=['Calculated_Total'], inplace=True)

        # 6. Clean Customer Satisfaction
        df['Customer_Satisfaction'] = pd.to_numeric(df['Customer_Satisfaction'], errors='coerce')
        missing_satisfaction = df['Customer_Satisfaction'].isna().sum()
        self.stats['missing_satisfaction_imputed'] = int(missing_satisfaction)
        
        # Mode satisfaction
        mode_satisfaction = df['Customer_Satisfaction'].mode()[0] if not df['Customer_Satisfaction'].mode().empty else 4.0
        df['Customer_Satisfaction'] = df['Customer_Satisfaction'].fillna(mode_satisfaction)
        
        # Handle outliers (satisfaction not in 1..5)
        satisfaction_outliers = (~df['Customer_Satisfaction'].isin([1, 2, 3, 4, 5])).sum()
        self.stats['satisfaction_outliers_corrected'] = int(satisfaction_outliers)
        df.loc[~df['Customer_Satisfaction'].isin([1, 2, 3, 4, 5]), 'Customer_Satisfaction'] = mode_satisfaction
        df['Customer_Satisfaction'] = df['Customer_Satisfaction'].astype(int)

        # 7. Additional Feature Engineering
        # Create Age Bin Groups
        bins = [0, 25, 35, 50, 65, 120]
        labels = ['18-25', '26-35', '36-50', '51-65', '65+']
        df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)
        
        self.stats['final_rows'] = len(df)
        self.stats['final_cols'] = len(df.columns)
        
        # Ensure directories exist and save
        os.makedirs(os.path.dirname(self.clean_path), exist_ok=True)
        df.to_csv(self.clean_path, index=False)
        print(f"Data cleaning complete! Cleaned dataset saved to: {self.clean_path}")
        self.log_stats()
        
        return df

    def log_stats(self):
        """Print and return cleaning statistics."""
        print("\n--- DATA CLEANING SUMMARY STATS ---")
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")
        print("-----------------------------------\n")
        return self.stats

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    raw_data_path = os.path.join(project_root, 'data', 'raw_sales_data.csv')
    cleaned_data_path = os.path.join(project_root, 'data', 'cleaned_sales_data.csv')
    
    cleaner = DataCleaner(raw_data_path, cleaned_data_path)
    cleaner.clean()
