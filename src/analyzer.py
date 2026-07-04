import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data_path, plots_dir):
        self.data_path = data_path
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Configure plotting styles
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
        # Sleek dark-blue/teal corporate color palette
        self.primary_color = "#1E3A8A"  # Dark Blue
        self.secondary_color = "#0D9488" # Teal
        self.accent_color = "#F59E0B"    # Amber/Yellow
        self.neutral_dark = "#1F2937"    # Dark grey

    def load_data(self):
        """Load the cleaned data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Cleaned data file not found at: {self.data_path}")
        return pd.read_csv(self.data_path)

    def run_analysis(self):
        """Generate and save all plots."""
        print("Running Exploratory Data Analysis (EDA)...")
        df = self.load_data()
        
        # Ensure correct datetime parsing
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        
        self.plot_monthly_sales_trend(df)
        self.plot_age_distribution(df)
        self.plot_revenue_by_category(df)
        self.plot_spend_vs_satisfaction(df)
        
        print(f"All plots successfully generated and saved to: {self.plots_dir}")

    def plot_monthly_sales_trend(self, df):
        """Plot total sales trend over time (monthly)."""
        # Group by Month-Year (using Period conversion then string for sorting/plotting)
        monthly_df = df.copy()
        monthly_df['Month_Year'] = monthly_df['Transaction_Date'].dt.to_period('M').dt.to_timestamp()
        trend = monthly_df.groupby('Month_Year')['Total_Spent'].sum().reset_index()
        
        plt.figure()
        sns.lineplot(
            data=trend, 
            x='Month_Year', 
            y='Total_Spent', 
            marker='o', 
            color=self.secondary_color, 
            linewidth=2.5
        )
        
        plt.title("Monthly Total Sales Trend (2025 - 2026)", pad=15, color=self.neutral_dark, fontweight='bold')
        plt.xlabel("Month", labelpad=10)
        plt.ylabel("Total Revenue (INR ₹)", labelpad=10)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, 'sales_trend.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_age_distribution(self, df):
        """Plot customer age distribution histogram."""
        plt.figure()
        sns.histplot(
            data=df, 
            x='Customer_Age', 
            kde=True, 
            color=self.primary_color, 
            edgecolor='white',
            bins=20
        )
        
        median_age = df['Customer_Age'].median()
        plt.axvline(
            median_age, 
            color=self.accent_color, 
            linestyle='--', 
            linewidth=2.5,
            label=f'Median Age: {median_age:.1f}'
        )
        
        plt.title("Customer Age Distribution", pad=15, color=self.neutral_dark, fontweight='bold')
        plt.xlabel("Age", labelpad=10)
        plt.ylabel("Count", labelpad=10)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, 'age_distribution.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_revenue_by_category(self, df):
        """Plot total revenue by product category."""
        revenue_cat = df.groupby('Product_Category')['Total_Spent'].sum().reset_index()
        revenue_cat = revenue_cat.sort_values(by='Total_Spent', ascending=False)
        
        plt.figure()
        sns.barplot(
            data=revenue_cat, 
            x='Product_Category', 
            y='Total_Spent', 
            palette="viridis",
            hue='Product_Category',
            legend=False
        )
        
        # Add labels on top of the bars
        for index, row in revenue_cat.iterrows():
            plt.text(
                index, 
                row['Total_Spent'] + (row['Total_Spent'] * 0.01), 
                                f"₹{row['Total_Spent']:,.0f}", 
                color='black', 
                ha="center", 
                va="bottom",
                fontsize=9
            )
            
        plt.title("Total Revenue by Product Category", pad=15, color=self.neutral_dark, fontweight='bold')
        plt.xlabel("Product Category", labelpad=10)
        plt.ylabel("Total Revenue (INR ₹)", labelpad=10)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, 'revenue_by_category.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_spend_vs_satisfaction(self, df):
        """Plot average spent by customer satisfaction score."""
        satisfy_spend = df.groupby('Customer_Satisfaction')['Total_Spent'].mean().reset_index()
        
        plt.figure()
        sns.barplot(
            data=satisfy_spend, 
            x='Customer_Satisfaction', 
            y='Total_Spent', 
            color=self.secondary_color
        )
        
        plt.title("Average Transaction Value by Customer Satisfaction", pad=15, color=self.neutral_dark, fontweight='bold')
        plt.xlabel("Customer Satisfaction Rating (1-5)", labelpad=10)
        plt.ylabel("Average Spend (INR ₹)", labelpad=10)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.2f}"))
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, 'satisfaction_vs_spend.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    cleaned_data_path = os.path.join(project_root, 'data', 'cleaned_sales_data.csv')
    plots_directory = os.path.join(project_root, 'plots')
    
    analyzer = DataAnalyzer(cleaned_data_path, plots_directory)
    analyzer.run_analysis()
