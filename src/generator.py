import os
import csv
import random
from datetime import datetime, timedelta

def generate_messy_data(output_path, num_records=250):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty']
    genders = ['Male', 'Female', 'Other']
    payment_methods = ['Credit Card', 'GPay', 'Debit Card', 'Cash']
    
    # Inconsistent representations for randomness
    gender_variations = {
        'Male': ['Male', 'M', 'male', 'm', 'Mlae'],
        'Female': ['Female', 'F', 'female', 'f', 'Femal'],
        'Other': ['Other', 'O', 'other']
    }
    
    category_variations = {
        'Electronics': ['Electronics', 'Elec', 'electronic', 'ELECTRONICS'],
        'Clothing': ['Clothing', 'Cloth', 'clothing', 'CLOTHING'],
        'Home & Kitchen': ['Home & Kitchen', 'Home', 'home & kitchen', 'Home & Kitch'],
        'Books': ['Books', 'Book', 'books'],
        'Beauty': ['Beauty', 'beauty', 'BEAUTY']
    }
    
    payment_variations = {
        'Credit Card': ['Credit Card', 'credit card', 'CREDIT CARD', 'CC'],
        'GPay': ['GPay', 'gpay', 'Gpay', 'Google Pay', 'google pay'],
        'Debit Card': ['Debit Card', 'debit card', 'DC'],
        'Cash': ['Cash', 'cash', 'CASH']
    }

    # Set random seed for reproducibility
    random.seed(101)
    
    records = []
    
    # Generate base transaction records
    for i in range(1, num_records + 1):
        tx_id = f"TX{100000 + i}"
        cust_id = f"CUST{1000 + random.randint(1, 200)}"
        
        # 1. Transaction Date
        # Generate a baseline date
        base_date = datetime(2025, 1, 1) + timedelta(
            days=random.randint(0, 500),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        date_rand = random.random()
        if date_rand < 0.70:
            # YYYY-MM-DD
            tx_date = base_date.strftime('%Y-%m-%d')
        elif date_rand < 0.80:
            # DD/MM/YYYY
            tx_date = base_date.strftime('%d/%m/%Y')
        elif date_rand < 0.90:
            # Month DD, YYYY
            tx_date = base_date.strftime('%B %d, %Y')
        elif date_rand < 0.95:
            # Missing
            tx_date = random.choice(['', 'N/A', 'unknown', 'nan'])
        else:
            # Out of bounds date or typo
            tx_date = f"2026-02-{random.randint(30, 31)}" # Invalid dates like 2026-02-30

        # 2. Customer Age
        age_rand = random.random()
        if age_rand < 0.85:
            cust_age = str(random.randint(18, 75))
        elif age_rand < 0.92:
            # Outliers (negative or extreme)
            cust_age = str(random.choice([-5, -12, 120, 150, 0]))
        else:
            # Missing
            cust_age = random.choice(['', 'NaN', 'N/A', 'NULL'])

        # 3. Customer Gender
        gender_rand = random.random()
        if gender_rand < 0.90:
            base_gender = random.choice(genders)
            cust_gender = random.choice(gender_variations[base_gender])
        else:
            cust_gender = random.choice(['', '?', 'N/A', 'unknown'])

        # 4. Product Category
        cat_rand = random.random()
        if cat_rand < 0.92:
            base_cat = random.choice(categories)
            prod_cat = random.choice(category_variations[base_cat])
        else:
            prod_cat = random.choice(['', 'nan', 'Other', 'unknown'])

        # 5. Quantity
        qty_rand = random.random()
        if qty_rand < 0.90:
            qty = random.randint(1, 5)
        elif qty_rand < 0.95:
            # Outliers
            qty = random.choice([-1, -3, 0, 100, 500])
        else:
            # Missing
            qty = random.choice(['', 'nan', 'N/A'])

        # 6. Unit Price
        # Normal range by category
        price_mapping = {
            'Electronics': (20.0, 1500.0),
            'Clothing': (10.0, 120.0),
            'Home & Kitchen': (15.0, 450.0),
            'Books': (5.0, 50.0),
            'Beauty': (8.0, 200.0)
        }
        
        # Find raw category to match base price mapping
        raw_cat = 'Electronics'
        for k, v in category_variations.items():
            if prod_cat in v:
                raw_cat = k
                break
        
        min_p, max_p = price_mapping.get(raw_cat, (10.0, 100.0))
        price_val = round(random.uniform(min_p, max_p), 2)
        
        price_rand = random.random()
        if price_rand < 0.90:
            unit_price = str(price_val)
        elif price_rand < 0.96:
            # Outliers
            unit_price = str(random.choice([-10.0, 0.0, 9999.99]))
        else:
            # Missing
            unit_price = random.choice(['', 'nan', 'NULL'])

        # 7. Total Spent (sometimes missing or incorrect)
        total_rand = random.random()
        try:
            q_val = int(qty)
            p_val = float(unit_price)
            calc_total = round(q_val * p_val, 2)
        except ValueError:
            calc_total = 0.0
            
        if total_rand < 0.80:
            total_spent = str(calc_total)
        elif total_rand < 0.90:
            # Missing
            total_spent = random.choice(['', 'nan', 'N/A'])
        else:
            # Mathematically incorrect total
            total_spent = str(round(calc_total * random.choice([0.5, 1.5, 2.0]), 2))

        # 8. Payment Method
        pay_rand = random.random()
        if pay_rand < 0.92:
            base_pay = random.choice(payment_methods)
            pay_method = random.choice(payment_variations[base_pay])
        else:
            pay_method = random.choice(['', 'unknown', 'N/A'])

        # 9. Customer Satisfaction
        satisfy_rand = random.random()
        if satisfy_rand < 0.88:
            satisfaction = str(random.randint(1, 5))
        elif satisfy_rand < 0.94:
            # Out of bounds
            satisfaction = str(random.choice([0, -1, 6, 10]))
        else:
            satisfaction = random.choice(['', 'nan', 'unknown'])

        records.append([
            tx_id, cust_id, tx_date, cust_age, cust_gender, 
            prod_cat, qty, unit_price, total_spent, pay_method, satisfaction
        ])
        
    # Inject Duplicates (exact duplicates)
    num_exact_dupes = 15
    for _ in range(num_exact_dupes):
        dup_record = random.choice(records).copy()
        records.append(dup_record)
        
    # Inject Duplicates with conflicting data for the same Transaction ID
    num_conflict_dupes = 10
    for _ in range(num_conflict_dupes):
        orig = random.choice(records)
        dup = orig.copy()
        # Change total_spent, age or price slightly
        dup[3] = str(random.randint(18, 75)) # change age
        dup[8] = str(float(dup[8]) + 10.0) if dup[8] and dup[8].replace('.','',1).isdigit() else "50.0"
        records.append(dup)

    # Shuffle records
    random.shuffle(records)
    
    # Write to CSV
    headers = [
        'Transaction_ID', 'Customer_ID', 'Transaction_Date', 'Customer_Age', 
        'Customer_Gender', 'Product_Category', 'Quantity', 'Unit_Price', 
        'Total_Spent', 'Payment_Method', 'Customer_Satisfaction'
    ]
    
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records)
        
    print(f"Dataset generated successfully! Total rows: {len(records)} (including duplicates)")
    print(f"Saved raw dataset to: {output_path}")

if __name__ == '__main__':
    # Set default path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_out = os.path.join(project_root, 'data', 'raw_sales_data.csv')
    generate_messy_data(default_out)
